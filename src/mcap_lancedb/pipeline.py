"""The Ray pipeline from per-scene MCAP files to the LanceDB ``frames`` table.

Shared by the local CLI (``mcap-lancedb-ingest``) and the cluster job
(``scripts/ingest_on_cluster.py``). They differ only in where the data lives,
how Ray starts, and how many embedding actors run.

1. Metadata: one Ray task per MCAP file reads the scene info, ego poses,
   calibrations and annotations into one row per camera keyframe.
2. ``ray.data.read_mcap`` reads the camera image topics, one file per task.
3. CPU tasks match images to keyframe metadata (dropping the sweeps between
   keyframes) and decode each JPEG into a thumbnail and the model input.
4. GPU actors embed the model input with SigLIP 2.
5. lancedb-ray writes the fragments in parallel and commits them together.

An existing table is replaced only once the minimum pool of embedding actors
fits on the cluster and the model loads on it, so a wrong device, model id or
cluster size leaves the old table in place.

The metadata is small (about 0.5 KB per frame), so it goes to every decode task
through the object store instead of being shuffled against the images.
"""

import functools
import io
import logging
import posixpath
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import ray
import torch
import transformers
from lancedb.index import Bitmap, BTree, IvfPq
from lancedb_ray import write_lancedb
from mcap.records import Schema
from mcap_protobuf.decoder import DecoderFactory
from PIL import Image
from ray.exceptions import GetTimeoutError, RayError
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from mcap_lancedb import (
    CAMERA_CHANNELS,
    DEFAULT_MODEL,
    META_EMBEDDING_DIM,
    META_EMBEDDING_MODEL,
    TABLE_NAME,
    THUMBNAIL_LONG_EDGE,
    THUMBNAIL_QUALITY,
)
from mcap_lancedb.embed import (
    ImageInputSpec,
    SiglipEncoder,
    image_input_spec,
    resolve_device,
)
from mcap_lancedb.mcap_io import channel_of, image_topic, read_scene
from mcap_lancedb.schema import (
    EMBEDDING_COLUMN,
    IMAGE_COLUMN,
    METADATA_SCHEMA,
    MODEL_INPUT_COLUMN,
    THUMBNAIL_COLUMN,
    embedding_field,
    frame_schema,
)

logger = logging.getLogger(__name__)

# Below this many rows an exact (brute-force) search is faster than IVF_PQ and
# has perfect recall, so --vector-index auto skips the index.
VECTOR_INDEX_MIN_ROWS = 100_000

# How long to wait for the minimum pool of embedding actors to fit, which
# includes an autoscaler bringing up GPU nodes.
DEFAULT_STARTUP_TIMEOUT_S = 600.0

# Settings Ray workers need too, not just the driver.
WORKER_ENV = {"TOKENIZERS_PARALLELISM": "false", "TRANSFORMERS_VERBOSITY": "error"}

# An image message is a frame's when all three match.
JOIN_KEYS = ["source_path", "channel", "mcap_log_time"]


@dataclass(frozen=True)
class IngestConfig:
    """Where the data lives, and how to run the pipeline on it.

    Attributes:
        mcap_uri: Directory of per-scene MCAP files, as an absolute local path
            or a URI such as ``s3://bucket/nuscenes-mcap``.
        db_uri: LanceDB directory, as a local path or a URI.
        model_id: Hugging Face id of a fixed-resolution SigLIP checkpoint.
        channels: Camera channels to ingest.
        limit: Ingest only this many frames, in file, camera and time order.
        batch_size: Frames per embedding batch.
        device: Torch device for the embedding actors, or ``None`` to pick.
        embed_actors: Embedding actors: a fixed count, or ``(min, max)`` to let
            Ray Data autoscale the pool.
        gpus_per_actor: GPUs each embedding actor reserves.
        vector_index: ``auto``, ``always`` or ``never``.
        startup_timeout_s: How long to wait for the minimum pool of embedding
            actors to fit on the cluster before giving up.
    """

    mcap_uri: str
    db_uri: str
    model_id: str = DEFAULT_MODEL
    channels: tuple[str, ...] = CAMERA_CHANNELS
    limit: int | None = None
    batch_size: int = 32
    device: str | None = None
    embed_actors: int | tuple[int, int] = 1
    gpus_per_actor: float = 0
    vector_index: str = "auto"
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S


def configure_logging() -> None:
    """Log at INFO and silence noise that doesn't indicate a problem."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    transformers.logging.set_verbosity_error()
    # Ray starts workers with fork+exec. The exec means no Lance or LanceDB
    # state is inherited, so both libraries' fork-safety warnings are false
    # alarms here. Only those two messages are filtered.
    warnings.filterwarnings(
        "ignore", message="lance is not fork-safe", category=UserWarning
    )
    warnings.filterwarnings(
        "ignore",
        message="lancedb fork support is experimental",
        category=RuntimeWarning,
    )


def list_mcap_files(mcap_uri: str) -> tuple[pafs.FileSystem, list[str]]:
    """Find the scene files under a local path or object-store URI.

    Args:
        mcap_uri: Absolute local path or URI of the MCAP directory.

    Returns:
        The filesystem, and the ``.mcap`` files directly in the directory,
        sorted by name.
    """
    filesystem, root = pafs.FileSystem.from_uri(mcap_uri)
    listing = filesystem.get_file_info(pafs.FileSelector(root, allow_not_found=True))
    files = sorted(
        info.path
        for info in listing
        if info.type == pafs.FileType.File and info.path.endswith(".mcap")
    )
    return filesystem, files


def read_scene_table(
    filesystem: pafs.FileSystem, path: str, channels: tuple[str, ...]
) -> pa.Table:
    """Read one scene file's keyframe metadata as a table.

    Args:
        filesystem: Filesystem the file lives on.
        path: The scene's MCAP file on that filesystem.
        channels: Camera channels to include.

    Returns:
        One row per camera keyframe, matching ``METADATA_SCHEMA``.
    """
    with filesystem.open_input_file(path) as handle:
        records = read_scene(handle, posixpath.basename(path), channels)
    return pa.Table.from_pylist(records, schema=METADATA_SCHEMA)


_read_scene_task = ray.remote(read_scene_table)


def read_metadata(
    filesystem: pafs.FileSystem,
    files: list[str],
    channels: tuple[str, ...],
    limit: int | None = None,
) -> pa.Table:
    """Read every scene's keyframe metadata, one Ray task per file.

    Args:
        filesystem: Filesystem the files live on.
        files: MCAP files, in name order.
        channels: Camera channels to include.
        limit: Keep only the first this many frames.

    Returns:
        One row per camera keyframe, ordered by file, camera and time.
    """
    tables = ray.get([_read_scene_task.remote(filesystem, p, channels) for p in files])
    metadata = pa.concat_tables([METADATA_SCHEMA.empty_table(), *tables])
    return metadata if limit is None else metadata.slice(0, limit)


def check_one_file_per_scene(metadata: pa.Table) -> None:
    """Refuse a scene whose frames come from more than one MCAP file.

    They would be written twice under the same frame_ids, which dedup merges
    on and the viewer looks frames up by. nuscenes2mcap names each file after
    its scene, so only a renamed or copied file does this.

    Args:
        metadata: Output of ``read_metadata``.

    Raises:
        ValueError: If any scene appears in two or more files.
    """
    files = metadata.group_by("scene_name").aggregate(
        [("source_path", "count_distinct")]
    )
    repeated = files.filter(pc.field("source_path_count_distinct") > 1)
    if repeated.num_rows:
        names = ", ".join(sorted(repeated["scene_name"].to_pylist()))
        msg = f"Scenes in more than one MCAP file: {names}. Keep one file per scene."
        raise ValueError(msg)


@functools.cache
def _protobuf_decoder(name: str, encoding: str, data: bytes) -> Callable[[bytes], Any]:
    """Build a decoder from a message's protobuf schema, once per schema."""
    decoder = DecoderFactory().decoder_for(
        "protobuf", Schema(id=0, name=name, encoding=encoding, data=data)
    )
    if decoder is None:
        msg = (
            f"{name} messages use {encoding} schemas; only protobuf images, as "
            "nuscenes2mcap writes them, are supported."
        )
        raise ValueError(msg)
    return decoder


def decode_frames(
    batch: pa.Table,
    metadata: "pa.Table | ray.ObjectRef",
    input_spec: ImageInputSpec,
) -> pa.Table:
    """CPU stage: match image messages to keyframes and decode them.

    Args:
        batch: Rows from ``ray.data.read_mcap`` with ``include_paths=True``.
        metadata: Keyframe metadata from ``read_metadata``, or a reference to
            it in the object store.
        input_spec: The embedding model's input size and resampling filter.

    Returns:
        The metadata of every keyframe in the batch, with ``thumbnail``,
        ``image`` and the transient model input appended. Sweeps are dropped.
    """
    if isinstance(metadata, ray.ObjectRef):
        metadata = ray.get(metadata)
    messages = pa.table(
        {
            "source_path": [posixpath.basename(p) for p in batch["path"].to_pylist()],
            "channel": [channel_of(topic) for topic in batch["topic"].to_pylist()],
            "mcap_log_time": batch["log_time"].cast(pa.int64()),
            "_message": pa.array(np.arange(batch.num_rows)),
        }
    )
    rows = metadata.select(JOIN_KEYS).append_column(
        "_row", pa.array(np.arange(metadata.num_rows))
    )
    matched = messages.join(rows, keys=JOIN_KEYS, join_type="inner").sort_by("_row")
    frames = metadata.take(matched["_row"])
    images = batch.take(matched["_message"])

    thumbnails: list[bytes] = []
    originals: list[bytes] = []
    model_inputs: list[bytes] = []
    for data, name, encoding, schema in zip(
        images["data"].to_pylist(),
        images["schema_name"].to_pylist(),
        images["schema_encoding"].to_pylist(),
        images["schema_data"].to_pylist(),
        strict=True,
    ):
        original = _protobuf_decoder(name, encoding, schema)(data).data
        with Image.open(io.BytesIO(original)) as image:
            rgb = image.convert("RGB")
        model_inputs.append(input_spec.prepare(rgb).tobytes())

        rgb.thumbnail(
            (THUMBNAIL_LONG_EDGE, THUMBNAIL_LONG_EDGE), Image.Resampling.LANCZOS
        )
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=THUMBNAIL_QUALITY)
        thumbnails.append(buffer.getvalue())
        originals.append(original)

    model_input_type = pa.binary(input_spec.nbytes)
    return (
        frames.append_column(THUMBNAIL_COLUMN, pa.array(thumbnails, pa.binary()))
        .append_column(IMAGE_COLUMN, pa.array(originals, pa.large_binary()))
        .append_column(
            pa.field(MODEL_INPUT_COLUMN, model_input_type),
            pa.array(model_inputs, model_input_type),
        )
    )


class EmbedFrames:
    """GPU stage: a Ray actor that holds one SigLIP 2 encoder.

    Batches stay in pyarrow format end to end. A numpy batch would turn the
    embeddings into Ray's tensor extension type, which won't cast to the
    ``fixed_size_list`` the table stores.
    """

    def __init__(self, model_id: str, device: str | None) -> None:
        """Load the encoder once per actor.

        Args:
            model_id: Hugging Face model id.
            device: Explicit device, or ``None`` to pick the best available.
        """
        self.encoder = SiglipEncoder(model_id, device=device)
        self.schema = frame_schema(self.encoder.dim)
        self.metadata = {
            META_EMBEDDING_MODEL.encode(): model_id.encode(),
            META_EMBEDDING_DIM.encode(): str(self.encoder.dim).encode(),
        }

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Embed a batch and swap the transient pixels for the embedding.

        Args:
            batch: Output of ``decode_frames``.

        Returns:
            Rows matching ``frame_schema``, tagged with the model id and
            embedding dimension.
        """
        spec = self.encoder.input_spec
        packed = bytearray().join(batch.column(MODEL_INPUT_COLUMN).to_pylist())
        pixels = np.frombuffer(packed, dtype=np.uint8).reshape(
            batch.num_rows, spec.height, spec.width, 3
        )
        embeddings = self.encoder.encode_pixels(pixels)
        vectors = pa.FixedSizeListArray.from_arrays(
            pa.array(embeddings.ravel(), pa.float32()), self.encoder.dim
        )
        out = batch.drop_columns([MODEL_INPUT_COLUMN]).append_column(
            embedding_field(self.encoder.dim), vectors
        )
        metadata = {**(batch.schema.metadata or {}), **self.metadata}
        return out.cast(self.schema).replace_schema_metadata(metadata)


def frames_dataset(
    config: IngestConfig,
    filesystem: pafs.FileSystem,
    files: list[str],
    metadata: pa.Table,
) -> ray.data.Dataset:
    """Chain the read, decode and embed stages.

    Args:
        config: Pipeline settings.
        filesystem: Filesystem the MCAP files live on.
        files: MCAP files holding the frames in ``metadata``.
        metadata: Output of ``read_metadata``.

    Returns:
        A lazy dataset of rows matching ``frame_schema``.
    """
    if isinstance(config.embed_actors, tuple):
        pool = ray.data.ActorPoolStrategy(
            min_size=config.embed_actors[0], max_size=config.embed_actors[1]
        )
    else:
        pool = ray.data.ActorPoolStrategy(size=config.embed_actors)
    return (
        ray.data.read_mcap(
            files,
            topics={image_topic(channel) for channel in config.channels},
            filesystem=filesystem,
            include_paths=True,
        )
        # One batch per block, so each task builds its keyframe lookup once.
        .map_batches(
            functools.partial(
                decode_frames,
                metadata=ray.put(metadata),
                input_spec=image_input_spec(config.model_id),
            ),
            batch_format="pyarrow",
            batch_size=None,
        )
        .map_batches(
            EmbedFrames,
            batch_format="pyarrow",
            batch_size=config.batch_size,
            fn_constructor_kwargs={
                "model_id": config.model_id,
                "device": config.device,
            },
            compute=pool,
            num_gpus=config.gpus_per_actor,
        )
    )


def load_encoder_device(model_id: str, device: str | None) -> str:
    """Load the model the way an embedding actor will, and report its device.

    Args:
        model_id: Hugging Face model id.
        device: Explicit device, or ``None`` to pick the best available.

    Returns:
        The device the model loaded on, for example ``cuda``.
    """
    # A device this node doesn't have fails here, before any weights download.
    torch.empty(0, device=resolve_device(device))
    return str(SiglipEncoder(model_id, device=device).device)


# One call per worker process: the worker exits afterwards and frees its copy of
# the model before the embedding actors load theirs.
_load_encoder_task = ray.remote(max_calls=1)(load_encoder_device)


def check_embedding_actors(config: IngestConfig) -> str:
    """Make sure the embedding actors can start, before anything is replaced.

    Reserves the minimum actor pool's resources in a placement group, which
    also asks an autoscaling cluster for them, then loads the model once inside
    that reservation, on the actors' device.

    Args:
        config: Pipeline settings.

    Returns:
        The device the model loaded on.

    Raises:
        RuntimeError: If the pool doesn't fit within
            ``config.startup_timeout_s``, or the model doesn't load.
    """
    actors = config.embed_actors
    minimum = actors[0] if isinstance(actors, tuple) else actors
    # A GPU actor reserves only its GPUs; a CPU-only one needs at most a CPU.
    gpus = config.gpus_per_actor
    bundle = {"GPU": gpus} if gpus else {"CPU": 1.0}
    unchanged = f"The {TABLE_NAME} table in {config.db_uri} is unchanged."
    group = placement_group([bundle] * minimum)
    try:
        try:
            ray.get(group.ready(), timeout=config.startup_timeout_s)
        except GetTimeoutError:
            msg = (
                f"The cluster couldn't fit {minimum} embedding actor(s) needing "
                f"{bundle} each within {config.startup_timeout_s:g}s. {unchanged}"
            )
            raise RuntimeError(msg) from None
        load = _load_encoder_task.options(
            num_cpus=0 if gpus else 1,
            num_gpus=gpus,
            scheduling_strategy=PlacementGroupSchedulingStrategy(group),
        )
        try:
            return ray.get(load.remote(config.model_id, config.device))
        # RayError also covers a worker killed for running out of memory.
        except RayError as error:
            msg = f"{config.model_id} failed to load for embedding. {unchanged}"
            raise RuntimeError(msg) from error
    finally:
        remove_placement_group(group)


def create_indexes(table: lancedb.table.Table, vector_index: str) -> None:
    """Create the scalar indexes, and the vector index when it pays off.

    Args:
        table: The frames table.
        vector_index: ``auto`` builds IVF_PQ at ``VECTOR_INDEX_MIN_ROWS`` rows
            or more; ``always`` and ``never`` override that.
    """
    # frame_id is the dedup merge key and the viewer's point-lookup key.
    table.create_index("frame_id", config=BTree())
    table.create_index("channel", config=Bitmap())
    table.create_index("location", config=Bitmap())
    table.create_index("scene_name", config=BTree())

    rows = table.count_rows()
    if vector_index == "always" or (
        vector_index == "auto" and rows >= VECTOR_INDEX_MIN_ROWS
    ):
        logger.info("Building an IVF_PQ cosine index over %d embeddings", rows)
        table.create_index(EMBEDDING_COLUMN, config=IvfPq(distance_type="cosine"))
    else:
        logger.info("Skipping the vector index: %d rows search faster exactly", rows)


def run(config: IngestConfig) -> int:
    """Run the whole pipeline on an already-initialized Ray cluster.

    Args:
        config: Pipeline settings.

    Returns:
        The number of frames written.

    Raises:
        FileNotFoundError: If there are no MCAP files or no camera keyframes.
        ValueError: If a scene appears in more than one MCAP file.
        RuntimeError: If the embedding actors can't start. An existing table
            is left as it was.
    """
    filesystem, files = list_mcap_files(config.mcap_uri)
    if not files:
        msg = f"No .mcap files in {config.mcap_uri}"
        raise FileNotFoundError(msg)
    metadata = read_metadata(filesystem, files, config.channels, config.limit)
    if metadata.num_rows == 0:
        msg = f"No camera keyframes in the MCAP files under {config.mcap_uri}"
        raise FileNotFoundError(msg)
    check_one_file_per_scene(metadata)
    # With a limit, only read the files that hold the chosen frames.
    wanted = set(metadata["source_path"].to_pylist())
    files = [path for path in files if posixpath.basename(path) in wanted]
    logger.info(
        "Found %d camera keyframes in %d MCAP file(s)", metadata.num_rows, len(files)
    )

    # Everything up to here leaves an existing table alone; the drop below
    # doesn't, and the pipeline only runs once it's gone.
    device = check_embedding_actors(config)
    logger.info(
        "%s loads on %s, so the embedding actors can start", config.model_id, device
    )

    frames = frames_dataset(config, filesystem, files, metadata)
    db = lancedb.connect(config.db_uri)
    if TABLE_NAME in db.list_tables().tables:
        # lancedb-ray's overwrite mode can't change a table's schema ("Append
        # with different schema"), and a different model changes it.
        logger.info("Replacing the existing %s table in %s", TABLE_NAME, config.db_uri)
        db.drop_table(TABLE_NAME)
    # Not Dataset.write_lance: Ray 2.58's Lance sink passes
    # storage_options_provider to lance.fragment.write_fragments, which pylance 12
    # removed. lancedb-ray also keeps the field and schema metadata.
    write_lancedb(frames, TABLE_NAME, uri=config.db_uri, mode="create")

    table = db.open_table(TABLE_NAME)
    create_indexes(table, config.vector_index)
    written = table.count_rows()
    logger.info("Wrote %d frames to %s/%s", written, config.db_uri, TABLE_NAME)
    return written
