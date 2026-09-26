"""Ingest nuScenes camera keyframes into LanceDB with Ray Data.

The pipeline has three stages:

1. Driver: read the JSON tables into one metadata record per camera keyframe.
2. CPU tasks: read each JPEG, keep its original bytes, make a thumbnail, and
   resize it to the model's input size.
3. GPU actors: embed the resized pixels with SigLIP 2, one actor per GPU.

lancedb-ray then writes every block as a Lance fragment in parallel and commits
them all in one transaction.

Usage:
    uv run mcap-lancedb-ingest --dataroot data/nuscenes --version v1.0-mini
"""

import argparse
import io
import logging
import math
import warnings
from collections.abc import Sequence
from datetime import timedelta
from functools import partial
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
import ray
import transformers
from lancedb.index import Bitmap, BTree, IvfPq
from lancedb_ray import write_lancedb
from PIL import Image

from mcap_lancedb import (
    CAMERA_CHANNELS,
    DEFAULT_DATAROOT,
    DEFAULT_DB,
    DEFAULT_MODEL,
    DEFAULT_VERSION,
    META_EMBEDDING_DIM,
    META_EMBEDDING_MODEL,
    TABLE_NAME,
    THUMBNAIL_LONG_EDGE,
    THUMBNAIL_QUALITY,
)
from mcap_lancedb.embed import ImageInputSpec, SiglipEncoder, image_input_spec
from mcap_lancedb.nuscenes_io import build_frame_records
from mcap_lancedb.schema import (
    EMBEDDING_COLUMN,
    IMAGE_COLUMN,
    MODEL_INPUT_COLUMN,
    THUMBNAIL_COLUMN,
    embedding_field,
    frame_schema,
    metadata_schema,
)

logger = logging.getLogger(__name__)

# Below this many rows an exact (brute-force) search is faster than IVF_PQ and
# has perfect recall, so --vector-index auto skips the index.
VECTOR_INDEX_MIN_ROWS = 100_000

# Rows per block for the CPU stage. Each row carries its original JPEG and its
# model-sized pixels, so 128 rows is a few tens of MB.
ROWS_PER_BLOCK = 128
DECODE_BATCH_SIZE = 16

# Settings Ray workers need too, not just the driver.
WORKER_ENV = {"TOKENIZERS_PARALLELISM": "false", "TRANSFORMERS_VERBOSITY": "error"}


def decode_frames(
    batch: pa.Table, dataroot: str, input_spec: ImageInputSpec
) -> pa.Table:
    """CPU stage: read JPEGs and add the thumbnail, original and model input.

    Args:
        batch: Metadata rows; ``source_path`` is relative to ``dataroot``.
        dataroot: Absolute path to the nuScenes root, visible to every worker.
        input_spec: The embedding model's input size and resampling filter.

    Returns:
        The batch with ``thumbnail``, ``image`` and the transient model input
        column appended.
    """
    root = Path(dataroot)
    thumbnails: list[bytes] = []
    originals: list[bytes] = []
    model_inputs: list[bytes] = []
    for relative_path in batch.column("source_path").to_pylist():
        original = (root / relative_path).read_bytes()
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
        batch.append_column(THUMBNAIL_COLUMN, pa.array(thumbnails, pa.binary()))
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
            Rows matching ``frame_schema``, with the model id and dimension
            added to the schema metadata the batch already carries.
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


def embedding_actor_plan(device: str, cluster_gpus: int) -> tuple[int, int]:
    """Decide how many embedding actors to run and how many GPUs each gets.

    Args:
        device: The ``--device`` flag: ``auto``, ``cuda``, ``mps`` or ``cpu``.
        cluster_gpus: GPUs Ray can see across the cluster. Ray 2.58 also
            reports an Apple silicon GPU here, as one GPU per Mac.

    Returns:
        ``(actors, gpus_per_actor)``: one actor per GPU when Ray sees any,
        otherwise a single actor that asks for none.

    Raises:
        SystemExit: If CUDA was requested but Ray sees no GPUs.
    """
    if device != "cpu" and cluster_gpus > 0:
        return cluster_gpus, 1
    if device == "cuda":
        msg = "--device cuda was requested, but Ray sees no GPUs."
        raise SystemExit(msg)
    return 1, 0


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments to parse. ``None`` reads ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="mcap-lancedb-ingest",
        description="Embed nuScenes camera keyframes into a LanceDB table.",
    )
    parser.add_argument("--dataroot", type=Path, default=DEFAULT_DATAROOT)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--channels", nargs="+", choices=CAMERA_CHANNELS, default=list(CAMERA_CHANNELS)
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Frames per embedding batch."
    )
    parser.add_argument("--limit", type=int, help="Ingest only the first N frames.")
    parser.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto"
    )
    parser.add_argument(
        "--vector-index", choices=["auto", "always", "never"], default="auto"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the ingest pipeline.

    Args:
        argv: Command-line arguments. ``None`` reads ``sys.argv``.
    """
    args = parse_args(argv)
    configure_logging()
    # Workers run in their own directories, so every path they see is absolute.
    # On a multi-node cluster both must be on shared storage.
    dataroot = args.dataroot.resolve()
    db_uri = args.db.resolve()

    tables = dataroot / args.version
    if not (tables / "sample_data.json").is_file():
        msg = (
            f"No nuScenes {args.version} tables in {tables}. Download the dataset "
            "(see the README) or point --dataroot at the folder that contains "
            f"{args.version}/ and samples/."
        )
        raise SystemExit(msg)
    records = build_frame_records(dataroot, args.version, args.channels, args.limit)
    if not records:
        msg = f"No camera keyframes found under {dataroot / args.version}."
        raise SystemExit(msg)
    logger.info("Found %d camera keyframes in %s", len(records), args.version)
    metadata = pa.Table.from_pylist(records, schema=metadata_schema(args.version))

    # Joins the cluster at RAY_ADDRESS when set, otherwise starts a local one.
    # Under `uv run`, Ray ships the project directory (minus .gitignore'd
    # paths such as data/) and rebuilds its environment on every node.
    ray.init(ignore_reinit_error=True, runtime_env={"env_vars": WORKER_ENV})
    resources = ray.cluster_resources()
    actors, gpus_per_actor = embedding_actor_plan(
        args.device, int(resources.get("GPU", 0))
    )
    logger.info(
        "Embedding with %s on %d actor(s), %d GPU(s) each",
        args.model,
        actors,
        gpus_per_actor,
    )

    # from_arrow yields a single block; split it so decoding runs in parallel.
    blocks = max(int(resources.get("CPU", 1)), math.ceil(len(records) / ROWS_PER_BLOCK))
    frames = (
        ray.data.from_arrow(metadata)
        .repartition(blocks)
        .map_batches(
            partial(
                decode_frames,
                dataroot=str(dataroot),
                input_spec=image_input_spec(args.model),
            ),
            batch_format="pyarrow",
            batch_size=DECODE_BATCH_SIZE,
        )
        .map_batches(
            EmbedFrames,
            batch_format="pyarrow",
            batch_size=args.batch_size,
            fn_constructor_kwargs={
                "model_id": args.model,
                "device": None if args.device == "auto" else args.device,
            },
            compute=ray.data.ActorPoolStrategy(size=actors),
            num_gpus=gpus_per_actor,
        )
    )

    db = lancedb.connect(db_uri)
    replacing = TABLE_NAME in db.list_tables().tables
    if replacing:
        logger.info("Replacing the existing %s table in %s", TABLE_NAME, db_uri)
    write_lancedb(frames, TABLE_NAME, uri=str(db_uri), mode="overwrite")

    table = db.open_table(TABLE_NAME)
    if replacing:
        # The replaced version's files would otherwise double the disk footprint.
        table.to_lance().cleanup_old_versions(older_than=timedelta(0))
    create_indexes(table, args.vector_index)
    logger.info("Wrote %d frames to %s/%s", table.count_rows(), db_uri, TABLE_NAME)


if __name__ == "__main__":
    main()
