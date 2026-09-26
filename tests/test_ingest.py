"""Tests for the ingest pipeline: its stages, both entry points, and planning."""

import io
import runpy
from pathlib import Path

import foxglove
import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pytest
import ray
from conftest import SyntheticMcap
from mcap.reader import make_reader
from PIL import Image

from mcap_lancedb import (
    DEFAULT_MODEL,
    META_EMBEDDING_DIM,
    META_EMBEDDING_MODEL,
    TABLE_NAME,
)
from mcap_lancedb.embed import SiglipEncoder, image_input_spec
from mcap_lancedb.ingest import embedding_actor_plan, main
from mcap_lancedb.mcap_io import build_frame_records, image_topic, read_scene
from mcap_lancedb.pipeline import (
    EmbedFrames,
    IngestConfig,
    create_indexes,
    decode_frames,
    read_scene_table,
    run,
)
from mcap_lancedb.schema import (
    EMBEDDING_COLUMN,
    IMAGE_COLUMN,
    METADATA_SCHEMA,
    MODEL_INPUT_COLUMN,
    THUMBNAIL_COLUMN,
    frame_schema,
)

CLUSTER_SCRIPT = Path(__file__).parents[1] / "scripts" / "ingest_on_cluster.py"

SCALAR_INDEXES = {
    "frame_id": "BTree",
    "channel": "Bitmap",
    "location": "Bitmap",
    "scene_name": "BTree",
}


def index_types(table: lancedb.table.Table) -> dict[str, str]:
    """Map each indexed column to its index type."""
    return {index.columns[0]: index.index_type for index in table.list_indices()}


def image_messages(path: Path) -> pa.Table:
    """A scene's image messages, shaped like ``ray.data.read_mcap`` rows."""
    rows = []
    with path.open("rb") as handle:
        for schema, channel, message in make_reader(handle).iter_messages(
            topics=[image_topic("CAM_FRONT"), image_topic("CAM_BACK")]
        ):
            assert schema is not None
            rows.append(
                {
                    "data": message.data,
                    "topic": channel.topic,
                    "log_time": message.log_time,
                    "schema_name": schema.name,
                    "schema_encoding": schema.encoding,
                    "schema_data": schema.data,
                    "path": str(path),
                }
            )
    return pa.Table.from_pylist(rows)


def test_actor_plan_uses_one_actor_per_gpu() -> None:
    """With GPUs, auto runs one single-GPU actor per GPU."""
    assert embedding_actor_plan("auto", cluster_gpus=2) == (2, 1)


def test_actor_plan_without_gpus_runs_one_actor() -> None:
    """Without GPUs, one actor asks for none."""
    assert embedding_actor_plan("auto", cluster_gpus=0) == (1, 0)


def test_actor_plan_cpu_ignores_gpus() -> None:
    """--device cpu never reserves a GPU."""
    assert embedding_actor_plan("cpu", cluster_gpus=2) == (1, 0)


def test_actor_plan_cuda_without_gpus_fails() -> None:
    """Asking for CUDA on a cluster without GPUs is an error, not a slow run."""
    with pytest.raises(SystemExit):
        embedding_actor_plan("cuda", cluster_gpus=0)


def test_decode_frames_keeps_keyframes_with_their_metadata(
    synthetic_mcap: SyntheticMcap,
) -> None:
    """The CPU stage drops sweeps and pairs each keyframe image with its row."""
    metadata = pa.Table.from_pylist(
        build_frame_records(synthetic_mcap.root), schema=METADATA_SCHEMA
    )
    batch = image_messages(synthetic_mcap.root / "nuscenes-scene-0001.mcap")
    spec = image_input_spec(DEFAULT_MODEL)
    out = decode_frames(batch, metadata, spec)

    assert batch.num_rows == 18  # 6 keyframe images and 12 sweeps
    assert out.num_rows == 6
    assert out.schema.names == [
        *METADATA_SCHEMA.names,
        THUMBNAIL_COLUMN,
        IMAGE_COLUMN,
        MODEL_INPUT_COLUMN,
    ]
    for row in out.to_pylist():
        assert row[IMAGE_COLUMN] == synthetic_mcap.jpegs[row["frame_id"]]
        assert len(row[MODEL_INPUT_COLUMN]) == spec.nbytes
        with Image.open(io.BytesIO(row[THUMBNAIL_COLUMN])) as thumbnail:
            assert thumbnail.format == "JPEG"


def test_embed_frames_writes_the_frame_schema(
    synthetic_mcap: SyntheticMcap, shared_encoder: SiglipEncoder
) -> None:
    """The GPU stage swaps the pixels for an embedding and tags the model.

    Ray workers aren't traced by coverage, so this runs the stage in-process.
    """
    metadata = pa.Table.from_pylist(
        build_frame_records(synthetic_mcap.root, limit=2), schema=METADATA_SCHEMA
    )
    batch = image_messages(synthetic_mcap.root / "nuscenes-scene-0001.mcap")
    decoded = decode_frames(batch, metadata, shared_encoder.input_spec)
    out = EmbedFrames(DEFAULT_MODEL, device=None)(decoded)

    assert out.schema.equals(frame_schema(shared_encoder.dim))
    tags = out.schema.metadata
    assert tags[META_EMBEDDING_MODEL.encode()] == DEFAULT_MODEL.encode()
    assert tags[META_EMBEDDING_DIM.encode()] == str(shared_encoder.dim).encode()
    vectors = np.stack(out.column(EMBEDDING_COLUMN).to_numpy(zero_copy_only=False))
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)


def test_create_indexes_builds_the_vector_index_only_when_asked(
    tmp_path: Path,
) -> None:
    """Scalar indexes always; IVF_PQ only for --vector-index always at this size."""
    n, dim = 300, 32
    vectors = np.random.default_rng(0).normal(size=(n, dim)).astype(np.float32)
    table = lancedb.connect(tmp_path).create_table(
        TABLE_NAME,
        pa.table(
            {
                "frame_id": [f"f{i}" for i in range(n)],
                "channel": ["CAM_FRONT", "CAM_BACK"] * (n // 2),
                "location": ["test-city"] * n,
                "scene_name": [f"scene-{i % 3}" for i in range(n)],
                EMBEDDING_COLUMN: pa.FixedSizeListArray.from_arrays(
                    pa.array(vectors.ravel()), dim
                ),
            }
        ),
    )
    create_indexes(table, "auto")
    assert index_types(table) == SCALAR_INDEXES
    create_indexes(table, "always")
    assert index_types(table) == {**SCALAR_INDEXES, EMBEDDING_COLUMN: "IvfPq"}


def test_ingest_writes_every_keyframe(
    pipeline_db: Path, synthetic_mcap: SyntheticMcap
) -> None:
    """The local CLI writes each keyframe once, with its original JPEG and tags."""
    table = lancedb.connect(pipeline_db).open_table(TABLE_NAME)
    tags = table.schema.metadata
    assert tags[META_EMBEDDING_MODEL.encode()] == DEFAULT_MODEL.encode()
    expected = frame_schema(int(tags[META_EMBEDDING_DIM.encode()]))
    assert [table.schema.field(name) for name in expected.names] == list(expected)
    assert index_types(table) == SCALAR_INDEXES

    rows = table.to_lance().to_table(columns=["frame_id", IMAGE_COLUMN]).to_pylist()
    assert {row["frame_id"]: row[IMAGE_COLUMN] for row in rows} == synthetic_mcap.jpegs


def test_cluster_job_writes_the_first_frames(
    cluster_db: Path, synthetic_mcap: SyntheticMcap
) -> None:
    """The cluster script runs the same pipeline; --limit takes the first frames."""
    table = lancedb.connect(cluster_db).open_table(TABLE_NAME)
    frame_ids = table.to_lance().to_table(columns=["frame_id"])["frame_id"]
    first_two = [r["frame_id"] for r in build_frame_records(synthetic_mcap.root)[:2]]
    assert sorted(frame_ids.to_pylist()) == sorted(first_two)
    assert (
        table.schema.metadata[META_EMBEDDING_MODEL.encode()] == DEFAULT_MODEL.encode()
    )


def test_ingest_without_mcap_files_says_how_to_get_them(tmp_path: Path) -> None:
    """An empty --mcap-dir fails before Ray starts, with a pointer to the fix."""
    with pytest.raises(SystemExit, match=r"convert_mini\.sh"):
        main(["--mcap-dir", str(tmp_path), "--db", str(tmp_path / "db")])


def test_pipeline_without_mcap_files_raises(tmp_path: Path) -> None:
    """The shared pipeline reports a missing input before starting any work."""
    config = IngestConfig(mcap_uri=str(tmp_path), db_uri=str(tmp_path / "db"))
    with pytest.raises(FileNotFoundError, match=r"No \.mcap files"):
        run(config)


def test_read_scene_table_reads_through_a_pyarrow_filesystem(
    synthetic_mcap: SyntheticMcap,
) -> None:
    """The metadata task opens files the way it would on S3, via pyarrow."""
    path = synthetic_mcap.root / "nuscenes-scene-0002.mcap"
    table = read_scene_table(pafs.LocalFileSystem(), str(path), ("CAM_FRONT",))
    assert table.schema.equals(METADATA_SCHEMA)
    with path.open("rb") as handle:
        expected = read_scene(handle, path.name, ("CAM_FRONT",))
    assert (
        table.to_pylist()
        == pa.Table.from_pylist(expected, schema=METADATA_SCHEMA).to_pylist()
    )


def test_decode_frames_rejects_non_protobuf_images(
    synthetic_mcap: SyntheticMcap,
) -> None:
    """ROS 2 (CDR) image messages fail with a clear message, not a parse error."""
    metadata = pa.Table.from_pylist(
        build_frame_records(synthetic_mcap.root), schema=METADATA_SCHEMA
    )
    batch = image_messages(synthetic_mcap.root / "nuscenes-scene-0001.mcap")
    index = batch.schema.get_field_index("schema_encoding")
    batch = batch.set_column(
        index, "schema_encoding", pa.array(["ros2msg"] * batch.num_rows)
    )
    with pytest.raises(ValueError, match="only protobuf"):
        decode_frames(batch, metadata, image_input_spec(DEFAULT_MODEL))


def test_pipeline_without_keyframes_raises(tmp_path: Path) -> None:
    """MCAP files with no annotated camera keyframes stop before any image work."""
    with foxglove.open_mcap(str(tmp_path / "empty.mcap"), allow_overwrite=True) as mcap:
        mcap.write_metadata(
            "scene-info",
            {
                "description": "Nothing here",
                "name": "scene-empty",
                "location": "test-city",
                "vehicle": "n000",
                "date_captured": "2018-08-01",
            },
        )
    config = IngestConfig(mcap_uri=str(tmp_path), db_uri=str(tmp_path / "db"))
    try:
        with pytest.raises(FileNotFoundError, match="No camera keyframes"):
            run(config)
    finally:
        ray.shutdown()


@pytest.mark.parametrize(
    "argv",
    [
        ["--mcap-uri", "data/mcap", "--db-uri", "/abs/db"],
        ["--mcap-uri", "/abs/mcap", "--db-uri", "data/lancedb"],
        ["--mcap-uri", "/abs/mcap", "--db-uri", "/abs/db", "--min-gpu-actors", "0"],
        [
            "--mcap-uri",
            "s3://bucket/mcap",
            "--db-uri",
            "s3://bucket/db",
            "--min-gpu-actors",
            "4",
            "--max-gpu-actors",
            "2",
        ],
    ],
)
def test_cluster_job_rejects_bad_arguments(argv: list[str]) -> None:
    """Relative paths and impossible actor bounds fail before Ray starts."""
    job = runpy.run_path(str(CLUSTER_SCRIPT))
    with pytest.raises(SystemExit):
        job["parse_args"](argv)


def test_cluster_job_accepts_uris_and_absolute_paths() -> None:
    """Object-store URIs and absolute paths both pass."""
    job = runpy.run_path(str(CLUSTER_SCRIPT))
    args = job["parse_args"](["--mcap-uri", "gs://b/mcap", "--db-uri", "/abs/db"])
    assert (args.mcap_uri, args.db_uri) == ("gs://b/mcap", "/abs/db")
