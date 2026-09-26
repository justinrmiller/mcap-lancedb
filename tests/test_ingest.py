"""Tests for the ingest pipeline's stages and planning."""

import io
from pathlib import Path

import pyarrow as pa
import pytest
from PIL import Image

from mcap_lancedb import THUMBNAIL_LONG_EDGE
from mcap_lancedb.embed import ImageInputSpec
from mcap_lancedb.ingest import decode_frames, embedding_actor_plan, main
from mcap_lancedb.schema import IMAGE_COLUMN, MODEL_INPUT_COLUMN, THUMBNAIL_COLUMN


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


def test_decode_frames_adds_media_columns(tmp_path: Path) -> None:
    """The CPU stage keeps the original bytes and adds the thumbnail and pixels."""
    buffer = io.BytesIO()
    Image.new("RGB", (1600, 900), (40, 90, 160)).save(buffer, format="JPEG")
    (tmp_path / "frame.jpg").write_bytes(buffer.getvalue())
    spec = ImageInputSpec(224, 224, Image.Resampling.BILINEAR)

    batch = pa.table({"source_path": ["frame.jpg"]})
    out = decode_frames(batch, str(tmp_path), spec)

    assert out.column(IMAGE_COLUMN)[0].as_py() == buffer.getvalue()
    assert len(out.column(MODEL_INPUT_COLUMN)[0].as_py()) == spec.nbytes
    with Image.open(io.BytesIO(out.column(THUMBNAIL_COLUMN)[0].as_py())) as thumb:
        assert max(thumb.size) == THUMBNAIL_LONG_EDGE
        assert thumb.format == "JPEG"


def test_ingest_without_the_dataset_says_how_to_get_it(tmp_path: Path) -> None:
    """A wrong --dataroot fails before Ray starts, with a pointer to the fix."""
    with pytest.raises(SystemExit, match="Download the dataset"):
        main(["--dataroot", str(tmp_path), "--db", str(tmp_path / "db")])
