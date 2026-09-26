"""Shared fixtures: synthetic nuScenes MCAP scenes, and the pipeline run on them.

The synthetic scenes are laid out the way nuscenes2mcap writes real ones, so
ingest (with Ray), dedup and the viewer all run without downloading or
converting nuScenes. The embedding model is the real default, downloaded once.
"""

import os

# Ray reads these when it's imported. Without the first, a test run started by
# `uv run` makes Ray rebuild the project environment for its workers.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")

import io
import math
import runpy
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import foxglove
import lancedb
import numpy as np
import pyarrow as pa
import pytest
import ray
from foxglove.messages import (
    CameraCalibration,
    CompressedImage,
    FrameTransform,
    ImageAnnotations,
    KeyValuePair,
    LocationFix,
    Point2,
    PointsAnnotation,
    PointsAnnotationType,
    Quaternion,
    Timestamp,
    Vector3,
)
from PIL import Image

from mcap_lancedb import DEFAULT_MODEL, TABLE_NAME, dedup, embed, ingest, pipeline
from mcap_lancedb.embed import SiglipEncoder

WIDTH, HEIGHT = 64, 36
KEYFRAMES = 3
SWEEPS_PER_KEYFRAME = 2
SECONDS_PER_KEYFRAME = 0.5
START_NS = 1_533_000_000_000_000_000
# Cameras capture a little after the keyframe's lidar sweep, like nuScenes.
CAMERA_DELAY_NS = {"CAM_FRONT": 12_000_000, "CAM_BACK": 37_000_000}
INTRINSIC = [32.0, 0.0, WIDTH / 2, 0.0, 32.0, HEIGHT / 2, 0.0, 0.0, 1.0]

# (description, start position, heading in radians, speed in m/s)
SCENES = {
    "scene-0001": ("Rain, turn right", (100.0, 200.0), math.pi / 2, 2.0),
    "scene-0002": ("Night, wait at intersection", (300.0, 400.0), 0.0, 0.0),
}

# What each camera's annotations list at every keyframe.
CATEGORIES = {
    "CAM_FRONT": [
        "human.pedestrian.adult",
        "human.pedestrian.adult",
        "vehicle.bicycle",
    ],
    "CAM_BACK": ["vehicle.car"],
}


class SyntheticMcap(NamedTuple):
    """Where the synthetic scenes live, and the JPEG behind every keyframe.

    Two scenes, each with CAM_FRONT and CAM_BACK streams of three keyframes:
    12 frames. scene-0001 ("Rain, turn right") drives north at 2 m/s with a new
    image every frame. scene-0002 ("Night, wait at intersection") stands still
    and repeats one image per camera. CAM_FRONT sees two pedestrians and a
    bicycle, CAM_BACK a car. Sweeps between keyframes, sensor transforms and a
    GPS topic must all be ignored.
    """

    root: Path
    jpegs: dict[str, bytes]


def jpeg(rng: np.random.Generator) -> bytes:
    """Encode a small image of random color blocks."""
    blocks = rng.integers(0, 256, size=(3, 4, 3), dtype=np.uint8)
    image = Image.fromarray(blocks).resize((WIDTH, HEIGHT), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def stamp(ns: int) -> Timestamp:
    """A foxglove timestamp from integer nanoseconds."""
    return Timestamp(sec=ns // 1_000_000_000, nsec=ns % 1_000_000_000)


def ego_pose(
    ns: int, start: Sequence[float], heading: float, travelled: float
) -> FrameTransform:
    """The map-to-car transform after driving ``travelled`` meters."""
    return FrameTransform(
        parent_frame_id="map",
        child_frame_id="base_link",
        timestamp=stamp(ns),
        translation=Vector3(
            x=start[0] + travelled * math.cos(heading),
            y=start[1] + travelled * math.sin(heading),
            z=0.0,
        ),
        rotation=Quaternion(
            x=0.0, y=0.0, z=math.sin(heading / 2), w=math.cos(heading / 2)
        ),
    )


def log_camera(channel: str, log_time: int, captured: int, data: bytes) -> None:
    """Log one camera frame's image and calibration."""
    sensor = channel.lower()
    foxglove.log(
        f"/{sensor}/image_rect_compressed",
        CompressedImage(
            timestamp=stamp(captured), frame_id=sensor, format="jpeg", data=data
        ),
        log_time=log_time,
    )
    foxglove.log(
        f"/{sensor}/camera_info",
        CameraCalibration(
            timestamp=stamp(captured),
            frame_id=sensor,
            width=WIDTH,
            height=HEIGHT,
            K=INTRINSIC,
        ),
        log_time=log_time,
    )


def log_annotations(channel: str, log_time: int) -> None:
    """Log a keyframe's annotation boxes, each tagged with its category."""
    boxes = [
        PointsAnnotation(
            type=PointsAnnotationType.LineList,
            points=[Point2(x=1.0, y=1.0), Point2(x=9.0, y=9.0)],
            metadata=[KeyValuePair(key="category", value=name)],
        )
        for name in CATEGORIES[channel]
    ]
    foxglove.log(
        f"/{channel.lower()}/annotations",
        ImageAnnotations(points=boxes),
        log_time=log_time,
    )


def write_scene(path: Path, scene: str, rng: np.random.Generator) -> dict[str, bytes]:
    """Write one scene the way nuscenes2mcap does; return its keyframe JPEGs."""
    description, start, heading, speed = SCENES[scene]
    repeated = {channel: jpeg(rng) for channel in CATEGORIES}
    step_ns = int(SECONDS_PER_KEYFRAME * 1e9)
    jpegs: dict[str, bytes] = {}
    with foxglove.open_mcap(str(path), allow_overwrite=True) as writer:
        writer.write_metadata(
            "scene-info",
            {
                "description": description,
                "name": scene,
                "location": "test-city",
                "vehicle": "n000",
                "date_captured": "2018-08-01",
            },
        )
        for i in range(KEYFRAMES):
            keyframe = START_NS + i * step_ns
            travelled = speed * i * SECONDS_PER_KEYFRAME
            foxglove.log(
                "/tf", ego_pose(keyframe, start, heading, travelled), log_time=keyframe
            )
            foxglove.log(
                "/gps", LocationFix(latitude=1.0, longitude=2.0), log_time=keyframe
            )
            for channel in CATEGORIES:
                foxglove.log(
                    "/tf",
                    FrameTransform(
                        parent_frame_id="base_link",
                        child_frame_id=channel.lower(),
                        timestamp=stamp(keyframe),
                    ),
                    log_time=keyframe,
                )
                captured = keyframe + CAMERA_DELAY_NS[channel]
                data = repeated[channel] if speed == 0 else jpeg(rng)
                jpegs[f"{scene}/{channel}/{captured // 1000}"] = data
                log_camera(channel, keyframe, captured, data)
                log_annotations(channel, keyframe)
            # Sweeps between keyframes: an image and calibration, no annotations.
            for sweep in range(1, SWEEPS_PER_KEYFRAME + 1):
                sweep_ns = keyframe + sweep * step_ns // (SWEEPS_PER_KEYFRAME + 1)
                for channel in CATEGORIES:
                    log_camera(channel, sweep_ns, sweep_ns, jpeg(rng))
    return jpegs


@pytest.fixture(scope="session")
def synthetic_mcap(tmp_path_factory: pytest.TempPathFactory) -> SyntheticMcap:
    """Write the synthetic scenes once per test session."""
    root = tmp_path_factory.mktemp("mcap")
    rng = np.random.default_rng(0)
    jpegs: dict[str, bytes] = {}
    for scene in SCENES:
        jpegs |= write_scene(root / f"nuscenes-{scene}.mcap", scene, rng)
    return SyntheticMcap(root, jpegs)


@pytest.fixture(scope="session")
def pipeline_db(
    synthetic_mcap: SyntheticMcap, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Ingest the synthetic scenes with Ray, then mark near-duplicates."""
    db = tmp_path_factory.mktemp("lancedb")
    try:
        ingest.main(["--mcap-dir", str(synthetic_mcap.root), "--db", str(db)])
    finally:
        ray.shutdown()
    dedup.main(["--db", str(db)])
    return db


@pytest.fixture(scope="session")
def cluster_db(
    synthetic_mcap: SyntheticMcap,
    tmp_path_factory: pytest.TempPathFactory,
    pipeline_db: Path,
) -> Path:
    """Run the cluster job script on a throwaway local cluster, two frames only.

    The database starts with a stale ``frames`` table of another schema, which
    the job must replace. It depends on ``pipeline_db`` only for ordering, so
    the two Ray runs (and their copies of the model) never overlap.
    """
    db = tmp_path_factory.mktemp("cluster-lancedb")
    lancedb.connect(db).create_table(TABLE_NAME, pa.table({"frame_id": ["stale"]}))
    script = Path(__file__).parents[1] / "scripts" / "ingest_on_cluster.py"
    job = runpy.run_path(str(script))
    try:
        job["main"](
            [
                "--mcap-uri",
                str(synthetic_mcap.root),
                "--db-uri",
                str(db),
                "--limit",
                "2",
                "--ray-address",
                "local",
                "--gpus-per-actor",
                "0",
            ]
        )
    finally:
        ray.shutdown()
    return db


@pytest.fixture(scope="session")
def siglip2_encoder(pipeline_db: Path, cluster_db: Path) -> SiglipEncoder:
    """The default model, loaded once for the whole test process.

    It depends on the Ray runs only for ordering: their actors' copies of the
    model are gone before this one loads, and a CI runner can't hold two.
    """
    return SiglipEncoder(DEFAULT_MODEL)


@pytest.fixture
def shared_encoder(
    monkeypatch: pytest.MonkeyPatch, siglip2_encoder: SiglipEncoder
) -> SiglipEncoder:
    """Make every in-process ``SiglipEncoder(DEFAULT_MODEL)`` reuse one instance.

    Covers ``EmbedFrames`` and the viewer's ``load_encoder``, which would each
    load their own 4.5 GB copy otherwise.
    """
    load = embed.SiglipEncoder

    def reuse(model_id: str = DEFAULT_MODEL, device: str | None = None) -> Any:  # noqa: ANN401
        if model_id == DEFAULT_MODEL:
            return siglip2_encoder
        return load(model_id, device)

    monkeypatch.setattr(embed, "SiglipEncoder", reuse)
    monkeypatch.setattr(pipeline, "SiglipEncoder", reuse)
    return siglip2_encoder
