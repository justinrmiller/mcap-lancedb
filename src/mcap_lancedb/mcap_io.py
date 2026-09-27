"""Read nuScenes scenes that Foxglove's nuscenes2mcap converted to MCAP.

The converter writes one MCAP file per scene. For every keyframe it logs, all at
the same log time: the ego pose on ``/tf``, and for each camera the original
JPEG (``/<camera>/image_rect_compressed``), its calibration
(``/<camera>/camera_info``) and the annotation boxes it can see
(``/<camera>/annotations``). Sweeps between keyframes get an image and a
calibration, but no annotations.

This module reads everything but the images, into one record per camera
keyframe; the ingest pipeline runs it as one Ray task per file. The images are
read by ``ray.data.read_mcap`` and matched to these records on (file, channel,
log time). Every record is fully denormalized because LanceDB has no joins.
"""

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import IO, Any, TypedDict

import numpy as np
import pyarrow as pa
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from mcap_lancedb import CAMERA_CHANNELS

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

SCENE_INFO = "scene-info"
TF_TOPIC = "/tf"
# The ego pose is the transform from the map to the car.
EGO_FRAMES = ("map", "base_link")


class FrameRecord(TypedDict):
    """Metadata for one camera keyframe, before any media or embedding is added."""

    frame_id: str
    scene_name: str
    source_path: str
    mcap_log_time: int
    scene_description: str
    scene_tags: list[str]
    is_night: bool
    is_rain: bool
    location: str
    log_date: date
    vehicle: str
    channel: str
    timestamp: datetime
    frame_index: int
    width: int
    height: int
    cam_intrinsic: list[float]
    ego_translation: list[float]
    ego_rotation: list[float]
    ego_speed_mps: float
    visible_categories: list[str]
    num_visible_objects: int
    num_pedestrians: int
    num_cyclists: int
    num_vehicles: int


def image_topic(channel: str) -> str:
    """The topic that carries a camera's JPEGs, for example ``/cam_front/...``."""
    return f"/{channel.lower()}/image_rect_compressed"


def channel_of(topic: str) -> str:
    """The camera channel a topic belongs to, for example ``CAM_FRONT``."""
    return topic.split("/")[1].upper()


def mcap_files(mcap_dir: Path) -> list[Path]:
    """List the scene files in a directory, in name order.

    Args:
        mcap_dir: Directory the converter wrote to.

    Returns:
        The ``.mcap`` files directly inside it, sorted by name.
    """
    return sorted(mcap_dir.glob("*.mcap"))


def microseconds(stamp: Any) -> int:  # noqa: ANN401
    """Convert a protobuf ``Timestamp`` to integer microseconds."""
    return stamp.seconds * 1_000_000 + stamp.nanos // 1_000


def ego_speeds(translations: np.ndarray, timestamps_us: np.ndarray) -> np.ndarray:
    """Estimate ego speed at each keyframe of a scene.

    Args:
        translations: Ego positions in meters, shape ``(n, 3)``, in time order.
        timestamps_us: Matching timestamps in microseconds, shape ``(n,)``.

    Returns:
        Speeds in meters per second, shape ``(n,)``. A single keyframe has no
        motion to measure and gets zero.
    """
    if len(translations) < 2:
        return np.zeros(len(translations))
    seconds = (timestamps_us - timestamps_us[0]) / 1e6
    velocity = np.gradient(translations, seconds, axis=0)
    return np.linalg.norm(velocity, axis=1)


def scene_tags(description: str) -> list[str]:
    """Split a free-text scene description into lowercase tags.

    Args:
        description: The scene's comma-separated description.

    Returns:
        Stripped, lowercased, non-empty tags in their original order.
    """
    return [tag.strip().lower() for tag in description.split(",") if tag.strip()]


def mentions(description: str, word: str) -> bool:
    """Check whether a description contains a whole word, ignoring case.

    A plain substring test would let "rain" match "train", "drain" and
    "terrain".

    Args:
        description: Free-text scene description.
        word: The word to look for.

    Returns:
        True if ``word`` appears as a whole word.
    """
    pattern = rf"\b{re.escape(word)}\b"
    return re.search(pattern, description, flags=re.IGNORECASE) is not None


def count_categories(categories: Iterable[str]) -> tuple[int, int, int]:
    """Count pedestrians, cyclists and other vehicles in a list of categories.

    Args:
        categories: nuScenes category names of visible objects.

    Returns:
        ``(pedestrians, cyclists, vehicles)``. Bicycles and motorcycles count as
        cyclists, not vehicles.
    """
    pedestrians = cyclists = vehicles = 0
    for name in categories:
        if name.startswith("human.pedestrian."):
            pedestrians += 1
        elif name in {"vehicle.bicycle", "vehicle.motorcycle"}:
            cyclists += 1
        elif name.startswith("vehicle."):
            vehicles += 1
    return pedestrians, cyclists, vehicles


def read_scene(
    handle: IO[bytes] | pa.NativeFile, source_path: str, channels: Sequence[str]
) -> list[FrameRecord]:
    """Build one record per camera keyframe from a scene's MCAP file.

    Args:
        handle: The scene's MCAP file, open for binary reading and seekable.
        source_path: The file's name, recorded as each frame's ``source_path``.
        channels: Camera channels to include. A repeated channel counts once.

    Returns:
        Records ordered by the order of ``channels``, then by time.

    Raises:
        ValueError: If the file has no ``scene-info`` metadata, which means
            nuscenes2mcap didn't write it.
    """
    # A repeat would emit every frame of that camera twice, under one frame_id.
    channels = list(dict.fromkeys(channels))
    topics = [TF_TOPIC]
    for channel in channels:
        topics += [f"/{channel.lower()}/camera_info", f"/{channel.lower()}/annotations"]

    poses: dict[int, Any] = {}
    calibrations: dict[tuple[str, int], Any] = {}
    categories: dict[tuple[str, int], list[str]] = {}
    reader = make_reader(handle, decoder_factories=[DecoderFactory()])
    info = next(
        (m.metadata for m in reader.iter_metadata() if m.name == SCENE_INFO), None
    )
    if info is None:
        msg = f"{source_path} has no {SCENE_INFO} metadata. Did nuscenes2mcap write it?"
        raise ValueError(msg)
    for _, channel, message, decoded in reader.iter_decoded_messages(topics=topics):
        if channel.topic == TF_TOPIC:
            if (decoded.parent_frame_id, decoded.child_frame_id) == EGO_FRAMES:
                poses[message.log_time] = decoded
            continue
        key = (channel_of(channel.topic), message.log_time)
        if channel.topic.endswith("/camera_info"):
            calibrations[key] = decoded
        else:
            categories[key] = [
                pair.value
                for box in decoded.points
                for pair in box.metadata
                if pair.key == "category"
            ]

    # Only keyframes carry annotations. Speed is measured along the keyframes'
    # ego poses, which every camera at that keyframe shares.
    keyframe_times = sorted({log_time for _, log_time in categories})
    ego = [poses[log_time] for log_time in keyframe_times]
    speeds = dict(
        zip(
            keyframe_times,
            ego_speeds(
                np.array(
                    [[p.translation.x, p.translation.y, p.translation.z] for p in ego]
                ),
                np.array([microseconds(p.timestamp) for p in ego], dtype=np.int64),
            ),
            strict=True,
        )
    )

    description: str = info["description"]
    streams: dict[str, list[int]] = defaultdict(list)
    for channel, log_time in sorted(categories):
        streams[channel].append(log_time)

    records: list[FrameRecord] = []
    for channel in channels:
        for frame_index, log_time in enumerate(streams[channel]):
            calibration = calibrations[channel, log_time]
            pose = poses[log_time]
            captured_us = microseconds(calibration.timestamp)
            visible = categories[channel, log_time]
            pedestrians, cyclists, vehicles = count_categories(visible)
            records.append(
                FrameRecord(
                    frame_id=f"{info['name']}/{channel}/{captured_us}",
                    scene_name=info["name"],
                    source_path=source_path,
                    mcap_log_time=log_time,
                    scene_description=description,
                    scene_tags=scene_tags(description),
                    is_night=mentions(description, "night"),
                    is_rain=mentions(description, "rain"),
                    location=info["location"],
                    log_date=date.fromisoformat(info["date_captured"]),
                    vehicle=info["vehicle"],
                    channel=channel,
                    timestamp=_EPOCH + timedelta(microseconds=captured_us),
                    frame_index=frame_index,
                    width=calibration.width,
                    height=calibration.height,
                    cam_intrinsic=list(calibration.K),
                    ego_translation=[
                        pose.translation.x,
                        pose.translation.y,
                        pose.translation.z,
                    ],
                    ego_rotation=[
                        pose.rotation.w,
                        pose.rotation.x,
                        pose.rotation.y,
                        pose.rotation.z,
                    ],
                    ego_speed_mps=float(speeds[log_time]),
                    visible_categories=sorted(set(visible)),
                    num_visible_objects=len(visible),
                    num_pedestrians=pedestrians,
                    num_cyclists=cyclists,
                    num_vehicles=vehicles,
                )
            )
    return records


def build_frame_records(
    mcap_dir: Path,
    channels: Sequence[str] = CAMERA_CHANNELS,
    limit: int | None = None,
) -> list[FrameRecord]:
    """Build one record per camera keyframe from every scene in a directory.

    Records are ordered by file name, then by the order of ``channels``, then by
    time, so a ``limit`` always selects the same frames.

    Args:
        mcap_dir: Directory of per-scene MCAP files.
        channels: Camera channels to include.
        limit: Stop after this many records. ``None`` keeps them all.

    Returns:
        Frame records ready for ``pa.Table.from_pylist``.
    """
    records: list[FrameRecord] = []
    for path in mcap_files(mcap_dir):
        with path.open("rb") as handle:
            records += read_scene(handle, path.name, channels)
        if limit is not None and len(records) >= limit:
            return records[:limit]
    return records
