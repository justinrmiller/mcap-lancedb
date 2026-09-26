"""Turn the raw nuScenes JSON tables into one flat record per camera keyframe.

This reads the relational tables directly instead of going through
nuscenes-devkit, which pulls in a large dependency tree for what amounts to a
handful of joins. Every record is fully denormalized because LanceDB has no joins:
scene context, ego pose, calibration and per-camera object counts all live on the
frame's own row.
"""

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import numpy as np

from mcap_lancedb import CAMERA_CHANNELS

type JsonRow = dict[str, Any]

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

# Annotation centers closer than this to the image plane are treated as behind
# the camera; dividing by a near-zero depth would throw them anywhere in the image.
MIN_DEPTH_M = 0.1


class FrameRecord(TypedDict):
    """Metadata for one camera keyframe, before any media or embedding is added."""

    frame_id: str
    sample_token: str
    scene_token: str
    scene_name: str
    source_path: str
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


class SampleObjects(NamedTuple):
    """Annotated object centers for one sample, in global coordinates."""

    centers: np.ndarray
    categories: np.ndarray


def load_table(dataroot: Path, version: str, name: str) -> list[JsonRow]:
    """Load one nuScenes JSON table.

    Args:
        dataroot: Directory that contains the version folder and ``samples/``.
        version: Dataset version folder, for example ``v1.0-mini``.
        name: Table name without the ``.json`` suffix.

    Returns:
        The table's rows as dictionaries.
    """
    path = dataroot / version / f"{name}.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def index_by_token(rows: Iterable[JsonRow]) -> dict[str, JsonRow]:
    """Key a table's rows by their ``token`` field.

    Args:
        rows: Rows from a nuScenes table.

    Returns:
        A mapping from token to row.
    """
    return {row["token"]: row for row in rows}


def quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert a nuScenes ``[w, x, y, z]`` quaternion to a rotation matrix.

    Args:
        quaternion: Rotation as ``[w, x, y, z]``. It need not be unit length.

    Returns:
        A 3x3 rotation matrix that maps local coordinates into the parent frame.
    """
    q = np.asarray(quaternion, dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def visible_in_camera(
    points: np.ndarray,
    ego_pose: JsonRow,
    calibration: JsonRow,
    width: int,
    height: int,
) -> np.ndarray:
    """Find which global points project inside a camera image.

    Points go global → ego → camera → pixels. The rotations map local to parent,
    so the inverse transform for row vectors is ``(p - t) @ R``.

    Args:
        points: Global coordinates, shape ``(n, 3)``.
        ego_pose: The ``ego_pose`` row at the camera's timestamp.
        calibration: The camera's ``calibrated_sensor`` row.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A boolean mask of shape ``(n,)``, true where the point is in front of the
        camera and lands inside the image bounds.
    """
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    p_ego = (points - np.asarray(ego_pose["translation"])) @ quaternion_to_matrix(
        ego_pose["rotation"]
    )
    p_cam = (p_ego - np.asarray(calibration["translation"])) @ quaternion_to_matrix(
        calibration["rotation"]
    )
    depth = p_cam[:, 2]
    in_front = depth > MIN_DEPTH_M
    pixels = p_cam @ np.asarray(calibration["camera_intrinsic"]).T
    safe_depth = np.where(in_front, depth, 1.0)
    u = pixels[:, 0] / safe_depth
    v = pixels[:, 1] / safe_depth
    return in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)


def ego_speeds(translations: np.ndarray, timestamps_us: np.ndarray) -> np.ndarray:
    """Estimate ego speed at each keyframe of one camera stream.

    Args:
        translations: Ego positions in meters, shape ``(n, 3)``, in time order.
        timestamps_us: Matching timestamps in microseconds, shape ``(n,)``.

    Returns:
        Speeds in meters per second, shape ``(n,)``. A single-frame stream has
        no motion to measure and gets zero.
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


def _objects_by_sample(
    annotations: Iterable[JsonRow],
    instances: dict[str, JsonRow],
    categories: dict[str, JsonRow],
) -> dict[str, SampleObjects]:
    """Group sensor-confirmed annotation centers by sample.

    Annotations no lidar or radar point touched are dropped: they are boxes the
    annotators interpolated through occlusion, not objects a sensor saw.
    """
    grouped: dict[str, list[JsonRow]] = defaultdict(list)
    for annotation in annotations:
        if annotation["num_lidar_pts"] + annotation["num_radar_pts"] > 0:
            grouped[annotation["sample_token"]].append(annotation)

    objects: dict[str, SampleObjects] = {}
    for sample_token, rows in grouped.items():
        names = [
            categories[instances[row["instance_token"]]["category_token"]]["name"]
            for row in rows
        ]
        objects[sample_token] = SampleObjects(
            centers=np.array([row["translation"] for row in rows], dtype=np.float64),
            categories=np.array(names),
        )
    return objects


def build_frame_records(
    dataroot: Path,
    version: str,
    channels: Sequence[str] = CAMERA_CHANNELS,
    limit: int | None = None,
) -> list[FrameRecord]:
    """Build one record per camera keyframe from the nuScenes JSON tables.

    Records are ordered by scene name, then by the order of ``channels``, then by
    time, so a ``limit`` always selects the same frames.

    Args:
        dataroot: Directory that contains the version folder and ``samples/``.
        version: Dataset version folder, for example ``v1.0-mini``.
        channels: Camera channels to include.
        limit: Stop after this many records. ``None`` keeps them all.

    Returns:
        Frame records ready for ``pa.Table.from_pylist``.
    """
    scenes = index_by_token(load_table(dataroot, version, "scene"))
    logs = index_by_token(load_table(dataroot, version, "log"))
    samples = index_by_token(load_table(dataroot, version, "sample"))
    sensors = index_by_token(load_table(dataroot, version, "sensor"))
    calibrations = index_by_token(load_table(dataroot, version, "calibrated_sensor"))
    poses = index_by_token(load_table(dataroot, version, "ego_pose"))
    objects = _objects_by_sample(
        load_table(dataroot, version, "sample_annotation"),
        index_by_token(load_table(dataroot, version, "instance")),
        index_by_token(load_table(dataroot, version, "category")),
    )

    wanted = set(channels)
    channel_of = {
        token: sensors[calibration["sensor_token"]]["channel"]
        for token, calibration in calibrations.items()
    }

    # Each (scene, channel) pair is one stream of keyframes; frame_index and ego
    # speed are both defined along that stream.
    streams: dict[tuple[str, str], list[JsonRow]] = defaultdict(list)
    for row in load_table(dataroot, version, "sample_data"):
        channel = channel_of[row["calibrated_sensor_token"]]
        if row["is_key_frame"] and channel in wanted:
            scene_token = samples[row["sample_token"]]["scene_token"]
            streams[scene_token, channel].append(row)

    channel_order = {name: i for i, name in enumerate(channels)}
    ordered_keys = sorted(
        streams, key=lambda key: (scenes[key[0]]["name"], channel_order[key[1]])
    )

    no_objects = SampleObjects(np.zeros((0, 3)), np.array([], dtype=str))
    records: list[FrameRecord] = []
    for scene_token, channel in ordered_keys:
        scene = scenes[scene_token]
        log = logs[scene["log_token"]]
        description: str = scene["description"]
        frames = sorted(streams[scene_token, channel], key=lambda r: r["timestamp"])
        frame_poses = [poses[row["ego_pose_token"]] for row in frames]
        speeds = ego_speeds(
            np.array([pose["translation"] for pose in frame_poses]),
            np.array([row["timestamp"] for row in frames], dtype=np.int64),
        )

        for frame_index, (row, pose, speed) in enumerate(
            zip(frames, frame_poses, speeds, strict=True)
        ):
            if limit is not None and len(records) >= limit:
                return records
            calibration = calibrations[row["calibrated_sensor_token"]]
            sample_objects = objects.get(row["sample_token"], no_objects)
            visible = sample_objects.categories[
                visible_in_camera(
                    sample_objects.centers,
                    pose,
                    calibration,
                    row["width"],
                    row["height"],
                )
            ].tolist()
            pedestrians, cyclists, vehicles = count_categories(visible)

            records.append(
                FrameRecord(
                    frame_id=row["token"],
                    sample_token=row["sample_token"],
                    scene_token=scene_token,
                    scene_name=scene["name"],
                    source_path=row["filename"],
                    scene_description=description,
                    scene_tags=scene_tags(description),
                    is_night=mentions(description, "night"),
                    is_rain=mentions(description, "rain"),
                    location=log["location"],
                    log_date=date.fromisoformat(log["date_captured"]),
                    vehicle=log["vehicle"],
                    channel=channel,
                    timestamp=_EPOCH + timedelta(microseconds=row["timestamp"]),
                    frame_index=frame_index,
                    width=row["width"],
                    height=row["height"],
                    cam_intrinsic=np.ravel(calibration["camera_intrinsic"]).tolist(),
                    ego_translation=list(pose["translation"]),
                    ego_rotation=list(pose["rotation"]),
                    ego_speed_mps=float(speed),
                    visible_categories=sorted(set(visible)),
                    num_visible_objects=len(visible),
                    num_pedestrians=pedestrians,
                    num_cyclists=cyclists,
                    num_vehicles=vehicles,
                )
            )
    return records
