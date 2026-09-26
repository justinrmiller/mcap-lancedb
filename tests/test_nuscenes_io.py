"""Tests for reading nuScenes tables and projecting annotations."""

from pathlib import Path

import numpy as np
import pytest

from mcap_lancedb import CAMERA_CHANNELS
from mcap_lancedb.nuscenes_io import (
    build_frame_records,
    count_categories,
    ego_speeds,
    mentions,
    quaternion_to_matrix,
    scene_tags,
    visible_in_camera,
)

MINI_ROOT = Path("data/nuscenes")
IDENTITY = [1.0, 0.0, 0.0, 0.0]
INTRINSIC = [[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]]


def test_quaternion_to_matrix_rotates_about_z() -> None:
    """A 90 degree yaw maps +x to +y."""
    half = np.sqrt(0.5)
    rotation = quaternion_to_matrix([half, 0.0, 0.0, half])
    np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)


def test_quaternion_to_matrix_normalizes() -> None:
    """A scaled quaternion gives the same rotation as the unit one."""
    np.testing.assert_allclose(quaternion_to_matrix([2.0, 0, 0, 0]), np.eye(3))


def test_visible_in_camera_checks_depth_and_bounds() -> None:
    """Only points in front of the camera and inside the image count."""
    pose = {"translation": [0.0, 0.0, 0.0], "rotation": IDENTITY}
    calibration = {
        "translation": [0.0, 0.0, 0.0],
        "rotation": IDENTITY,
        "camera_intrinsic": INTRINSIC,
    }
    points = np.array(
        [
            [0.0, 0.0, 10.0],  # straight ahead: center pixel
            [0.0, 0.0, -10.0],  # behind the camera
            [0.0, 0.0, 0.05],  # closer than the minimum depth
            [100.0, 0.0, 10.0],  # far off to the side
        ]
    )
    mask = visible_in_camera(points, pose, calibration, width=1600, height=900)
    assert mask.tolist() == [True, False, False, False]


def test_visible_in_camera_applies_ego_pose() -> None:
    """A point is moved into the ego frame before the camera frame."""
    pose = {"translation": [5.0, 0.0, 0.0], "rotation": IDENTITY}
    calibration = {
        "translation": [0.0, 0.0, 0.0],
        "rotation": IDENTITY,
        "camera_intrinsic": INTRINSIC,
    }
    # 10 m ahead of the camera once the ego offset is removed.
    mask = visible_in_camera(np.array([[5.0, 0.0, 10.0]]), pose, calibration, 1600, 900)
    assert mask.tolist() == [True]


def test_ego_speeds_constant_velocity() -> None:
    """Straight-line motion at 2 m/s reads as 2 m/s everywhere."""
    timestamps = np.array([0, 500_000, 1_000_000, 1_500_000])
    positions = np.column_stack([[0.0, 1.0, 2.0, 3.0], np.zeros(4), np.zeros(4)])
    np.testing.assert_allclose(ego_speeds(positions, timestamps), 2.0)


def test_ego_speeds_single_frame_is_zero() -> None:
    """A one-frame stream has no measurable motion."""
    assert ego_speeds(np.zeros((1, 3)), np.array([0])).tolist() == [0.0]


def test_scene_tags_splits_and_lowercases() -> None:
    """Tags are comma-separated, stripped and lowercased."""
    assert scene_tags("Night, big street,  Bus stop,") == [
        "night",
        "big street",
        "bus stop",
    ]


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Night, after rain, many peds", True),
        ("Rain, parked cars", True),
        ("Train station, drain, rough terrain", False),
        ("Raining", False),
    ],
)
def test_mentions_matches_whole_words_only(description: str, expected: bool) -> None:
    """The rain flag matches the word, not every word containing it."""
    assert mentions(description, "rain") is expected


def test_count_categories_separates_cyclists_from_vehicles() -> None:
    """Bicycles and motorcycles count as cyclists, other vehicles as vehicles."""
    names = [
        "human.pedestrian.adult",
        "human.pedestrian.child",
        "vehicle.bicycle",
        "vehicle.motorcycle",
        "vehicle.car",
        "vehicle.bus.rigid",
        "movable_object.trafficcone",
    ]
    assert count_categories(names) == (2, 2, 2)


@pytest.mark.skipif(
    not (MINI_ROOT / "v1.0-mini").exists(), reason="needs nuScenes mini"
)
def test_build_frame_records_on_mini() -> None:
    """Mini has 404 samples, so 404 keyframes per camera."""
    records = build_frame_records(MINI_ROOT, "v1.0-mini")
    assert len(records) == 404 * len(CAMERA_CHANNELS)
    assert len({r["frame_id"] for r in records}) == len(records)
    stationary = [r for r in records if r["scene_name"] == "scene-0553"]
    assert all(r["ego_speed_mps"] < 0.5 for r in stationary)
    assert {r["scene_name"] for r in records if r["is_night"]} == {
        "scene-1077",
        "scene-1094",
        "scene-1100",
    }


@pytest.mark.skipif(
    not (MINI_ROOT / "v1.0-mini").exists(), reason="needs nuScenes mini"
)
def test_build_frame_records_limit_is_a_stable_prefix() -> None:
    """A limit selects the first records of the full ordering."""
    full = build_frame_records(MINI_ROOT, "v1.0-mini")
    limited = build_frame_records(MINI_ROOT, "v1.0-mini", limit=50)
    assert [r["frame_id"] for r in limited] == [r["frame_id"] for r in full[:50]]
