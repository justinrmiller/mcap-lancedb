"""Tests for reading nuScenes scenes out of MCAP files."""

from pathlib import Path

import foxglove
import numpy as np
import pytest
from conftest import SyntheticMcap

from mcap_lancedb import CAMERA_CHANNELS
from mcap_lancedb.mcap_io import (
    build_frame_records,
    channel_of,
    count_categories,
    ego_speeds,
    image_topic,
    mentions,
    read_scene,
    scene_tags,
)

MINI_MCAP = Path(__file__).parents[1] / "data" / "mcap"


def test_topics_round_trip_to_channels() -> None:
    """A camera's image topic names its channel."""
    assert image_topic("CAM_FRONT") == "/cam_front/image_rect_compressed"
    assert channel_of(image_topic("CAM_BACK_LEFT")) == "CAM_BACK_LEFT"


def test_ego_speeds_constant_velocity() -> None:
    """Straight-line motion at 2 m/s reads as 2 m/s everywhere."""
    timestamps = np.array([0, 500_000, 1_000_000, 1_500_000])
    positions = np.column_stack([[0.0, 1.0, 2.0, 3.0], np.zeros(4), np.zeros(4)])
    np.testing.assert_allclose(ego_speeds(positions, timestamps), 2.0)


def test_ego_speeds_single_frame_is_zero() -> None:
    """A single keyframe has no measurable motion."""
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


def test_build_frame_records_on_the_synthetic_scenes(
    synthetic_mcap: SyntheticMcap,
) -> None:
    """Keyframes only, in stream order, with speed, flags and per-camera objects."""
    records = build_frame_records(synthetic_mcap.root)
    assert [(r["scene_name"], r["channel"], r["frame_index"]) for r in records] == [
        (scene, channel, i)
        for scene in ("scene-0001", "scene-0002")
        for channel in ("CAM_FRONT", "CAM_BACK")
        for i in range(3)
    ]
    assert {r["frame_id"] for r in records} == set(synthetic_mcap.jpegs)
    for record in records:
        moving = record["scene_name"] == "scene-0001"
        assert record["source_path"] == f"nuscenes-{record['scene_name']}.mcap"
        assert record["ego_speed_mps"] == pytest.approx(2.0 if moving else 0.0)
        assert record["is_rain"] is moving
        assert record["is_night"] is not moving
        assert (record["width"], record["height"]) == (64, 36)
        if record["channel"] == "CAM_FRONT":
            assert record["visible_categories"] == [
                "human.pedestrian.adult",
                "vehicle.bicycle",
            ]
            assert record["num_visible_objects"] == 3
            assert (record["num_pedestrians"], record["num_cyclists"]) == (2, 1)
        else:
            assert record["visible_categories"] == ["vehicle.car"]
            assert record["num_vehicles"] == 1


def test_build_frame_records_limit_is_a_stable_prefix(
    synthetic_mcap: SyntheticMcap,
) -> None:
    """A limit selects the first records of the full ordering."""
    full = build_frame_records(synthetic_mcap.root)
    limited = build_frame_records(synthetic_mcap.root, limit=5)
    assert [r["frame_id"] for r in limited] == [r["frame_id"] for r in full[:5]]


def test_read_scene_counts_a_repeated_channel_once(
    synthetic_mcap: SyntheticMcap,
) -> None:
    """``--channels CAM_FRONT CAM_FRONT`` doesn't write each frame twice."""
    path = synthetic_mcap.root / "nuscenes-scene-0001.mcap"
    with path.open("rb") as handle:
        once = read_scene(handle, path.name, ["CAM_FRONT"])
    with path.open("rb") as handle:
        repeated = read_scene(handle, path.name, ["CAM_FRONT", "CAM_FRONT"])
    assert [r["frame_id"] for r in repeated] == [r["frame_id"] for r in once]
    assert len(once) == 3


def test_read_scene_rejects_files_without_scene_info(tmp_path: Path) -> None:
    """An MCAP file nuscenes2mcap didn't write fails with a clear message."""
    path = tmp_path / "other.mcap"
    with foxglove.open_mcap(str(path), allow_overwrite=True):
        pass
    with path.open("rb") as handle, pytest.raises(ValueError, match="scene-info"):
        read_scene(handle, path.name, CAMERA_CHANNELS)


@pytest.mark.skipif(
    not any(MINI_MCAP.glob("*.mcap")), reason="needs nuScenes mini as MCAP"
)
def test_build_frame_records_on_mini() -> None:
    """Mini has 404 samples, so 404 keyframes per camera."""
    records = build_frame_records(MINI_MCAP)
    assert len(records) == 404 * len(CAMERA_CHANNELS)
    assert len({r["frame_id"] for r in records}) == len(records)
    stationary = [r for r in records if r["scene_name"] == "scene-0553"]
    assert all(r["ego_speed_mps"] < 0.5 for r in stationary)
    assert {r["scene_name"] for r in records if r["is_night"]} == {
        "scene-1077",
        "scene-1094",
        "scene-1100",
    }
