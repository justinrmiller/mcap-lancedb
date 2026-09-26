"""Headless tests for the Streamlit viewer.

The AppTest cases drive the real app against the table at ``MCAP_LANCEDB_DB``
(default ``data/lancedb``) and are skipped until ingest and dedup have run.
"""

import io
import os
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from mcap_lancedb import META_DEDUP_THRESHOLD, META_EMBEDDING_MODEL, TABLE_NAME, dedup
from mcap_lancedb.app import (
    EXAMPLE_QUERIES,
    build_where,
    edge_similarity,
    load_catalog,
    suppress_at,
)

APP = Path(__file__).parents[1] / "src" / "mcap_lancedb" / "app.py"
DB = Path(os.environ.get("MCAP_LANCEDB_DB", "data/lancedb"))
TIMEOUT_S = 300

needs_table = pytest.mark.skipif(
    not (DB / "frames.lance").exists(), reason="run mcap-lancedb-ingest first"
)


def where_for_locations(locations: list[str]) -> str | None:
    """Build a prefilter that only filters on location."""
    return build_where(
        cameras=[],
        locations=locations,
        time_of_day="Any",
        min_pedestrians=0,
        categories=[],
        hide_duplicates=False,
    )


def test_build_where_without_filters_is_none() -> None:
    """No filters means no prefilter at all."""
    assert where_for_locations([]) is None


def test_build_where_combines_every_filter() -> None:
    """Each widget contributes one clause, joined with AND."""
    where = build_where(
        cameras=["CAM_FRONT", "CAM_BACK"],
        locations=["boston-seaport"],
        time_of_day="Night",
        min_pedestrians=2,
        categories=["vehicle.car", "human.pedestrian.adult"],
        hide_duplicates=True,
        exclude_frame="abc",
    )
    assert where == (
        "channel IN ('CAM_FRONT', 'CAM_BACK') AND location IN ('boston-seaport') "
        "AND is_night = true AND num_pedestrians >= 2 "
        "AND array_has(visible_categories, 'vehicle.car') "
        "AND array_has(visible_categories, 'human.pedestrian.adult') "
        "AND dup_of IS NULL AND frame_id != 'abc'"
    )


def test_build_where_escapes_quotes() -> None:
    """Single quotes are doubled, not passed through."""
    assert where_for_locations(["o'hare"]) == "location IN ('o''hare')"


def test_app_without_a_table_says_which_command_to_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty database gets an empty state, not a traceback."""
    monkeypatch.setenv("MCAP_LANCEDB_DB", str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S).run()
    assert not app.exception
    assert "mcap-lancedb-ingest" in app.info[0].value


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """Run the viewer once against the real table."""
    monkeypatch.setenv("MCAP_LANCEDB_DB", str(DB))
    started = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S).run()
    assert not started.exception
    return started


@needs_table
def test_text_search_renders_results(app: AppTest) -> None:
    """A typed query fills the grid with thumbnails."""
    app.text_input(key="query").input("a bus at a bus stop").run()
    assert not app.exception
    assert len(app.image) > 0
    assert any(b.key and b.key.startswith("like-") for b in app.button)


@needs_table
def test_example_query_fills_the_query_box(app: AppTest) -> None:
    """Choosing an example copies it into the query and searches."""
    app.pills(key="example").set_value(EXAMPLE_QUERIES[0]).run()
    assert not app.exception
    assert app.text_input(key="query").value == EXAMPLE_QUERIES[0]


@needs_table
def test_filters_and_hide_duplicates(app: AppTest) -> None:
    """Prefiltered search runs with every filter set."""
    app.text_input(key="query").input("pedestrians on a crosswalk").run()
    app.multiselect[0].set_value(["CAM_FRONT"])
    app.segmented_control[0].set_value("Day")
    app.toggle[0].set_value(True)
    app.run()
    assert not app.exception


@needs_table
def test_more_like_this_and_full_resolution(app: AppTest) -> None:
    """Image search and the full-resolution dialog both run cleanly."""
    app.text_input(key="query").input("parking lot with parked cars").run()
    like = next(b for b in app.button if b.key and b.key.startswith("like-"))
    like.click().run()
    assert not app.exception
    assert app.session_state["anchor"] == str(like.key).removeprefix("like-")

    full = next(b for b in app.button if b.key and b.key.startswith("full-"))
    full.click().run()
    assert not app.exception


@needs_table
def test_threshold_slider_recomputes_removals(app: AppTest) -> None:
    """Moving the threshold re-runs suppression and redraws the tab."""
    removed_before = app.metric[2].value
    app.slider(key="threshold").set_value(0.95).run()
    assert not app.exception
    assert app.metric[2].value != removed_before


@needs_table
def test_viewer_reproduces_dedup_at_the_stored_threshold() -> None:
    """At dedup's own threshold, the slider's re-run marks exactly what dedup did.

    The viewer and the CLI build their visit order from different sources
    (pandas and Arrow), so this pins that they agree frame for frame.
    """
    table = lancedb.connect(DB).open_table(TABLE_NAME)
    stored = (table.schema.metadata or {}).get(META_DEDUP_THRESHOLD.encode())
    if stored is None:
        pytest.skip("run mcap-lancedb-dedup first")
    version = table.version
    frame_ids = load_catalog(str(DB), version)["frame_id"].to_numpy()
    dup_of = suppress_at(str(DB), version, float(stored))

    marked = (
        table.to_lance()
        .checkout_version(version)
        .to_table(columns=["frame_id", "dup_of"])
    )
    assert marked.column("frame_id").to_pylist() == frame_ids.tolist()
    assert marked.column("dup_of").to_pylist() == [
        frame_ids[i] if i >= 0 else None for i in dup_of
    ]


def make_viewer_table(root: Path) -> None:
    """Create an 8-frame table the viewer can open without a model or dataset.

    f1 and f2 are near-copies of f0 (cosine 0.990 and 0.958); the rest are
    orthogonal to everything. So 0.98 removes one frame and 0.95 removes two.
    """
    basis = np.eye(8, dtype=np.float32)
    vectors = basis.copy()
    vectors[1] = basis[0] + 0.14 * basis[1]
    vectors[2] = basis[0] + 0.30 * basis[2]
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    buffer = io.BytesIO()
    Image.new("RGB", (32, 18), (90, 90, 90)).save(buffer, format="JPEG")
    n = len(vectors)
    table = pa.table(
        {
            "frame_id": [f"f{i}" for i in range(n)],
            "scene_name": ["scene-test"] * n,
            "scene_description": ["Synthetic"] * n,
            "channel": ["CAM_FRONT"] * n,
            "location": ["test-town"] * n,
            "is_night": [False] * n,
            "timestamp": pa.array(range(n), pa.timestamp("us", tz="UTC")),
            "ego_speed_mps": pa.array([0.0] * n, pa.float32()),
            "num_visible_objects": pa.array([0] * n, pa.int32()),
            "visible_categories": pa.array([[]] * n, pa.list_(pa.string())),
            "thumbnail": [buffer.getvalue()] * n,
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.ravel(), pa.float32()), 8
            ),
        }
    ).replace_schema_metadata({META_EMBEDDING_MODEL: "test-model"})
    lancedb.connect(root).create_table(TABLE_NAME, table)


def test_threshold_slider_follows_a_dedup_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After dedup reruns, the slider and metrics move to the new threshold.

    A keyed slider otherwise keeps its old position, leaving the tab showing
    one threshold while the stored marks use another.
    """
    make_viewer_table(tmp_path)
    run_dedup = ["--db", str(tmp_path), "--device", "cpu", "--threshold"]
    dedup.main([*run_dedup, "0.98"])
    monkeypatch.setenv("MCAP_LANCEDB_DB", str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S).run()
    assert not app.exception
    assert app.slider(key="threshold").value == pytest.approx(0.98)
    assert app.metric[2].value == "12.5%"

    dedup.main([*run_dedup, "0.95"])
    app.run()
    assert not app.exception
    assert app.slider(key="threshold").value == pytest.approx(0.95)
    assert app.metric[2].value == "25.0%"
    # Only the cameras the table holds are offered as filters.
    assert app.multiselect[0].options == ["Front"]


def test_edge_similarity_reads_either_direction() -> None:
    """The pair's similarity is found whichever frame lists the other."""
    indices = np.array([[1], [2], [0]])
    similarities = np.array([[0.99], [0.97], [0.95]], dtype=np.float32)
    assert edge_similarity(indices, similarities, frame=1, kept=0) == pytest.approx(
        0.99
    )
    assert edge_similarity(indices, similarities, frame=0, kept=2) == pytest.approx(
        0.95
    )
    with pytest.raises(ValueError, match="not neighbors"):
        edge_similarity(np.array([[1], [0], [0]]), similarities, frame=2, kept=1)
