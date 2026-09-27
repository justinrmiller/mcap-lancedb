"""Headless tests for the Streamlit viewer.

The AppTest cases drive the real app against the synthetic pipeline table, and
again against ``data/lancedb`` when ingest and dedup have run on nuScenes mini.
"""

import io
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
    db_path,
    edge_similarity,
    load_catalog,
    snapshot_of,
    suppress_at,
)
from mcap_lancedb.embed import SiglipEncoder

REPO = Path(__file__).parents[1]
APP = REPO / "src" / "mcap_lancedb" / "app.py"
MINI_DB = REPO / "data" / "lancedb"
TIMEOUT_S = 300


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
def mini_db() -> Path:
    """The nuScenes mini table, or a skip if it hasn't been built."""
    if not (MINI_DB / f"{TABLE_NAME}.lance").exists():
        pytest.skip("run mcap-lancedb-ingest and mcap-lancedb-dedup on mini first")
    return MINI_DB


@pytest.fixture(params=["synthetic", "mini"])
def viewer_db(request: pytest.FixtureRequest) -> Path:
    """The synthetic pipeline table, then the mini table when there is one."""
    return request.getfixturevalue(
        "pipeline_db" if request.param == "synthetic" else "mini_db"
    )


def start_viewer(db: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """Run the viewer once against a table."""
    monkeypatch.setenv("MCAP_LANCEDB_DB", str(db))
    started = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S).run()
    assert not started.exception
    return started


@pytest.fixture
def app(
    viewer_db: Path, shared_encoder: SiglipEncoder, monkeypatch: pytest.MonkeyPatch
) -> AppTest:
    """The viewer on each table, reusing the test session's encoder."""
    return start_viewer(viewer_db, monkeypatch)


def test_text_search_renders_results(app: AppTest) -> None:
    """A typed query fills the grid with thumbnails."""
    app.text_input(key="query").input("a bus at a bus stop").run()
    assert not app.exception
    assert len(app.image) > 0
    assert any(b.key and b.key.startswith("like-") for b in app.button)


def test_example_query_fills_the_query_box(app: AppTest) -> None:
    """Choosing an example copies it into the query and searches."""
    app.pills(key="example").set_value(EXAMPLE_QUERIES[0]).run()
    assert not app.exception
    assert app.text_input(key="query").value == EXAMPLE_QUERIES[0]


def test_filters_and_hide_duplicates(app: AppTest) -> None:
    """Prefiltered search runs with every filter set."""
    app.text_input(key="query").input("pedestrians on a crosswalk").run()
    app.multiselect[0].set_value(["CAM_FRONT"])
    app.segmented_control[0].set_value("Day")
    app.toggle[0].set_value(True)
    app.run()
    assert not app.exception


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


def test_threshold_slider_recomputes_removals(
    mini_db: Path, shared_encoder: SiglipEncoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Moving the threshold re-runs suppression and redraws the tab.

    Mini only: whether 0.95 removes more depends on the data.
    """
    app = start_viewer(mini_db, monkeypatch)
    removed_before = app.metric[2].value
    app.slider(key="threshold").set_value(0.95).run()
    assert not app.exception
    assert app.metric[2].value != removed_before


def test_viewer_reproduces_dedup_at_the_stored_threshold(viewer_db: Path) -> None:
    """At dedup's own threshold, the slider's re-run marks exactly what dedup did.

    The viewer and the CLI build their visit order from different sources
    (pandas and Arrow), so this pins that they agree frame for frame.
    """
    table = lancedb.connect(viewer_db).open_table(TABLE_NAME)
    stored = (table.schema.metadata or {}).get(META_DEDUP_THRESHOLD.encode())
    if stored is None:
        pytest.skip("run mcap-lancedb-dedup first")
    dataset = table.to_lance()
    snapshot = snapshot_of(str(viewer_db), dataset)
    frame_ids = load_catalog(snapshot)["frame_id"].to_numpy()
    dup_of = suppress_at(snapshot, float(stored))

    marked = dataset.to_table(columns=["frame_id", "dup_of"])
    assert marked.column("frame_id").to_pylist() == frame_ids.tolist()
    assert marked.column("dup_of").to_pylist() == [
        frame_ids[i] if i >= 0 else None for i in dup_of
    ]


def make_viewer_table(root: Path, model: str | None = "test-model") -> None:
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
    )
    if model is not None:
        table = table.replace_schema_metadata({META_EMBEDDING_MODEL: model})
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


def test_viewer_follows_a_reingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recreated table at the same version number isn't served from cache.

    Ingest drops and recreates the table, so its version numbers restart, and
    the same steps reach the same version again.
    """
    make_viewer_table(tmp_path)
    run_dedup = ["--db", str(tmp_path), "--device", "cpu", "--threshold"]
    dedup.main([*run_dedup, "0.98"])
    app = start_viewer(tmp_path, monkeypatch)
    before = lancedb.connect(tmp_path).open_table(TABLE_NAME).version
    assert app.metric[2].value == "12.5%"

    lancedb.connect(tmp_path).drop_table(TABLE_NAME)
    make_viewer_table(tmp_path)
    dedup.main([*run_dedup, "0.95"])
    assert lancedb.connect(tmp_path).open_table(TABLE_NAME).version == before
    app.run()
    assert not app.exception
    assert app.slider(key="threshold").value == pytest.approx(0.95)
    assert app.metric[2].value == "25.0%"


def test_db_path_keeps_uris_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    """An object-store URI reaches LanceDB with its "//" intact."""
    monkeypatch.setattr("sys.argv", ["app.py"])
    monkeypatch.setenv("MCAP_LANCEDB_DB", "s3://bucket/lancedb")
    assert db_path() == "s3://bucket/lancedb"
    monkeypatch.setattr("sys.argv", ["app.py", "--db", "gs://bucket/db"])
    assert db_path() == "gs://bucket/db"


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


def start_on_viewer_table(
    root: Path, monkeypatch: pytest.MonkeyPatch, model: str | None = "test-model"
) -> AppTest:
    """Run the viewer once on a fresh 8-frame table."""
    make_viewer_table(root, model)
    return start_viewer(root, monkeypatch)


def infos(app: AppTest) -> str:
    """Every info box's text, joined."""
    return "\n".join(str(box.value) for box in app.info)


def test_viewer_before_dedup_explains_how_to_mark_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without dedup, the tab says what to run and the filter can't be turned on."""
    app = start_on_viewer_table(tmp_path, monkeypatch)
    assert "haven't been marked yet" in infos(app)
    assert app.toggle[0].disabled


def test_viewer_rejects_a_table_without_its_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Text queries can't be embedded to match an unknown model, so say so."""
    app = start_on_viewer_table(tmp_path, monkeypatch, model=None)
    assert not app.exception
    assert "doesn't record its embedding model" in app.error[0].value


def test_image_search_recovers_from_a_missing_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A "more like this" frame that's gone (say, after re-ingest) resets search."""
    app = start_on_viewer_table(tmp_path, monkeypatch)
    app.session_state["anchor"] = "gone"
    app.run()
    assert not app.exception
    assert app.session_state["anchor"] is None


def test_image_search_with_no_matches_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filters that exclude every frame explain why the grid is empty."""
    app = start_on_viewer_table(tmp_path, monkeypatch)
    app.session_state["anchor"] = "f0"
    app.segmented_control[0].set_value("Night")
    app.run()
    assert not app.exception
    assert "No frames match these filters" in infos(app)


def test_threshold_that_removes_nothing_shows_no_clusters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Above every pair's similarity, the clusters section says so."""
    make_viewer_table(tmp_path)
    dedup.main(["--db", str(tmp_path), "--device", "cpu", "--threshold", "0.98"])
    app = start_viewer(tmp_path, monkeypatch)
    app.slider(key="threshold").set_value(0.999).run()
    assert not app.exception
    assert app.metric[2].value == "0.0%"
    assert "Nothing is removed at this threshold" in infos(app)
