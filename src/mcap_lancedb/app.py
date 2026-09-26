"""Streamlit viewer for semantic search and interactive near-duplicate removal.

Run from the project root so Streamlit picks up ``.streamlit/config.toml``:

    uv run streamlit run src/mcap_lancedb/app.py
    uv run streamlit run src/mcap_lancedb/app.py -- --db data/lancedb

The viewer never loads the image or embedding columns in bulk. Grids read inline
thumbnails; full-resolution images and embeddings are fetched per frame, on
demand, through the ``frame_id`` index.
"""

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import lance
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
import streamlit as st
import transformers
from lancedb.query import LanceVectorQueryBuilder

from mcap_lancedb import (
    DEFAULT_DB,
    META_DEDUP_THRESHOLD,
    META_EMBEDDING_MODEL,
    TABLE_NAME,
)
from mcap_lancedb.dedup import (
    default_threshold,
    graph_from_columns,
    greedy_suppress,
    suppression_order,
)
from mcap_lancedb.embed import SiglipEncoder
from mcap_lancedb.schema import (
    DEDUP_COLUMNS,
    EMBEDDING_COLUMN,
    IMAGE_COLUMN,
    THUMBNAIL_COLUMN,
)

# Examples chosen to work on nuScenes mini. Mini has no rain scene (only
# "after rain"), so there is deliberately no rain example.
EXAMPLE_QUERIES = (
    "a city street at night",
    "construction zone with traffic cones",
    "bus at a bus stop",
    "parking lot with parked cars",
    "pedestrians on a crosswalk",
)

CAMERA_LABELS = {
    "CAM_FRONT": "Front",
    "CAM_FRONT_RIGHT": "Front right",
    "CAM_BACK_RIGHT": "Back right",
    "CAM_BACK": "Back",
    "CAM_BACK_LEFT": "Back left",
    "CAM_FRONT_LEFT": "Front left",
}

# Light columns cached for filters, charts and tables. Never image or embedding.
CATALOG_COLUMNS = [
    "frame_id",
    "scene_name",
    "scene_description",
    "channel",
    "location",
    "is_night",
    "timestamp",
    "ego_speed_mps",
    "num_visible_objects",
    "visible_categories",
]
RESULT_COLUMNS = [
    "frame_id",
    "scene_name",
    "scene_description",
    "channel",
    "ego_speed_mps",
    THUMBNAIL_COLUMN,
    # Named explicitly: Lance is phasing out projecting it automatically.
    "_distance",
]

SPEED_BINS = [-np.inf, 0.5, 3.0, 8.0, np.inf]
SPEED_LABELS = ["Under 0.5 m/s", "0.5 to 3 m/s", "3 to 8 m/s", "Over 8 m/s"]

REFINE_FACTOR = 10

ACCENT = "#1F4E9E"
GRID_COLUMNS = 4
CLUSTER_MEMBERS_SHOWN = 7

SIGN_CSS = """
<style>
.sign {
  background: #1F4E9E;
  color: #FFFFFF;
  border-radius: 12px;
  padding: 6px;
  margin-bottom: 0.5rem;
}
.sign-face {
  border: 3px solid #FFFFFF;
  border-radius: 8px;
  padding: 0.9rem 1.2rem 0.8rem;
}
.sign h1 {
  color: #FFFFFF;
  font-size: 2.1rem;
  font-weight: 700;
  line-height: 1.1;
  margin: 0;
  padding: 0;
}
.sign p {
  color: #FFFFFF;
  font-size: 1.05rem;
  margin: 0.35rem 0 0;
}
</style>
"""


def db_path() -> str:
    """Resolve the database path from ``--db``, then the environment.

    Returns:
        ``--db`` if given after ``--`` on the command line, else
        ``MCAP_LANCEDB_DB``, else the default ``data/lancedb``.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", default=os.environ.get("MCAP_LANCEDB_DB"))
    known, _ = parser.parse_known_args(sys.argv[1:])
    return str(Path(known.db or DEFAULT_DB))


@st.cache_resource
def connect(path: str) -> lancedb.DBConnection:
    """Connect once per database path.

    A zero consistency interval makes every read check for a newer version,
    so a dedup rerun shows up without restarting the viewer.
    """
    return lancedb.connect(path, read_consistency_interval=timedelta(0))


def open_table(path: str) -> lancedb.table.Table | None:
    """Open the frames table, or return ``None`` if ingest hasn't run."""
    db = connect(path)
    if TABLE_NAME not in db.list_tables().tables:
        return None
    return db.open_table(TABLE_NAME)


def dataset_at(path: str, version: int) -> lance.LanceDataset:
    """Open the table's Lance dataset pinned to one version."""
    table = connect(path).open_table(TABLE_NAME)
    return table.to_lance().checkout_version(version)


@st.cache_resource(show_spinner="Loading the embedding model")
def load_encoder(model_id: str) -> SiglipEncoder:
    """Load the model that embedded the table, once per process."""
    # The SigLIP 2 configs carry out-of-vocabulary bos/eos ids that transformers
    # warns about on every load; the ids are never used for embedding.
    transformers.logging.set_verbosity_error()
    return SiglipEncoder(model_id)


@st.cache_data(show_spinner="Reading frame metadata")
def load_catalog(path: str, version: int) -> pd.DataFrame:
    """Load the light metadata columns for every frame.

    Args:
        path: Database path.
        version: Table version, part of the cache key so a rerun of dedup or
            ingest invalidates it.

    Returns:
        One row per frame, in table order.
    """
    return dataset_at(path, version).to_table(columns=CATALOG_COLUMNS).to_pandas()


@st.cache_data(show_spinner="Reading the neighbor graph")
def load_graph(path: str, version: int) -> tuple[np.ndarray, np.ndarray]:
    """Load the stored kNN graph as positions into the catalog's row order.

    Args:
        path: Database path.
        version: Table version, for cache invalidation.

    Returns:
        ``(indices, similarities)``, each of shape ``(n, k)``.
    """
    graph_table = dataset_at(path, version).to_table(
        columns=["frame_id", "nn_frame_ids", "nn_similarity"]
    )
    graph = graph_from_columns(
        graph_table.column("frame_id").combine_chunks(),
        graph_table.column("nn_frame_ids").combine_chunks(),
        graph_table.column("nn_similarity").combine_chunks(),
    )
    return graph.indices, graph.similarities


@st.cache_data(show_spinner="Removing near-duplicates")
def suppress_at(path: str, version: int, threshold: float) -> np.ndarray:
    """Re-run greedy suppression on the stored graph at a new threshold.

    Args:
        path: Database path.
        version: Table version, for cache invalidation.
        threshold: Cosine similarity for a near-duplicate.

    Returns:
        For each frame, the catalog position of the frame that suppressed it,
        or ``-1`` if kept.
    """
    catalog = load_catalog(path, version)
    indices, similarities = load_graph(path, version)
    order = suppression_order(
        catalog["num_visible_objects"].to_numpy(),
        catalog["timestamp"].to_numpy(),
        catalog["frame_id"].to_numpy(),
    )
    return greedy_suppress(indices, similarities, threshold, order)


@st.cache_data(show_spinner=False)
def embed_query(model_id: str, text: str) -> np.ndarray:
    """Embed a text query with the table's model."""
    return load_encoder(model_id).encode_text([text])[0]


def sql_string(value: str) -> str:
    """Quote a string literal for a Lance SQL filter."""
    return "'" + value.replace("'", "''") + "'"


def sql_in(column: str, values: Sequence[str]) -> str:
    """Build a ``column IN (...)`` clause."""
    return f"{column} IN ({', '.join(sql_string(v) for v in values)})"


def build_where(
    *,
    cameras: Sequence[str],
    locations: Sequence[str],
    time_of_day: str,
    min_pedestrians: int,
    categories: Sequence[str],
    hide_duplicates: bool,
    exclude_frame: str | None = None,
) -> str | None:
    """Turn the filter widgets into one SQL prefilter.

    Args:
        cameras: Channels to keep. Empty keeps all.
        locations: Locations to keep. Empty keeps all.
        time_of_day: ``Any``, ``Day`` or ``Night``.
        min_pedestrians: Minimum pedestrians visible in the frame.
        categories: Object categories that must all be visible.
        hide_duplicates: Keep only frames dedup kept.
        exclude_frame: A frame to leave out, such as the query frame itself.

    Returns:
        A SQL predicate, or ``None`` when nothing is filtered.
    """
    clauses: list[str] = []
    if cameras:
        clauses.append(sql_in("channel", cameras))
    if locations:
        clauses.append(sql_in("location", locations))
    if time_of_day != "Any":
        clauses.append(f"is_night = {str(time_of_day == 'Night').lower()}")
    if min_pedestrians > 0:
        clauses.append(f"num_pedestrians >= {int(min_pedestrians)}")
    clauses.extend(
        f"array_has(visible_categories, {sql_string(name)})" for name in categories
    )
    if hide_duplicates:
        clauses.append("dup_of IS NULL")
    if exclude_frame:
        clauses.append(f"frame_id != {sql_string(exclude_frame)}")
    return " AND ".join(clauses) or None


def search(
    table: lancedb.table.Table, vector: np.ndarray, where: str | None, limit: int
) -> pd.DataFrame:
    """Run a prefiltered cosine vector search.

    Args:
        table: The frames table.
        vector: L2-normalized query embedding.
        where: SQL prefilter, applied before the nearest-neighbor search so the
            limit is always filled when enough frames match.
        limit: Number of results.

    Returns:
        Result rows with a ``similarity`` column, most similar first.
    """
    # A vector argument always yields a vector query builder; search() is just
    # typed as the base class.
    builder = cast(
        "LanceVectorQueryBuilder",
        table.search(vector, vector_column_name=EMBEDDING_COLUMN),
    )
    # With an IVF_PQ index, distances are PQ approximations and can be far off.
    # Refining re-ranks limit * REFINE_FACTOR candidates by exact distance, so
    # the similarity shown is exact. Without an index it changes nothing.
    query = (
        builder.distance_type("cosine")
        .refine_factor(REFINE_FACTOR)
        .select(RESULT_COLUMNS)
        .limit(limit)
    )
    if where:
        query = query.where(where, prefilter=True)
    results = query.to_pandas()
    results["similarity"] = 1.0 - results["_distance"]
    return results


def fetch_frames(
    table: lancedb.table.Table, frame_ids: Sequence[str], columns: Sequence[str]
) -> pa.Table:
    """Fetch a few frames by id through the ``frame_id`` index.

    Args:
        table: The frames table.
        frame_ids: Frames to fetch.
        columns: Columns to read, in addition to ``frame_id``.

    Returns:
        The matching rows, in no particular order.
    """
    return table.to_lance().to_table(
        columns=["frame_id", *columns], filter=sql_in("frame_id", frame_ids)
    )


def camera_label(channel: str) -> str:
    """Human-readable camera name, such as ``Front left``."""
    return CAMERA_LABELS.get(channel, channel)


@st.dialog("Full resolution", width="large")
def show_full_resolution(frame_id: str, caption: str) -> None:
    """Load one original JPEG from the table and show it."""
    table = open_table(db_path())
    if table is None:
        return
    image = fetch_frames(table, [frame_id], [IMAGE_COLUMN]).column(IMAGE_COLUMN)
    st.image(image[0].as_py(), width="stretch")
    st.caption(caption)


def set_anchor(frame_id: str | None) -> None:
    """Switch to image search from a frame, or back to text with ``None``."""
    st.session_state.anchor = frame_id


def set_query(text: str) -> None:
    """Replace the query text and leave image search."""
    st.session_state.query = text
    set_anchor(None)


def pick_example() -> None:
    """Copy the chosen example into the query box, then clear the pill."""
    choice = st.session_state.example
    if choice:
        set_query(choice)
    st.session_state.example = None


def render_header(catalog: pd.DataFrame, model_id: str) -> None:
    """Draw the sign-style header and a one-line summary of the table."""
    st.html(
        SIGN_CSS + '<div class="sign"><div class="sign-face">'
        "<h1>mcap-lancedb</h1>"
        "<p>Find the right 1% of your camera frames</p>"
        "</div></div>"
    )
    st.caption(
        f"{len(catalog):,} frames from {catalog['scene_name'].nunique()} scenes, "
        f"embedded with {model_id}"
    )


def render_filters(catalog: pd.DataFrame, dedup_threshold: float | None) -> str | None:
    """Draw the search filters and return the SQL prefilter they describe.

    Args:
        catalog: Light metadata for every frame, for the filter options.
        dedup_threshold: Threshold the stored ``dup_of`` column was built at,
            or ``None`` if dedup hasn't run.

    Returns:
        The SQL prefilter, or ``None`` when nothing is filtered.
    """
    has_dedup = dedup_threshold is not None
    st.subheader("Filters")
    # Only the cameras this table has; ingest may have run with --channels.
    present = set(catalog["channel"].unique())
    cameras = st.multiselect(
        "Cameras",
        [channel for channel in CAMERA_LABELS if channel in present],
        format_func=camera_label,
        placeholder="All cameras",
    )
    locations = st.multiselect(
        "Locations",
        sorted(catalog["location"].unique()),
        placeholder="All locations",
    )
    time_of_day = st.segmented_control(
        "Time of day", ["Any", "Day", "Night"], default="Any", required=True
    )
    min_pedestrians = st.number_input(
        "Minimum pedestrians in view", min_value=0, max_value=50, value=0
    )
    categories = st.multiselect(
        "Must show",
        sorted({c for cats in catalog["visible_categories"] for c in cats}),
        placeholder="Any objects",
        help="Every chosen object category must be visible in the frame.",
    )
    hide_duplicates = st.toggle(
        "Hide near-duplicates",
        disabled=not has_dedup,
        help=(
            f"Hide the frames `mcap-lancedb-dedup` marked at threshold "
            f"{dedup_threshold:.3f}. The slider on the Near-duplicates tab "
            "doesn't change this; rerun dedup with `--threshold` to."
            if has_dedup
            else "Run `uv run mcap-lancedb-dedup` to turn this on."
        ),
    )
    return build_where(
        cameras=cameras,
        locations=locations,
        time_of_day=time_of_day or "Any",
        min_pedestrians=int(min_pedestrians),
        categories=categories,
        hide_duplicates=hide_duplicates and has_dedup,
        exclude_frame=st.session_state.anchor,
    )


def render_result_card(row: dict[str, Any]) -> None:
    """Draw one search result with its actions."""
    frame_id = row["frame_id"]
    camera = camera_label(row["channel"])
    with st.container(border=True):
        st.image(row[THUMBNAIL_COLUMN], width="stretch")
        st.markdown(f"**{row['scene_name']}**, {camera.lower()} camera")
        st.markdown(
            f"Similarity {row['similarity']:.3f}  \n"
            f"Ego speed {row['ego_speed_mps']:.1f} m/s"
        )
        st.caption(row["scene_description"])
        # Stacked, not side by side: a quarter-width card truncates two labels.
        st.button(
            "More like this",
            key=f"like-{frame_id}",
            icon=":material/image_search:",
            on_click=set_anchor,
            args=(frame_id,),
            width="stretch",
        )
        if st.button(
            "Full resolution",
            key=f"full-{frame_id}",
            icon=":material/open_in_full:",
            width="stretch",
        ):
            show_full_resolution(
                frame_id, f"{row['scene_name']}, {camera.lower()} camera"
            )


def render_results(results: pd.DataFrame) -> None:
    """Lay out search results as a thumbnail grid."""
    if results.empty:
        st.info("No frames match these filters. Loosen a filter and try again.")
        return
    records = results.to_dict("records")
    for start in range(0, len(records), GRID_COLUMNS):
        columns = st.columns(GRID_COLUMNS)
        batch = records[start : start + GRID_COLUMNS]
        for column, row in zip(columns, batch, strict=False):
            with column:
                render_result_card(row)


def anchor_vector(table: lancedb.table.Table, frame_id: str) -> np.ndarray:
    """Read one frame's stored embedding for image-to-image search."""
    column = fetch_frames(table, [frame_id], [EMBEDDING_COLUMN]).column(
        EMBEDDING_COLUMN
    )
    return np.asarray(column[0].as_py(), dtype=np.float32)


def render_search_tab(
    table: lancedb.table.Table,
    catalog: pd.DataFrame,
    model_id: str,
    dedup_threshold: float | None,
) -> None:
    """Draw the text and image search tab."""
    filters_column, results_column = st.columns([1, 3], gap="large")
    with filters_column:
        where = render_filters(catalog, dedup_threshold)
    with results_column:
        st.text_input(
            "Describe the frames you want",
            key="query",
            placeholder="For example, a truck turning left at an intersection",
            on_change=set_anchor,
            args=(None,),
        )
        st.pills(
            "Or try an example",
            EXAMPLE_QUERIES,
            key="example",
            on_change=pick_example,
            wrap=True,
        )
        limit = st.slider(
            "Results", min_value=8, max_value=64, value=24, step=4, key="limit"
        )

        anchor = st.session_state.anchor
        matches = catalog.loc[catalog["frame_id"] == anchor]
        if anchor and matches.empty:
            # The frame is gone, for example after a re-ingest.
            set_anchor(None)
            anchor = None
        if anchor:
            anchor_row = matches.iloc[0]
            st.markdown(
                f"Showing frames like **{anchor_row['scene_name']}**, "
                f"{camera_label(anchor_row['channel']).lower()} camera."
            )
            st.button(
                "Back to text search",
                icon=":material/arrow_back:",
                on_click=set_anchor,
                args=(None,),
            )
            vector = anchor_vector(table, anchor)
        elif st.session_state.query.strip():
            vector = embed_query(model_id, st.session_state.query.strip())
            st.caption(
                "SigLIP scores text against images low in absolute terms, "
                "usually 0.1 to 0.2. Compare results by rank, not by score."
            )
        else:
            st.info("Type a description or pick an example to search the frames.")
            return
        render_results(search(table, vector, where, limit))


def render_dedup_summary(catalog: pd.DataFrame, removed: np.ndarray) -> None:
    """Draw the removal metrics, the speed chart and the per-scene table."""
    total, dropped = len(catalog), int(removed.sum())
    frames, kept, share = st.columns(3)
    frames.metric("Frames", f"{total:,}", border=True)
    kept.metric("Kept", f"{total - dropped:,}", border=True)
    share.metric("Removed", f"{dropped / total:.1%}", border=True)

    frame = catalog.assign(
        removed=removed,
        speed=pd.cut(
            catalog["ego_speed_mps"], SPEED_BINS, labels=SPEED_LABELS, right=False
        ),
    )
    by_speed = (
        frame.groupby("speed", observed=False)
        .agg(frames=("removed", "size"), rate=("removed", "mean"))
        .reset_index()
    )
    by_scene = (
        frame.groupby(["scene_name", "scene_description"])
        .agg(frames=("removed", "size"), removed=("removed", "sum"))
        .reset_index()
    )
    by_scene["kept"] = by_scene["frames"] - by_scene["removed"]
    by_scene["share"] = by_scene["removed"] / by_scene["frames"]

    chart_column, table_column = st.columns([2, 3], gap="large")
    with chart_column:
        st.markdown("**Removal rate by ego speed**")
        st.bar_chart(
            by_speed.assign(rate=100 * by_speed["rate"].fillna(0.0)),
            x="speed",
            y="rate",
            # Labels follow the data columns, not the screen axes.
            x_label="Ego speed",
            y_label="Frames removed (%)",
            color=ACCENT,
            horizontal=True,
            sort=False,
            height=260,
        )
    table_column.dataframe(
        by_scene.sort_values("share", ascending=False),
        hide_index=True,
        # The description goes last, where truncating it on narrow screens
        # costs nothing.
        column_order=["scene_name", "share", "frames", "kept", "scene_description"],
        column_config={
            "scene_name": st.column_config.TextColumn("Scene", width=110),
            "scene_description": "Description",
            "frames": "Frames",
            "kept": "Kept",
            "share": st.column_config.ProgressColumn(
                "Removed", format="percent", min_value=0.0, max_value=1.0, width=150
            ),
        },
    )


def edge_similarity(
    indices: np.ndarray, similarities: np.ndarray, frame: int, kept: int
) -> float:
    """Look up a duplicate's similarity to the frame that suppressed it.

    Suppression only follows edges of the stored kNN graph, so the pair is always
    an edge: the duplicate lists the kept frame, the kept frame lists the
    duplicate, or both. Cosine similarity is symmetric, so either entry is exact.

    Args:
        indices: Stored graph neighbor positions, shape ``(n, k)``.
        similarities: Stored graph similarities, shape ``(n, k)``.
        frame: Position of the duplicate.
        kept: Position of the kept frame that suppressed it.

    Returns:
        Their cosine similarity.

    Raises:
        ValueError: If the pair is not an edge, meaning the graph and the
            suppression result don't belong together.
    """
    for source, target in ((frame, kept), (kept, frame)):
        hits = np.flatnonzero(indices[source] == target)
        if hits.size:
            return float(similarities[source, hits[0]])
    msg = f"Frames {frame} and {kept} are not neighbors in the stored graph."
    raise ValueError(msg)


def render_clusters(
    table: lancedb.table.Table,
    catalog: pd.DataFrame,
    dup_of: np.ndarray,
    graph: tuple[np.ndarray, np.ndarray],
) -> None:
    """Show the largest clusters as a kept frame plus its duplicates.

    Similarities come from the stored graph, and thumbnails are fetched for the
    frames on screen only, so no embeddings are read at all.
    """
    st.subheader("Largest clusters")
    sizes = np.bincount(dup_of[dup_of >= 0], minlength=len(catalog))
    count_column, _ = st.columns([1, 4])
    count = int(
        count_column.number_input(
            "Clusters to show", min_value=1, max_value=20, value=5
        )
    )
    representatives = [int(i) for i in np.argsort(-sizes)[:count] if sizes[i]]
    if not representatives:
        st.info("Nothing is removed at this threshold. Lower it to form clusters.")
        return

    indices, similarities = graph
    clusters = {
        rep: sorted(
            (edge_similarity(indices, similarities, int(member), rep), int(member))
            for member in np.flatnonzero(dup_of == rep)
        )
        for rep in representatives
    }
    frame_ids = catalog["frame_id"].to_numpy()
    on_screen = [
        frame_ids[i]
        for rep, scored in clusters.items()
        for i in [rep, *(member for _, member in scored[:CLUSTER_MEMBERS_SHOWN])]
    ]
    fetched = fetch_frames(table, on_screen, [THUMBNAIL_COLUMN])
    thumbnails = dict(
        zip(
            fetched.column("frame_id").to_pylist(),
            fetched.column(THUMBNAIL_COLUMN).to_pylist(),
            strict=True,
        )
    )

    for rep, scored in clusters.items():
        row = catalog.iloc[rep]
        st.markdown(
            f"**{row['scene_name']}**, {camera_label(row['channel']).lower()} camera: "
            f"kept 1, removed {len(scored)}. Least similar duplicates first."
        )
        columns = st.columns(CLUSTER_MEMBERS_SHOWN + 1)
        columns[0].image(thumbnails[frame_ids[rep]], width="stretch")
        columns[0].markdown("**Kept**")
        shown = scored[:CLUSTER_MEMBERS_SHOWN]
        for column, (similarity, member) in zip(columns[1:], shown, strict=False):
            column.image(thumbnails[frame_ids[member]], width="stretch")
            column.caption(f"Similarity {similarity:.3f}")
        if len(scored) > CLUSTER_MEMBERS_SHOWN:
            st.caption(f"And {len(scored) - CLUSTER_MEMBERS_SHOWN} more.")


def render_dedup_tab(
    table: lancedb.table.Table,
    *,
    catalog: pd.DataFrame,
    path: str,
    version: int,
    model_id: str,
    stored_threshold: float | None,
) -> None:
    """Draw the near-duplicate tab."""
    if stored_threshold is None:
        st.info(
            "Near-duplicates haven't been marked yet. Run "
            "`uv run mcap-lancedb-dedup`, then reload this page."
        )
        return
    # A keyed slider keeps its position across reruns and ignores a changed
    # default, so after dedup reruns it would still show the old threshold.
    # Move it to the stored value whenever the table itself changes.
    if st.session_state.get("threshold_source") != (path, version):
        st.session_state.threshold = stored_threshold
        st.session_state.threshold_source = (path, version)
    threshold = st.slider(
        "Near-duplicate threshold",
        key="threshold",
        # Widened when dedup ran with a --threshold outside the usual range, so
        # the stored value is always a valid starting position.
        min_value=min(0.90, stored_threshold),
        max_value=max(0.999, stored_threshold),
        step=0.001,
        format="%.3f",
        help=(
            f"Cosine similarity at or above which two frames count as "
            f"near-duplicates. Dedup used {stored_threshold:.3f}; the default "
            f"for {model_id} is {default_threshold(model_id):.3f}."
        ),
    )
    st.caption(
        "Moving the slider re-runs greedy suppression on the stored neighbor "
        "graph. Nothing in the table changes."
    )
    dup_of = suppress_at(path, version, threshold)
    render_dedup_summary(catalog, dup_of >= 0)
    render_clusters(table, catalog, dup_of, load_graph(path, version))


def main() -> None:
    """Render the viewer."""
    st.set_page_config(
        page_title="mcap-lancedb", page_icon=":material/signpost:", layout="wide"
    )
    st.session_state.setdefault("query", "")
    st.session_state.setdefault("anchor", None)

    path = db_path()
    table = open_table(path)
    if table is None:
        st.info(
            f"There is no {TABLE_NAME} table in `{path}` yet. Create it with "
            "`uv run mcap-lancedb-ingest`."
        )
        return

    version = table.version
    schema = table.schema
    metadata = schema.metadata or {}
    model = metadata.get(META_EMBEDDING_MODEL.encode())
    if model is None:
        st.error(
            f"The {TABLE_NAME} table in `{path}` doesn't record its embedding "
            "model, so text queries can't be embedded to match it. Recreate it "
            "with `uv run mcap-lancedb-ingest`."
        )
        return
    model_id = model.decode()

    # The threshold alone isn't proof: a failed rerun of dedup can leave the
    # metadata behind after dropping the columns.
    stored = metadata.get(META_DEDUP_THRESHOLD.encode())
    columns_present = all(name in schema.names for name in DEDUP_COLUMNS)
    dedup_threshold = float(stored) if stored is not None and columns_present else None

    catalog = load_catalog(path, version)
    render_header(catalog, model_id)
    search_tab, dedup_tab = st.tabs(["Search", "Near-duplicates"])
    with search_tab:
        render_search_tab(table, catalog, model_id, dedup_threshold)
    with dedup_tab:
        render_dedup_tab(
            table,
            catalog=catalog,
            path=path,
            version=version,
            model_id=model_id,
            stored_threshold=dedup_threshold,
        )


if __name__ == "__main__":
    main()
