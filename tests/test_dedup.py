"""Tests for the kNN graph and greedy near-duplicate suppression."""

from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
import pytest
import torch

from mcap_lancedb import (
    DEFAULT_MODEL,
    META_DEDUP_K,
    META_DEDUP_THRESHOLD,
    META_EMBEDDING_MODEL,
    TABLE_NAME,
    dedup,
)
from mcap_lancedb.dedup import (
    default_threshold,
    graph_from_columns,
    greedy_suppress,
    knn_graph,
    suppression_order,
)

CPU = torch.device("cpu")


def random_unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Draw ``n`` random L2-normalized vectors."""
    vectors = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_knn_graph_matches_brute_force() -> None:
    """Neighbors equal a full argsort of the similarity matrix, minus self."""
    vectors = random_unit_vectors(50, 16)
    graph = knn_graph(vectors, k=5, device=CPU)
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    expected = np.argsort(-similarity, axis=1)[:, :5]
    np.testing.assert_array_equal(graph.indices, expected)
    np.testing.assert_allclose(
        graph.similarities, np.take_along_axis(similarity, expected, 1), rtol=1e-5
    )


def test_knn_graph_chunking_does_not_change_the_result() -> None:
    """A tiny budget forces one-row chunks and gives the same graph."""
    vectors = random_unit_vectors(40, 8, seed=1)
    whole = knn_graph(vectors, k=4, device=CPU)
    chunked = knn_graph(vectors, k=4, device=CPU, budget_bytes=1)
    np.testing.assert_array_equal(whole.indices, chunked.indices)


def test_knn_graph_caps_k_and_excludes_self() -> None:
    """Neighbors are capped at n - 1 and never include the frame itself."""
    graph = knn_graph(random_unit_vectors(3, 4), k=10, device=CPU)
    assert graph.indices.shape == (3, 2)
    assert all(i not in row for i, row in enumerate(graph.indices))


def test_greedy_suppress_does_not_chain() -> None:
    """A~B and B~C with A and C dissimilar keeps C.

    Connected components would put all three in one cluster; greedy
    suppression only lets kept frames suppress.
    """
    indices = np.array([[1, 2], [0, 2], [1, 0]])
    similarities = np.array([[0.99, 0.50], [0.99, 0.99], [0.99, 0.50]])
    dup_of = greedy_suppress(indices, similarities, 0.95, order=np.array([0, 1, 2]))
    assert dup_of.tolist() == [-1, 0, -1]


def test_greedy_suppress_symmetrizes_edges() -> None:
    """An edge stored only on the later frame still lets the earlier one suppress."""
    indices = np.array([[2], [2], [0]])
    similarities = np.array([[0.10], [0.10], [0.99]])
    dup_of = greedy_suppress(indices, similarities, 0.95, order=np.array([0, 1, 2]))
    assert dup_of.tolist() == [-1, -1, 0]


def test_greedy_suppress_respects_order() -> None:
    """Whichever frame is visited first is the one kept."""
    indices = np.array([[1], [0]])
    similarities = np.array([[0.99], [0.99]])
    assert greedy_suppress(indices, similarities, 0.95, np.array([1, 0])).tolist() == [
        1,
        -1,
    ]


def test_suppression_order_prefers_objects_then_time() -> None:
    """More visible objects wins; ties go to the earliest frame."""
    order = suppression_order(
        np.array([1, 5, 5, 0]),
        np.array([10, 30, 20, 0]),
        np.array(["a", "b", "c", "d"]),
    )
    assert order.tolist() == [2, 1, 0, 3]


def test_graph_from_columns_round_trips() -> None:
    """Stored neighbor ids map back to the same positions."""
    frame_ids = pa.array(["a", "b", "c"])
    nn_ids = pa.array([["b", "c"], ["a", "c"], ["b", "a"]])
    nn_sims = pa.array([[0.9, 0.8], [0.9, 0.7], [0.7, 0.8]], pa.list_(pa.float32()))
    graph = graph_from_columns(frame_ids, nn_ids, nn_sims)
    assert graph.indices.tolist() == [[1, 2], [0, 2], [1, 0]]
    np.testing.assert_allclose(graph.similarities[0], [0.9, 0.8])


def make_frames_table(root: Path, embeddings: np.ndarray, objects: list[int]) -> None:
    """Create a minimal frames table holding just the columns dedup reads."""
    n, dim = embeddings.shape
    table = pa.table(
        {
            "frame_id": [f"f{i}" for i in range(n)],
            "num_visible_objects": pa.array(objects, pa.int32()),
            "timestamp": pa.array(range(n), pa.timestamp("us", tz="UTC")),
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(embeddings.ravel(), pa.float32()), dim
            ),
        }
    ).replace_schema_metadata({META_EMBEDDING_MODEL: "test-model"})
    lancedb.connect(root).create_table(TABLE_NAME, table)


def read_frames(root: Path) -> pa.Table:
    """Read the frames table back, with its schema metadata."""
    return lancedb.connect(root).open_table(TABLE_NAME).to_lance().to_table()


def test_dedup_cli_marks_merges_and_reruns(tmp_path: Path) -> None:
    """The CLI keeps the frame with more objects and records how it decided.

    A rerun replaces the dedup columns instead of failing on them, and the
    table's existing schema metadata survives both runs.
    """
    base = random_unit_vectors(3, 16, seed=2)
    near_copy = base[0] + 0.001 * random_unit_vectors(1, 16, seed=3)[0]
    embeddings = np.vstack([base[0], near_copy / np.linalg.norm(near_copy), base[1:]])
    make_frames_table(tmp_path, embeddings, objects=[1, 5, 0, 0])

    args = ["--db", str(tmp_path), "--device", "cpu", "--k", "2", "--threshold", "0.99"]
    dedup.main(args)
    dedup.main(args)

    frames = read_frames(tmp_path)
    assert frames.column("dup_of").to_pylist() == ["f1", None, None, None]
    assert frames.column("nn_frame_ids")[0].as_py()[0] == "f1"
    assert frames.schema.names.count("dup_of") == 1
    metadata = frames.schema.metadata
    assert metadata[META_EMBEDDING_MODEL.encode()] == b"test-model"
    assert metadata[META_DEDUP_THRESHOLD.encode()] == b"0.99"
    assert metadata[META_DEDUP_K.encode()] == b"2"


def test_dedup_cli_handles_a_single_frame(tmp_path: Path) -> None:
    """One frame has no neighbors: it is kept and its neighbor lists are empty."""
    make_frames_table(tmp_path, random_unit_vectors(1, 8), objects=[0])
    dedup.main(["--db", str(tmp_path), "--device", "cpu"])
    frames = read_frames(tmp_path)
    assert frames.column("dup_of").to_pylist() == [None]
    assert frames.column("nn_frame_ids").to_pylist() == [[]]


@pytest.mark.parametrize(
    "argv", [["--k", "0"], ["--threshold", "0"], ["--threshold", "1.5"]]
)
def test_dedup_cli_rejects_bad_arguments(argv: list[str]) -> None:
    """Out-of-range --k and --threshold fail at parse time, not mid-run."""
    with pytest.raises(SystemExit):
        dedup.parse_args(argv)


def test_graph_from_columns_rejects_unknown_neighbors() -> None:
    """A neighbor id missing from the table fails loudly, not as position -1."""
    with pytest.raises(ValueError, match="doesn't match"):
        graph_from_columns(
            pa.array(["a", "b"]),
            pa.array([["b"], ["gone"]]),
            pa.array([[0.9], [0.9]], pa.list_(pa.float32())),
        )


def test_dedup_collapses_a_stationary_stream(pipeline_db: Path) -> None:
    """On the pipeline table, a camera that sees the same image keeps one frame.

    scene-0002 stands still and repeats one image per camera. Other merges
    depend on the model, so only these streams have a fixed answer.
    """
    table = lancedb.connect(pipeline_db).open_table(TABLE_NAME)
    frames = (
        table.to_lance()
        .to_table(columns=["frame_id", "scene_name", "channel", "dup_of"])
        .to_pylist()
    )
    kept = {frame["frame_id"] for frame in frames if frame["dup_of"] is None}
    assert all(frame["dup_of"] in kept for frame in frames if frame["dup_of"])
    for channel in ("CAM_FRONT", "CAM_BACK"):
        stream = [
            frame
            for frame in frames
            if frame["scene_name"] == "scene-0002" and frame["channel"] == channel
        ]
        assert len(stream) == 3
        assert sum(frame["dup_of"] is None for frame in stream) <= 1
    stored = table.schema.metadata[META_DEDUP_THRESHOLD.encode()]
    assert float(stored) == default_threshold(DEFAULT_MODEL)


def test_dedup_cli_without_a_table_says_to_ingest(tmp_path: Path) -> None:
    """An empty database fails with the command to run first, not a traceback."""
    with pytest.raises(SystemExit, match="Run mcap-lancedb-ingest first"):
        dedup.main(["--db", str(tmp_path), "--device", "cpu"])


def test_dedup_cli_rejects_an_empty_table(tmp_path: Path) -> None:
    """A table with no frames has nothing to compare."""
    make_frames_table(tmp_path, np.zeros((0, 4), dtype=np.float32), objects=[])
    with pytest.raises(SystemExit, match="is empty"):
        dedup.main(["--db", str(tmp_path), "--device", "cpu"])
