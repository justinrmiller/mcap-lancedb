"""Mark near-duplicate frames with an exact kNN graph and greedy suppression.

Near-duplicates are marked, never deleted: each frame gets its k nearest
neighbors and a ``dup_of`` column that points at the frame that suppressed it
(null means kept). The viewer re-runs the suppression at any threshold from the
stored graph, so choosing a threshold never needs another pass over embeddings.

Usage:
    uv run mcap-lancedb-dedup --db data/lancedb
"""

import argparse
import logging
import time
from collections.abc import Sequence
from typing import NamedTuple

import lance
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from mcap_lancedb import (
    DEFAULT_DB,
    DEFAULT_MODEL,
    FAST_MODEL,
    META_DEDUP_K,
    META_DEDUP_THRESHOLD,
    META_EMBEDDING_MODEL,
    TABLE_NAME,
)
from mcap_lancedb.cli import positive_int
from mcap_lancedb.embed import resolve_device
from mcap_lancedb.schema import DEDUP_COLUMNS, DEDUP_FIELDS, EMBEDDING_COLUMN

logger = logging.getLogger(__name__)

DEFAULT_K = 32

# Each chunk of the similarity matrix is (rows x n) float32; this caps it.
SIMILARITY_BUDGET_BYTES = 256 * 1024**2

# Calibrated per model on nuScenes mini, because similarity scales differ between
# checkpoints. Consecutive 2 Hz keyframes are very alike even in motion (median
# nearest-neighbor similarity is ~0.97), so a generic 0.95 would remove half of
# the moving frames. These values sit where pairs stop showing visible change
# and where removals concentrate most in stationary frames. See the README.
DEFAULT_THRESHOLDS: dict[str, float] = {DEFAULT_MODEL: 0.985, FAST_MODEL: 0.98}
FALLBACK_THRESHOLD = 0.98

PRIORITY_COLUMNS = ("frame_id", "num_visible_objects", "timestamp")


class KnnGraph(NamedTuple):
    """Top-k cosine neighbors of every frame, most similar first.

    Attributes:
        indices: Row positions of each frame's neighbors, shape ``(n, k)``.
        similarities: Matching cosine similarities, shape ``(n, k)``.
    """

    indices: np.ndarray
    similarities: np.ndarray


def default_threshold(model_id: str) -> float:
    """Return the calibrated near-duplicate threshold for a model.

    Args:
        model_id: The embedding model recorded in the table's schema metadata.

    Returns:
        The calibrated threshold, or ``FALLBACK_THRESHOLD`` for other models.
    """
    return DEFAULT_THRESHOLDS.get(model_id, FALLBACK_THRESHOLD)


def knn_graph(
    embeddings: np.ndarray,
    k: int,
    device: torch.device,
    budget_bytes: int = SIMILARITY_BUDGET_BYTES,
) -> KnnGraph:
    """Build an exact top-k cosine kNN graph with chunked matrix multiplies.

    Exact search keeps the graph reproducible run to run, which an ANN index
    does not guarantee, and it takes minutes on a GPU even at trainval scale.

    Args:
        embeddings: L2-normalized vectors, shape ``(n, dim)``.
        k: Neighbors per frame. Capped at ``n - 1``.
        device: Device for the matrix multiplies.
        budget_bytes: Cap on the size of each similarity-matrix chunk.

    Returns:
        The graph, excluding self-matches.
    """
    n = len(embeddings)
    k = min(k, n - 1)
    if k < 1:
        return KnnGraph(np.zeros((n, 0), np.int64), np.zeros((n, 0), np.float32))

    matrix = torch.tensor(embeddings, dtype=torch.float32, device=device)
    rows_per_chunk = max(1, budget_bytes // (4 * n))
    indices = np.empty((n, k), dtype=np.int64)
    similarities = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, rows_per_chunk):
        stop = min(start + rows_per_chunk, n)
        chunk = matrix[start:stop] @ matrix.T
        diagonal = torch.arange(stop - start, device=device)
        chunk[diagonal, diagonal + start] = -torch.inf
        values, positions = torch.topk(chunk, k, dim=1)
        similarities[start:stop] = values.cpu().numpy()
        indices[start:stop] = positions.cpu().numpy()
    return KnnGraph(indices, similarities)


def suppression_order(
    num_visible_objects: np.ndarray, timestamps: np.ndarray, frame_ids: np.ndarray
) -> np.ndarray:
    """Order frames by how much they deserve to be kept.

    Args:
        num_visible_objects: Objects visible in each frame.
        timestamps: Capture times, any sortable dtype.
        frame_ids: Frame ids, used only to break exact ties deterministically.

    Returns:
        Row positions: most visible objects first, then earliest capture.
    """
    return np.lexsort((frame_ids, timestamps, -num_visible_objects))


def greedy_suppress(
    indices: np.ndarray,
    similarities: np.ndarray,
    threshold: float,
    order: np.ndarray,
) -> np.ndarray:
    """Mark near-duplicates with greedy, NMS-style suppression.

    Edges at or above ``threshold`` are made symmetric. Frames are visited in
    ``order``; each unvisited frame is kept and suppresses its unvisited
    neighbors. A suppressed frame never suppresses anything itself, which is
    what stops the A~B~C chaining that makes connected components drift until
    A and C look nothing alike.

    Args:
        indices: kNN neighbor positions, shape ``(n, k)``.
        similarities: kNN similarities, shape ``(n, k)``.
        threshold: Minimum cosine similarity for a near-duplicate.
        order: Visit order from ``suppression_order``.

    Returns:
        For each frame, the position of the kept frame that suppressed it, or
        ``-1`` if the frame is kept.
    """
    n = len(indices)
    rows, cols = np.nonzero(similarities >= threshold)
    neighbors = indices[rows, cols]
    sources = np.concatenate([rows, neighbors])
    targets = np.concatenate([neighbors, rows])
    by_source = np.argsort(sources, kind="stable")
    targets = targets[by_source]
    offsets = np.searchsorted(sources[by_source], np.arange(n + 1))

    dup_of = np.full(n, -1, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    for node in order:
        if visited[node]:
            continue
        visited[node] = True
        adjacent = targets[offsets[node] : offsets[node + 1]]
        adjacent = adjacent[~visited[adjacent]]
        visited[adjacent] = True
        dup_of[adjacent] = node
    return dup_of


def graph_from_columns(
    frame_ids: pa.Array, nn_frame_ids: pa.ListArray, nn_similarity: pa.ListArray
) -> KnnGraph:
    """Rebuild a positional ``KnnGraph`` from the stored dedup columns.

    Args:
        frame_ids: Frame ids in table order.
        nn_frame_ids: ``list<string>`` neighbor ids, one list per frame.
        nn_similarity: ``list<float32>`` similarities, one list per frame.

    Returns:
        The graph with neighbor ids mapped to positions in ``frame_ids``.

    Raises:
        ValueError: If the lists differ in length or name a frame that isn't in
            ``frame_ids``. Either means the columns are stale, and a missing id
            would otherwise become position -1, silently the last frame.
    """
    n = len(frame_ids)
    flat_ids = nn_frame_ids.flatten().to_numpy(zero_copy_only=False)
    positions = pd.Index(frame_ids.to_numpy(zero_copy_only=False)).get_indexer(flat_ids)
    k = len(flat_ids) // n if n else 0
    lengths = nn_frame_ids.value_lengths().to_numpy(zero_copy_only=False)
    if (lengths != k).any() or (positions < 0).any():
        msg = "The stored neighbor graph doesn't match the table's frames."
        raise ValueError(msg)
    return KnnGraph(
        positions.astype(np.int64).reshape(n, k),
        nn_similarity.flatten().to_numpy().reshape(n, k),
    )


def write_dedup_columns(
    dataset: lance.LanceDataset,
    frame_ids: pa.Array,
    graph: KnnGraph,
    dup_of: np.ndarray,
    threshold: float,
) -> None:
    """Merge the kNN graph and ``dup_of`` into the table as new columns.

    Existing dedup columns are dropped first, so reruns replace rather than
    fail. The threshold and k are recorded in the table's schema metadata.

    Each step is its own commit. The old threshold is cleared before anything
    else and the new one recorded last, so no version of the table pairs a
    threshold with columns it didn't produce, even if a step fails.

    Args:
        dataset: The frames table's Lance dataset.
        frame_ids: Frame ids in the order the graph was built.
        graph: The kNN graph.
        dup_of: Output of ``greedy_suppress``.
        threshold: Threshold that produced ``dup_of``.
    """
    recorded = dataset.schema.metadata or {}
    if META_DEDUP_THRESHOLD.encode() in recorded:
        dataset.update_schema_metadata({META_DEDUP_THRESHOLD: None, META_DEDUP_K: None})
    stale = [name for name in DEDUP_COLUMNS if name in dataset.schema.names]
    if stale:
        dataset.drop_columns(stale)

    n, k = graph.indices.shape
    # Every frame has exactly k neighbors, so list i spans [i * k, (i + 1) * k).
    # k is 0 for a one-frame table, which leaves every list empty.
    offsets = pa.array(np.arange(n + 1, dtype=np.int32) * k)
    columns = pa.Table.from_arrays(
        [
            frame_ids,
            pa.ListArray.from_arrays(
                offsets, frame_ids.take(pa.array(graph.indices.ravel()))
            ),
            pa.ListArray.from_arrays(
                offsets, pa.array(graph.similarities.ravel(), pa.float32())
            ),
            frame_ids.take(pa.array(dup_of, mask=dup_of < 0)),
        ],
        schema=pa.schema([pa.field("frame_id", pa.string()), *DEDUP_FIELDS]),
    )
    dataset.merge(columns, left_on="frame_id")
    # str() is the shortest text that parses back to the same float, so the
    # viewer's slider starts at exactly the threshold that produced dup_of.
    dataset.update_schema_metadata(
        {META_DEDUP_THRESHOLD: str(threshold), META_DEDUP_K: str(k)}
    )


def similarity_threshold(value: str) -> float:
    """Parse a cosine-similarity threshold in (0, 1].

    Args:
        value: The raw argument.

    Returns:
        The parsed threshold.

    Raises:
        argparse.ArgumentTypeError: If the value is outside (0, 1].
    """
    threshold = float(value)
    if not 0.0 < threshold <= 1.0:
        msg = f"must be in (0, 1], got {threshold}"
        raise argparse.ArgumentTypeError(msg)
    return threshold


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments to parse. ``None`` reads ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="mcap-lancedb-dedup",
        description="Mark near-duplicate frames in the frames table.",
    )
    # A plain string: pathlib would collapse the "//" in s3://bucket/lancedb.
    parser.add_argument(
        "--db", default=str(DEFAULT_DB), help="LanceDB directory or URI."
    )
    parser.add_argument(
        "--k", type=positive_int, default=DEFAULT_K, help="Neighbors per frame."
    )
    parser.add_argument(
        "--threshold",
        type=similarity_threshold,
        help="Cosine similarity for a near-duplicate. Defaults to the value "
        "calibrated for the table's embedding model.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Build the kNN graph, suppress near-duplicates and merge the result.

    Args:
        argv: Command-line arguments. ``None`` reads ``sys.argv``.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    db = lancedb.connect(args.db)
    if TABLE_NAME not in db.list_tables().tables:
        msg = f"No {TABLE_NAME} table in {args.db}. Run mcap-lancedb-ingest first."
        raise SystemExit(msg)
    dataset = db.open_table(TABLE_NAME).to_lance()
    model_id = (dataset.schema.metadata or {}).get(META_EMBEDDING_MODEL.encode(), b"")
    threshold = (
        args.threshold
        if args.threshold is not None
        else default_threshold(model_id.decode())
    )

    frames = dataset.to_table(columns=[*PRIORITY_COLUMNS, EMBEDDING_COLUMN])
    if frames.num_rows == 0:
        msg = f"The {TABLE_NAME} table in {args.db} is empty."
        raise SystemExit(msg)
    vectors = frames.column(EMBEDDING_COLUMN).combine_chunks()
    embeddings = vectors.flatten().to_numpy().reshape(len(frames), -1)
    device = resolve_device(args.device)

    started = time.perf_counter()
    graph = knn_graph(embeddings, args.k, device)
    logger.info(
        "Built an exact %d-NN graph over %d frames on %s in %.1fs",
        graph.indices.shape[1],
        len(frames),
        device,
        time.perf_counter() - started,
    )
    if graph.similarities.shape[1]:
        # Logged every run so a new model or dataset can be recalibrated.
        nearest = graph.similarities[:, 0]
        logger.info(
            "Nearest-neighbor similarity percentiles p10/p50/p90/p99: %s",
            " / ".join(f"{v:.3f}" for v in np.percentile(nearest, [10, 50, 90, 99])),
        )

    frame_ids = frames.column("frame_id").combine_chunks()
    order = suppression_order(
        frames.column("num_visible_objects").to_numpy(),
        frames.column("timestamp").to_numpy(),
        frame_ids.to_numpy(zero_copy_only=False),
    )
    dup_of = greedy_suppress(graph.indices, graph.similarities, threshold, order)
    removed = int((dup_of >= 0).sum())
    logger.info(
        "Threshold %.3f marks %d of %d frames as near-duplicates (%.1f%%)",
        threshold,
        removed,
        len(frames),
        100 * removed / len(frames),
    )

    write_dedup_columns(dataset, frame_ids, graph, dup_of, threshold)
    logger.info("Merged %s into %s", ", ".join(DEDUP_COLUMNS), TABLE_NAME)


if __name__ == "__main__":
    main()
