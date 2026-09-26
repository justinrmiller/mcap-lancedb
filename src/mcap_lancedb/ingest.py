"""Ingest nuScenes MCAP scenes into LanceDB with Ray Data, on this machine.

The pipeline itself lives in ``mcap_lancedb.pipeline``; this CLI points it at
local directories, starts Ray (or joins ``RAY_ADDRESS``), and runs one
embedding actor per GPU Ray can see. ``scripts/ingest_on_cluster.py`` runs the
same pipeline as a Ray job on a cluster.

Usage:
    uv run mcap-lancedb-ingest --mcap-dir data/mcap --db data/lancedb
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import ray

from mcap_lancedb import (
    CAMERA_CHANNELS,
    DEFAULT_DB,
    DEFAULT_MCAP_DIR,
    DEFAULT_MODEL,
)
from mcap_lancedb.pipeline import WORKER_ENV, IngestConfig, configure_logging, run

logger = logging.getLogger(__name__)


def embedding_actor_plan(device: str, cluster_gpus: int) -> tuple[int, int]:
    """Decide how many embedding actors to run and how many GPUs each gets.

    Args:
        device: The ``--device`` flag: ``auto``, ``cuda``, ``mps`` or ``cpu``.
        cluster_gpus: GPUs Ray can see across the cluster. Ray 2.58 also
            reports an Apple silicon GPU here, as one GPU per Mac.

    Returns:
        ``(actors, gpus_per_actor)``: one actor per GPU when Ray sees any,
        otherwise a single actor that asks for none.

    Raises:
        SystemExit: If CUDA was requested but Ray sees no GPUs.
    """
    if device != "cpu" and cluster_gpus > 0:
        return cluster_gpus, 1
    if device == "cuda":
        msg = "--device cuda was requested, but Ray sees no GPUs."
        raise SystemExit(msg)
    return 1, 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments to parse. ``None`` reads ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="mcap-lancedb-ingest",
        description="Embed nuScenes MCAP camera keyframes into a LanceDB table.",
    )
    parser.add_argument("--mcap-dir", type=Path, default=DEFAULT_MCAP_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--channels", nargs="+", choices=CAMERA_CHANNELS, default=list(CAMERA_CHANNELS)
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Frames per embedding batch."
    )
    parser.add_argument("--limit", type=int, help="Ingest only the first N frames.")
    parser.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto"
    )
    parser.add_argument(
        "--vector-index", choices=["auto", "always", "never"], default="auto"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the ingest pipeline locally.

    Args:
        argv: Command-line arguments. ``None`` reads ``sys.argv``.
    """
    args = parse_args(argv)
    configure_logging()
    # Ray workers run in their own directories, so every path they see is
    # absolute. On a multi-node cluster both must be on shared storage.
    mcap_dir = args.mcap_dir.resolve()
    if not any(mcap_dir.glob("*.mcap")):
        msg = (
            f"No .mcap files in {mcap_dir}. Convert nuScenes with "
            "scripts/convert_mini.sh (see the README) or point --mcap-dir at "
            "the converted scenes."
        )
        raise SystemExit(msg)

    # Joins the cluster at RAY_ADDRESS when set, otherwise starts a local one.
    # Under `uv run`, Ray ships the project directory (minus .gitignore'd
    # paths such as data/) and rebuilds its environment on every node.
    ray.init(ignore_reinit_error=True, runtime_env={"env_vars": WORKER_ENV})
    actors, gpus_per_actor = embedding_actor_plan(
        args.device, int(ray.cluster_resources().get("GPU", 0))
    )
    logger.info(
        "Embedding with %s on %d actor(s), %d GPU(s) each",
        args.model,
        actors,
        gpus_per_actor,
    )
    run(
        IngestConfig(
            mcap_uri=str(mcap_dir),
            db_uri=str(args.db.resolve()),
            model_id=args.model,
            channels=tuple(args.channels),
            limit=args.limit,
            batch_size=args.batch_size,
            device=None if args.device == "auto" else args.device,
            embed_actors=actors,
            gpus_per_actor=gpus_per_actor,
            vector_index=args.vector_index,
        )
    )


if __name__ == "__main__":
    main()
