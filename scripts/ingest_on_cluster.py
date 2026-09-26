r"""Run the ingest pipeline as a Ray job on a cluster.

The same pipeline as ``mcap-lancedb-ingest`` (``mcap_lancedb.pipeline``),
pointed at object storage and an existing cluster, with an autoscaling pool of
GPU embedding actors. It doesn't touch the viewer or dedup. Submit it from the
repository root:

    ray job submit --address http://HEAD:8265 --working-dir . -- \
        uv run scripts/ingest_on_cluster.py \
        --mcap-uri s3://BUCKET/nuscenes-mcap --db-uri s3://BUCKET/lancedb \
        --min-gpu-actors 2 --max-gpu-actors 16

Under ``uv run``, Ray rebuilds the project's locked environment on each worker
node. Each node downloads the model into its Hugging Face cache the first time
an actor starts there.
"""

import argparse
import logging
from collections.abc import Sequence

import ray

from mcap_lancedb import CAMERA_CHANNELS, DEFAULT_MODEL
from mcap_lancedb.pipeline import WORKER_ENV, IngestConfig, configure_logging, run

logger = logging.getLogger("ingest_on_cluster")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments to parse. ``None`` reads ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Embed nuScenes MCAP keyframes into LanceDB on a Ray cluster."
    )
    parser.add_argument(
        "--mcap-uri",
        required=True,
        help="Directory of per-scene MCAP files, e.g. s3://bucket/nuscenes-mcap.",
    )
    parser.add_argument(
        "--db-uri", required=True, help="LanceDB directory, e.g. s3://bucket/lancedb."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--channels", nargs="+", choices=CAMERA_CHANNELS, default=list(CAMERA_CHANNELS)
    )
    parser.add_argument("--limit", type=int, help="Ingest only the first N frames.")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Frames per embedding batch."
    )
    parser.add_argument(
        "--min-gpu-actors",
        type=int,
        default=1,
        help="Embedding actors to keep running.",
    )
    parser.add_argument(
        "--max-gpu-actors",
        type=int,
        help="Most embedding actors to scale up to. Default: the GPUs the "
        "cluster has when the job starts.",
    )
    parser.add_argument(
        "--gpus-per-actor",
        type=float,
        default=1.0,
        help="GPUs each actor reserves; 0.5 packs two actors per GPU, 0 runs on CPU.",
    )
    parser.add_argument(
        "--ray-address",
        default="auto",
        help='Cluster to join. "auto" is the one running this job; "local" '
        "starts a throwaway cluster on this machine.",
    )
    parser.add_argument(
        "--vector-index", choices=["auto", "always", "never"], default="auto"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Join the cluster and run the pipeline.

    Args:
        argv: Command-line arguments. ``None`` reads ``sys.argv``.
    """
    args = parse_args(argv)
    configure_logging()
    ray.init(address=args.ray_address, runtime_env={"env_vars": WORKER_ENV})
    gpus = ray.cluster_resources().get("GPU", 0)
    max_actors = args.max_gpu_actors or max(
        args.min_gpu_actors,
        int(gpus / args.gpus_per_actor) if args.gpus_per_actor else 1,
    )
    logger.info(
        "Cluster has %s CPUs and %s GPUs; embedding on %d to %d actor(s)",
        ray.cluster_resources().get("CPU", 0),
        gpus,
        args.min_gpu_actors,
        max_actors,
    )
    run(
        IngestConfig(
            mcap_uri=args.mcap_uri,
            db_uri=args.db_uri,
            model_id=args.model,
            channels=tuple(args.channels),
            limit=args.limit,
            batch_size=args.batch_size,
            embed_actors=(args.min_gpu_actors, max_actors),
            gpus_per_actor=args.gpus_per_actor,
            vector_index=args.vector_index,
        )
    )


if __name__ == "__main__":
    main()
