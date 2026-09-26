"""Embedding-based curation of nuScenes camera frames stored in LanceDB.

The package reads nuScenes scenes converted to MCAP, ingests their camera
keyframes into a single denormalized LanceDB table, embeds them with SigLIP 2,
marks near-duplicates, and serves a Streamlit viewer for semantic search and
interactive near-duplicate removal.
"""

from pathlib import Path

# SigLIP 2 at 384 px is the quality default. The base 224 px checkpoint embeds
# several times faster and is the practical choice for CPU-only runs.
DEFAULT_MODEL = "google/siglip2-so400m-patch16-384"
FAST_MODEL = "google/siglip2-base-patch16-224"

TABLE_NAME = "frames"
DEFAULT_DB = Path("data/lancedb")
DEFAULT_MCAP_DIR = Path("data/mcap")

CAMERA_CHANNELS: tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)

# Result grids never touch the full-resolution image column; they read this
# inline JPEG thumbnail instead.
THUMBNAIL_LONG_EDGE = 320
THUMBNAIL_QUALITY = 85

# Table-level schema metadata keys. The viewer reads the embedding model from
# here so text queries are always embedded by the model that embedded the frames.
META_EMBEDDING_MODEL = "mcap_lancedb.embedding_model"
META_EMBEDDING_DIM = "mcap_lancedb.embedding_dim"
META_DEDUP_THRESHOLD = "mcap_lancedb.dedup_threshold"
META_DEDUP_K = "mcap_lancedb.dedup_k"
