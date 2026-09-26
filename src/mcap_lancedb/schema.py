"""Arrow schema for the ``frames`` table.

One row per camera keyframe, fully denormalized. The schema is built up in three
stages that mirror the pipeline: metadata from the MCAP files (driver), media
(CPU stage), then the embedding (GPU stage). The dedup columns are merged in
later by ``mcap-lancedb-dedup``.
"""

import pyarrow as pa

EMBEDDING_COLUMN = "embedding"
IMAGE_COLUMN = "image"
THUMBNAIL_COLUMN = "thumbnail"

# Model-sized raw RGB pixels, passed from the CPU stage to the GPU stage and
# dropped before the write. Resizing on CPU ships ~0.4 MB per frame to the GPU
# actors instead of a decoded 1600x900 frame (~4.3 MB).
MODEL_INPUT_COLUMN = "_model_input"

METADATA_SCHEMA = pa.schema(
    [
        # IDs and provenance. frame_id is scene/channel/capture time in µs.
        ("frame_id", pa.string()),
        ("scene_name", pa.string()),
        # The scene's MCAP file, relative to --mcap-dir, and the log time of
        # this frame's image message in it.
        ("source_path", pa.string()),
        ("mcap_log_time", pa.int64()),
        # Scene context, copied onto every frame because LanceDB has no joins.
        ("scene_description", pa.string()),
        ("scene_tags", pa.list_(pa.string())),
        ("is_night", pa.bool_()),
        ("is_rain", pa.bool_()),
        ("location", pa.string()),
        ("log_date", pa.date32()),
        ("vehicle", pa.string()),
        # Camera.
        ("channel", pa.string()),
        ("timestamp", pa.timestamp("us", tz="UTC")),
        ("frame_index", pa.int32()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("cam_intrinsic", pa.list_(pa.float32(), 9)),
        # Ego pose at the keyframe. Rotation is [w, x, y, z].
        ("ego_translation", pa.list_(pa.float64(), 3)),
        ("ego_rotation", pa.list_(pa.float64(), 4)),
        ("ego_speed_mps", pa.float32()),
        # Annotated objects with any box corner inside this camera's image.
        ("visible_categories", pa.list_(pa.string())),
        ("num_visible_objects", pa.int32()),
        ("num_pedestrians", pa.int32()),
        ("num_cyclists", pa.int32()),
        ("num_vehicles", pa.int32()),
    ]
)

MEDIA_FIELDS: tuple[pa.Field, ...] = (
    # Small inline JPEG for result grids.
    pa.field(THUMBNAIL_COLUMN, pa.binary()),
    # The original JPEG bytes, untouched. Lance reads only the columns a query
    # projects, so scans and searches that leave this out never pay for it.
    pa.field(IMAGE_COLUMN, pa.large_binary()),
)

DEDUP_FIELDS: tuple[pa.Field, ...] = (
    # The frame's k nearest neighbors by cosine similarity, most similar first.
    pa.field("nn_frame_ids", pa.list_(pa.string())),
    pa.field("nn_similarity", pa.list_(pa.float32())),
    # The frame this one duplicates. Null means the frame is kept.
    pa.field("dup_of", pa.string()),
)
DEDUP_COLUMNS: tuple[str, ...] = tuple(field.name for field in DEDUP_FIELDS)


def embedding_field(dim: int) -> pa.Field:
    """Build the fixed-width embedding field.

    Args:
        dim: Embedding dimension.

    Returns:
        A ``fixed_size_list<float32, dim>`` field named ``embedding``.
    """
    return pa.field(EMBEDDING_COLUMN, pa.list_(pa.float32(), dim))


def frame_schema(dim: int) -> pa.Schema:
    """Full schema of a written row, before dedup columns are merged in.

    Args:
        dim: Embedding dimension.

    Returns:
        Metadata fields, then media, then the embedding. No table metadata.
    """
    return pa.schema([*METADATA_SCHEMA, *MEDIA_FIELDS, embedding_field(dim)])
