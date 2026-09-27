# LanceDB schema

mcap-lancedb stores everything in one LanceDB table, `frames`, with one row per
camera keyframe: 2,424 rows for nuScenes mini (404 samples × 6 cameras). The
columns are defined in [src/mcap_lancedb/schema.py](src/mcap_lancedb/schema.py)
and arrive in the order the pipeline builds them:

1. **Metadata**, read from each scene's MCAP file by one Ray task per file.
2. **Media**, added by the CPU stage that decodes the JPEGs.
3. **The embedding**, added by the GPU actors.
4. **Dedup columns**, merged in later by `mcap-lancedb-dedup`. They're absent
   until it runs.

To print the schema of your own table:

```python
import lancedb

table = lancedb.connect("data/lancedb").open_table("frames")
print(table.schema)
print(table.list_indices())
```

Arrow marks every column nullable, but only `dup_of` is ever null.

## Columns

### IDs and provenance

| Column | Type | Contents |
| --- | --- | --- |
| `frame_id` | `string` | `<scene>/<channel>/<capture time in µs>`, for example `scene-0001/CAM_FRONT/1533000000012000` in the test data. Unique per row, so it serves as the primary key: dedup merges on it and the viewer fetches frames by it. BTree index. |
| `scene_name` | `string` | The nuScenes scene, for example `scene-0553`. BTree index. |
| `source_path` | `string` | File name of the scene's MCAP file, without its directory, for example `nuscenes-scene-0553.mcap`. |
| `mcap_log_time` | `int64` | Log time, in nanoseconds, of this frame's image message in that file. nuscenes2mcap logs a keyframe's image, calibration, annotations and ego pose at one log time, so ingest matches images to rows on (`source_path`, `channel`, `mcap_log_time`). It also finds the exact message again in the MCAP file. |

### Scene context

The same for every frame of a scene. They're copied onto each row because
LanceDB has no joins.

| Column | Type | Contents |
| --- | --- | --- |
| `scene_description` | `string` | nuScenes' free-text description, a comma-separated list such as `Night, wait at intersection`. Every result card in the viewer shows it. |
| `scene_tags` | `list<string>` | The description split on commas, trimmed and lowercased: `["night", "wait at intersection"]`. |
| `is_night` | `bool` | The description contains the whole word "night", in any case. Not detected from the images. |
| `is_rain` | `bool` | The same for "rain": "Rain" and "after rain" count; "raining", "train" and "terrain" don't. No mini scene was actually raining. |
| `location` | `string` | The nuScenes map, for example `boston-seaport` or `singapore-hollandvillage`. Bitmap index, since there are only a few values. |
| `log_date` | `date32` | The date the log was captured. |
| `vehicle` | `string` | The recording vehicle's id, for example `n015`. |

### Camera

| Column | Type | Contents |
| --- | --- | --- |
| `channel` | `string` | `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_BACK_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT` or `CAM_FRONT_LEFT`. Bitmap index. |
| `timestamp` | `timestamp[us, UTC]` | When the camera captured the image, from its calibration message. Cameras fire shortly after the keyframe's lidar sweep, so this is a little later than `mcap_log_time`. |
| `frame_index` | `int32` | The frame's position among this camera's keyframes in the scene, from 0. Sweeps aren't counted. |
| `width`, `height` | `int32` | The original image size: 1600 × 900 on nuScenes. |
| `cam_intrinsic` | `fixed_size_list<float32>[9]` | The 3 × 3 camera matrix K, row by row: `[fx, 0, cx, 0, fy, cy, 0, 0, 1]`. |

### Ego vehicle

| Column | Type | Contents |
| --- | --- | --- |
| `ego_translation` | `fixed_size_list<float64>[3]` | The car's position in the map frame, in meters: `[x, y, z]`. |
| `ego_rotation` | `fixed_size_list<float64>[4]` | The car's orientation as a quaternion: `[w, x, y, z]`. |
| `ego_speed_mps` | `float32` | Speed in m/s, from the scene's keyframe poses at 2 Hz: a central difference inside the scene, one-sided at its first and last keyframe. Short stops and starts are smoothed out. The viewer's *Removal rate by ego speed* chart bins it. |

### Objects in this camera

nuscenes2mcap projects every annotation box into every camera and logs the
boxes that land in the image, so these count objects in *this* camera's view,
not around the car. A box counts when any of its corners lands in the image;
nothing checks whether the object is hidden behind something else.

| Column | Type | Contents |
| --- | --- | --- |
| `visible_categories` | `list<string>` | The distinct nuScenes categories in view, sorted, for example `["human.pedestrian.adult", "vehicle.car"]`. The viewer's *Must show* filter matches on it. |
| `num_visible_objects` | `int32` | The number of boxes in view, of any category, including cones, barriers and animals. Dedup keeps frames with more objects first. |
| `num_pedestrians` | `int32` | Boxes in `human.pedestrian.*`. The viewer's *Minimum pedestrians in view* filter uses it. |
| `num_cyclists` | `int32` | Boxes in `vehicle.bicycle` and `vehicle.motorcycle`, ridden or parked: the MCAP files don't carry nuScenes' rider attributes. |
| `num_vehicles` | `int32` | Boxes in every other `vehicle.*` category. |

### Media

| Column | Type | Contents |
| --- | --- | --- |
| `thumbnail` | `binary` | A JPEG with a 320 px long edge, quality 85. Result grids and cluster views read only this. |
| `image` | `large_binary` | The original JPEG bytes from nuScenes, unchanged. Only the *Full resolution* view reads it. `large_binary` has 64-bit offsets, so one array can hold more than 2 GB of images. |

### Embedding

| Column | Type | Contents |
| --- | --- | --- |
| `embedding` | `fixed_size_list<float32>[D]` | The SigLIP 2 image embedding, L2-normalized so a dot product is the cosine similarity. D depends on the model: 1152 for `google/siglip2-so400m-patch16-384` (the default), 768 for `google/siglip2-base-patch16-224`. Embeddings from different models aren't comparable. |

### Near-duplicates

Added by `mcap-lancedb-dedup`, and replaced each time it runs.

| Column | Type | Contents |
| --- | --- | --- |
| `nn_frame_ids` | `list<string>` | The `frame_id`s of the frame's k nearest neighbors by cosine similarity, most similar first. k is `--k` (default 32), capped at one less than the number of frames. |
| `nn_similarity` | `list<float32>` | The cosine similarity to each of those neighbors. The viewer's threshold slider re-runs suppression from these two columns without reading any embeddings. |
| `dup_of` | `string` | The `frame_id` of the kept frame that suppressed this one, or null if this frame is kept. *Hide near-duplicates* is the filter `dup_of IS NULL`. |

The pipeline also passes a temporary `_model_input` column of resized pixels
from the CPU stage to the GPU actors. It's dropped before the write and never
stored.

## Table metadata

Key-value pairs on the table's schema:

| Key | Example | Contents |
| --- | --- | --- |
| `mcap_lancedb.embedding_model` | `google/siglip2-so400m-patch16-384` | The model that embedded the frames. The viewer embeds text queries with it, so queries and frames always share a space. Do the same when querying from Python. |
| `mcap_lancedb.embedding_dim` | `1152` | D, the width of `embedding`. |
| `mcap_lancedb.dedup_threshold` | `0.985` | The cosine similarity threshold that produced `dup_of`. Absent until dedup runs. A rerun clears it before replacing the dedup columns and records the new value last, so a stored threshold always matches the columns beside it. |
| `mcap_lancedb.dedup_k` | `32` | The number of neighbors stored per frame, after the cap. |

## Indexes

| Column | Index | Why |
| --- | --- | --- |
| `frame_id` | BTree | The dedup merge key, and the viewer's point lookups: thumbnails, full-resolution images and *More like this*. |
| `scene_name` | BTree | Per-scene filters and lookups. |
| `channel` | Bitmap | The camera filter. Six values. |
| `location` | Bitmap | The location filter. Four values on nuScenes. |
| `embedding` | IVF_PQ, cosine | Only at 100,000 rows or more, or with `--vector-index always`. |

## Design choices

- **Fully denormalized.** LanceDB has no joins, so scene, pose and calibration
  data are copied onto every frame. A filter like "night frames at
  singapore-hollandvillage with two or more pedestrians" is one prefilter on
  one table.
- **Originals inline, but never read by accident.** `image` is a plain
  `large_binary` column. Lance reads only the columns a query projects, so
  searches and scans that leave it out (all of them, apart from the
  full-resolution view) never pay for it. Grids read the small `thumbnail`
  column instead.
- **Near-duplicates are marked, never deleted.** Dedup only adds columns, so
  "hide near-duplicates" is just `dup_of IS NULL`, and the viewer can re-run
  suppression at any threshold from the stored graph.
- **The vector index is conditional.** Below 100k rows an exact search is faster
  than IVF_PQ and has perfect recall, so `--vector-index auto` skips it. The
  viewer re-ranks candidates by exact distance either way, so the similarity it
  shows is exact even with the index.
