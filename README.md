# mcap-lancedb

Embedding-based curation for robotics and AV fleet data: find the right 1% of
camera frames. mcap-lancedb ingests nuScenes camera keyframes into LanceDB with
Ray Data, embeds them with SigLIP 2, marks near-duplicates, and ships a
Streamlit viewer for:

1. **Text-to-image search** with metadata filters: camera, location, day or
   night, pedestrians in view, object categories, and hiding near-duplicates.
2. **Interactive near-duplicate removal**: move a threshold slider and watch
   removals concentrate where the car was standing still.

## Step-by-step guide

This walks through the whole demo on nuScenes v1.0-mini: 10 scenes and 2,424
camera frames. Run every command from this directory (`mcap-lancedb/`).

### Before you start

| You need | Notes |
| --- | --- |
| [uv](https://docs.astral.sh/uv/) | It installs Python 3.12 and every dependency for you. |
| About 8 GB of disk | 1.6 GB environment, 4.5 GB model weights, 450 MB of dataset, 450 MB per table. |
| 16 GB of RAM | The default model is 4.5 GB in fp32, and the viewer loads it too. |
| A GPU (optional) | NVIDIA (CUDA) or Apple silicon (MPS). Without one, use the faster base model. |

nuScenes is licensed for non-commercial use (CC BY-NC-SA 4.0). Read the terms
at [nuscenes.org](https://www.nuscenes.org/terms-of-use) before downloading.

### 1. Install

```bash
uv sync
```

This creates `.venv/` with the package, its three commands, and the dev tools.

### 2. Download nuScenes mini

The tarball is 4.2 GB, but it's streamed straight into `tar`, and only the JSON
tables and camera keyframes are kept, about 450 MB on disk.

```bash
mkdir -p data/nuscenes
```

On Linux (GNU tar):

```bash
curl -s https://www.nuscenes.org/data/v1.0-mini.tgz | tar -xz -C data/nuscenes --wildcards 'v1.0-mini/*' 'samples/CAM_*'
```

On macOS (bsdtar, which matches patterns without the flag):

```bash
curl -s https://www.nuscenes.org/data/v1.0-mini.tgz | tar -xz -C data/nuscenes 'v1.0-mini/*' 'samples/CAM_*'
```

Check it arrived. This should print `2424`:

```bash
find data/nuscenes/samples -name '*.jpg' | wc -l
```

`data/` is in `.gitignore`, which also keeps it out of the environment Ray
uploads to its workers (see [Troubleshooting](#troubleshooting)).

### 3. Run a smoke test

Before the full run, push 200 frames through the whole pipeline with the small
model into a throwaway database:

```bash
uv run mcap-lancedb-ingest --model google/siglip2-base-patch16-224 --limit 200 --db data/lancedb-smoke
```

The first run downloads the model (1.5 GB). It ends with:

```
INFO mcap_lancedb.ingest: Found 200 camera keyframes in v1.0-mini
INFO mcap_lancedb.ingest: Embedding with google/siglip2-base-patch16-224 on 1 actor(s), 1 GPU(s) each
INFO mcap_lancedb.ingest: Skipping the vector index: 200 rows search faster exactly
INFO mcap_lancedb.ingest: Wrote 200 frames to .../data/lancedb-smoke/frames
```

The actor line depends on your hardware: `0 GPU(s)` on a CPU-only machine,
one actor per GPU on a CUDA box. Ray also prints its own progress in between.
Once it works, delete the throwaway table:

```bash
rm -rf data/lancedb-smoke
```

### 4. Ingest every frame

```bash
uv run mcap-lancedb-ingest
```

The defaults are `--dataroot data/nuscenes --version v1.0-mini --db
data/lancedb` and the quality model, `google/siglip2-so400m-patch14-384` (a
4.5 GB download the first time). Expect about 7 minutes on an Apple M4 Pro.
Without a GPU, use the base model instead; it takes under 3 minutes on 12 CPU
cores:

```bash
uv run mcap-lancedb-ingest --model google/siglip2-base-patch16-224
```

It finishes with `Wrote 2424 frames to .../data/lancedb/frames`. Running ingest
again replaces the table, including any dedup results, so rerun dedup
afterwards.

### 5. Mark near-duplicates

```bash
uv run mcap-lancedb-dedup
```

This takes seconds. With the default model you should see:

```
INFO mcap_lancedb.dedup: Built an exact 32-NN graph over 2424 frames on mps in 0.4s
INFO mcap_lancedb.dedup: Nearest-neighbor similarity percentiles p10/p50/p90/p99: 0.940 / 0.978 / 0.996 / 0.999
INFO mcap_lancedb.dedup: Threshold 0.985 marks 623 of 2424 frames as near-duplicates (25.7%)
INFO mcap_lancedb.dedup: Merged nn_frame_ids, nn_similarity, dup_of into frames
```

With the base model it's 631 frames at 0.98. Frames are marked, never deleted.
Rerun with a different `--threshold` at any time; it replaces the previous
marks.

### 6. Explore in the viewer

```bash
uv run streamlit run src/mcap_lancedb/app.py
```

Open <http://localhost:8501>. The first search loads the model, which takes a
few seconds.

**Search tab**

1. Pick an example query, such as *pedestrians on a crosswalk*. The grid fills
   with scene-0553, whose description reads "peds crossing crosswalk". Every
   card shows the scene description so you can check results against the
   ground truth.
2. Narrow it down with the filters on the left. They run before the vector
   search (a prefilter), so you still get a full page whenever enough frames
   match. *Must show* keeps only frames where every chosen object category is
   in view of that camera, not just somewhere around the car.
3. Turn on *Hide near-duplicates* to search only the frames dedup kept (1,801
   of 2,424 at the default threshold). Don't expect a stationary scene to
   vanish: scene-0553 is stopped at a busy crossing where pedestrians keep
   moving, so it still keeps 70 of its 246 frames.
4. *More like this* searches by image, using that frame's stored embedding.
   *Back to text search* returns to your query.
5. *Full resolution* opens the original 1600×900 JPEG, read from the table on
   demand.

Text-to-image similarities are low in absolute terms, around 0.1 to 0.2. That's
normal for SigLIP; only the ranking matters.

**Near-duplicates tab**

1. The metrics show how many frames the stored threshold keeps.
2. *Removal rate by ego speed* shows removals piling up under 0.5 m/s, where
   the car stood still.
3. The per-scene table puts scene-0553 and scene-1100, the two stationary
   scenes, at the top.
4. *Largest clusters* shows each kept frame beside its duplicates, least
   similar first, so you can judge the threshold's weakest calls.
5. Drag the threshold slider. Everything above recomputes from the stored
   neighbor graph in a moment, and nothing in the table changes. To make a new
   threshold stick for *Hide near-duplicates*, rerun step 5 with `--threshold`.

### 7. Query the table from Python (optional)

The table is ordinary LanceDB, so anything the viewer does is a few lines of
code:

```python
import lancedb
from mcap_lancedb.embed import SiglipEncoder

table = lancedb.connect("data/lancedb").open_table("frames")
model = table.schema.metadata[b"mcap_lancedb.embedding_model"].decode()
query = SiglipEncoder(model).encode_text(["a bus at a bus stop"])[0]

hits = (
    table.search(query, vector_column_name="embedding")
    .distance_type("cosine")
    .where("dup_of IS NULL AND is_night = false", prefilter=True)
    .select(["scene_name", "channel", "scene_description", "_distance"])
    .limit(5)
    .to_pandas()
)
```

Always embed queries with the model named in the schema metadata, as above.
Embeddings from different models aren't comparable.

## Command reference

`mcap-lancedb-ingest`:

| Flag | Default | Notes |
| --- | --- | --- |
| `--dataroot` | `data/nuscenes` | The folder that contains `v1.0-mini/` and `samples/`. |
| `--version` | `v1.0-mini` | `v1.0-trainval` works without code changes. |
| `--db` | `data/lancedb` | LanceDB directory. The table is always named `frames`. |
| `--model` | `google/siglip2-so400m-patch14-384` | `google/siglip2-base-patch16-224` is several times faster. Fixed-resolution SigLIP checkpoints only. |
| `--device` | `auto` | CUDA, then MPS, then CPU. fp16 on CUDA, fp32 elsewhere. |
| `--channels` | all six | Any subset of `CAM_FRONT` … `CAM_FRONT_LEFT`. |
| `--limit` | none | Ingest only the first N frames, in scene, camera and time order. |
| `--batch-size` | 32 | Frames per embedding batch. Lower it if the GPU runs out of memory. |
| `--vector-index` | `auto` | IVF_PQ only at 100k rows or more; `always` or `never` override. |

`mcap-lancedb-dedup`:

| Flag | Default | Notes |
| --- | --- | --- |
| `--db` | `data/lancedb` | Same directory as ingest. |
| `--threshold` | calibrated per model | 0.985 for so400m, 0.98 for base and any other model. Must be in (0, 1]. |
| `--k` | 32 | Neighbors per frame in the stored graph. Raise it for long stops. |
| `--device` | `auto` | Where the exact kNN matrix multiplies run. |

The viewer takes `--db` after a `--`, or the `MCAP_LANCEDB_DB` environment
variable:

```bash
uv run streamlit run src/mcap_lancedb/app.py -- --db data/lancedb-base
```

## Running on other hardware

- **CUDA box with two GPUs**: Ray reports both, so the embedding stage runs one
  actor per GPU (`ActorPoolStrategy(size=2)`, one GPU each). Decoding and
  resizing stay on CPU tasks.
- **Apple silicon**: Ray 2.58 reports the Mac's GPU as one GPU, so one actor
  runs on MPS.
- **CPU only**: one actor. Ray fuses the decode and embed stages because their
  resources match, which is fine here.
- **A Ray cluster**: set `RAY_ADDRESS` and ingest joins it. `--dataroot` and
  `--db` must be on shared storage (a NAS) mounted at the same path on every
  node.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No nuScenes v1.0-mini tables in ...` | `--dataroot` must point at the folder containing `v1.0-mini/` and `samples/`. Redo step 2 if it's empty. |
| `tar: Option --wildcards is not supported` | You're on macOS. Use the bsdtar command in step 2. |
| Ray warns the runtime_env package is "approaching the maximum upload size" | Under `uv run`, Ray uploads the current directory to its workers, minus anything in `.gitignore`. Run from `mcap-lancedb/`, and keep datasets in `data/` or outside the project. |
| Ingest hangs before any progress | The embedding actor reserves a CPU, so a 1-CPU machine deadlocks. Use at least 2 CPUs. |
| Out of memory while embedding | Lower `--batch-size`, or use the base model. |
| `You are sending unauthenticated requests to the HF Hub` | Harmless. Set `HF_TOKEN` for faster downloads and higher rate limits. |
| The viewer says there's no `frames` table | It reads `data/lancedb` relative to where you launched it. Launch from `mcap-lancedb/`, or pass `-- --db PATH`. |
| *Hide near-duplicates* is greyed out, or the Near-duplicates tab is empty | Run step 5, then reload the page. |
| Others on your network can open the viewer | Streamlit listens on every interface by default. Add `--server.address localhost` to keep it local. |
| The viewer doesn't rerun when you edit `app.py` | Streamlit's file watcher is off in `.streamlit/config.toml`. Its module scan touches transformers' lazy aliases, which import torchvision (not installed), and logs a traceback for each. For rerun-on-save, add `--server.fileWatcherType auto` and ignore that noise. |

## How it works

```
JSON tables ──driver──▶ one metadata row per camera keyframe
                          │ ray.data.from_arrow(...).repartition(n)
                          ▼
                   CPU tasks: read JPEG, keep original bytes,
                   320 px thumbnail, resize to model input
                          │ ~0.4 MB per frame, not ~4.3 MB decoded
                          ▼
                   GPU actors: SigLIP 2 → L2-normalized embedding
                          │
                          ▼
             lancedb-ray: fragments written in parallel,
             committed as one transaction
```

- **No nuscenes-devkit.** `nuscenes_io.py` reads the JSON tables directly and
  joins them in dictionaries.
- **Objects per camera, not per sample.** Annotation box centers are projected
  global → ego → camera → pixels, and only boxes a lidar or radar point touched
  count. So a frame's `num_pedestrians` means pedestrians in *that* image.
- **Preprocessing matches the processor exactly.** The CPU stage resizes with
  the checkpoint's own filter, read from `processor.image_processor`: bilinear
  for SigLIP 2, bicubic for SigLIP 1. Normalization runs on the GPU.
  Embeddings match the Hugging Face processor path to cosine 0.9999999.
- **Text is padded to 64 tokens and lowercased**, which is how SigLIP 2 was
  trained. Other padding silently degrades retrieval. These checkpoints load a
  case-sensitive `GemmaTokenizer`, so the lowercasing has to happen in code.
- **pyarrow batches end to end.** Numpy batches would turn embeddings into
  Ray's tensor extension type, which won't cast to `fixed_size_list`.
- **lancedb-ray, not `Dataset.write_lance`.** Ray's built-in Lance sink passes
  `storage_options_provider` to `lance.fragment.write_fragments`, which pylance
  12 removed. lancedb-ray delegates to `lance-ray`, which only passes that
  argument on pylance 4.x, and it preserves field and schema metadata.
- **Exact similarities in the viewer.** Searches re-rank candidates by exact
  distance (`refine_factor`), so the similarity shown is exact even when an
  IVF_PQ index is in play.

## Schema

One table, `frames`, with one row per camera keyframe.

| Group | Columns |
| --- | --- |
| IDs and provenance | `frame_id` (sample_data token), `sample_token`, `scene_token`, `scene_name`, `source_path` (relative to the dataroot) |
| Scene context | `scene_description`, `scene_tags`, `is_night`, `is_rain`, `location`, `log_date`, `vehicle` |
| Camera | `channel`, `timestamp` (µs, UTC), `frame_index`, `width`, `height`, `cam_intrinsic` (9 × float32) |
| Ego | `ego_translation` (3 × float64), `ego_rotation` ([w, x, y, z]), `ego_speed_mps` |
| Objects in this camera | `visible_categories`, `num_visible_objects`, `num_pedestrians`, `num_cyclists`, `num_vehicles` |
| Media | `thumbnail` (inline JPEG, 320 px long edge, q85), `image` (original JPEG bytes, `large_binary`) |
| Embedding | `embedding` (`fixed_size_list<float32, D>`, L2-normalized) |
| Dedup, merged later | `nn_frame_ids`, `nn_similarity` (top-k neighbors), `dup_of` (null means kept) |

Table-level schema metadata records the embedding model, its dimension, the
nuScenes version, and, after dedup, the threshold and k. The viewer embeds text
queries with the model named there, so queries and frames always share a space.

Why it looks like this:

- **Fully denormalized.** LanceDB has no joins, so scene, log, pose and
  calibration data are copied onto every frame. A filter like "night frames at
  boston-seaport with two or more pedestrians" is one prefilter on one table.
- **Originals inline, but never read by accident.** `image` is a plain
  `large_binary` column. Lance reads only the columns a query projects, so
  searches and scans that leave it out (all of them, apart from the
  full-resolution view) never pay for it. Grids read the small `thumbnail`
  column instead.
- **Near-duplicates are marked, never deleted.** Dedup merges its columns in
  with `LanceDataset.merge`, after dropping old ones on reruns. The raw data
  stays intact, "hide near-duplicates" is just `dup_of IS NULL`, and the viewer
  can re-run suppression at any threshold from the stored graph.
- **The vector index is conditional.** Below 100k rows an exact search is faster
  than IVF_PQ and has perfect recall, so `--vector-index auto` skips it. Scalar
  indexes always exist: BITMAP on `channel` and `location`, BTREE on
  `scene_name` and on `frame_id`, which is the merge key and the viewer's
  point-lookup key.

## Near-duplicate removal

`mcap-lancedb-dedup` builds an **exact** top-k cosine kNN graph (k=32) with
chunked matrix multiplies, capped at a 256 MB similarity matrix per chunk. Exact
search makes the graph reproducible; it takes under a second for mini and
minutes on a GPU for trainval.

Suppression is **greedy, NMS-style**, not connected components. Components chain
A~B~C until A and C look nothing alike. Instead, edges at or above the threshold
are made symmetric and frames are visited in priority order (most visible
objects first, then earliest). Each unvisited frame is kept and marks its
unvisited neighbors `dup_of` itself. A suppressed frame never suppresses others.
The viewer runs the same function, and at the stored threshold it reproduces
the stored `dup_of` frame for frame (a test pins this).

### Calibrating the threshold

The threshold was chosen from the data, not assumed. Consecutive 2 Hz keyframes
from one camera are very similar even while driving. The median
nearest-neighbor similarity on mini is 0.973 (base) and 0.978 (so400m), so a
generic 0.95 would remove about half of the frames taken at over 3 m/s.

Removal rates on mini, so400m, stationary (< 0.5 m/s) vs moving (≥ 3 m/s):

| Threshold | All frames | Stationary | Moving | Gap |
| --- | ---: | ---: | ---: | ---: |
| 0.950 | 65.5% | 90.1% | 54.4% | 36 pts |
| 0.970 | 47.0% | 82.7% | 31.3% | 51 pts |
| 0.980 | 33.3% | 73.7% | 16.9% | 57 pts |
| **0.985** | **25.7%** | **66.5%** | **9.7%** | **57 pts** |
| 0.990 | 17.3% | 56.8% | 2.3% | 55 pts |

Pairs below 0.98 show visible change when you look at them (pedestrians have
moved, a car has passed). From 0.985 up they are near-identical. The defaults
are therefore **0.985 for so400m** and **0.98 for the base model**, where its
gap peaks. Other models fall back to 0.98. Dedup logs the nearest-neighbor
percentiles on every run, so a new model or dataset can be recalibrated the
same way.

At the default, the stationary scenes dominate: scene-0553 loses 72% of its
frames and scene-1100 68%, 55% of all removals between them. scene-0757,
which stops for its last 22 of 41 keyframes (59% of frames under 0.5 m/s), is
next at 43%.

## Development

```bash
uv run ruff check
uv run ruff format --check
uv run --group dev ty check
uv run pytest
uv run --group dev pre-commit run --all-files
```

The tests cover the projection math, ego speed, the kNN graph, greedy
suppression, the dedup CLI end to end on synthetic tables, the SQL prefilter
builder and the CPU decode stage. They also drive the viewer headlessly with
`streamlit.testing.v1.AppTest` against `data/lancedb`; those cases skip until
steps 4 and 5 have run.

## Known limitations

- **Center-only projection.** An object counts as visible when its box center
  lands in the image. A large truck whose center is just off-frame is missed,
  and a pedestrian fully hidden behind a bus still counts.
- **`num_cyclists` counts bicycles and motorcycles, ridden or parked.** It
  follows the category, not the rider: on mini, 61% of these boxes are marked
  `cycle.without_rider`. The rider attribute is in each annotation's
  `attribute_tokens` if you need riders only.
- **Night and rain come from the scene description** (a whole-word match), not
  the pixels. "After rain" counts as rain, and mini has no scene that was
  actually raining, so rain queries aren't a good showcase on mini.
- **kNN-k caps suppression.** A kept frame can only suppress frames in its own
  top-k or frames that have it in their top-k. A stationary stretch much longer
  than k frames therefore splits into several kept frames rather than one.
  Raise `--k` for long stops.
- **Dark frames embed alike.** Night frames sit closer together in SigLIP space
  than their content warrants, so night scenes lose somewhat more frames to
  dedup at the same threshold.
- **Ego speed is derived from keyframe poses** (a central difference at 2 Hz),
  so short stops and starts are smoothed.
- **Trainval loads its JSON on the driver.** `sample_data.json` and
  `ego_pose.json` take a few GB of RAM for v1.0-trainval.

## Follow-ups (not in scope)

- Swap the embedding stage for a [Geneva](https://lancedb.com/docs/geneva/) UDF,
  so re-embedding with a new model is a column backfill rather than a re-ingest.
- An MCAP ingest path: nuScenes → MCAP with Foxglove's converter, then
  `ray.data.read_mcap` into the same schema.
