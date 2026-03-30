# SceneFrame

Extract frame pairs from video scenes for image/video generation model training.

SceneFrame detects scene boundaries in videos, extracts frame pairs (_A and _B), and provides a cleaning + control image pipeline to produce high-quality training datasets.

## Installation

```bash
# Base (CPU scene detection)
pip install -e .

# With GPU support (TransNetV2, depth maps, NSFW filter)
pip install -e ".[gpu]"

# Character detection (YOLO)
pip install ultralytics
```

## Pipeline

```
extract → clean → control
```

### 1. Extract

Detect scenes and extract frame pairs from videos.

```bash
# Basic extraction (all modes, CPU)
sceneframe extract /path/to/videos -o /path/to/output

# GPU scene detection (TransNetV2), single mode
sceneframe extract /path/to/videos -o /output -m inter-seq --engine transnetv2

# From a .txt file listing directories
sceneframe extract videos.txt -o /output --min-duration 10
```

**Modes:**
- `intra` — first + last frame of each scene (same scene, different time)
- `inter-seq` — consecutive scene pairs, no overlap (different scenes)
- `inter-slide` — consecutive scenes, sliding window (overlapping pairs)
- `all` — run all three modes (default)

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `-m, --mode` | `all` | Extraction mode |
| `-e, --engine` | `pyscenedetect` | Scene detection: `pyscenedetect` (CPU) or `transnetv2` (GPU) |
| `--min-duration` | `10.0` | Minimum video duration in seconds |
| `--max-pairs` | unlimited | Max pairs per video per mode |
| `-w, --workers` | auto | Parallel workers (16 for transnetv2, cpu-2 for pyscenedetect) |
| `--recursive/--no-recursive` | recursive | Search subdirectories |
| `--resume/--no-resume` | resume | Skip already-processed videos |
| `--redetect` | off | Re-segment long scenes (>20s) with AdaptiveDetector |

Videos that fail to decode are moved to `{output}/skipped/`.

### 2. Clean

Remove bad pairs: solid colors, blur, duplicates, NSFW/character filtering.

```bash
# Basic cleaning
sceneframe clean /path/to/output --similarity 0.92

# With blur detection and NSFW filter (keep NSFW, remove SFW)
sceneframe clean /path/to/output --blur --nsfw --keep-nsfw --similarity 0.92

# With character detection (YOLO)
sceneframe clean /path/to/output --blur --character --similarity 0.92

# Smart filter (experimental): each image must be NSFW or have a character
sceneframe clean /path/to/output --blur --smart-filter --similarity 0.92
```

**Pipeline order:** solid → blur (retry) → duplicates → [smart-filter | character + NSFW] → orphans

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--no-solid` | | Skip solid-color removal |
| `--no-duplicates` | | Skip duplicate removal |
| `--similarity` | `0.96` | Cosine similarity threshold (lower = more aggressive) |
| `--blur` | off | Remove blurry _A frames (Laplacian variance) |
| `--blur-threshold` | `100.0` | Laplacian variance below this = blurry |
| `--character` | off | Remove pairs without characters in _A (YOLO) |
| `--character-percentage` | `100.0` | % of pairs to check |
| `--nsfw/--no-nsfw` | off | NSFW filter (Falconsai/nsfw_image_detection) |
| `--keep-nsfw/--remove-nsfw` | keep | Keep NSFW pairs (reverse filter for NSFW datasets) |
| `--smart-filter` | off | Experimental: NSFW→YOLO cascade on both _A and _B |
| `--dry-run` | | Show what would be removed without deleting |

**Smart filter** (`--smart-filter`): Each image (_A and _B) must individually pass at least one check — NSFW detected OR character/person detected by YOLO. Failed images get up to 3 retries with nearby frames. Cannot be used with `--nsfw` or `--character`.

### 3. Control

Generate control images for conditional training (ControlNet, IP-Adapter, etc.).

```bash
# 100% depth maps
sceneframe control /path/to/output

# Mixed: 30% depth + 30% canny + 40% image_base
sceneframe control /path/to/output -p 100 --depth 30 --canny 30 --image-base 40

# 50% of images get control, all depth
sceneframe control /path/to/output -p 50
```

**Control types:**
- `depth` — Depth maps via Depth Anything V2 Large (GPU, `_C.jpg`)
- `canny` — Canny edge detection (CPU, `_C.jpg`)
- `image-base` — Copy of _A or _B as `_image_base.jpg`

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `-p, --percentage` | `100` | % of images that get control |
| `--depth` | `100` | % of selected for depth maps |
| `--canny` | `0` | % of selected for canny edges |
| `--image-base` | `0` | % of selected for image copy |
| `--image-base-source` | `A` | Which image to copy (A or B) |
| `-b, --batch-size` | `32` | Depth GPU batch size |
| `--canny-low` | `100` | Canny lower threshold |
| `--canny-high` | `200` | Canny upper threshold |

`--depth`, `--canny`, and `--image-base` must sum to 100.

## Upload

Upload dataset to HuggingFace Hub (ZIP archives + XET for speed).

```bash
export HF_TOKEN=hf_xxxxx
python upload_dataset.py /path/to/dataset

# High performance mode (64GB+ RAM)
HF_XET_HIGH_PERFORMANCE=1 python upload_dataset.py /path/to/dataset

# Custom repo name and zip size
python upload_dataset.py /path/to/dataset --repo-name my-dataset --max-zip-gb 10
```

## Output Structure

```
output/
├── intra/                    # intra-scene pairs
│   ├── 000001_A.jpg
│   ├── 000001_B.jpg
│   ├── 000001_C.jpg          # control image (depth or canny)
│   ├── 000001_image_base.jpg  # image base copy
│   └── ...
├── inter-seq/                # inter-scene sequential pairs
├── inter-slide/              # inter-scene sliding pairs
├── pairs_metadata.jsonl      # source video, frame indices, scene bounds
├── processed_videos.log      # resume tracking
└── skipped/                  # videos that failed to decode
```

## Hardware Recommendations

| Hardware | Extract workers | Depth batch | YOLO batch | NSFW batch |
|----------|----------------|-------------|------------|------------|
| RTX 5090 (32GB) | 16 | 64 | 32-64 | 64 |
| RTX PRO 6000 (96GB) | 16 | 128-256 | 64-128 | 128-256 |
| CPU only | cpu_count - 2 | N/A | N/A | N/A |

## Dependencies

**Base:** click, scenedetect, opencv-python, numpy, tqdm

**GPU (optional):** torch, transformers, Pillow, transnetv2-pytorch

**Character detection:** ultralytics (YOLOv8/v11)

## License

MIT
