# YOLO26 Knowledge Distillation

Train a small, fast YOLO26 student model (e.g. `yolo26n`) from a large teacher (e.g. `yolo26x`) with
knowledge distillation, then export it to ONNX and OpenVINO and verify the exported model still
predicts the same as the PyTorch one.

Supports three tasks: **detect** (boxes), **segment** (boxes + masks) and **pose** (boxes + keypoints).
Can be used from the command line or from a desktop GUI.

## Why

A nano model is far cheaper to run than an x-large one and fits on CPUs, edge devices and
OpenVINO-accelerated Intel hardware, but trained from scratch it loses accuracy. Distillation lets
the student learn from the teacher's soft predictions, recovering much of that accuracy while keeping
the small model's speed. This repository wraps the whole pipeline into one reproducible command, so
the same setup works on any machine.

## Features

- **Distillation for detect, segment and pose**, with a task-specific KD loss for each head
- **End-to-end pipeline**: dataset preparation → training → export → parity check in one run
- **Export** to ONNX (FP32) and OpenVINO (FP16 or INT8 quantized)
- **Export parity check**: compares PyTorch vs ONNX/OpenVINO predictions (detection count, confidence,
  masks or keypoints) on validation images
- **NDJSON datasets**: Ultralytics NDJSON exports are converted to a local YOLO dataset automatically,
  with parallel image download
- **PySide6 GUI** with per-task presets, live log, and saved settings
- **Structured logs** (`[DISTILL]`, `[PROGRESS]`, `[DATA]`) for every stage, easy to grep or parse

## Project structure

```
├── distill_yolo26_seg.py           # CLI entry point (detect / segment / pose)
├── distill_yolo26_seg_gui.py       # PySide6 GUI that builds and runs the CLI command
├── distill/                        # Core library
│   ├── yolo_distill.py             #   config, KD criteria, trainers, export, parity check
│   ├── detect_ndjson_to_yolo.py    #   detect NDJSON → YOLO dataset
│   └── pose_ndjson_to_yolo.py      #   pose NDJSON → YOLO dataset
├── tools/                          # Standalone dataset utilities
│   ├── merge_detect_ndjson_to_yolo.py  # merge several detect NDJSON exports into one dataset
│   ├── prepare_pose_640.py         #   center-crop a pose dataset to 640×640 and drop near-duplicates
│   ├── extract_images.py           #   pick every 5th image from a folder
│   └── tile_images.py              #   cut images into 320×320 tiles
├── pyproject.toml                  # Dependencies (uv, PyTorch CUDA 12.8 wheels)
└── uv.lock
```

## Requirements

- Python 3.11–3.12, managed with [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU with a recent driver (the PyTorch `cu128` wheels ship their own CUDA runtime;
  a full CUDA toolkit is not required)
- Main packages: `ultralytics`, `torch`, `torchvision`, `onnx`, `onnxslim`, `openvino`, `PySide6`

## Quick start

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the environment
uv sync

# Check that PyTorch sees the GPU
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Start the GUI
uv run distill_yolo26_seg_gui.py

# Or see all CLI options
uv run distill_yolo26_seg.py --help
```

## CLI examples

### Segment

```bash
uv run distill_yolo26_seg.py \
  --task segment \
  --teacher yolo26x-seg.pt \
  --student-init yolo26n-seg.pt \
  --data /path/to/data.yaml \
  --imgsz 320 --epochs 120 --batch 16 --workers 8 --device auto \
  --outdir ./runs_distill
```

### Detect from an NDJSON export

```bash
uv run distill_yolo26_seg.py \
  --task detect \
  --teacher yolo26x.pt \
  --data /path/to/detect-export.ndjson \
  --imgsz 320 --epochs 120 --batch 16 --workers 8 --device auto \
  --outdir ./runs_detect_distill
```

### Pose from an NDJSON export

```bash
uv run distill_yolo26_seg.py \
  --task pose \
  --teacher yolo26x-pose.pt \
  --data /path/to/pose-export.ndjson \
  --imgsz 416 --epochs 350 --batch 16 --workers 8 --device auto \
  --outdir ./runs_pose_distill
```

### OpenVINO INT8 export

```bash
uv run distill_yolo26_seg.py \
  --task detect \
  --teacher yolo26x.pt --data ... \
  --export onnx openvino --openvino-int8 \
  --outdir ./runs_int8
```

## How the KD loss works

Each task uses its own distillation criterion:

| Task    | Criterion                           | KD terms                          |
|---------|-------------------------------------|-----------------------------------|
| detect  | `DetectionDistillationCriterion`    | cls + box                         |
| segment | `SegmentationDistillationCriterion` | cls + box + mask + proto + semseg |
| pose    | `PoseDistillationCriterion`         | cls + box + kpt                   |

Total loss: `loss = α · hard_loss + (1 − α) · kd_gain · KD`, where `KD` is a weighted sum of the
per-head terms (`--kd-cls-weight`, `--kd-box-weight`, `--kd-mask-weight`, `--kd-proto-weight`,
`--kd-semseg-weight`, `--kd-kpt-weight`) and soft targets use `--temperature`.

## NDJSON datasets

When `--data` points to an `.ndjson` file, it is converted to a YOLO dataset before training:

- `distill/detect_ndjson_to_yolo.py` for detect, `distill/pose_ndjson_to_yolo.py` for pose
- `--ndjson-download-workers` (default 32) sets parallel image downloads
- For detect and pose, training does not start if any image fails to download

To merge several detect exports into one dataset (classes are merged by name and label ids remapped):

```bash
uv run python tools/merge_detect_ndjson_to_yolo.py \
  --inputs a.ndjson b.ndjson --outdir merge/merged_detect_yolo --clean --workers 24
```

## GUI

- **Task** selector (detect / segment / pose)
- **Presets**: "Apply detect defaults" and "Apply pose accuracy defaults"
- **KD parameters** for every head, including `kd_kpt_w` for pose
- **Export**: ONNX (FP32), OpenVINO (FP16 / INT8)
- **NDJSON** datasets are prepared automatically before the run, with progress in the log
- **Heartbeat**: if the process prints nothing for 60 s, the GUI logs `[GUI] Process is still running`
- All fields are saved to `~/.distill_yolo26_seg_gui.json`

Pose accuracy preset:

```yaml
student: yolo26n-pose.pt
imgsz: 416
epochs: 350  patience: 80  batch: 16
alpha: 0.65  temperature: 3.0  kd_gain: 1.1
kd_box_w: 0.40  kd_kpt_w: 0.90
OpenVINO: FP16 (INT8 off)
```

Detect preset:

```yaml
student: yolo26n.pt
imgsz: 320  epochs: 120  batch: 16
alpha: 0.8  temperature: 2.0  kd_gain: 1.0
kd_cls_w: 1.0  kd_box_w: 0.35  (mask/proto/semseg/kpt = 0)
```

## Export and verification

After training, the best student is exported to ONNX (FP32) and OpenVINO (FP16 by default, INT8 with
`--openvino-int8`). The parity check then runs PyTorch and each exported model on validation images and
compares detection count, confidence and, depending on the task, masks or keypoints. Use
`--strict-parity` / `--strict-export-validation` to fail the run on a mismatch.

## Reading the logs

- `[DISTILL] Stage=cli_start|prepare|train_begin|train_done|export_begin|export_done|parity_begin|parity_done`
- `[PROGRESS] Stage=epoch_end epoch=X/Y train_loss=... kd_loss=... lr=...`
- `[PROGRESS] Stage=val_end mAP50=... mAP50-95=...`
- `[DATA] Prepared dataset: images_total=... downloaded=... failed=...`
- `[DEVICE] auto -> 0 (CUDA)`
- `[GUI] Process is still running; no output for ...s`

## Troubleshooting

```bash
nvidia-smi                                                        # driver and GPU visible?
uv run python -c "import ultralytics, onnx, openvino, torch; print('ok')"  # packages installed?
```
