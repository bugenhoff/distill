#!/usr/bin/env python3
"""Distill YOLO26 detect/segment/pose teacher into nano student and export ONNX/OpenVINO.

Examples:
  .venv/bin/python scripts/distill_yolo26_seg.py \
      --teacher /path/to/teacher.pt \
      --student-init yolo26n-seg.pt \
      --data /path/to/dataset.yaml \
      --imgsz 320 --epochs 120 --batch 16 --device auto
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.yolo_distill import DistillationConfig, run_distillation  # noqa: E402


def _configure_console_logging() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    try:
        from ultralytics.utils import LOGGER
    except Exception:
        return

    LOGGER.setLevel(logging.INFO)
    if not LOGGER.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.INFO)
        LOGGER.addHandler(handler)
    for handler in LOGGER.handlers:
        handler.setLevel(logging.INFO)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill YOLO26 teacher to nano student with export pipeline (detect, segment, or pose).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--task", choices=["detect", "segment", "pose"], default="segment", help="Distillation task")

    parser.add_argument("--teacher", required=True, help="Teacher model path or alias (e.g. yolo26x-seg.pt)")
    parser.add_argument(
        "--student-init",
        default="yolo26n-seg.pt",
        help="Student initialization model path or alias (e.g. yolo26n-seg.pt)",
    )
    parser.add_argument("--data", required=True, help="Dataset YAML, or detect/pose NDJSON export")
    parser.add_argument("--imgsz", type=int, default=320, help="Training and export image size")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda[:id]")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr0", type=float, default=0.003)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--alpha", type=float, default=0.8, help="Hard/KD blend: total = alpha*hard + (1-alpha)*kd")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kd-gain", type=float, default=1.0, help="Global multiplier for KD component")
    parser.add_argument("--kd-cls-weight", type=float, default=1.0, help="KD weight for class logits")
    parser.add_argument("--kd-box-weight", type=float, default=0.35, help="KD weight for box regression")
    parser.add_argument("--kd-mask-weight", type=float, default=0.6, help="KD weight for mask coefficients")
    parser.add_argument("--kd-proto-weight", type=float, default=0.6, help="KD weight for proto masks")
    parser.add_argument("--kd-semseg-weight", type=float, default=0.4, help="KD weight for semantic head")
    parser.add_argument("--kd-kpt-weight", type=float, default=0.6, help="KD weight for pose keypoints")

    parser.add_argument(
        "--export",
        nargs="*",
        choices=["onnx", "openvino"],
        default=["onnx", "openvino"],
        help="Artifacts to export from best distilled nano model",
    )
    parser.add_argument("--onnx-opset", type=int, default=17)
    parser.add_argument("--export-dynamic", action="store_true", help="Export with dynamic shapes")
    parser.add_argument("--export-batch", type=int, default=1)
    parser.add_argument("--no-export-simplify", action="store_true", help="Disable ONNX graph simplify")
    parser.add_argument("--export-nms", action="store_true", help="Embed NMS into exported graph")
    parser.add_argument(
        "--export-end2end",
        action="store_true",
        help="Export one-to-one end2end head (YOLO26 default); by default script uses end2end=False for better parity.",
    )
    parser.add_argument("--parity-samples", type=int, default=12)
    parser.add_argument("--strict-export-validation", action="store_true")
    parser.add_argument("--strict-parity", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument("--openvino-int8", action="store_true", help="Export OpenVINO artifact in INT8 mode")
    parser.add_argument(
        "--ndjson-download-workers",
        type=int,
        default=32,
        help="Parallel image download workers when --data points to a detect/pose NDJSON export",
    )

    parser.add_argument(
        "--outdir",
        default=".",
        help="Output folder (default: current working directory where script is launched)",
    )
    parser.add_argument("--run-name", default="distill_yolo26_seg")

    return parser.parse_args()


def _prepare_ndjson_data_source(args: argparse.Namespace) -> str:
    data_path = Path(args.data).expanduser()
    if data_path.suffix.lower() != ".ndjson":
        return args.data
    if not data_path.exists():
        raise RuntimeError(f"NDJSON not found: {data_path}")
    if args.task == "segment":
        raise RuntimeError("Segment NDJSON conversion is not supported; use dataset YAML for segment task.")

    outdir = Path(args.outdir).expanduser().resolve()
    dataset_dir = outdir / "datasets" / f"{data_path.stem}_{args.task}_yolo"
    print(
        f"[DATA] Converting {args.task} NDJSON: {data_path} -> {dataset_dir} "
        f"(download_workers={int(args.ndjson_download_workers)})",
        flush=True,
    )

    if args.task == "detect":
        from utils.detect_ndjson_to_yolo import convert
    else:
        from utils.pose_ndjson_to_yolo import convert

    summary = convert(
        data_path.resolve(),
        dataset_dir,
        download_images=True,
        timeout=30.0,
        overwrite_images=False,
        workers=int(args.ndjson_download_workers),
        progress_every=100,
        progress_callback=lambda done, total: print(f"[DATA] downloaded {done}/{total}", flush=True),
    )
    yaml_path = summary.get("yaml")
    if not isinstance(yaml_path, str) or not yaml_path:
        raise RuntimeError("NDJSON conversion failed: data.yaml was not produced")

    print(
        "[DATA] Prepared dataset: "
        f"images_total={summary.get('images_total')} downloaded={summary.get('images_downloaded')} "
        f"failed={summary.get('images_failed')} yaml={yaml_path}",
        flush=True,
    )
    failed = int(summary.get("images_failed") or 0)
    if failed > 0:
        raise RuntimeError(f"Failed to download {failed} images from NDJSON.")
    return yaml_path


def _apply_task_student_default(args: argparse.Namespace) -> None:
    if args.student_init != "yolo26n-seg.pt":
        return
    if args.task == "detect":
        args.student_init = "yolo26n.pt"
    elif args.task == "pose":
        args.student_init = "yolo26n-pose.pt"


def main() -> None:
    _configure_console_logging()
    args = _parse_args()
    _apply_task_student_default(args)
    print("[DISTILL] Stage=cli_start", flush=True)
    effective_data = _prepare_ndjson_data_source(args)

    config = DistillationConfig(
        task=args.task,
        teacher=args.teacher,
        student_init=args.student_init,
        data=effective_data,
        outdir=Path(args.outdir),
        run_name=args.run_name,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        lr0=args.lr0,
        lrf=args.lrf,
        seed=args.seed,
        alpha=args.alpha,
        temperature=args.temperature,
        kd_gain=args.kd_gain,
        kd_cls_weight=args.kd_cls_weight,
        kd_box_weight=args.kd_box_weight,
        kd_mask_weight=args.kd_mask_weight,
        kd_proto_weight=args.kd_proto_weight,
        kd_semseg_weight=args.kd_semseg_weight,
        kd_kpt_weight=args.kd_kpt_weight,
        export_formats=tuple(args.export),
        onnx_opset=args.onnx_opset,
        export_dynamic=args.export_dynamic,
        export_batch=args.export_batch,
        export_simplify=not args.no_export_simplify,
        export_nms=args.export_nms,
        export_end2end=args.export_end2end,
        parity_samples=args.parity_samples,
        strict_export_validation=args.strict_export_validation,
        strict_parity=args.strict_parity,
        amp=not args.no_amp,
        openvino_int8=args.openvino_int8,
    )

    summary = run_distillation(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
