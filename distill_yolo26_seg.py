#!/usr/bin/env python3
"""Distill YOLO26 segmentation teacher into nano student and export ONNX/OpenVINO.

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
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.yolo_distill import DistillationConfig, run_distillation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill YOLO26x-seg teacher to YOLO26n-seg student with export pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--teacher", required=True, help="Teacher model path or alias (e.g. yolo26x-seg.pt)")
    parser.add_argument(
        "--student-init",
        default="yolo26n-seg.pt",
        help="Student initialization model path or alias (e.g. yolo26n-seg.pt)",
    )
    parser.add_argument("--data", required=True, help="Dataset YAML used for teacher training")
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

    parser.add_argument(
        "--outdir",
        default=".",
        help="Output folder (default: current working directory where script is launched)",
    )
    parser.add_argument("--run-name", default="distill_yolo26_seg")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config = DistillationConfig(
        teacher=args.teacher,
        student_init=args.student_init,
        data=args.data,
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
    )

    summary = run_distillation(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
