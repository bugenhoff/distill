from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.torch_utils import unwrap_model

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(slots=True)
class DistillationConfig:
    teacher: str
    student_init: str
    data: str
    outdir: Path
    run_name: str
    imgsz: int = 320
    epochs: int = 120
    batch: int = 16
    workers: int = 8
    device: str = "auto"
    patience: int = 50
    lr0: float = 0.003
    lrf: float = 0.01
    alpha: float = 0.8
    temperature: float = 2.0
    kd_gain: float = 1.0
    kd_cls_weight: float = 1.0
    kd_box_weight: float = 0.35
    kd_mask_weight: float = 0.6
    kd_proto_weight: float = 0.6
    kd_semseg_weight: float = 0.4
    seed: int = 42
    amp: bool = True
    export_formats: tuple[str, ...] = ("onnx", "openvino")
    onnx_opset: int = 17
    export_dynamic: bool = False
    export_batch: int = 1
    export_simplify: bool = True
    export_nms: bool = False
    export_end2end: bool = False
    parity_samples: int = 12
    strict_export_validation: bool = False
    strict_parity: bool = False


class SegmentationDistillationCriterion:
    """Wraps Ultralytics segmentation criterion and adds teacher-student KD loss."""

    def __init__(
        self,
        base_criterion: Any,
        teacher_model: torch.nn.Module,
        student_model: torch.nn.Module,
        *,
        alpha: float,
        temperature: float,
        kd_gain: float,
        kd_cls_weight: float,
        kd_box_weight: float,
        kd_mask_weight: float,
        kd_proto_weight: float,
        kd_semseg_weight: float,
    ) -> None:
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.student_model = student_model

        self.alpha = float(alpha)
        self.temperature = max(float(temperature), 1e-6)
        self.kd_gain = max(float(kd_gain), 0.0)

        self.kd_cls_weight = max(float(kd_cls_weight), 0.0)
        self.kd_box_weight = max(float(kd_box_weight), 0.0)
        self.kd_mask_weight = max(float(kd_mask_weight), 0.0)
        self.kd_proto_weight = max(float(kd_proto_weight), 0.0)
        self.kd_semseg_weight = max(float(kd_semseg_weight), 0.0)
        self._mismatch_warnings: set[str] = set()

        self.last_kd_loss = 0.0

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hard_loss, hard_items = self.base_criterion(preds, batch)

        # Keep validation and pure-supervised path unchanged.
        if not self.student_model.training or self.alpha >= 1.0 or self.kd_gain <= 0.0:
            self.last_kd_loss = 0.0
            return hard_loss, hard_items

        with torch.no_grad():
            teacher_raw = self.teacher_model(batch["img"])

        student_head = _extract_head_for_kd(preds)
        teacher_head = _extract_head_for_kd(teacher_raw)
        kd_loss = self._compute_kd_loss(student_head, teacher_head)

        total_loss = self.alpha * hard_loss + (1.0 - self.alpha) * self.kd_gain * kd_loss
        self.last_kd_loss = float(kd_loss.detach().cpu())
        return total_loss, hard_items

    def update(self) -> None:
        if hasattr(self.base_criterion, "update"):
            self.base_criterion.update()

    def _warn_mismatch_once(self, key: str, student_shape: Any, teacher_shape: Any) -> None:
        if key in self._mismatch_warnings:
            return
        self._mismatch_warnings.add(key)
        LOGGER.warning(
            "[DISTILL] Skip KD term '%s' due to incompatible shapes: student=%s teacher=%s",
            key,
            student_shape,
            teacher_shape,
        )

    def _compute_kd_loss(self, student_head: dict[str, Any], teacher_head: dict[str, Any]) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        weights: list[float] = []

        cls_loss = self._kd_cls(student_head, teacher_head)
        if cls_loss is not None and self.kd_cls_weight > 0:
            losses.append(cls_loss)
            weights.append(self.kd_cls_weight)

        box_loss = self._kd_boxes(student_head, teacher_head)
        if box_loss is not None and self.kd_box_weight > 0:
            losses.append(box_loss)
            weights.append(self.kd_box_weight)

        mask_loss = self._kd_mask_coeff(student_head, teacher_head)
        if mask_loss is not None and self.kd_mask_weight > 0:
            losses.append(mask_loss)
            weights.append(self.kd_mask_weight)

        proto_loss = self._kd_proto(student_head, teacher_head)
        if proto_loss is not None and self.kd_proto_weight > 0:
            losses.append(proto_loss)
            weights.append(self.kd_proto_weight)

        semseg_loss = self._kd_semseg(student_head, teacher_head)
        if semseg_loss is not None and self.kd_semseg_weight > 0:
            losses.append(semseg_loss)
            weights.append(self.kd_semseg_weight)

        if not losses:
            return _zero_like(student_head)

        total_weight = sum(weights)
        weighted = sum(loss * weight for loss, weight in zip(losses, weights))
        return weighted / max(total_weight, 1e-12)

    def _kd_cls(self, student_head: dict[str, Any], teacher_head: dict[str, Any]) -> torch.Tensor | None:
        s_scores = student_head.get("scores")
        t_scores = teacher_head.get("scores")
        if not isinstance(s_scores, torch.Tensor) or not isinstance(t_scores, torch.Tensor):
            return None

        if s_scores.shape != t_scores.shape:
            self._warn_mismatch_once("cls", tuple(s_scores.shape), tuple(t_scores.shape))
            return None

        if s_scores.shape[1] == 1:
            return F.binary_cross_entropy_with_logits(
                s_scores / self.temperature,
                torch.sigmoid(t_scores / self.temperature),
            )

        s_log_probs = F.log_softmax((s_scores / self.temperature).transpose(1, 2), dim=-1)
        t_probs = F.softmax((t_scores / self.temperature).transpose(1, 2), dim=-1)
        return F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (self.temperature**2)

    def _kd_boxes(self, student_head: dict[str, Any], teacher_head: dict[str, Any]) -> torch.Tensor | None:
        s_boxes = student_head.get("boxes")
        t_boxes = teacher_head.get("boxes")
        if not isinstance(s_boxes, torch.Tensor) or not isinstance(t_boxes, torch.Tensor):
            return None

        if s_boxes.shape != t_boxes.shape:
            self._warn_mismatch_once("box", tuple(s_boxes.shape), tuple(t_boxes.shape))
            return None
        return F.smooth_l1_loss(s_boxes, t_boxes.detach())

    def _kd_mask_coeff(self, student_head: dict[str, Any], teacher_head: dict[str, Any]) -> torch.Tensor | None:
        s_coeff = student_head.get("mask_coefficient")
        t_coeff = teacher_head.get("mask_coefficient")
        if not isinstance(s_coeff, torch.Tensor) or not isinstance(t_coeff, torch.Tensor):
            return None

        if s_coeff.shape != t_coeff.shape:
            self._warn_mismatch_once("mask_coeff", tuple(s_coeff.shape), tuple(t_coeff.shape))
            return None
        return F.smooth_l1_loss(s_coeff, t_coeff.detach())

    def _kd_proto(self, student_head: dict[str, Any], teacher_head: dict[str, Any]) -> torch.Tensor | None:
        s_proto, _ = _split_proto_and_semseg(student_head.get("proto"))
        t_proto, _ = _split_proto_and_semseg(teacher_head.get("proto"))
        if s_proto is None or t_proto is None:
            return None

        if s_proto.ndim != 4 or t_proto.ndim != 4:
            self._warn_mismatch_once("proto_ndim", tuple(s_proto.shape), tuple(t_proto.shape))
            return None
        if s_proto.shape[:2] != t_proto.shape[:2]:
            self._warn_mismatch_once("proto_channels", tuple(s_proto.shape), tuple(t_proto.shape))
            return None
        if s_proto.shape[-2:] != t_proto.shape[-2:]:
            t_proto = F.interpolate(t_proto, size=s_proto.shape[-2:], mode="bilinear", align_corners=False)
        return F.mse_loss(s_proto, t_proto)

    def _kd_semseg(self, student_head: dict[str, Any], teacher_head: dict[str, Any]) -> torch.Tensor | None:
        _, s_semseg = _split_proto_and_semseg(student_head.get("proto"))
        _, t_semseg = _split_proto_and_semseg(teacher_head.get("proto"))
        if s_semseg is None or t_semseg is None:
            return None

        if s_semseg.ndim != 4 or t_semseg.ndim != 4:
            self._warn_mismatch_once("semseg_ndim", tuple(s_semseg.shape), tuple(t_semseg.shape))
            return None
        if s_semseg.shape[:2] != t_semseg.shape[:2]:
            self._warn_mismatch_once("semseg_channels", tuple(s_semseg.shape), tuple(t_semseg.shape))
            return None
        if s_semseg.shape[-2:] != t_semseg.shape[-2:]:
            t_semseg = F.interpolate(t_semseg, size=s_semseg.shape[-2:], mode="bilinear", align_corners=False)
        return F.mse_loss(s_semseg, t_semseg)


class DistillationSegTrainer(SegmentationTrainer):
    """SegmentationTrainer with integrated teacher distillation and simple mAP-based best checkpoint selection."""

    def __init__(
        self,
        *,
        teacher_weights: str,
        alpha: float,
        temperature: float,
        kd_gain: float,
        kd_cls_weight: float,
        kd_box_weight: float,
        kd_mask_weight: float,
        kd_proto_weight: float,
        kd_semseg_weight: float,
        overrides: dict[str, Any],
    ) -> None:
        self.teacher_weights = teacher_weights
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.kd_gain = float(kd_gain)
        self.kd_cls_weight = float(kd_cls_weight)
        self.kd_box_weight = float(kd_box_weight)
        self.kd_mask_weight = float(kd_mask_weight)
        self.kd_proto_weight = float(kd_proto_weight)
        self.kd_semseg_weight = float(kd_semseg_weight)

        self._teacher_model: torch.nn.Module | None = None
        self._best_selection_fitness = -math.inf

        super().__init__(overrides=overrides)
        self._register_progress_callbacks()

    def _register_progress_callbacks(self) -> None:
        self.add_callback("on_train_start", self._cb_on_train_start)
        self.add_callback("on_train_epoch_start", self._cb_on_train_epoch_start)
        self.add_callback("on_train_epoch_end", self._cb_on_train_epoch_end)
        self.add_callback("on_val_end", self._cb_on_val_end)
        self.add_callback("on_train_end", self._cb_on_train_end)

    def _cb_on_train_start(self, trainer: "DistillationSegTrainer") -> None:
        LOGGER.info(
            "[PROGRESS] Stage=train_start epochs=%s batch=%s imgsz=%s device=%s",
            getattr(trainer, "epochs", "?"),
            getattr(trainer, "batch_size", "?"),
            getattr(getattr(trainer, "args", None), "imgsz", "?"),
            getattr(trainer, "device", "?"),
        )

    def _cb_on_train_epoch_start(self, trainer: "DistillationSegTrainer") -> None:
        epoch_idx = int(getattr(trainer, "epoch", -1)) + 1
        total_epochs = int(getattr(trainer, "epochs", 0))
        LOGGER.info("[PROGRESS] Stage=epoch_start epoch=%d/%d", epoch_idx, max(total_epochs, 1))

    def _cb_on_train_epoch_end(self, trainer: "DistillationSegTrainer") -> None:
        epoch_idx = int(getattr(trainer, "epoch", -1)) + 1
        total_epochs = int(getattr(trainer, "epochs", 0))

        lr = None
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is not None and getattr(optimizer, "param_groups", None):
            lr = optimizer.param_groups[0].get("lr")

        tloss = getattr(trainer, "tloss", None)
        train_loss = None
        if isinstance(tloss, torch.Tensor):
            values = tloss.detach().float().cpu().flatten().tolist()
            if values:
                train_loss = float(sum(values) / len(values))
        elif isinstance(tloss, (list, tuple)):
            values = [float(x) for x in tloss]
            if values:
                train_loss = float(sum(values) / len(values))

        criterion = getattr(unwrap_model(trainer.model), "criterion", None)
        kd_loss = float(getattr(criterion, "last_kd_loss", 0.0)) if criterion is not None else 0.0

        LOGGER.info(
            "[PROGRESS] Stage=epoch_end epoch=%d/%d train_loss=%s kd_loss=%.6f lr=%s",
            epoch_idx,
            max(total_epochs, 1),
            f"{train_loss:.6f}" if train_loss is not None else "n/a",
            kd_loss,
            f"{float(lr):.8f}" if lr is not None else "n/a",
        )

    def _cb_on_val_end(self, trainer: "DistillationSegTrainer") -> None:
        metrics = getattr(trainer, "metrics", None)
        if not isinstance(metrics, dict):
            return

        map50 = metrics.get("metrics/mAP50(M)", metrics.get("metrics/mAP50(B)"))
        map5095 = metrics.get("metrics/mAP50-95(M)", metrics.get("metrics/mAP50-95(B)"))
        if map50 is None and map5095 is None:
            return

        map50_str = f"{float(map50):.5f}" if map50 is not None else "n/a"
        map5095_str = f"{float(map5095):.5f}" if map5095 is not None else "n/a"
        LOGGER.info("[PROGRESS] Stage=val_end mAP50=%s mAP50-95=%s", map50_str, map5095_str)

    def _cb_on_train_end(self, trainer: "DistillationSegTrainer") -> None:
        best = getattr(trainer, "best", None)
        best_fitness = getattr(trainer, "best_fitness", None)
        best_fitness_str = f"{float(best_fitness):.6f}" if best_fitness is not None else "n/a"
        LOGGER.info("[PROGRESS] Stage=train_end best_fitness=%s best=%s", best_fitness_str, best)

    def _setup_train(self):
        super()._setup_train()
        self._setup_distillation()

    def _setup_distillation(self) -> None:
        LOGGER.info("[DISTILL] Loading teacher model: %s", self.teacher_weights)
        teacher = YOLO(self.teacher_weights, task="segment", verbose=False)
        teacher_model = teacher.model
        teacher_model.to(self.device)
        teacher_model.eval()
        teacher_model.requires_grad_(False)

        student_model = unwrap_model(self.model)
        base_criterion = student_model.init_criterion()
        student_model.criterion = SegmentationDistillationCriterion(
            base_criterion=base_criterion,
            teacher_model=teacher_model,
            student_model=student_model,
            alpha=self.alpha,
            temperature=self.temperature,
            kd_gain=self.kd_gain,
            kd_cls_weight=self.kd_cls_weight,
            kd_box_weight=self.kd_box_weight,
            kd_mask_weight=self.kd_mask_weight,
            kd_proto_weight=self.kd_proto_weight,
            kd_semseg_weight=self.kd_semseg_weight,
        )
        self._teacher_model = teacher_model

        LOGGER.info(
            "[DISTILL] Enabled KD: alpha=%.3f temp=%.3f kd_gain=%.3f weights(cls=%.2f box=%.2f mask=%.2f proto=%.2f semseg=%.2f)",
            self.alpha,
            self.temperature,
            self.kd_gain,
            self.kd_cls_weight,
            self.kd_box_weight,
            self.kd_mask_weight,
            self.kd_proto_weight,
            self.kd_semseg_weight,
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys

        loss_list = [round(float(x), 5) for x in loss_items]
        if len(loss_list) < len(keys):
            loss_list.extend([0.0] * (len(keys) - len(loss_list)))
        return dict(zip(keys, loss_list))

    def validate(self):
        metrics, default_fitness = super().validate()
        if metrics is None:
            return None, None

        map_seg = float(metrics.get("metrics/mAP50-95(M)", metrics.get("metrics/mAP50-95(B)", 0.0)))
        LOGGER.info("[DISTILL] Validation mAP50-95(M)=%.5f", map_seg)

        self._append_selection_log(
            {
                "epoch": int(getattr(self, "epoch", -1)) + 1,
                "map50_95_m": map_seg,
                "fitness": map_seg,
            }
        )

        self._best_selection_fitness = max(self._best_selection_fitness, map_seg)
        self.best_fitness = self._best_selection_fitness

        if default_fitness is None:
            return metrics, map_seg
        return metrics, map_seg

    def _append_selection_log(self, payload: dict[str, Any]) -> None:
        log_path = Path(self.save_dir) / "selection_metrics.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_distillation(config: DistillationConfig) -> dict[str, Any]:
    outdir = Path(config.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("[DISTILL] Stage=prepare outdir=%s", outdir)
    resolved_device = _resolve_ultralytics_device(config.device)
    resolved_data = _resolve_data_source(config.data, outdir=outdir)
    train_fliplr = _resolve_train_fliplr_for_data_yaml(resolved_data, default_fliplr=0.5)
    if str(config.device).strip() != resolved_device:
        LOGGER.info("[DISTILL] Device resolved: requested=%r -> effective=%r", config.device, resolved_device)
    LOGGER.info("[DISTILL] Augment setup: fliplr=%.3f (auto)", train_fliplr)

    overrides = {
        "task": "segment",
        "model": _resolve_model_source(config.student_init),
        "data": resolved_data,
        "imgsz": int(config.imgsz),
        "epochs": int(config.epochs),
        "batch": int(config.batch),
        "workers": int(config.workers),
        "device": resolved_device,
        "project": str(outdir),
        "name": config.run_name,
        "exist_ok": True,
        "patience": int(config.patience),
        "lr0": float(config.lr0),
        "lrf": float(config.lrf),
        "cos_lr": True,
        "amp": bool(config.amp),
        "save": True,
        "val": True,
        "seed": int(config.seed),
        "verbose": True,
        # Conservative augmentation setup improves mask quality on noisy small objects.
        "hsv_h": 0.010,
        "hsv_s": 0.45,
        "hsv_v": 0.20,
        "degrees": 2.0,
        "translate": 0.03,
        "scale": 0.20,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": float(train_fliplr),
        "mosaic": 0.10,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 25,
        "multi_scale": 0.0,
    }

    trainer = DistillationSegTrainer(
        teacher_weights=_resolve_model_source(config.teacher),
        alpha=config.alpha,
        temperature=config.temperature,
        kd_gain=config.kd_gain,
        kd_cls_weight=config.kd_cls_weight,
        kd_box_weight=config.kd_box_weight,
        kd_mask_weight=config.kd_mask_weight,
        kd_proto_weight=config.kd_proto_weight,
        kd_semseg_weight=config.kd_semseg_weight,
        overrides=overrides,
    )
    LOGGER.info("[DISTILL] Stage=train_begin run_name=%s", config.run_name)
    trainer.train()
    LOGGER.info("[DISTILL] Stage=train_done")

    run_dir = Path(trainer.save_dir).resolve()
    best_from_run = Path(trainer.best).resolve()
    last_from_run = Path(trainer.last).resolve()
    if not best_from_run.exists():
        best_from_run = last_from_run

    selected_checkpoint = best_from_run
    checkpoint_probes = {
        str(best_from_run): {"selected": True, "reason": "trainer_best_by_map50_95M"},
        str(last_from_run): {"selected": bool(last_from_run == best_from_run), "reason": "trainer_last"},
    }

    best_distilled = outdir / "best_distilled.pt"
    shutil.copy2(selected_checkpoint, best_distilled)

    export_reports: list[dict[str, Any]] = []
    exported_paths: dict[str, str] = {}
    if config.export_formats:
        LOGGER.info("[DISTILL] Stage=export_begin formats=%s", list(config.export_formats))
        exported_paths, export_reports = export_model_artifacts(
            model_path=best_distilled,
            outdir=outdir,
            imgsz=config.imgsz,
            export_formats=config.export_formats,
            onnx_opset=config.onnx_opset,
            dynamic=config.export_dynamic,
            batch=config.export_batch,
            simplify=config.export_simplify,
            nms=config.export_nms,
            end2end=config.export_end2end,
            strict_validation=config.strict_export_validation,
        )
        LOGGER.info("[DISTILL] Stage=export_done artifacts=%s", exported_paths)

    LOGGER.info("[DISTILL] Stage=parity_begin samples=%d", int(config.parity_samples))
    parity = run_export_parity_check(
        model_path=best_distilled,
        exported=exported_paths,
        data_yaml=resolved_data,
        imgsz=config.imgsz,
        max_samples=config.parity_samples,
    )
    LOGGER.info("[DISTILL] Stage=parity_done ok=%s", None if parity is None else parity.get("ok"))
    if config.strict_parity and parity and not parity.get("ok", True):
        raise RuntimeError(f"Export parity check failed: {parity}")

    summary = {
        "run_dir": str(run_dir),
        "best_from_run": str(best_from_run),
        "last_from_run": str(last_from_run),
        "selected_checkpoint": str(selected_checkpoint),
        "checkpoint_probes": checkpoint_probes,
        "best_distilled": str(best_distilled),
        "data_requested": str(config.data),
        "data_effective": resolved_data,
        "device_requested": str(config.device),
        "device_effective": resolved_device,
        "train_fliplr_effective": float(train_fliplr),
        "export_end2end": bool(config.export_end2end),
        "exported": exported_paths,
        "export_reports": export_reports,
        "parity": parity,
    }

    summary_path = outdir / "distill_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("[DISTILL] Summary saved: %s", summary_path)
    return summary


def export_model_artifacts(
    *,
    model_path: Path,
    outdir: Path,
    imgsz: int,
    export_formats: Iterable[str],
    onnx_opset: int,
    dynamic: bool,
    batch: int,
    simplify: bool,
    nms: bool,
    end2end: bool,
    strict_validation: bool,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    formats = tuple(dict.fromkeys(fmt.lower().strip() for fmt in export_formats if fmt))
    model = YOLO(str(model_path), task="segment", verbose=False)

    exported_paths: dict[str, str] = {}
    reports: list[dict[str, Any]] = []

    if "onnx" in formats:
        kwargs = {
            "format": "onnx",
            "imgsz": int(imgsz),
            "opset": int(onnx_opset),
            "dynamic": bool(dynamic),
            "simplify": bool(simplify),
            "half": False,
            "batch": int(batch),
            "nms": bool(nms),
            "end2end": bool(end2end),
        }
        LOGGER.info("[EXPORT] ONNX args: %s", kwargs)
        onnx_path = Path(model.export(**kwargs))
        onnx_path = _move_artifact_if_needed(onnx_path, outdir)
        ok, message = validate_onnx_artifact(onnx_path)
        reports.append({"format": "onnx", "artifact": str(onnx_path), "ok": ok, "message": message})
        exported_paths["onnx"] = str(onnx_path)
        if strict_validation and not ok:
            raise RuntimeError(f"ONNX validation failed: {message}")

    if "openvino" in formats:
        kwargs = {
            "format": "openvino",
            "imgsz": int(imgsz),
            "dynamic": bool(dynamic),
            "half": True,
            "int8": False,
            "batch": int(batch),
            "nms": bool(nms),
            "end2end": bool(end2end),
        }
        LOGGER.info("[EXPORT] OpenVINO args: %s", kwargs)
        ov_path = Path(model.export(**kwargs))
        ov_path = _move_artifact_if_needed(ov_path, outdir)
        ok, message = validate_openvino_artifact(ov_path)
        reports.append({"format": "openvino", "artifact": str(ov_path), "ok": ok, "message": message})
        exported_paths["openvino"] = str(ov_path)
        if strict_validation and not ok:
            raise RuntimeError(f"OpenVINO validation failed: {message}")

    return exported_paths, reports


def validate_onnx_artifact(artifact_path: Path) -> tuple[bool, str]:
    if not artifact_path.exists():
        return False, f"ONNX file not found: {artifact_path}"
    try:
        import onnx
    except Exception as exc:
        return False, f"onnx package unavailable: {exc}"

    try:
        model = onnx.load(str(artifact_path))
        onnx.checker.check_model(model)
        return True, "ONNX checker: OK"
    except Exception as exc:
        return False, f"ONNX checker error: {exc}"


def validate_openvino_artifact(artifact_path: Path) -> tuple[bool, str]:
    if not artifact_path.exists():
        return False, f"OpenVINO artifact not found: {artifact_path}"
    try:
        try:
            from openvino.runtime import Core  # type: ignore[attr-defined]
        except Exception:
            from openvino import Core  # OpenVINO 2026+
    except Exception as exc:
        return False, f"openvino Core unavailable: {exc}"

    try:
        xml_path = _resolve_openvino_xml_path(artifact_path)
        core = Core()
        core.read_model(str(xml_path))
        return True, f"OpenVINO read_model: OK ({xml_path.name})"
    except Exception as exc:
        return False, f"OpenVINO validation error: {exc}"


def run_export_parity_check(
    *,
    model_path: Path,
    exported: dict[str, str],
    data_yaml: str,
    imgsz: int,
    max_samples: int,
) -> dict[str, Any] | None:
    if not exported:
        return None

    sample_images = discover_sample_images_from_data_yaml(data_yaml, split="val", limit=max_samples)
    if not sample_images:
        return {"ok": False, "reason": "No validation images discovered for parity check"}

    ref_model = YOLO(str(model_path), task="segment", verbose=False)
    ref_stats = _collect_prediction_signature(ref_model, sample_images, imgsz)

    target_reports: dict[str, Any] = {}
    ok = True
    for fmt, artifact in exported.items():
        target_model = YOLO(str(artifact), task="segment", verbose=False)
        target_stats = _collect_prediction_signature(target_model, sample_images, imgsz)

        mean_abs_count_gap = abs(ref_stats["mean_count"] - target_stats["mean_count"])
        mean_abs_conf_gap = abs(ref_stats["mean_conf"] - target_stats["mean_conf"])
        mean_abs_mask_area_gap = abs(ref_stats["mean_mask_area"] - target_stats["mean_mask_area"])
        mask_area_rel_gap = mean_abs_mask_area_gap / max(ref_stats["mean_mask_area"], 1e-6)

        # Confidence scales can differ across backends; use count+mask parity as stable criteria.
        fmt_ok = mean_abs_count_gap <= 1.0 and mask_area_rel_gap <= 0.20
        ok = ok and fmt_ok
        target_reports[fmt] = {
            "ok": fmt_ok,
            "artifact": artifact,
            "mean_abs_count_gap": mean_abs_count_gap,
            "mean_abs_conf_gap": mean_abs_conf_gap,
            "mean_abs_mask_area_gap": mean_abs_mask_area_gap,
            "mask_area_rel_gap": mask_area_rel_gap,
            "reference": ref_stats,
            "target": target_stats,
        }

    return {"ok": ok, "samples": len(sample_images), "targets": target_reports}


def discover_sample_images_from_data_yaml(data_yaml: str, split: str = "val", limit: int = 16) -> list[Path]:
    yaml_path = Path(data_yaml).expanduser()
    if not yaml_path.exists():
        return []

    cfg = YAML.load(yaml_path)
    if not isinstance(cfg, dict):
        return []

    data_root = cfg.get("path")
    if data_root:
        root = Path(data_root)
        if not root.is_absolute():
            root = (yaml_path.parent / root).resolve()
    else:
        root = yaml_path.parent.resolve()

    split_value = cfg.get(split)
    if split_value is None:
        return []

    paths = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for entry in paths:
        if not isinstance(entry, str):
            continue
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()

        if candidate.is_dir():
            images.extend(sorted(_iter_images(candidate)))
            continue

        if candidate.is_file() and candidate.suffix.lower() in _IMAGE_SUFFIXES:
            images.append(candidate)
            continue

        if candidate.is_file() and candidate.suffix.lower() == ".txt":
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                img_path = Path(line)
                if not img_path.is_absolute():
                    img_path = (root / img_path).resolve()
                if img_path.exists() and img_path.suffix.lower() in _IMAGE_SUFFIXES:
                    images.append(img_path)

    unique_images = []
    seen: set[Path] = set()
    for image_path in images:
        image_path = image_path.resolve()
        if image_path in seen:
            continue
        seen.add(image_path)
        unique_images.append(image_path)
        if len(unique_images) >= max(limit, 1):
            break

    return unique_images


def _collect_prediction_signature(model: YOLO, images: list[Path], imgsz: int) -> dict[str, float]:
    total_count = 0.0
    total_conf = 0.0
    total_masks = 0.0

    for image_path in images:
        result = model.predict(source=str(image_path), imgsz=imgsz, verbose=False)[0]
        if result.boxes is not None:
            count = int(len(result.boxes))
            conf = float(result.boxes.conf.mean().item()) if count else 0.0
        else:
            count = 0
            conf = 0.0

        mask_area = 0.0
        if result.masks is not None and hasattr(result.masks, "data") and result.masks.data is not None:
            mask_area = float(result.masks.data.float().sum().item())

        total_count += count
        total_conf += conf
        total_masks += mask_area

    n = max(len(images), 1)
    return {
        "mean_count": total_count / n,
        "mean_conf": total_conf / n,
        "mean_mask_area": total_masks / n,
    }


def _resolve_data_source(value: str, *, outdir: Path | None = None) -> str:
    path = Path(value).expanduser()
    if not path.exists():
        return value
    path = path.resolve()
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return str(path)
    return _autofix_data_yaml(path, outdir=outdir)


def _resolve_train_fliplr_for_data_yaml(data_yaml: str, default_fliplr: float = 0.5) -> float:
    yaml_path = Path(data_yaml).expanduser()
    if not yaml_path.exists():
        return float(default_fliplr)
    try:
        cfg = YAML.load(yaml_path)
    except Exception:
        return float(default_fliplr)
    if not isinstance(cfg, dict):
        return float(default_fliplr)

    names_raw = cfg.get("names")
    if isinstance(names_raw, dict):
        names = [str(v).lower() for _, v in sorted(names_raw.items(), key=lambda kv: int(kv[0]))]
    elif isinstance(names_raw, list):
        names = [str(v).lower() for v in names_raw]
    else:
        names = []

    directional = any(("left" in n or "right" in n) for n in names)
    if directional and default_fliplr > 0:
        LOGGER.warning("[DISTILL] Directional classes detected (left/right). Setting fliplr=0.0.")
        return 0.0
    return float(default_fliplr)


def _resolve_model_source(value: str) -> str:
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else value


def _resolve_ultralytics_device(value: str | None) -> str:
    raw = "" if value is None else str(value).strip()
    normalized = raw.lower()

    if not raw or normalized == "auto":
        if torch.cuda.is_available():
            return "0"
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False.")

    if normalized == "cuda":
        if torch.cuda.is_available():
            return "0"
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False.")
    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False.")
        index = normalized.split(":", 1)[1].strip()
        return index or "0"

    return raw


def _autofix_data_yaml(yaml_path: Path, *, outdir: Path | None = None) -> str:
    cfg = YAML.load(yaml_path)
    if not isinstance(cfg, dict):
        return str(yaml_path)

    dataset_root = yaml_path.parent.resolve()
    changed = False

    def _rewrite_path_like(value: Any, *, fallback_to_root: bool = False) -> Any:
        nonlocal changed
        if not isinstance(value, str):
            return value
        p = Path(value).expanduser()
        if not p.is_absolute() or p.exists():
            return value

        parts = list(p.parts)
        for anchor in ("train", "valid", "val", "test"):
            if anchor not in parts:
                continue
            idx = parts.index(anchor)
            rel_from_anchor = Path(*parts[idx:])
            anchored_candidate = (dataset_root / rel_from_anchor).resolve()
            if anchored_candidate.exists():
                changed = True
                return str(anchored_candidate)

        candidate = dataset_root / p.name
        if candidate.exists():
            changed = True
            return str(candidate.resolve())
        if fallback_to_root:
            changed = True
            return str(dataset_root)
        return value

    cfg["path"] = _rewrite_path_like(cfg.get("path"), fallback_to_root=True)
    for key in ("train", "val", "test", "minival"):
        cfg[key] = _rewrite_path_like(cfg.get(key))

    roboflow = cfg.get("roboflow")
    if isinstance(roboflow, dict):
        for key, value in list(roboflow.items()):
            if not key.endswith("_ann"):
                continue
            roboflow[key] = _rewrite_path_like(value)

    if not changed:
        return str(yaml_path)

    target_dir = outdir.resolve() if outdir else dataset_root
    target_dir.mkdir(parents=True, exist_ok=True)
    patched_path = target_dir / f"{yaml_path.stem}.autofix.local{yaml_path.suffix}"
    YAML.save(file=patched_path, data=cfg)
    LOGGER.warning("[DISTILL] Auto-fixed dataset yaml paths: %s -> %s", yaml_path, patched_path)
    return str(patched_path)


def _extract_head_for_kd(preds: Any) -> dict[str, Any]:
    if isinstance(preds, tuple):
        preds = preds[1]

    if isinstance(preds, dict):
        if "one2one" in preds and isinstance(preds["one2one"], dict):
            return preds["one2one"]
        if "one2many" in preds and isinstance(preds["one2many"], dict):
            return preds["one2many"]
        return preds

    raise TypeError(f"Unsupported predictions type for KD: {type(preds)!r}")


def _split_proto_and_semseg(value: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if isinstance(value, torch.Tensor):
        return value, None
    if isinstance(value, tuple) and len(value) == 2:
        proto, semseg = value
        proto_tensor = proto if isinstance(proto, torch.Tensor) else None
        semseg_tensor = semseg if isinstance(semseg, torch.Tensor) else None
        return proto_tensor, semseg_tensor
    return None, None


def _iter_images(folder: Path):
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            yield path


def _move_artifact_if_needed(artifact_path: Path, outdir: Path) -> Path:
    artifact_path = artifact_path.resolve()
    outdir = outdir.resolve()

    if artifact_path.parent == outdir:
        return artifact_path

    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / artifact_path.name
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    shutil.move(str(artifact_path), str(target))
    return target


def _resolve_openvino_xml_path(artifact_path: Path) -> Path:
    if artifact_path.is_file() and artifact_path.suffix.lower() == ".xml":
        return artifact_path
    if artifact_path.is_dir():
        xml_files = sorted(artifact_path.glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No .xml file found in OpenVINO artifact: {artifact_path}")
        return xml_files[0]
    raise FileNotFoundError(f"OpenVINO artifact not found: {artifact_path}")


def _zero_like(head: dict[str, Any]) -> torch.Tensor:
    for key in ("scores", "boxes", "mask_coefficient"):
        value = head.get(key)
        if isinstance(value, torch.Tensor):
            return value.sum() * 0.0
    raise RuntimeError("Cannot infer device for zero KD tensor")
