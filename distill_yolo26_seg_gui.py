#!/usr/bin/env python3
"""PySide6 GUI launcher for distill_yolo26_seg.py."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

DETECT_DEFAULTS = {
    "student": "yolo26n.pt",
    "imgsz": "320",
    "epochs": "120",
    "batch": "16",
    "workers": "8",
    "download_workers": "32",
    "patience": "50",
    "alpha": "0.8",
    "temperature": "2.0",
    "kd_gain": "1.0",
    "kd_cls_weight": "1.0",
    "kd_box_weight": "0.35",
    "kd_mask_weight": "0.0",
    "kd_proto_weight": "0.0",
    "kd_semseg_weight": "0.0",
    "kd_kpt_weight": "0.0",
}

SEGMENT_DEFAULTS = {
    "student": "yolo26n-seg.pt",
    "imgsz": "320",
    "epochs": "120",
    "batch": "16",
    "workers": "8",
    "download_workers": "32",
    "patience": "50",
    "alpha": "0.8",
    "temperature": "2.0",
    "kd_gain": "1.0",
    "kd_cls_weight": "1.0",
    "kd_box_weight": "0.35",
    "kd_mask_weight": "0.6",
    "kd_proto_weight": "0.6",
    "kd_semseg_weight": "0.4",
    "kd_kpt_weight": "0.6",
}

POSE_DEFAULTS = {
    "student": "yolo26n-pose.pt",
    "imgsz": "416",
    "epochs": "350",
    "batch": "16",
    "workers": "8",
    "download_workers": "32",
    "patience": "80",
    "alpha": "0.65",
    "temperature": "3.0",
    "kd_gain": "1.1",
    "kd_cls_weight": "1.0",
    "kd_box_weight": "0.40",
    "kd_mask_weight": "0.6",
    "kd_proto_weight": "0.6",
    "kd_semseg_weight": "0.4",
    "kd_kpt_weight": "0.90",
}


def _resolve_gui_device(value: str, python_exec: str) -> str:
    raw = value.strip()
    normalized = raw.lower()
    if normalized not in {"", "auto"}:
        return raw

    probe = subprocess.run(
        [python_exec, "-c", "import torch; print(int(torch.cuda.is_available()))"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        details = (probe.stderr or probe.stdout or "").strip()
        if details:
            raise RuntimeError(f"PyTorch is unavailable in target interpreter: {details}")
        raise RuntimeError("PyTorch is unavailable in target interpreter. CUDA-only mode requires torch with CUDA.")

    if probe.stdout.strip() == "1":
        return "0"
    raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False.")


def _resolve_python_executable(script_dir: Path) -> str:
    venv_python = script_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _check_python_runtime_requirements(python_exec: str, *, need_onnx: bool, need_openvino: bool) -> tuple[bool, str]:
    check_script = (
        "import importlib.util as u\n"
        "required=['torch','ultralytics']\n"
        "if int(__import__('os').environ.get('NEED_ONNX','0')): required.append('onnx')\n"
        "if int(__import__('os').environ.get('NEED_OPENVINO','0')): required.append('openvino')\n"
        "missing=[m for m in required if u.find_spec(m) is None]\n"
        "cuda_ok=False\n"
        "torch_error=''\n"
        "try:\n"
        "  import torch\n"
        "  cuda_ok=bool(torch.cuda.is_available())\n"
        "except Exception as e:\n"
        "  torch_error=str(e)\n"
        "import json\n"
        "print(json.dumps({'missing':missing,'cuda_ok':cuda_ok,'torch_error':torch_error}, ensure_ascii=False))\n"
    )

    env = os.environ.copy()
    env["NEED_ONNX"] = "1" if need_onnx else "0"
    env["NEED_OPENVINO"] = "1" if need_openvino else "0"
    probe = subprocess.run(
        [python_exec, "-c", check_script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    if probe.returncode != 0:
        details = (probe.stderr or probe.stdout or "").strip()
        msg = "Failed to check Python packages in the target environment."
        if details:
            msg = f"{msg}\n\nDetails:\n{details}"
        return False, msg

    try:
        payload = json.loads(probe.stdout.strip())
    except Exception:
        return False, "Failed to parse the package check result. Run: uv sync"

    missing = payload.get("missing") if isinstance(payload, dict) else None
    cuda_ok = payload.get("cuda_ok") if isinstance(payload, dict) else False
    torch_error = payload.get("torch_error") if isinstance(payload, dict) else ""

    if not isinstance(missing, list):
        return False, "Unexpected package check output format. Run: uv sync"

    problems: list[str] = []
    if missing:
        problems.append("Missing Python packages: " + ", ".join(str(x) for x in missing))
    if not cuda_ok:
        problems.append("CUDA is not available in torch (torch.cuda.is_available() == False)")
    if torch_error:
        problems.append(f"torch error: {torch_error}")

    if problems:
        install_hint = (
            "Install dependencies and check the GPU:\n"
            "1) uv sync\n"
            '2) uv run python -c "import torch; print(torch.cuda.is_available())"'
        )
        return False, "\n\n".join(["\n".join(problems), install_hint])

    return True, ""


class DistillGui(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO26 Distillation")
        self.resize(980, 760)
        self.setMinimumSize(900, 660)

        self._process: QProcess | None = None
        self._last_output_at = 0.0
        self._settings_path = Path.home() / ".distill_yolo26_seg_gui.json"

        self._build_ui()
        self._load_settings()
        self._on_task_changed(apply_defaults=False)
        self._sync_openvino_precision_state()

        self._heartbeat_timer = QTimer(self)
        self._heartbeat_timer.setInterval(30_000)
        self._heartbeat_timer.timeout.connect(self._on_heartbeat)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QGridLayout()
        layout.addLayout(form)

        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "segment", "pose"])
        self.task_combo.setCurrentText("segment")
        self.task_combo.currentTextChanged.connect(lambda _value: self._on_task_changed(apply_defaults=True))
        form.addWidget(QLabel("Task"), 0, 0)
        form.addWidget(self.task_combo, 0, 1)

        self.teacher_edit = QLineEdit()
        self.student_edit = QLineEdit("yolo26n-seg.pt")
        self.data_edit = QLineEdit()
        self.outdir_edit = QLineEdit(os.getcwd())
        self._add_path_row(form, 1, "Teacher (.pt)", self.teacher_edit, self._pick_teacher)
        self._add_path_row(form, 2, "Student init (.pt or alias)", self.student_edit, self._pick_student)
        self._add_path_row(form, 3, "Dataset (YAML/NDJSON)", self.data_edit, self._pick_data)
        self._add_path_row(form, 4, "Output folder", self.outdir_edit, self._pick_outdir)
        form.setColumnStretch(1, 1)

        training_box = QGroupBox("Training params")
        training_layout = QGridLayout(training_box)
        layout.addWidget(training_box)

        self.imgsz_edit = QLineEdit("320")
        self.epochs_edit = QLineEdit("120")
        self.batch_edit = QLineEdit("16")
        self.workers_edit = QLineEdit("8")
        self.download_workers_edit = QLineEdit("32")
        self.patience_edit = QLineEdit("50")
        self.device_edit = QLineEdit("auto")
        self.alpha_edit = QLineEdit("0.8")
        self.temperature_edit = QLineEdit("2.0")
        self._add_edit_grid(
            training_layout,
            [
                ("imgsz", self.imgsz_edit),
                ("epochs", self.epochs_edit),
                ("batch", self.batch_edit),
                ("workers", self.workers_edit),
                ("download_workers", self.download_workers_edit),
                ("patience", self.patience_edit),
                ("device", self.device_edit),
                ("alpha", self.alpha_edit),
                ("temperature", self.temperature_edit),
            ],
            columns=5,
        )

        kd_box = QGroupBox("KD params")
        kd_layout = QGridLayout(kd_box)
        layout.addWidget(kd_box)

        self.kd_gain_edit = QLineEdit("1.0")
        self.kd_cls_weight_edit = QLineEdit("1.0")
        self.kd_box_weight_edit = QLineEdit("0.35")
        self.kd_mask_weight_edit = QLineEdit("0.6")
        self.kd_proto_weight_edit = QLineEdit("0.6")
        self.kd_semseg_weight_edit = QLineEdit("0.4")
        self.kd_kpt_weight_edit = QLineEdit("0.6")
        self._add_edit_grid(
            kd_layout,
            [
                ("kd_gain", self.kd_gain_edit),
                ("kd_cls_w", self.kd_cls_weight_edit),
                ("kd_box_w", self.kd_box_weight_edit),
                ("kd_mask_w", self.kd_mask_weight_edit),
                ("kd_proto_w", self.kd_proto_weight_edit),
                ("kd_semseg_w", self.kd_semseg_weight_edit),
                ("kd_kpt_w", self.kd_kpt_weight_edit),
            ],
            columns=4,
        )

        export_box = QGroupBox("Export")
        export_layout = QGridLayout(export_box)
        layout.addWidget(export_box)

        self.export_onnx_cb = QCheckBox("ONNX (FP32)")
        self.export_onnx_cb.setChecked(True)
        self.export_openvino_cb = QCheckBox("OpenVINO (FP16)")
        self.export_openvino_cb.setChecked(True)
        self.export_openvino_int8_cb = QCheckBox("OpenVINO (INT8)")
        self.export_openvino_int8_cb.toggled.connect(self._sync_openvino_precision_state)
        self.dynamic_cb = QCheckBox("Dynamic shapes")
        self.raw_detect_cb = QCheckBox("Legacy raw detect output")
        self.raw_detect_cb.setToolTip(
            "Off (recommended): native YOLO26 NMS-free output [batch, 300, 6]. "
            "On: legacy raw output [batch, 4+classes, anchors] for classic YOLO decoders."
        )
        self.strict_validation_cb = QCheckBox("Strict export validation")
        self.strict_parity_cb = QCheckBox("Strict parity")

        export_layout.addWidget(self.export_onnx_cb, 0, 0)
        export_layout.addWidget(self.export_openvino_cb, 0, 1)
        export_layout.addWidget(self.export_openvino_int8_cb, 0, 2)
        export_layout.addWidget(self.dynamic_cb, 0, 3)
        export_layout.addWidget(self.strict_validation_cb, 1, 0)
        export_layout.addWidget(self.strict_parity_cb, 1, 1)
        export_layout.addWidget(self.raw_detect_cb, 1, 2, 1, 2)

        controls = QHBoxLayout()
        layout.addLayout(controls)
        self.start_btn = QPushButton("Start distillation")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        self.pose_defaults_btn = QPushButton("Apply pose accuracy defaults")
        self.pose_defaults_btn.clicked.connect(lambda: self._apply_task_defaults("pose"))
        self.detect_defaults_btn = QPushButton("Apply detect defaults")
        self.detect_defaults_btn.clicked.connect(lambda: self._apply_task_defaults("detect"))
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.detect_defaults_btn)
        controls.addWidget(self.pose_defaults_btn)
        controls.addStretch(1)

        self.cmd_label = QLabel()
        self.cmd_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.cmd_label.setWordWrap(True)
        layout.addWidget(self.cmd_label)

        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)
        layout.addWidget(log_box, 1)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(30_000)
        log_layout.addWidget(self.log_text)

    def _add_path_row(self, layout: QGridLayout, row: int, label: str, edit: QLineEdit, slot) -> None:
        button = QPushButton("Browse")
        button.clicked.connect(slot)
        layout.addWidget(QLabel(label), row, 0)
        layout.addWidget(edit, row, 1)
        layout.addWidget(button, row, 2)

    def _add_edit_grid(self, layout: QGridLayout, entries: list[tuple[str, QLineEdit]], *, columns: int) -> None:
        for idx, (label, edit) in enumerate(entries):
            row = idx // columns
            col = (idx % columns) * 2
            edit.setMaximumWidth(110)
            layout.addWidget(QLabel(label), row, col)
            layout.addWidget(edit, row, col + 1)

    def _on_task_changed(self, *, apply_defaults: bool) -> None:
        task = self.task_combo.currentText().strip().lower() or "segment"
        self.setWindowTitle(f"YOLO26 Distillation ({task})")
        self.raw_detect_cb.setEnabled(task == "detect")
        if task != "detect":
            self.raw_detect_cb.setChecked(False)

        student = self.student_edit.text().strip().lower()
        known_defaults = {"", "yolo26n.pt", "yolo26n-seg.pt", "yolo26n-pose.pt"}
        if task == "detect" and student in known_defaults:
            self.student_edit.setText("yolo26n.pt")
        elif task == "pose" and student in known_defaults:
            self.student_edit.setText("yolo26n-pose.pt")
        elif task == "segment" and student in known_defaults:
            self.student_edit.setText("yolo26n-seg.pt")

        if apply_defaults:
            self._apply_task_defaults(task)

    def _apply_task_defaults(self, task: str) -> None:
        defaults = {"detect": DETECT_DEFAULTS, "segment": SEGMENT_DEFAULTS, "pose": POSE_DEFAULTS}[task]
        if self.task_combo.currentText() != task:
            self.task_combo.blockSignals(True)
            self.task_combo.setCurrentText(task)
            self.task_combo.blockSignals(False)
            self.setWindowTitle(f"YOLO26 Distillation ({task})")
        for key, value in defaults.items():
            self._field_map()[key].setText(value)

        if task == "pose":
            self.export_onnx_cb.setChecked(True)
            self.export_openvino_cb.setChecked(True)
            self.export_openvino_int8_cb.setChecked(False)
            self.dynamic_cb.setChecked(False)
            self.strict_validation_cb.setChecked(False)
            self.strict_parity_cb.setChecked(False)
            self._append_log(
                "[GUI] Pose defaults applied: imgsz=416, epochs=350, OpenVINO FP16; "
                "INT8 is off to preserve accuracy until you validate mAP.\n"
            )
        elif task == "detect":
            self.raw_detect_cb.setChecked(False)
            self._append_log(
                "[GUI] Detect defaults applied: student=yolo26n.pt, KD uses cls+box terms, "
                "export output is native YOLO26 NMS-free [batch, 300, 6].\n"
            )

    def _field_map(self) -> dict[str, QLineEdit]:
        return {
            "student": self.student_edit,
            "imgsz": self.imgsz_edit,
            "epochs": self.epochs_edit,
            "batch": self.batch_edit,
            "workers": self.workers_edit,
            "download_workers": self.download_workers_edit,
            "patience": self.patience_edit,
            "alpha": self.alpha_edit,
            "temperature": self.temperature_edit,
            "kd_gain": self.kd_gain_edit,
            "kd_cls_weight": self.kd_cls_weight_edit,
            "kd_box_weight": self.kd_box_weight_edit,
            "kd_mask_weight": self.kd_mask_weight_edit,
            "kd_proto_weight": self.kd_proto_weight_edit,
            "kd_semseg_weight": self.kd_semseg_weight_edit,
            "kd_kpt_weight": self.kd_kpt_weight_edit,
        }

    def _sync_openvino_precision_state(self) -> None:
        int8_enabled = self.export_openvino_int8_cb.isChecked()
        if int8_enabled:
            self.export_openvino_cb.setChecked(False)
        self.export_openvino_cb.setEnabled(not int8_enabled)

    def _pick_teacher(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select teacher model", "", "PyTorch model (*.pt);;All (*)")
        if path:
            self.teacher_edit.setText(path)

    def _pick_student(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select student init model", "", "PyTorch model (*.pt);;All (*)")
        if path:
            self.student_edit.setText(path)

    def _pick_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select dataset YAML or NDJSON",
            "",
            "Dataset (*.yaml *.yml *.ndjson);;All (*)",
        )
        if path:
            self.data_edit.setText(path)

    def _pick_outdir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.outdir_edit.setText(path)

    def _append_log(self, text: str) -> None:
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text.replace("\r", "\n"))
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()

    def _validate_inputs(self) -> bool:
        task = self.task_combo.currentText().strip().lower() or "segment"
        if not self.teacher_edit.text().strip():
            QMessageBox.critical(self, "Missing teacher", "Select teacher model path or alias.")
            return False
        if not self.data_edit.text().strip():
            QMessageBox.critical(self, "Missing dataset", "Select dataset YAML or NDJSON.")
            return False
        if task == "segment" and self.data_edit.text().strip().lower().endswith(".ndjson"):
            QMessageBox.critical(
                self,
                "Invalid dataset for segment",
                "NDJSON is supported only for detect and pose tasks. Switch task or choose dataset YAML.",
            )
            return False
        if not (
            self.export_onnx_cb.isChecked()
            or self.export_openvino_cb.isChecked()
            or self.export_openvino_int8_cb.isChecked()
        ):
            answer = QMessageBox.question(self, "No export selected", "Continue without export artifacts?")
            if answer != QMessageBox.Yes:
                return False
        return True

    def _prepare_data_source_for_run(self) -> str:
        task = self.task_combo.currentText().strip().lower() or "segment"
        data_source = self.data_edit.text().strip()
        data_path = Path(data_source).expanduser()

        if task not in {"detect", "pose"} or data_path.suffix.lower() != ".ndjson":
            return data_source

        if not data_path.exists():
            raise RuntimeError(f"Pose NDJSON not found: {data_path}")

        outdir = Path(self.outdir_edit.text().strip() or os.getcwd()).expanduser().resolve()
        dataset_dir = outdir / "datasets" / f"{data_path.stem}_{task}_yolo"
        download_workers = self._ndjson_download_workers()
        self._append_log(
            f"[DATA] Converting {task} NDJSON: {data_path} -> {dataset_dir} "
            f"(download_workers={download_workers})\n"
        )

        if task == "detect":
            from distill.detect_ndjson_to_yolo import convert
        else:
            from distill.pose_ndjson_to_yolo import convert

        def on_download_progress(done: int, total: int) -> None:
            self._append_log(f"[DATA] downloaded {done}/{total}\n")
            QApplication.processEvents()

        summary = convert(
            data_path.resolve(),
            dataset_dir,
            download_images=True,
            timeout=30.0,
            overwrite_images=False,
            workers=download_workers,
            progress_every=100,
            progress_callback=on_download_progress,
        )
        yaml_path = summary.get("yaml")
        if not isinstance(yaml_path, str) or not yaml_path:
            raise RuntimeError("NDJSON conversion failed: data.yaml was not produced")

        self._append_log(
            "[DATA] Prepared dataset: "
            f"images_total={summary.get('images_total')} downloaded={summary.get('images_downloaded')} "
            f"failed={summary.get('images_failed')} yaml={yaml_path}\n"
        )

        failed = int(summary.get("images_failed") or 0)
        if failed > 0:
            answer = QMessageBox.question(
                self,
                "Image download issues",
                f"Failed to download {failed} images from NDJSON. Continue anyway?",
            )
            if answer != QMessageBox.Yes:
                raise RuntimeError("Aborted: incomplete dataset conversion")

        return yaml_path

    def _ndjson_download_workers(self) -> int:
        raw = self.download_workers_edit.text().strip()
        try:
            value = int(raw)
        except Exception as exc:
            raise RuntimeError(f"download_workers must be an integer, got: {raw!r}") from exc
        if value <= 0:
            raise RuntimeError("download_workers must be greater than 0")
        return value

    def _build_command(self, *, data_source: str) -> list[str]:
        script_path = Path(__file__).resolve().with_name("distill_yolo26_seg.py")
        py_exec = _resolve_python_executable(script_path.parent)
        task = self.task_combo.currentText().strip().lower() or "segment"
        effective_device = _resolve_gui_device(self.device_edit.text().strip(), py_exec)
        cmd = [
            py_exec,
            "-u",
            str(script_path),
            "--task",
            task,
            "--teacher",
            self.teacher_edit.text().strip(),
            "--student-init",
            self.student_edit.text().strip(),
            "--data",
            data_source,
            "--imgsz",
            self.imgsz_edit.text().strip(),
            "--epochs",
            self.epochs_edit.text().strip(),
            "--batch",
            self.batch_edit.text().strip(),
            "--workers",
            self.workers_edit.text().strip(),
            "--device",
            effective_device,
            "--patience",
            self.patience_edit.text().strip(),
            "--alpha",
            self.alpha_edit.text().strip(),
            "--temperature",
            self.temperature_edit.text().strip(),
            "--kd-gain",
            self.kd_gain_edit.text().strip(),
            "--kd-cls-weight",
            self.kd_cls_weight_edit.text().strip(),
            "--kd-box-weight",
            self.kd_box_weight_edit.text().strip(),
            "--kd-mask-weight",
            self.kd_mask_weight_edit.text().strip(),
            "--kd-proto-weight",
            self.kd_proto_weight_edit.text().strip(),
            "--kd-semseg-weight",
            self.kd_semseg_weight_edit.text().strip(),
            "--kd-kpt-weight",
            self.kd_kpt_weight_edit.text().strip(),
            "--outdir",
            self.outdir_edit.text().strip(),
        ]

        export_formats: list[str] = []
        if self.export_onnx_cb.isChecked():
            export_formats.append("onnx")
        if self.export_openvino_cb.isChecked() or self.export_openvino_int8_cb.isChecked():
            export_formats.append("openvino")
        cmd.append("--export")
        cmd.extend(export_formats)
        if self.export_openvino_int8_cb.isChecked():
            cmd.append("--openvino-int8")

        if self.dynamic_cb.isChecked():
            cmd.append("--export-dynamic")
        if self.raw_detect_cb.isChecked():
            cmd.append("--export-raw-detect")
        if self.strict_validation_cb.isChecked():
            cmd.append("--strict-export-validation")
        if self.strict_parity_cb.isChecked():
            cmd.append("--strict-parity")

        return cmd

    def _start(self) -> None:
        if self._process is not None:
            return
        if not self._validate_inputs():
            return

        script_path = Path(__file__).resolve().with_name("distill_yolo26_seg.py")
        py_exec = _resolve_python_executable(script_path.parent)
        ok, preflight_message = _check_python_runtime_requirements(
            py_exec,
            need_onnx=self.export_onnx_cb.isChecked(),
            need_openvino=self.export_openvino_cb.isChecked() or self.export_openvino_int8_cb.isChecked(),
        )
        if not ok:
            QMessageBox.critical(self, "Missing dependencies", preflight_message)
            self._append_log(f"[PRECHECK ERROR] {preflight_message}\n")
            return

        try:
            effective_data = self._prepare_data_source_for_run()
            cmd = self._build_command(data_source=effective_data)
        except Exception as exc:
            QMessageBox.critical(self, "Start error", str(exc))
            self._append_log(f"[START ERROR] {exc}\n")
            return

        cmd_preview = " ".join(shlex.quote(part) for part in cmd)
        self.cmd_label.setText(f"Command: {cmd_preview}")
        if (self.device_edit.text().strip() or "auto").lower() == "auto":
            self._append_log("[DEVICE] auto -> 0 (CUDA)\n")
        self._append_log(f"\n[START] {cmd_preview}\n")

        run_cwd = Path(self.outdir_edit.text().strip() or os.getcwd()).resolve()
        run_cwd.mkdir(parents=True, exist_ok=True)

        process = QProcess(self)
        process.setProgram(cmd[0])
        process.setArguments(cmd[1:])
        process.setWorkingDirectory(str(run_cwd))
        process.setProcessChannelMode(QProcess.MergedChannels)

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("ULTRALYTICS_VERBOSE", "1")
        env.insert("YOLO_VERBOSE", "1")
        process.setProcessEnvironment(env)

        process.readyReadStandardOutput.connect(self._read_process_output)
        process.readyReadStandardError.connect(self._read_process_output)
        process.errorOccurred.connect(self._on_process_error)
        process.finished.connect(self._on_process_finished)

        self._process = process
        self._last_output_at = time.monotonic()
        process.start()
        if not process.waitForStarted(5000):
            self._append_log(f"[ERROR] Failed to start: {process.errorString()}\n")
            self._process = None
            return

        self._append_log(f"[PROC] PID: {process.processId()}\n")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._heartbeat_timer.start()

    def _read_process_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput())
        if not data:
            data = bytes(self._process.readAllStandardError())
        if not data:
            return
        self._last_output_at = time.monotonic()
        self._append_log(data.decode("utf-8", errors="replace"))

    def _on_process_error(self, _error: QProcess.ProcessError) -> None:
        if self._process is not None:
            self._append_log(f"\n[PROCESS ERROR] {self._process.errorString()}\n")

    def _on_process_finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        self._read_process_output()
        self._append_log(f"\n[END] Exit code: {code}\n")
        self._process = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._heartbeat_timer.stop()

    def _on_heartbeat(self) -> None:
        if self._process is None:
            return
        idle_seconds = int(time.monotonic() - self._last_output_at)
        if idle_seconds >= 60:
            self._append_log(f"[GUI] Process is still running; no output for {idle_seconds}s.\n")

    def _stop(self) -> None:
        if self._process is None:
            return
        self._append_log("\n[STOP] Terminating process...\n")
        self._process.terminate()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._process is not None:
            answer = QMessageBox.question(self, "Process running", "Distillation is running. Stop and exit?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self._process.terminate()
        self._save_settings()
        event.accept()

    def _load_settings(self) -> None:
        if not self._settings_path.exists():
            return
        try:
            payload = json.loads(self._settings_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return

        text_fields = {
            "teacher": self.teacher_edit,
            "student": self.student_edit,
            "data": self.data_edit,
            "outdir": self.outdir_edit,
            "imgsz": self.imgsz_edit,
            "epochs": self.epochs_edit,
            "batch": self.batch_edit,
            "workers": self.workers_edit,
            "download_workers": self.download_workers_edit,
            "patience": self.patience_edit,
            "device": self.device_edit,
            "alpha": self.alpha_edit,
            "temperature": self.temperature_edit,
            "kd_gain": self.kd_gain_edit,
            "kd_cls_weight": self.kd_cls_weight_edit,
            "kd_box_weight": self.kd_box_weight_edit,
            "kd_mask_weight": self.kd_mask_weight_edit,
            "kd_proto_weight": self.kd_proto_weight_edit,
            "kd_semseg_weight": self.kd_semseg_weight_edit,
            "kd_kpt_weight": self.kd_kpt_weight_edit,
        }
        for key, edit in text_fields.items():
            value = payload.get(key)
            if isinstance(value, str):
                edit.setText(value)

        task = payload.get("task")
        if isinstance(task, str) and task in {"detect", "segment", "pose"}:
            self.task_combo.blockSignals(True)
            self.task_combo.setCurrentText(task)
            self.task_combo.blockSignals(False)

        bool_fields = {
            "export_onnx": self.export_onnx_cb,
            "export_openvino": self.export_openvino_cb,
            "export_openvino_int8": self.export_openvino_int8_cb,
            "dynamic": self.dynamic_cb,
            "export_raw_detect": self.raw_detect_cb,
            "strict_validation": self.strict_validation_cb,
            "strict_parity": self.strict_parity_cb,
        }
        for key, checkbox in bool_fields.items():
            value = payload.get(key)
            if isinstance(value, bool):
                checkbox.setChecked(value)

    def _save_settings(self) -> None:
        payload = {
            "task": self.task_combo.currentText().strip(),
            "teacher": self.teacher_edit.text().strip(),
            "student": self.student_edit.text().strip(),
            "data": self.data_edit.text().strip(),
            "outdir": self.outdir_edit.text().strip(),
            "imgsz": self.imgsz_edit.text().strip(),
            "epochs": self.epochs_edit.text().strip(),
            "batch": self.batch_edit.text().strip(),
            "workers": self.workers_edit.text().strip(),
            "download_workers": self.download_workers_edit.text().strip(),
            "patience": self.patience_edit.text().strip(),
            "device": self.device_edit.text().strip(),
            "alpha": self.alpha_edit.text().strip(),
            "temperature": self.temperature_edit.text().strip(),
            "kd_gain": self.kd_gain_edit.text().strip(),
            "kd_cls_weight": self.kd_cls_weight_edit.text().strip(),
            "kd_box_weight": self.kd_box_weight_edit.text().strip(),
            "kd_mask_weight": self.kd_mask_weight_edit.text().strip(),
            "kd_proto_weight": self.kd_proto_weight_edit.text().strip(),
            "kd_semseg_weight": self.kd_semseg_weight_edit.text().strip(),
            "kd_kpt_weight": self.kd_kpt_weight_edit.text().strip(),
            "export_onnx": self.export_onnx_cb.isChecked(),
            "export_openvino": self.export_openvino_cb.isChecked(),
            "export_openvino_int8": self.export_openvino_int8_cb.isChecked(),
            "dynamic": self.dynamic_cb.isChecked(),
            "export_raw_detect": self.raw_detect_cb.isChecked(),
            "strict_validation": self.strict_validation_cb.isChecked(),
            "strict_parity": self.strict_parity_cb.isChecked(),
        }
        try:
            self._settings_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self._append_log(f"[WARN] Failed to save GUI settings: {exc}\n")


def main() -> None:
    app = QApplication(sys.argv)
    gui = DistillGui()
    gui.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
