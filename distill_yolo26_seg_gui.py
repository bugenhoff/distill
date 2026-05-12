#!/usr/bin/env python3
"""Minimal GUI launcher for distill_yolo26_seg.py."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


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


class DistillGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("YOLO26 Seg Distillation")
        self.geometry("920x700")
        self.minsize(860, 620)

        self._proc: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._settings_path = Path.home() / ".distill_yolo26_seg_gui.json"

        self._build_state()
        self._load_settings()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_state(self) -> None:
        self.teacher_var = tk.StringVar()
        self.student_var = tk.StringVar(value="yolo26n-seg.pt")
        self.data_var = tk.StringVar()
        self.device_var = tk.StringVar(value="auto")
        self.outdir_var = tk.StringVar(value=os.getcwd())

        self.imgsz_var = tk.StringVar(value="320")
        self.epochs_var = tk.StringVar(value="120")
        self.batch_var = tk.StringVar(value="16")
        self.workers_var = tk.StringVar(value="8")
        self.patience_var = tk.StringVar(value="50")

        self.alpha_var = tk.StringVar(value="0.8")
        self.temperature_var = tk.StringVar(value="2.0")

        self.export_onnx_var = tk.BooleanVar(value=True)
        self.export_openvino_var = tk.BooleanVar(value=True)
        self.dynamic_var = tk.BooleanVar(value=False)
        self.strict_validation_var = tk.BooleanVar(value=False)
        self.strict_parity_var = tk.BooleanVar(value=False)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        form = ttk.Frame(root)
        form.pack(fill="x", expand=False)

        row = 0
        self._add_path_row(form, row, "Teacher (.pt)", self.teacher_var, self._pick_teacher)
        row += 1
        self._add_path_row(form, row, "Student init (.pt or alias)", self.student_var, self._pick_student)
        row += 1
        self._add_path_row(form, row, "Dataset YAML", self.data_var, self._pick_data)
        row += 1
        self._add_path_row(form, row, "Output folder", self.outdir_var, self._pick_outdir)
        row += 1

        params = ttk.LabelFrame(root, text="Training params", padding=10)
        params.pack(fill="x", pady=(10, 0))
        self._add_simple_grid(params)

        exports = ttk.LabelFrame(root, text="Export", padding=10)
        exports.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(exports, text="ONNX (FP32)", variable=self.export_onnx_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(exports, text="OpenVINO (FP16)", variable=self.export_openvino_var).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(exports, text="Dynamic shapes", variable=self.dynamic_var).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(exports, text="Strict export validation", variable=self.strict_validation_var).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Checkbutton(exports, text="Strict parity", variable=self.strict_parity_var).grid(row=1, column=1, sticky="w")

        controls = ttk.Frame(root)
        controls.pack(fill="x", pady=(12, 6))
        self.start_btn = ttk.Button(controls, text="Start distillation", command=self._start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(controls, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))

        self.cmd_label = ttk.Label(root, text="")
        self.cmd_label.pack(fill="x", pady=(4, 4))

        log_frame = ttk.LabelFrame(root, text="Log", padding=8)
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_frame, wrap="word", height=20)
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

    def _add_simple_grid(self, parent: ttk.LabelFrame) -> None:
        entries = [
            ("imgsz", self.imgsz_var),
            ("epochs", self.epochs_var),
            ("batch", self.batch_var),
            ("workers", self.workers_var),
            ("patience", self.patience_var),
            ("device", self.device_var),
            ("alpha", self.alpha_var),
            ("temperature", self.temperature_var),
        ]

        for idx, (label, var) in enumerate(entries):
            r = idx // 5
            c = (idx % 5) * 2
            ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w", padx=(0, 4), pady=2)
            ttk.Entry(parent, textvariable=var, width=10).grid(row=r, column=c + 1, sticky="w", padx=(0, 16), pady=2)

    def _add_path_row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar, browse_command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8, pady=3)
        ttk.Button(parent, text="Browse", command=browse_command).grid(row=row, column=2, sticky="w", pady=3)
        parent.grid_columnconfigure(1, weight=1)

    def _pick_teacher(self) -> None:
        path = filedialog.askopenfilename(title="Select teacher model", filetypes=[("PyTorch model", "*.pt"), ("All", "*")])
        if path:
            self.teacher_var.set(path)

    def _pick_student(self) -> None:
        path = filedialog.askopenfilename(title="Select student init model", filetypes=[("PyTorch model", "*.pt"), ("All", "*")])
        if path:
            self.student_var.set(path)

    def _pick_data(self) -> None:
        path = filedialog.askopenfilename(title="Select dataset yaml", filetypes=[("YAML", "*.yaml *.yml"), ("All", "*")])
        if path:
            self.data_var.set(path)

    def _pick_outdir(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.outdir_var.set(path)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _validate_inputs(self) -> bool:
        if not self.teacher_var.get().strip():
            messagebox.showerror("Missing teacher", "Select teacher model path or alias.")
            return False
        if not self.data_var.get().strip():
            messagebox.showerror("Missing dataset", "Select dataset YAML.")
            return False
        if not (self.export_onnx_var.get() or self.export_openvino_var.get()):
            if not messagebox.askyesno("No export selected", "Continue without export artifacts?"):
                return False
        return True

    def _build_command(self) -> list[str]:
        script_path = Path(__file__).resolve().with_name("distill_yolo26_seg.py")
        py_exec = _resolve_python_executable(script_path.parent)
        requested_device = self.device_var.get().strip()
        effective_device = _resolve_gui_device(requested_device, py_exec)
        cmd = [
            py_exec,
            "-u",
            str(script_path),
            "--teacher",
            self.teacher_var.get().strip(),
            "--student-init",
            self.student_var.get().strip(),
            "--data",
            self.data_var.get().strip(),
            "--imgsz",
            self.imgsz_var.get().strip(),
            "--epochs",
            self.epochs_var.get().strip(),
            "--batch",
            self.batch_var.get().strip(),
            "--workers",
            self.workers_var.get().strip(),
            "--device",
            effective_device,
            "--patience",
            self.patience_var.get().strip(),
            "--alpha",
            self.alpha_var.get().strip(),
            "--temperature",
            self.temperature_var.get().strip(),
            "--outdir",
            self.outdir_var.get().strip(),
        ]

        export_formats: list[str] = []
        if self.export_onnx_var.get():
            export_formats.append("onnx")
        if self.export_openvino_var.get():
            export_formats.append("openvino")
        cmd.append("--export")
        cmd.extend(export_formats)

        if self.dynamic_var.get():
            cmd.append("--export-dynamic")
        if self.strict_validation_var.get():
            cmd.append("--strict-export-validation")
        if self.strict_parity_var.get():
            cmd.append("--strict-parity")

        return cmd

    def _start(self) -> None:
        if self._proc is not None:
            return
        if not self._validate_inputs():
            return

        try:
            cmd = self._build_command()
        except Exception as exc:
            messagebox.showerror("Device error", str(exc))
            self._append_log(f"[DEVICE ERROR] {exc}\n")
            return
        cmd_preview = " ".join(shlex.quote(part) for part in cmd)
        self.cmd_label.configure(text=f"Command: {cmd_preview}")
        requested_device = self.device_var.get().strip()
        if (requested_device or "auto").lower() == "auto":
            self._append_log("[DEVICE] auto -> 0 (CUDA)\n")
        self._append_log(f"\n[START] {cmd_preview}\n")

        try:
            run_cwd = Path(self.outdir_var.get().strip() or os.getcwd()).resolve()
            run_cwd.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("PYTHONIOENCODING", "utf-8")
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(run_cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            self._append_log(f"[PROC] PID: {self._proc.pid}\n")
        except Exception as exc:
            self._append_log(f"[ERROR] Failed to start: {exc}\n")
            self._proc = None
            return

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        self._reader_thread = threading.Thread(target=self._stream_output, daemon=True)
        self._reader_thread.start()

    def _stream_output(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None

        buffer: list[str] = []
        while True:
            chunk = self._proc.stdout.read(1)
            if chunk == "":
                if buffer:
                    self.after(0, self._append_log, "".join(buffer))
                break

            if chunk in {"\n", "\r"}:
                line = "".join(buffer)
                buffer.clear()
                line += "\n"
                self.after(0, self._append_log, line)
                continue

            buffer.append(chunk)

        code = self._proc.wait()
        self.after(0, self._on_process_exit, code)

    def _on_process_exit(self, code: int) -> None:
        self._append_log(f"\n[END] Exit code: {code}\n")
        self._proc = None
        self._reader_thread = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def _stop(self) -> None:
        if self._proc is None:
            return
        self._append_log("\n[STOP] Terminating process...\n")
        self._proc.terminate()

    def _on_close(self) -> None:
        if self._proc is not None:
            if not messagebox.askyesno("Process running", "Distillation is running. Stop and exit?"):
                return
            self._proc.terminate()
        self._save_settings()
        self.destroy()

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
            "teacher": self.teacher_var,
            "student": self.student_var,
            "data": self.data_var,
            "outdir": self.outdir_var,
            "imgsz": self.imgsz_var,
            "epochs": self.epochs_var,
            "batch": self.batch_var,
            "workers": self.workers_var,
            "patience": self.patience_var,
            "device": self.device_var,
            "alpha": self.alpha_var,
            "temperature": self.temperature_var,
        }
        bool_fields = {
            "export_onnx": self.export_onnx_var,
            "export_openvino": self.export_openvino_var,
            "dynamic": self.dynamic_var,
            "strict_validation": self.strict_validation_var,
            "strict_parity": self.strict_parity_var,
        }
        for key, var in text_fields.items():
            value = payload.get(key)
            if isinstance(value, str):
                var.set(value)
        for key, var in bool_fields.items():
            value = payload.get(key)
            if isinstance(value, bool):
                var.set(value)

    def _save_settings(self) -> None:
        payload = {
            "teacher": self.teacher_var.get().strip(),
            "student": self.student_var.get().strip(),
            "data": self.data_var.get().strip(),
            "outdir": self.outdir_var.get().strip(),
            "imgsz": self.imgsz_var.get().strip(),
            "epochs": self.epochs_var.get().strip(),
            "batch": self.batch_var.get().strip(),
            "workers": self.workers_var.get().strip(),
            "patience": self.patience_var.get().strip(),
            "device": self.device_var.get().strip(),
            "alpha": self.alpha_var.get().strip(),
            "temperature": self.temperature_var.get().strip(),
            "export_onnx": bool(self.export_onnx_var.get()),
            "export_openvino": bool(self.export_openvino_var.get()),
            "dynamic": bool(self.dynamic_var.get()),
            "strict_validation": bool(self.strict_validation_var.get()),
            "strict_parity": bool(self.strict_parity_var.get()),
        }
        try:
            self._settings_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self._append_log(f"[WARN] Failed to save GUI settings: {exc}\n")


def main() -> None:
    app = DistillGui()
    app.mainloop()


if __name__ == "__main__":
    main()
