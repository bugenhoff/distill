#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(slots=True)
class DatasetMeta:
    names: list[str]
    kpt_shape: tuple[int, int]


@dataclass(slots=True)
class DownloadJob:
    url: str
    target: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Ultralytics pose NDJSON export to local YOLO pose dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to pose NDJSON file")
    parser.add_argument("--outdir", required=True, help="Output directory for YOLO dataset")
    parser.add_argument("--timeout", type=float, default=30.0, help="Image download timeout in seconds")
    parser.add_argument("--overwrite-images", action="store_true", help="Re-download existing images")
    parser.add_argument("--no-download", action="store_true", help="Skip image download, write labels/yaml only")
    parser.add_argument("--workers", type=int, default=16, help="Parallel image download workers")
    parser.add_argument("--progress-every", type=int, default=100, help="Print download progress every N images")
    return parser.parse_args()


def _format_float(value: float) -> str:
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text or "0"


def _ordered_names(raw: Any) -> list[str]:
    if isinstance(raw, dict):
        pairs: list[tuple[int, str]] = []
        for key, value in raw.items():
            try:
                idx = int(key)
            except Exception:
                continue
            pairs.append((idx, str(value)))
        pairs.sort(key=lambda x: x[0])
        return [name for _, name in pairs]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def _parse_meta(first_record: dict[str, Any]) -> DatasetMeta:
    names = _ordered_names(first_record.get("class_names"))
    kpt_shape_raw = first_record.get("kpt_shape")
    if isinstance(kpt_shape_raw, list) and len(kpt_shape_raw) == 2:
        try:
            kpt_shape = (int(kpt_shape_raw[0]), int(kpt_shape_raw[1]))
        except Exception:
            kpt_shape = (0, 0)
    else:
        kpt_shape = (0, 0)
    if kpt_shape[0] <= 0 or kpt_shape[1] <= 0:
        raise ValueError("Invalid or missing kpt_shape in NDJSON dataset header")
    return DatasetMeta(names=names, kpt_shape=kpt_shape)


def _download_image(url: str, target: Path, timeout: float) -> tuple[bool, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "distill-pose-dataset-converter/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except urllib.error.URLError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return True, "ok"


def _write_yaml(outdir: Path, meta: DatasetMeta, seen_splits: set[str]) -> Path:
    yaml_path = outdir / "data.yaml"
    lines: list[str] = []
    lines.append(f"path: {outdir.resolve()}")
    for split in ("train", "val", "test"):
        if split in seen_splits:
            lines.append(f"{split}: images/{split}")
    lines.append("names:")
    for idx, name in enumerate(meta.names):
        lines.append(f"  {idx}: {name}")
    lines.append(f"kpt_shape: [{meta.kpt_shape[0]}, {meta.kpt_shape[1]}]")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def _to_label_lines(pose_rows: Any, expected_values: int) -> tuple[list[str], int]:
    if not isinstance(pose_rows, list):
        return [], 0

    lines: list[str] = []
    dropped = 0
    for row in pose_rows:
        if not isinstance(row, list) or not row:
            dropped += 1
            continue
        try:
            cls_idx = int(row[0])
            values = [float(x) for x in row[1:]]
        except Exception:
            dropped += 1
            continue

        if len(values) < expected_values:
            dropped += 1
            continue
        if len(values) > expected_values:
            values = values[:expected_values]

        parts = [str(cls_idx)] + [_format_float(v) for v in values]
        lines.append(" ".join(parts))
    return lines, dropped


def _download_jobs(
    jobs: list[DownloadJob],
    *,
    timeout: float,
    workers: int,
    progress_every: int,
    progress_callback: Callable[[int, int], None] | None,
) -> tuple[int, int]:
    if not jobs:
        return 0, 0

    images_downloaded = 0
    images_failed = 0
    completed = 0
    total = len(jobs)
    max_workers = max(1, int(workers))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_image, job.url, job.target, timeout): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                ok, _ = future.result()
            except Exception:
                ok = False

            if ok:
                images_downloaded += 1
            else:
                images_failed += 1

            completed += 1
            if progress_callback is not None and (
                progress_every > 0 and (completed % progress_every == 0 or completed == total)
            ):
                progress_callback(completed, total)

    return images_downloaded, images_failed


def convert(
    input_path: Path,
    outdir: Path,
    *,
    download_images: bool,
    timeout: float,
    overwrite_images: bool,
    workers: int = 16,
    progress_every: int = 100,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)

    seen_splits: set[str] = set()
    images_total = 0
    images_downloaded = 0
    images_failed = 0
    labels_written = 0
    rows_dropped = 0
    download_jobs: list[DownloadJob] = []

    with input_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
        if not first_line:
            raise ValueError("NDJSON is empty")
        first_record = json.loads(first_line)
        if not isinstance(first_record, dict) or first_record.get("type") != "dataset":
            raise ValueError("First NDJSON line must be dataset metadata")
        meta = _parse_meta(first_record)

        expected_values = 4 + (meta.kpt_shape[0] * meta.kpt_shape[1])

        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            if not isinstance(record, dict) or record.get("type") != "image":
                continue

            file_name = record.get("file")
            if not isinstance(file_name, str) or not file_name:
                continue

            split_raw = record.get("split")
            split = str(split_raw).strip().lower() if isinstance(split_raw, str) and split_raw.strip() else "train"
            seen_splits.add(split)
            images_total += 1

            image_target = outdir / "images" / split / file_name
            label_target = outdir / "labels" / split / f"{Path(file_name).stem}.txt"
            label_target.parent.mkdir(parents=True, exist_ok=True)

            if download_images:
                image_url = record.get("url")
                if not isinstance(image_url, str) or not image_url:
                    images_failed += 1
                else:
                    if overwrite_images or not image_target.exists():
                        download_jobs.append(DownloadJob(url=image_url, target=image_target))

            pose_rows = None
            annotations = record.get("annotations")
            if isinstance(annotations, dict):
                pose_rows = annotations.get("pose")

            label_lines, dropped = _to_label_lines(pose_rows, expected_values)
            rows_dropped += dropped
            label_target.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            labels_written += 1

    downloaded, failed = _download_jobs(
        download_jobs,
        timeout=timeout,
        workers=workers,
        progress_every=progress_every,
        progress_callback=progress_callback,
    )
    images_downloaded += downloaded
    images_failed += failed

    yaml_path = _write_yaml(outdir, meta, seen_splits)
    return {
        "outdir": str(outdir.resolve()),
        "yaml": str(yaml_path.resolve()),
        "images_total": images_total,
        "images_downloaded": images_downloaded,
        "images_failed": images_failed,
        "labels_written": labels_written,
        "rows_dropped": rows_dropped,
        "splits": sorted(seen_splits),
    }


def main() -> None:
    args = _parse_args()
    summary = convert(
        Path(args.input).expanduser().resolve(),
        Path(args.outdir).expanduser().resolve(),
        download_images=not args.no_download,
        timeout=float(args.timeout),
        overwrite_images=bool(args.overwrite_images),
        workers=int(args.workers),
        progress_every=int(args.progress_every),
        progress_callback=lambda done, total: print(f"downloaded {done}/{total}", flush=True),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
