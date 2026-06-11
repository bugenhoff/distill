#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUTS = (Path("merge/out320.ndjson"), Path("merge/player-mob.ndjson"))
DEFAULT_OUTDIR = Path("merge/merged_detect_yolo")


@dataclass(slots=True)
class DatasetMeta:
    name: str
    names: list[str]


@dataclass(slots=True)
class SourceSummary:
    input: str
    name: str
    prefix: str
    class_mapping: dict[int, int]
    images_total: int = 0
    images_downloaded: int = 0
    images_failed: int = 0
    labels_written: int = 0
    rows_dropped: int = 0


@dataclass(slots=True)
class DownloadJob:
    source_idx: int
    url: str
    target: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge several Ultralytics detect NDJSON exports into one YOLO detect dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[str(path) for path in DEFAULT_INPUTS],
        help="Input detect NDJSON files",
    )
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory for merged YOLO dataset")
    parser.add_argument("--timeout", type=float, default=30.0, help="Image download timeout in seconds")
    parser.add_argument("--overwrite-images", action="store_true", help="Re-download existing images")
    parser.add_argument("--no-download", action="store_true", help="Skip image download, write labels/yaml/report only")
    parser.add_argument("--clean", action="store_true", help="Remove output directory before writing")
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
        pairs.sort(key=lambda item: item[0])
        return [name for _, name in pairs]
    if isinstance(raw, list):
        return [str(value) for value in raw]
    return []


def _parse_meta(first_record: dict[str, Any], input_path: Path) -> DatasetMeta:
    task = first_record.get("task")
    if isinstance(task, str) and task.strip() and task.strip().lower() != "detect":
        raise ValueError(f"{input_path}: expected detect NDJSON, got task={task!r}")

    names = _ordered_names(first_record.get("class_names"))
    if not names:
        raise ValueError(f"{input_path}: invalid or missing class_names in NDJSON dataset header")

    raw_name = first_record.get("name")
    name = str(raw_name).strip() if isinstance(raw_name, str) and raw_name.strip() else input_path.stem
    return DatasetMeta(name=name, names=names)


def _slugify(value: str, fallback: str) -> str:
    chars: list[str] = []
    for char in value.strip().lower():
        if char.isalnum():
            chars.append(char)
        elif char in {"-", "_", ".", " "}:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or fallback


def _prefixed_file_name(prefix: str, file_name: str) -> str:
    source_name = Path(file_name).name
    return f"{prefix}__{source_name}"


def _download_image(url: str, target: Path, timeout: float) -> tuple[bool, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "distill-detect-dataset-merger/1.0"})
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


def _build_union(inputs: list[Path]) -> tuple[list[DatasetMeta], list[str], list[dict[int, int]]]:
    metas: list[DatasetMeta] = []
    union_names: list[str] = []
    union_index: dict[str, int] = {}
    mappings: list[dict[int, int]] = []

    for input_path in inputs:
        with input_path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline()
        if not first_line:
            raise ValueError(f"{input_path}: NDJSON is empty")

        first_record = json.loads(first_line)
        if not isinstance(first_record, dict) or first_record.get("type") != "dataset":
            raise ValueError(f"{input_path}: first NDJSON line must be dataset metadata")

        meta = _parse_meta(first_record, input_path)
        mapping: dict[int, int] = {}
        for source_idx, class_name in enumerate(meta.names):
            if class_name not in union_index:
                union_index[class_name] = len(union_names)
                union_names.append(class_name)
            mapping[source_idx] = union_index[class_name]

        metas.append(meta)
        mappings.append(mapping)

    return metas, union_names, mappings


def _to_label_lines(box_rows: Any, class_mapping: dict[int, int]) -> tuple[list[str], int]:
    if not isinstance(box_rows, list):
        return [], 0

    lines: list[str] = []
    dropped = 0
    for row in box_rows:
        if not isinstance(row, list) or not row:
            dropped += 1
            continue
        try:
            source_cls_idx = int(row[0])
            cls_idx = class_mapping[source_cls_idx]
            values = [float(value) for value in row[1:]]
        except Exception:
            dropped += 1
            continue

        if len(values) < 4:
            dropped += 1
            continue
        if len(values) > 4:
            values = values[:4]

        parts = [str(cls_idx)] + [_format_float(value) for value in values]
        lines.append(" ".join(parts))
    return lines, dropped


def _write_yaml(outdir: Path, names: list[str], seen_splits: set[str]) -> Path:
    yaml_path = outdir / "data.yaml"
    lines: list[str] = [f"path: {outdir.resolve()}"]
    for split in ("train", "val", "test"):
        if split in seen_splits:
            lines.append(f"{split}: images/{split}")
    lines.append("names:")
    for idx, name in enumerate(names):
        lines.append(f"  {idx}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def merge(
    input_paths: list[Path],
    outdir: Path,
    *,
    download_images: bool,
    timeout: float,
    overwrite_images: bool,
    clean: bool,
    workers: int,
    progress_every: int,
) -> dict[str, Any]:
    if clean and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metas, union_names, mappings = _build_union(input_paths)
    prefixes = [_slugify(meta.name, input_path.stem) for meta, input_path in zip(metas, input_paths, strict=True)]

    used_prefixes: set[str] = set()
    for idx, prefix in enumerate(prefixes):
        unique_prefix = prefix
        suffix = 2
        while unique_prefix in used_prefixes:
            unique_prefix = f"{prefix}_{suffix}"
            suffix += 1
        prefixes[idx] = unique_prefix
        used_prefixes.add(unique_prefix)

    seen_splits: set[str] = set()
    sources = [
        SourceSummary(
            input=str(input_path.resolve()),
            name=meta.name,
            prefix=prefix,
            class_mapping=class_mapping,
        )
        for input_path, meta, class_mapping, prefix in zip(input_paths, metas, mappings, prefixes, strict=True)
    ]
    download_jobs: list[DownloadJob] = []

    for source_idx, (input_path, class_mapping, prefix) in enumerate(
        zip(input_paths, mappings, prefixes, strict=True)
    ):
        summary = sources[source_idx]
        with input_path.open("r", encoding="utf-8") as handle:
            next(handle)
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
                summary.images_total += 1

                merged_file_name = _prefixed_file_name(prefix, file_name)
                image_target = outdir / "images" / split / merged_file_name
                label_target = outdir / "labels" / split / f"{Path(merged_file_name).stem}.txt"
                label_target.parent.mkdir(parents=True, exist_ok=True)

                if download_images:
                    image_url = record.get("url")
                    if not isinstance(image_url, str) or not image_url:
                        summary.images_failed += 1
                    elif overwrite_images or not image_target.exists():
                        download_jobs.append(DownloadJob(source_idx=source_idx, url=image_url, target=image_target))

                box_rows = None
                annotations = record.get("annotations")
                if isinstance(annotations, dict):
                    box_rows = annotations.get("boxes")

                label_lines, dropped = _to_label_lines(box_rows, class_mapping)
                summary.rows_dropped += dropped
                label_target.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
                summary.labels_written += 1

    if download_images and download_jobs:
        total_jobs = len(download_jobs)
        completed_jobs = 0
        max_workers = max(1, int(workers))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_download_image, job.url, job.target, timeout): job for job in download_jobs
            }
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    ok, _ = future.result()
                except Exception:
                    ok = False
                if ok:
                    sources[job.source_idx].images_downloaded += 1
                else:
                    sources[job.source_idx].images_failed += 1

                completed_jobs += 1
                if progress_every > 0 and (completed_jobs % progress_every == 0 or completed_jobs == total_jobs):
                    print(f"downloaded {completed_jobs}/{total_jobs}", file=sys.stderr, flush=True)

    yaml_path = _write_yaml(outdir, union_names, seen_splits)
    report_path = outdir / "merge_report.json"
    report = {
        "outdir": str(outdir.resolve()),
        "yaml": str(yaml_path.resolve()),
        "class_names": {idx: name for idx, name in enumerate(union_names)},
        "splits": sorted(seen_splits),
        "sources": [
            {
                "input": source.input,
                "name": source.name,
                "prefix": source.prefix,
                "class_mapping": source.class_mapping,
                "images_total": source.images_total,
                "images_downloaded": source.images_downloaded,
                "images_failed": source.images_failed,
                "labels_written": source.labels_written,
                "rows_dropped": source.rows_dropped,
            }
            for source in sources
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path.resolve())
    return report


def main() -> None:
    args = _parse_args()
    input_paths = [Path(value).expanduser().resolve() for value in args.inputs]
    report = merge(
        input_paths,
        Path(args.outdir).expanduser().resolve(),
        download_images=not args.no_download,
        timeout=float(args.timeout),
        overwrite_images=bool(args.overwrite_images),
        clean=bool(args.clean),
        workers=int(args.workers),
        progress_every=int(args.progress_every),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
