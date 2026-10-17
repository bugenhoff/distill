#!/usr/bin/env python3
"""Build a clean 640x640 center-cropped YOLO-pose dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - exercised by CLI users.
    raise SystemExit("Pillow is required. Run with: uv run python tools/prepare_pose_640.py") from exc

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is available in the uv env.
    yaml = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SOURCE = Path("datasets/pose")
DEFAULT_OUTPUT = Path("datasets/pose_640")
SOURCE_SIZE = (640, 640)
CROP_SIZE = (640, 640)
SPLIT_ORDER = {"train": 0, "val": 1, "valid": 1, "test": 2}


@dataclass
class SplitStats:
    seen: int = 0
    accepted: int = 0
    skipped_wrong_size: int = 0
    skipped_empty_label: int = 0
    skipped_duplicate: int = 0
    missing_label: int = 0
    malformed_labels: int = 0
    dropped_objects: int = 0


@dataclass
class RunStats:
    source: str
    output: str
    source_size: tuple[int, int]
    crop_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    dedupe_threshold: int
    dry_run: bool
    splits: dict[str, SplitStats] = field(default_factory=dict)
    duplicate_examples: list[dict[str, Any]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a YOLO-pose dataset by source image size, center-crop accepted images to "
            "640x640, update labels, and optionally remove near-duplicate crops."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help=f"Source dataset root. Default: {DEFAULT_SOURCE}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output dataset root. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--overwrite", action="store_true", help="Remove the output directory before writing.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without writing files.")
    parser.add_argument("--source-width", type=int, default=SOURCE_SIZE[0], help="Required source image width.")
    parser.add_argument("--source-height", type=int, default=SOURCE_SIZE[1], help="Required source image height.")
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE[0], help="Square center crop size.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for written crops.")
    parser.add_argument(
        "--dedupe-threshold",
        type=int,
        default=4,
        help=(
            "dHash Hamming distance for near-duplicate filtering after crop. "
            "Use -1 to disable. 0 keeps only exact perceptual-hash matches."
        ),
    )
    parser.add_argument(
        "--min-box-visible-ratio",
        type=float,
        default=0.05,
        help="Drop an object if less than this fraction of its bbox remains inside the crop.",
    )
    parser.add_argument(
        "--keep-empty-labels",
        action="store_true",
        help="Keep images even when no labels remain after crop. Default drops them.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N images per split. Use 0 to disable.",
    )
    return parser.parse_args()


def center_crop_box(source_size: tuple[int, int], crop_size: int) -> tuple[int, int, int, int]:
    width, height = source_size
    if crop_size > width or crop_size > height:
        raise ValueError(f"Crop size {crop_size} does not fit into source size {width}x{height}.")
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return left, top, left + crop_size, top + crop_size


def discover_splits(source: Path) -> list[str]:
    images_dir = source / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Expected images directory: {images_dir}")
    splits = [p.name for p in images_dir.iterdir() if p.is_dir()]
    return sorted(splits, key=lambda name: (SPLIT_ORDER.get(name, 99), name))


def iter_images(split_dir: Path) -> list[Path]:
    return sorted(path for path in split_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def format_number(value: float) -> str:
    value = max(0.0, min(1.0, value))
    if abs(value) < 1e-12:
        value = 0.0
    if abs(value - 1.0) < 1e-12:
        value = 1.0
    return f"{value:.6f}".rstrip("0").rstrip(".")


def format_visibility(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return str(int(rounded))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def transform_label_line(
    line: str,
    source_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
    min_visible_ratio: float,
) -> tuple[str | None, bool]:
    parts = line.split()
    if len(parts) < 5:
        return None, True

    class_id = parts[0]
    try:
        values = [float(part) for part in parts[1:]]
    except ValueError:
        return None, True

    if len(values) < 4:
        return None, True
    if len(values) > 4 and (len(values) - 4) % 3 != 0:
        return None, True

    width, height = source_size
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top

    cx, cy, box_w, box_h = values[:4]
    x1 = (cx - box_w / 2.0) * width
    y1 = (cy - box_h / 2.0) * height
    x2 = (cx + box_w / 2.0) * width
    y2 = (cy + box_h / 2.0) * height

    original_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if original_area <= 0.0:
        return None, False

    clipped_x1 = max(x1, crop_left)
    clipped_y1 = max(y1, crop_top)
    clipped_x2 = min(x2, crop_right)
    clipped_y2 = min(y2, crop_bottom)
    clipped_area = max(0.0, clipped_x2 - clipped_x1) * max(0.0, clipped_y2 - clipped_y1)
    if clipped_area <= 0.0 or clipped_area / original_area < min_visible_ratio:
        return None, False

    new_x1 = clipped_x1 - crop_left
    new_y1 = clipped_y1 - crop_top
    new_x2 = clipped_x2 - crop_left
    new_y2 = clipped_y2 - crop_top
    new_cx = ((new_x1 + new_x2) / 2.0) / crop_width
    new_cy = ((new_y1 + new_y2) / 2.0) / crop_height
    new_box_w = (new_x2 - new_x1) / crop_width
    new_box_h = (new_y2 - new_y1) / crop_height

    output = [
        class_id,
        format_number(new_cx),
        format_number(new_cy),
        format_number(new_box_w),
        format_number(new_box_h),
    ]

    keypoints = values[4:]
    for idx in range(0, len(keypoints), 3):
        kpt_x, kpt_y, visibility = keypoints[idx : idx + 3]
        pixel_x = kpt_x * width
        pixel_y = kpt_y * height
        inside_crop = crop_left <= pixel_x < crop_right and crop_top <= pixel_y < crop_bottom
        if visibility <= 0 or not inside_crop:
            output.extend(["0", "0", "0"])
            continue
        output.extend(
            [
                format_number((pixel_x - crop_left) / crop_width),
                format_number((pixel_y - crop_top) / crop_height),
                format_visibility(visibility),
            ]
        )

    return " ".join(output), False


def transform_label_file(
    label_path: Path,
    source_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
    min_visible_ratio: float,
) -> tuple[list[str], int, int]:
    output_lines: list[str] = []
    malformed = 0
    dropped = 0
    if not label_path.exists():
        return output_lines, malformed, dropped

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        transformed, is_malformed = transform_label_line(line, source_size, crop_box, min_visible_ratio)
        if is_malformed:
            malformed += 1
        elif transformed is None:
            dropped += 1
        else:
            output_lines.append(transformed)
    return output_lines, malformed, dropped


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    gray = image.convert("L").resize((hash_size + 1, hash_size), resampling)
    pixels = gray.tobytes()
    value = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            value <<= 1
            if pixels[row_start + col] > pixels[row_start + col + 1]:
                value |= 1
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def find_duplicate(
    image_hash: int,
    accepted_hashes: list[tuple[int, str]],
    threshold: int,
) -> tuple[str, int] | None:
    if threshold < 0:
        return None
    for previous_hash, previous_relpath in accepted_hashes:
        distance = hamming_distance(image_hash, previous_hash)
        if distance <= threshold:
            return previous_relpath, distance
    return None


def safe_prepare_output(output: Path, source: Path, overwrite: bool, dry_run: bool) -> None:
    output_resolved = output.resolve()
    source_resolved = source.resolve()
    if output_resolved == source_resolved:
        raise ValueError("Output must differ from source to avoid damaging the original dataset.")
    if dry_run:
        return
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}. Use --overwrite or choose another --output.")
        shutil.rmtree(output)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)


def write_data_yaml(source: Path, output: Path, splits: list[str], dry_run: bool) -> None:
    if dry_run:
        return

    data: dict[str, Any] = {}
    source_yaml = source / "data.yaml"
    if source_yaml.exists() and yaml is not None:
        loaded = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data.update(loaded)
    elif source_yaml.exists():
        for line in source_yaml.read_text(encoding="utf-8").splitlines():
            if line.startswith("path:"):
                continue
            if ":" in line and not line.startswith(" "):
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()

    data["path"] = str(output.resolve())
    if "train" in splits:
        data["train"] = "images/train"
    val_split = "val" if "val" in splits else "valid" if "valid" in splits else None
    if val_split:
        data["val"] = f"images/{val_split}"
    if "test" in splits:
        data["test"] = "images/test"

    if "names" not in data:
        data["names"] = {0: "class_0"}

    if yaml is not None:
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    else:
        lines = [f"path: {data['path']}"]
        for key in ("train", "val", "test"):
            if key in data:
                lines.append(f"{key}: {data[key]}")
        if "names" in data:
            lines.append(f"names: {data['names']}")
        if "kpt_shape" in data:
            lines.append(f"kpt_shape: {data['kpt_shape']}")
        text = "\n".join(lines) + "\n"
    (output / "data.yaml").write_text(text, encoding="utf-8")


def process_dataset(args: argparse.Namespace) -> RunStats:
    source = args.source.resolve()
    output = args.output.resolve()
    source_size = (args.source_width, args.source_height)
    crop_box = center_crop_box(source_size, args.crop_size)
    crop_size = (args.crop_size, args.crop_size)
    splits = discover_splits(source)
    safe_prepare_output(output, source, args.overwrite, args.dry_run)

    stats = RunStats(
        source=str(source),
        output=str(output),
        source_size=source_size,
        crop_size=crop_size,
        crop_box=crop_box,
        dedupe_threshold=args.dedupe_threshold,
        dry_run=args.dry_run,
    )
    accepted_hashes: list[tuple[int, str]] = []

    for split in splits:
        split_stats = SplitStats()
        stats.splits[split] = split_stats
        input_images_dir = source / "images" / split
        input_labels_dir = source / "labels" / split
        output_images_dir = output / "images" / split
        output_labels_dir = output / "labels" / split
        if not args.dry_run:
            output_images_dir.mkdir(parents=True, exist_ok=True)
            output_labels_dir.mkdir(parents=True, exist_ok=True)

        for image_path in iter_images(input_images_dir):
            split_stats.seen += 1
            rel_image_path = image_path.relative_to(source)
            label_path = input_labels_dir / image_path.relative_to(input_images_dir).with_suffix(".txt")
            if not label_path.exists():
                split_stats.missing_label += 1

            with Image.open(image_path) as image:
                if image.size != source_size:
                    split_stats.skipped_wrong_size += 1
                    continue

                label_lines, malformed, dropped = transform_label_file(
                    label_path,
                    source_size,
                    crop_box,
                    args.min_box_visible_ratio,
                )
                split_stats.malformed_labels += malformed
                split_stats.dropped_objects += dropped
                if not label_lines and not args.keep_empty_labels:
                    split_stats.skipped_empty_label += 1
                    continue

                crop = image.crop(crop_box)
                image_hash = dhash(crop)
                duplicate = find_duplicate(image_hash, accepted_hashes, args.dedupe_threshold)
                if duplicate is not None:
                    previous_relpath, distance = duplicate
                    split_stats.skipped_duplicate += 1
                    if len(stats.duplicate_examples) < 20:
                        stats.duplicate_examples.append(
                            {
                                "skipped": str(rel_image_path),
                                "matched": previous_relpath,
                                "distance": distance,
                            }
                        )
                    continue

                split_stats.accepted += 1
                accepted_hashes.append((image_hash, str(rel_image_path)))

                if not args.dry_run:
                    output_image_path = output_images_dir / image_path.relative_to(input_images_dir)
                    output_label_path = output_labels_dir / label_path.relative_to(input_labels_dir)
                    output_image_path.parent.mkdir(parents=True, exist_ok=True)
                    output_label_path.parent.mkdir(parents=True, exist_ok=True)
                    save_kwargs: dict[str, Any] = {}
                    if output_image_path.suffix.lower() in {".jpg", ".jpeg"}:
                        save_kwargs = {"quality": args.jpeg_quality, "subsampling": 0}
                        crop = crop.convert("RGB")
                    crop.save(output_image_path, **save_kwargs)
                    output_label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

            if args.progress_every and split_stats.seen % args.progress_every == 0:
                print(
                    f"{split}: seen={split_stats.seen} accepted={split_stats.accepted} "
                    f"wrong_size={split_stats.skipped_wrong_size} dup={split_stats.skipped_duplicate}"
                )

    write_data_yaml(source, output, splits, args.dry_run)
    if not args.dry_run:
        report_path = output / "prepare_report.json"
        report_path.write_text(json.dumps(asdict(stats), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    args = parse_args()
    stats = process_dataset(args)
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


# yolo detect train \
#   model=320x320.pt \
#   data=merge/merged_detect_yolo/data.yaml \
#   imgsz=320 \
#   epochs=100 \
#   batch=-1
