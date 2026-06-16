#!/usr/bin/env python3
"""Build a 320x320 YOLO dataset with player-frame crops.

The input dataset is expected to use the legacy class layout:

    0 mini_game
    1 bobber
    2 mini_fish
    3 nick
    4 self

The output dataset preserves all five classes. Existing labeled 320x320
samples are normalized to 320x320, while full-frame player images are split
into 320x320 crops. Crops without labels are discarded.

Example:

    python utils/build_player_bar_dataset.py \
        --source /home/bugen/distill/merge/merged_detect_yolo \
        --output /home/bugen/distill/merge/merged_detect_yolo_320
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

MINIGAME_CLASSES = {0, 1, 2}
PLAYER_CLASSES = {3, 4}
SUPPORTED_CLASSES = MINIGAME_CLASSES | PLAYER_CLASSES
OUTPUT_NAMES = ("mini_game", "bobber", "mini_fish", "nick", "self")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
OUTPUT_SIZE = 320


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class CropWindow:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return int(self.x2 - self.x1)

    @property
    def height(self) -> int:
        return int(self.y2 - self.y1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/bugen/distill/merge/merged_detect_yolo"),
        help="Path to the legacy YOLO dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output YOLO dataset root.",
    )
    parser.add_argument("--stride", type=int, default=160, help="Stride for the fixed 320x320 crop grid.")
    parser.add_argument(
        "--object-jitter",
        type=int,
        default=3,
        help="Extra object-centered crop variants per player frame.",
    )
    parser.add_argument(
        "--min-visible",
        type=float,
        default=0.60,
        help="Minimum original bbox area fraction visible inside a crop.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def read_labels(path: Path) -> list[YoloBox]:
    if not path.exists():
        return []
    labels: list[YoloBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        labels.append(
            YoloBox(
                class_id=int(float(parts[0])),
                x=float(parts[1]),
                y=float(parts[2]),
                w=float(parts[3]),
                h=float(parts[4]),
            )
        )
    return labels


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def box_to_pixels(box: YoloBox, width: int, height: int) -> tuple[float, float, float, float]:
    cx = box.x * width
    cy = box.y * height
    bw = box.w * width
    bh = box.h * height
    return (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0)


def clip_box_to_crop(
    pixel_box: tuple[float, float, float, float],
    crop: CropWindow,
) -> tuple[tuple[float, float, float, float] | None, float]:
    x1, y1, x2, y2 = pixel_box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0.0:
        return None, 0.0

    ix1 = max(x1, float(crop.x1))
    iy1 = max(y1, float(crop.y1))
    ix2 = min(x2, float(crop.x2))
    iy2 = min(y2, float(crop.y2))
    inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter_area <= 0.0:
        return None, 0.0
    visible = inter_area / area
    return (ix1 - crop.x1, iy1 - crop.y1, ix2 - crop.x1, iy2 - crop.y1), visible


def crop_labels(
    labels: list[YoloBox],
    *,
    crop: CropWindow,
    width: int,
    height: int,
    min_visible: float,
) -> list[str]:
    output: list[str] = []
    for label in labels:
        if label.class_id not in PLAYER_CLASSES:
            continue
        clipped, visible = clip_box_to_crop(box_to_pixels(label, width, height), crop)
        if clipped is None or visible < min_visible:
            continue
        x1, y1, x2, y2 = clipped
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 1.0 or bh <= 1.0:
            continue
        cx = (x1 + x2) / 2.0 / crop.width
        cy = (y1 + y2) / 2.0 / crop.height
        output.append(f"{label.class_id} {cx:.6f} {cy:.6f} {bw / crop.width:.6f} {bh / crop.height:.6f}")
    return output


def axis_positions(size: int, crop_size: int, stride: int) -> list[int]:
    if size <= crop_size:
        return [0]
    positions = list(range(0, max(1, size - crop_size + 1), max(1, stride)))
    last = size - crop_size
    if positions[-1] != last:
        positions.append(last)
    return sorted(set(int(value) for value in positions))


def grid_windows(width: int, height: int, crop_size: int, stride: int) -> list[CropWindow]:
    crop_w = min(crop_size, width)
    crop_h = min(crop_size, height)
    return [
        CropWindow(x, y, x + crop_w, y + crop_h)
        for y in axis_positions(height, crop_h, stride)
        for x in axis_positions(width, crop_w, stride)
    ]


def object_centered_windows(
    labels: list[YoloBox],
    *,
    width: int,
    height: int,
    crop_size: int,
    variants: int,
    rng: random.Random,
) -> Iterable[CropWindow]:
    crop_w = min(crop_size, width)
    crop_h = min(crop_size, height)
    max_x = width - crop_w
    max_y = height - crop_h
    for label in labels:
        if label.class_id not in PLAYER_CLASSES:
            continue
        cx = label.x * width
        cy = label.y * height
        for _index in range(max(0, variants)):
            jitter_x = rng.uniform(-0.25, 0.25) * crop_w
            jitter_y = rng.uniform(-0.25, 0.25) * crop_h
            x1 = int(round(cx - crop_w / 2.0 + jitter_x))
            y1 = int(round(cy - crop_h / 2.0 + jitter_y))
            x1 = max(0, min(x1, max_x))
            y1 = max(0, min(y1, max_y))
            yield CropWindow(x1, y1, x1 + crop_w, y1 + crop_h)


def write_image_and_labels(
    image,
    labels: list[str],
    *,
    image_out: Path,
    label_out: Path,
) -> None:
    if not labels:
        raise ValueError(f"refusing to write an empty label file: {label_out}")
    resized = cv2.resize(image, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_out), resized):
        raise RuntimeError(f"failed to write image: {image_out}")
    label_out.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")


def copy_minigame_sample(
    image_path: Path,
    labels: list[YoloBox],
    *,
    split: str,
    output: Path,
) -> int:
    kept = [
        f"{label.class_id} {label.x:.6f} {label.y:.6f} {label.w:.6f} {label.h:.6f}"
        for label in labels
        if label.class_id in SUPPORTED_CLASSES
    ]
    if not kept:
        return 0
    image_out = output / "images" / split / image_path.name
    label_out = output / "labels" / split / f"{image_path.stem}.txt"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    write_image_and_labels(image, kept, image_out=image_out, label_out=label_out)
    return 1


def build_player_crops(
    image_path: Path,
    labels: list[YoloBox],
    *,
    split: str,
    output: Path,
    stride: int,
    object_jitter: int,
    min_visible: float,
    rng: random.Random,
) -> tuple[int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    height, width = image.shape[:2]

    windows = grid_windows(width, height, OUTPUT_SIZE, stride)
    windows.extend(
        object_centered_windows(
            labels,
            width=width,
            height=height,
            crop_size=OUTPUT_SIZE,
            variants=object_jitter,
            rng=rng,
        )
    )
    unique_windows = list(dict.fromkeys(windows))

    positives: list[tuple[CropWindow, list[str]]] = []
    for window in unique_windows:
        crop_label_lines = crop_labels(labels, crop=window, width=width, height=height, min_visible=min_visible)
        if crop_label_lines:
            positives.append((window, crop_label_lines))

    written_total = 0
    for index, (window, crop_label_lines) in enumerate(positives):
        crop_img = image[window.y1 : window.y2, window.x1 : window.x2]
        if crop_img.size == 0:
            continue
        suffix = f"crop{OUTPUT_SIZE}_{window.x1}_{window.y1}_{index:03d}"
        image_out = output / "images" / split / f"{image_path.stem}__{suffix}.jpg"
        label_out = output / "labels" / split / f"{image_path.stem}__{suffix}.txt"
        write_image_and_labels(
            crop_img,
            crop_label_lines,
            image_out=image_out,
            label_out=label_out,
        )
        written_total += 1
    discarded_empty = len(unique_windows) - len(positives)
    return written_total, discarded_empty


def discover_splits(source: Path) -> list[str]:
    splits = [
        split
        for split in ("train", "val", "valid", "test")
        if (source / "images" / split).is_dir() and (source / "labels" / split).is_dir()
    ]
    if "train" not in splits:
        raise FileNotFoundError(f"train split not found under {source}")
    if "val" in splits and "valid" in splits:
        raise ValueError("source contains both 'val' and 'valid' splits; keep only one validation split")
    return splits


def write_data_yaml(output: Path, splits: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(OUTPUT_NAMES))
    validation_split = "val" if "val" in splits else "valid" if "valid" in splits else None
    paths = [f"path: {output}", "train: images/train"]
    if validation_split is not None:
        paths.append(f"val: images/{validation_split}")
    if "test" in splits:
        paths.append("test: images/test")
    (output / "data.yaml").write_text(
        "\n".join(paths) + f"\nnames:\n{names}\n",
        encoding="utf-8",
    )


def build_dataset(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    rng = random.Random(int(args.seed))
    splits = discover_splits(source)
    totals = {"minigame": 0, "player_crops": 0, "discarded_empty_crops": 0, "discarded_empty_sources": 0}
    for split in splits:
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = find_image(image_dir, label_path.stem)
            if image_path is None:
                continue
            labels = read_labels(label_path)
            if not labels:
                totals["discarded_empty_sources"] += 1
                continue
            class_ids = {label.class_id for label in labels}
            if class_ids & MINIGAME_CLASSES:
                totals["minigame"] += copy_minigame_sample(image_path, labels, split=split, output=output)
            if class_ids & PLAYER_CLASSES:
                total, discarded_empty = build_player_crops(
                    image_path,
                    labels,
                    split=split,
                    output=output,
                    stride=max(1, min(OUTPUT_SIZE, int(args.stride))),
                    object_jitter=max(0, int(args.object_jitter)),
                    min_visible=max(0.0, min(1.0, float(args.min_visible))),
                    rng=rng,
                )
                totals["player_crops"] += total
                totals["discarded_empty_crops"] += discarded_empty

    write_data_yaml(output, splits)
    print(
        "done: "
        f"minigame={totals['minigame']} "
        f"player_crops={totals['player_crops']} "
        f"discarded_empty_crops={totals['discarded_empty_crops']} "
        f"discarded_empty_sources={totals['discarded_empty_sources']} "
        f"output={output}"
    )


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()
