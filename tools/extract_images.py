#!/usr/bin/env python3
import pathlib
import shutil
import sys

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.avif'}

def main():
    if len(sys.argv) != 4:
        print("Usage: extract_images.py <source_dir> <output_dir> <count>", file=sys.stderr)
        sys.exit(1)

    src = pathlib.Path(sys.argv[1])
    dst = pathlib.Path(sys.argv[2])
    try:
        n = int(sys.argv[3])
    except ValueError:
        print("Error: count must be an integer", file=sys.stderr)
        sys.exit(1)

    if not src.is_dir():
        print(f"Error: source '{src}' is not a directory", file=sys.stderr)
        sys.exit(1)

    images = sorted([f for f in src.iterdir() if f.suffix.lower() in IMG_EXTS and f.is_file()])
    selected = [images[i] for i in range(0, len(images), 5)]

    if len(selected) < n:
        print(f"Error: only {len(selected)} images available (every 5th from {len(images)} total), need {n}", file=sys.stderr)
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    for f in selected[:n]:
        shutil.copy2(f, dst / f.name)
        print(f"Copied: {f.name}")

    print(f"\nDone: {n} images copied to '{dst}'")

if __name__ == '__main__':
    main()
