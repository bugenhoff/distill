#!/usr/bin/env python3
import pathlib
import sys

from PIL import Image

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.avif'}
TILE = 320

def main():
    if len(sys.argv) != 3:
        print("Usage: tile_images.py <source_dir> <output_dir>", file=sys.stderr)
        sys.exit(1)

    src = pathlib.Path(sys.argv[1])
    dst = pathlib.Path(sys.argv[2])

    if not src.is_dir():
        print(f"Error: source '{src}' is not a directory", file=sys.stderr)
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    images = sorted([f for f in src.iterdir() if f.suffix.lower() in IMG_EXTS and f.is_file()])

    if not images:
        print("No images found in source directory", file=sys.stderr)
        sys.exit(1)

    total = 0
    for img_path in images:
        img = Image.open(img_path)
        w, h = img.size
        cols = w // TILE
        rows = h // TILE
        if cols == 0 or rows == 0:
            print(f"Skipped (too small): {img_path.name} ({w}x{h})")
            continue
        stem = img_path.stem
        ext = img_path.suffix
        for r in range(rows):
            for c in range(cols):
                left = c * TILE
                top = r * TILE
                tile = img.crop((left, top, left + TILE, top + TILE))
                out_name = f"{stem}_{r}_{c}{ext}"
                tile.save(dst / out_name)
                total += 1
        print(f"Tiled: {img_path.name} ({w}x{h} -> {rows}x{cols} tiles)")

    print(f"\nDone: {total} tiles saved to '{dst}'")

if __name__ == '__main__':
    main()
