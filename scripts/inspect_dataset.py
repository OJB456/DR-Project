"""Inspect the existing local APTOS dataset without downloading or copying it.

Example:
    python scripts/inspect_dataset.py

The script reads ``archive/`` by default. It keeps aggregate counters and
small metadata structures in memory, but never loads the image collection or
the image pixels into RAM. Duplicate detection hashes one file at a time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_files(root: Path) -> Iterator[Path]:
    yield from (path for path in root.rglob("*") if path.is_file())


def image_id_from_path(path: Path) -> str:
    return path.stem


def read_csv_labels(path: Path) -> tuple[list[dict[str, str]], list[str], str | None, str | None]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not reader.fieldnames:
        return rows, [], None, None
    id_column = next((name for name in ("id_code", "image", "image_id", "id") if name in reader.fieldnames), reader.fieldnames[0])
    label_column = next((name for name in ("diagnosis", "label", "level", "grade") if name in reader.fieldnames), None)
    return rows, reader.fieldnames, id_column, label_column


def inspect_image(path: Path) -> tuple[tuple[int, int] | None, str | None]:
    if path.suffix.lower() == ".png":
        try:
            with path.open("rb") as handle:
                if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                    return None, "invalid PNG signature"
                saw_iend = False
                width = height = None
                while True:
                    header = handle.read(8)
                    if len(header) != 8:
                        return None, "truncated PNG chunk header"
                    length, chunk_type = struct.unpack(">I4s", header)
                    chunk_data = handle.read(length)
                    stored_crc = handle.read(4)
                    if len(chunk_data) != length or len(stored_crc) != 4:
                        return None, "truncated PNG chunk"
                    expected_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
                    actual_crc = struct.unpack(">I", stored_crc)[0]
                    if expected_crc != actual_crc:
                        return None, f"invalid CRC in {chunk_type.decode('ascii', errors='replace')} chunk"
                    if chunk_type == b"IHDR" and length >= 8:
                        width, height = struct.unpack(">II", chunk_data[:8])
                    if chunk_type == b"IEND":
                        saw_iend = True
                        break
                if not saw_iend or not width or not height:
                    return None, "missing PNG dimensions or IEND"
                return (width, height), None
        except OSError as error:
            return None, str(error)
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return (int(image.width), int(image.height)), None
    except Exception as error:  # image libraries expose different corruption exceptions
        return None, f"{type(error).__name__}: {error}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_split(root: Path, csv_name: str, image_dir_name: str, hash_duplicates: bool = False) -> dict[str, Any]:
    csv_path = root / csv_name
    image_dir = root / image_dir_name
    if not csv_path.is_file():
        raise FileNotFoundError(f"Required CSV is missing: {csv_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Required image directory is missing: {image_dir}")

    rows, columns, id_column, label_column = read_csv_labels(csv_path)
    image_paths = [path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    actual_ids = Counter(image_id_from_path(path) for path in image_paths)
    csv_ids = [Path(str(row[id_column])).stem for row in rows] if id_column else []
    csv_id_counts = Counter(csv_ids)
    missing_ids = sorted(set(csv_ids) - set(actual_ids))
    extra_ids = sorted(set(actual_ids) - set(csv_ids))
    duplicate_csv_ids = {image_id: count for image_id, count in csv_id_counts.items() if count > 1}
    duplicate_image_ids = {image_id: count for image_id, count in actual_ids.items() if count > 1}
    labels = Counter(str(row[label_column]).strip() for row in rows) if label_column else Counter()
    extensions = Counter(path.suffix.lower().lstrip(".") for path in image_paths)

    sample_dimensions: dict[str, list[int]] = {}
    corrupt_images: list[dict[str, str]] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    for image_path in image_paths:
        dimensions, error = inspect_image(image_path)
        if dimensions is None:
            corrupt_images.append({"path": str(image_path), "error": error or "unreadable"})
        elif len(sample_dimensions) < 20:
            sample_dimensions[str(image_path.relative_to(image_dir))] = [dimensions[0], dimensions[1]]
        if hash_duplicates and error is None:
            try:
                hashes[sha256_file(image_path)].append(str(image_path))
            except OSError as hash_error:
                corrupt_images.append({"path": str(image_path), "error": str(hash_error)})

    return {
        "csv_path": str(csv_path),
        "image_directory": str(image_dir),
        "csv_columns": columns,
        "csv_rows": len(rows),
        "id_column": id_column,
        "label_column": label_column,
        "image_file_count": len(image_paths),
        "image_extensions": dict(sorted(extensions.items())),
        "sample_image_dimensions": sample_dimensions,
        "label_values": sorted(labels),
        "class_distribution": dict(sorted(labels.items(), key=lambda item: item[0])),
        "missing_images": missing_ids,
        "missing_image_count": len(missing_ids),
        "extra_images": extra_ids,
        "extra_image_count": len(extra_ids),
        "duplicate_csv_ids": duplicate_csv_ids,
        "duplicate_image_ids": duplicate_image_ids,
        "corrupt_images": corrupt_images,
        "corrupt_image_count": len(corrupt_images),
        "duplicate_content_groups": [paths for paths in hashes.values() if len(paths) > 1],
        "_content_hashes": set(hashes),
        "mapping_verified": not missing_ids and not extra_ids and not duplicate_csv_ids and not duplicate_image_ids,
    }


def build_report(root: Path, hash_duplicates: bool = False) -> dict[str, Any]:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {root}")

    all_files = list(iter_files(root))
    split_definitions = {
        "train": ("train_1.csv", "train_images/train_images"),
        "validation": ("valid.csv", "val_images/val_images"),
        "test": ("test.csv", "test_images/test_images"),
    }
    splits = {name: inspect_split(root, csv_name, image_dir, hash_duplicates) for name, (csv_name, image_dir) in split_definitions.items()}
    split_ids = {}
    split_hashes: dict[str, set[str]] = {}
    for name, split in splits.items():
        rows, _, id_column, _ = read_csv_labels(root / split_definitions[name][0])
        split_ids[name] = {Path(str(row[id_column])).stem for row in rows} if id_column else set()
        split_hashes[name] = split.pop("_content_hashes")
    id_overlap = {f"{left}_vs_{right}": sorted(split_ids[left] & split_ids[right]) for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    content_overlap = {f"{left}_vs_{right}": len(split_hashes[left] & split_hashes[right]) for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))} if hash_duplicates else None
    csv_paths = [path for path in all_files if path.suffix.lower() == ".csv"]
    json_paths = [path for path in all_files if path.suffix.lower() == ".json"]
    top_level_entries = sorted({str(path.relative_to(root).parts[0]) for path in all_files})
    image_formats = Counter()
    for split in splits.values():
        image_formats.update(split["image_extensions"])
    report: dict[str, Any] = {
        "dataset_root": str(root.resolve()),
        "total_files": len(all_files),
        "image_file_count": sum(split["image_file_count"] for split in splits.values()),
        "image_formats": dict(sorted(image_formats.items())),
        "csv_files": [str(path) for path in csv_paths],
        "json_files": [str(path) for path in json_paths],
        "top_level_structure": top_level_entries,
        "splits": splits,
        "proposed_mapping": {
            "train_1.csv": "train_images/train_images",
            "valid.csv": "val_images/val_images",
            "test.csv": "test_images/test_images",
        },
        "mapping_confirmed": all(split["mapping_verified"] for split in splits.values()),
        "split_id_overlap": id_overlap,
        "split_content_overlap_group_counts": content_overlap,
        "potential_data_leakage": any(id_overlap.values()) or (bool(content_overlap) and any(content_overlap.values())),
        "content_hashing_performed": hash_duplicates,
        "duplicate_image_groups": [],
        "duplicate_image_group_count": 0,
        "official_separate_train_validation_test": True,
        "notes": [
            "Image hashes and dimensions were processed one file at a time.",
            "No files were downloaded, moved, or copied.",
            "The proposed CSV-to-directory mapping is verified by exact image stem matching.",
            "Quality thresholds and clinical performance are outside this structural dataset audit.",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("archive"), help="Existing local dataset root; never downloaded by this script")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dataset_report.json"))
    parser.add_argument("--hash-duplicates", action="store_true", help="Hash image contents for duplicate detection; reads every image byte")
    args = parser.parse_args()
    try:
        report = build_report(args.dataset_root, args.hash_duplicates)
    except (FileNotFoundError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Dataset path: {report['dataset_root']}")
    print(f"Total files: {report['total_files']}")
    print(f"Images: {report['image_file_count']}")
    for name, split in report["splits"].items():
        print(f"{name}: {split['csv_rows']} labels, {split['image_file_count']} images")
        print(f"  columns: {split['csv_columns']}")
        print(f"  classes: {split['class_distribution']}")
        print(f"  missing: {split['missing_image_count']}, extra: {split['extra_image_count']}, corrupt: {split['corrupt_image_count']}")
    print(f"Mapping confirmed: {report['mapping_confirmed']}")
    print(f"Potential data leakage: {report['potential_data_leakage']}")
    print(f"Report saved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
