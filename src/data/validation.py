"""Small, non-destructive dataset validation helpers."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import IMAGE_EXTENSIONS, read_records


def summarize_split(csv_path: str | Path, image_dir: str | Path) -> dict[str, Any]:
    """Return mapping and label counts without loading the full image collection."""
    frame = pd.read_csv(csv_path)
    records = read_records(csv_path, image_dir)
    labels = Counter(label for _, label in records)
    return {
        "csv_path": str(csv_path),
        "image_directory": str(image_dir),
        "csv_rows": int(len(frame)),
        "image_file_count": int(len(records)),
        "class_distribution": {str(key): value for key, value in sorted(labels.items())},
        "supported_extensions": sorted(IMAGE_EXTENSIONS),
        "mapping_verified": len(records) == len(frame),
    }
