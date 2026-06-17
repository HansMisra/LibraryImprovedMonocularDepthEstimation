import json
from pathlib import Path

import cv2
import numpy as np


def _read_jsonl(path):
    records = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_idx}: {exc}") from exc

    return records


def validate_corpus_manifest(manifest_path, max_records=None):
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records = _read_jsonl(manifest_path)

    if max_records is not None:
        records = records[:max_records]

    if len(records) == 0:
        print(f"No records found in manifest: {manifest_path}")
        return False

    errors = []
    seen_ids = set()

    required_keys = [
        "sample_id",
        "left_image",
        "right_image",
        "pseudo_disparity",
        "semantic_map",
        "semantic_overlay",
    ]

    for idx, record in enumerate(records, start=1):
        sample_id = record.get("sample_id", f"record_{idx}")

        for key in required_keys:
            if key not in record:
                errors.append(f"{sample_id}: missing key '{key}'")

        if sample_id in seen_ids:
            errors.append(f"{sample_id}: duplicate sample_id")
        seen_ids.add(sample_id)

        missing_key = any(key not in record for key in required_keys)
        if missing_key:
            continue

        paths = {
            "left_image": Path(record["left_image"]),
            "right_image": Path(record["right_image"]),
            "pseudo_disparity": Path(record["pseudo_disparity"]),
            "semantic_map": Path(record["semantic_map"]),
            "semantic_overlay": Path(record["semantic_overlay"]),
        }

        for label, path in paths.items():
            if not path.exists():
                errors.append(f"{sample_id}: {label} not found: {path}")

        if any(not path.exists() for path in paths.values()):
            continue

        left_img = cv2.imread(str(paths["left_image"]), cv2.IMREAD_COLOR)
        right_img = cv2.imread(str(paths["right_image"]), cv2.IMREAD_COLOR)
        disp_img = cv2.imread(str(paths["pseudo_disparity"]), cv2.IMREAD_UNCHANGED)
        overlay_img = cv2.imread(str(paths["semantic_overlay"]), cv2.IMREAD_COLOR)

        if left_img is None:
            errors.append(f"{sample_id}: could not read left image")
            continue

        if right_img is None:
            errors.append(f"{sample_id}: could not read right image")

        if disp_img is None:
            errors.append(f"{sample_id}: could not read pseudo disparity")

        if overlay_img is None:
            errors.append(f"{sample_id}: could not read semantic overlay")

        try:
            semantic_map = np.load(paths["semantic_map"])
        except Exception as exc:
            errors.append(f"{sample_id}: could not load semantic map: {exc}")
            continue

        height, width = left_img.shape[:2]

        if semantic_map.shape[:2] != (height, width):
            errors.append(
                f"{sample_id}: semantic map shape {semantic_map.shape[:2]} "
                f"does not match left image {(height, width)}"
            )

        if right_img is not None and right_img.shape[:2] != (height, width):
            errors.append(
                f"{sample_id}: right image shape {right_img.shape[:2]} "
                f"does not match left image {(height, width)}"
            )

        if disp_img is not None and disp_img.shape[:2] != (height, width):
            errors.append(
                f"{sample_id}: disparity shape {disp_img.shape[:2]} "
                f"does not match left image {(height, width)}"
            )

        if overlay_img is not None and overlay_img.shape[:2] != (height, width):
            errors.append(
                f"{sample_id}: overlay shape {overlay_img.shape[:2]} "
                f"does not match left image {(height, width)}"
            )

    if errors:
        print(f"Corpus validation failed with {len(errors)} issue(s).")
        print("First few issues:")

        for error in errors[:20]:
            print(f"- {error}")

        return False

    print(f"Corpus validation passed for {len(records)} record(s).")
    return True