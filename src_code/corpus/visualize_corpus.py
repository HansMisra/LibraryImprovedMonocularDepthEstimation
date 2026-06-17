import json
from pathlib import Path

import cv2
import numpy as np


def _read_jsonl(path):
    records = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def _normalize_disparity(disparity):
    disparity = disparity.astype(np.float32)

    min_value = float(np.min(disparity))
    max_value = float(np.max(disparity))

    if max_value <= min_value:
        return np.zeros_like(disparity, dtype=np.uint8)

    normalized = (disparity - min_value) / (max_value - min_value)
    return (normalized * 255).astype(np.uint8)


def _add_label(image, label):
    labeled = image.copy()

    cv2.rectangle(
        labeled,
        (0, 0),
        (260, 34),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        labeled,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    return labeled


def save_corpus_visualizations(
    manifest_path,
    output_dir,
    limit=5,
):
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(manifest_path)

    if limit is not None:
        records = records[:limit]

    if len(records) == 0:
        print(f"No records found in manifest: {manifest_path}")
        return []

    saved_paths = []

    for idx, record in enumerate(records, start=1):
        sample_id = record["sample_id"]

        left = cv2.imread(record["left_image"], cv2.IMREAD_COLOR)
        disparity = cv2.imread(record["pseudo_disparity"], cv2.IMREAD_UNCHANGED)
        overlay = cv2.imread(record["semantic_overlay"], cv2.IMREAD_COLOR)

        if left is None:
            print(f"[{idx}/{len(records)}] skipped {sample_id}: missing left image")
            continue

        if disparity is None:
            print(f"[{idx}/{len(records)}] skipped {sample_id}: missing disparity")
            continue

        if overlay is None:
            print(f"[{idx}/{len(records)}] skipped {sample_id}: missing overlay")
            continue

        height, width = left.shape[:2]

        if overlay.shape[:2] != (height, width):
            overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_NEAREST)

        disp_norm = _normalize_disparity(disparity)
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_PLASMA)

        left_labeled = _add_label(left, "Left RGB")
        disp_labeled = _add_label(disp_color, "Pseudo Disparity")
        overlay_labeled = _add_label(overlay, "Semantic Overlay")

        grid = cv2.hconcat([left_labeled, disp_labeled, overlay_labeled])

        output_path = output_dir / f"{sample_id}_grid.png"
        cv2.imwrite(str(output_path), grid)

        saved_paths.append(str(output_path))
        print(f"[{idx}/{len(records)}] saved {output_path.name}")

    print(f"Saved {len(saved_paths)} visualization grid(s) to: {output_dir}")
    return saved_paths