import json
from pathlib import Path


def build_corpus_manifest(
    left_image_dir,
    right_image_dir,
    disparity_dir,
    semantic_map_dir,
    semantic_overlay_dir,
    output_path,
    limit=None,
    require_all=True,
    frame_suffix="_10",
):
    left_image_dir = Path(left_image_dir)
    right_image_dir = Path(right_image_dir)
    disparity_dir = Path(disparity_dir)
    semantic_map_dir = Path(semantic_map_dir)
    semantic_overlay_dir = Path(semantic_overlay_dir)
    output_path = Path(output_path)

    required_dirs = {
        "left_image_dir": left_image_dir,
        "right_image_dir": right_image_dir,
        "disparity_dir": disparity_dir,
        "semantic_map_dir": semantic_map_dir,
        "semantic_overlay_dir": semantic_overlay_dir,
    }

    for label, directory in required_dirs.items():
        if not directory.exists():
            raise FileNotFoundError(f"{label} not found: {directory}")

    left_images = sorted(left_image_dir.glob("*.png"))

    if frame_suffix is not None:
        left_images = [
            path for path in left_images
            if path.stem.endswith(frame_suffix)
        ]

    if limit is not None:
        left_images = left_images[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    missing = []

    for left_path in left_images:
        sample_id = left_path.stem

        right_path = right_image_dir / left_path.name
        disparity_path = disparity_dir / left_path.name
        semantic_map_path = semantic_map_dir / f"{sample_id}.npy"
        semantic_overlay_path = semantic_overlay_dir / f"{sample_id}.png"

        paths = {
            "right_image": right_path,
            "pseudo_disparity": disparity_path,
            "semantic_map": semantic_map_path,
            "semantic_overlay": semantic_overlay_path,
        }

        missing_paths = [
            str(path)
            for path in paths.values()
            if not path.exists()
        ]

        if missing_paths:
            missing.append(
                {
                    "sample_id": sample_id,
                    "missing": missing_paths,
                }
            )

            if require_all:
                continue

        record = {
            "sample_id": sample_id,
            "left_image": str(left_path),
            "right_image": str(right_path),
            "pseudo_disparity": str(disparity_path),
            "semantic_map": str(semantic_map_path),
            "semantic_overlay": str(semantic_overlay_path),
        }

        records.append(record)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(records)} corpus records to: {output_path}")

    if missing:
        print(f"Skipped or flagged {len(missing)} samples with missing files.")
        print("First few missing samples:")

        for item in missing[:5]:
            print(item)

    return records