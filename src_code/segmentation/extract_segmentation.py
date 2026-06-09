from pathlib import Path

import cv2
import numpy as np

from segmentation.segmenter import SegFormerSegmenter


def make_overlay(image_path, class_map):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    norm_map = class_map.astype(np.float32)
    max_value = max(float(norm_map.max()), 1.0)
    norm_map = (norm_map / max_value * 255).astype(np.uint8)

    color_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.60, color_map, 0.40, 0.0)

    return overlay


def generate_segmentation(
    image_dir="kitti_data/data_scene_flow/testing/image_2",
    output_dir="outputs/semantic_maps",
    limit=None,
    model_name="nvidia/segformer-b0-finetuned-ade-512-512",
    device=None,
):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    semantic_map_dir = output_dir
    overlay_dir = output_dir.parent / "semantic_overlays"

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}\n"
            "Expected KITTI left images. Pass a valid --image-dir."
        )

    semantic_map_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(image_dir.glob("*.png")) +
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.jpeg"))
    )

    if limit is not None:
        image_paths = image_paths[:limit]

    if len(image_paths) == 0:
        print(f"No images found in: {image_dir}")
        return []

    print(f"Found {len(image_paths)} images.")
    print(f"Loading segmentation model...")

    segmenter = SegFormerSegmenter(
        model_name=model_name,
        device=device,
    )

    saved_paths = []

    for idx, image_path in enumerate(image_paths, start=1):
        stem = image_path.stem

        class_map = segmenter.predict(image_path)

        map_path = semantic_map_dir / f"{stem}.npy"
        overlay_path = overlay_dir / f"{stem}.png"

        np.save(map_path, class_map)

        overlay = make_overlay(image_path, class_map)
        cv2.imwrite(str(overlay_path), overlay)

        saved_paths.append(
            {
                "image": str(image_path),
                "semantic_map": str(map_path),
                "overlay": str(overlay_path),
            }
        )

        print(f"[{idx}/{len(image_paths)}] saved {map_path.name}")

    print("Semantic segmentation generation complete.")
    return saved_paths