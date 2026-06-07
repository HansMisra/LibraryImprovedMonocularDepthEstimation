from pathlib import Path


def generate_segmentation(
    image_dir="kitti_data/data_scene_flow/testing/image_2",
    output_dir="outputs/semantic_maps",
    limit=None,
):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}\n"
            "Expected KITTI left images. Pass a valid --image-dir."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(image_dir.glob("*.png")) +
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.jpeg"))
    )

    if limit is not None:
        image_paths = image_paths[:limit]

    print(f"Found {len(image_paths)} images.")
    print("Semantic segmentation placeholder ran successfully.")
    print("Next step: connect SegFormer model inference here.")

    return image_paths