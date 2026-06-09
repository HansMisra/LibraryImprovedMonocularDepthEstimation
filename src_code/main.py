import argparse
import os

from train import train
from run_evaluation import run_evaluation
from create_test_disp import save_disparity_maps


def get_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    paths = {
        "script_dir": script_dir,
        "train_images": os.path.join(
            script_dir,
            "kitti_data",
            "data_scene_flow",
            "training",
            "image_2"
        ),
        "train_disparities": os.path.join(
            script_dir,
            "kitti_data",
            "data_scene_flow",
            "training",
            "disp_occ_0"
        ),
        "test_left_images": os.path.join(
            script_dir,
            "kitti_data",
            "data_scene_flow",
            "testing",
            "image_2"
        ),
        "test_right_images": os.path.join(
            script_dir,
            "kitti_data",
            "data_scene_flow",
            "testing",
            "image_3"
        ),
        "test_disparities": os.path.join(
            script_dir,
            "kitti_data",
            "data_scene_flow",
            "testing",
            "test_disp"
        ),
        "semantic_maps": os.path.join(
            script_dir,
            "outputs",
            "semantic_maps"
        ),
        "semantic_overlays": os.path.join(
            script_dir,
            "outputs",
            "semantic_overlays"
        ),
        "corpus_manifest": os.path.join(
            script_dir,
            "outputs",
            "corpus_manifest.jsonl"
        ),
        "model_weights": os.path.join(script_dir, "model_weights.pth")
    }

    return paths


def check_dir(path, label):
    if not os.path.exists(path):
        print(f"{label} not found: {path}")
        return False
    return True


def train_model(paths, epochs, batch_size):
    if not check_dir(paths["train_images"], "Training image directory"):
        return

    if not check_dir(paths["train_disparities"], "Training disparity directory"):
        return

    train(
        paths["train_images"],
        paths["train_disparities"],
        epochs=epochs,
        batch_size=batch_size,
        save_path=paths["model_weights"]
    )


def generate_test_disparities(paths):
    if not check_dir(paths["test_left_images"], "Test left image directory"):
        return

    if not check_dir(paths["test_right_images"], "Test right image directory"):
        return

    os.makedirs(paths["test_disparities"], exist_ok=True)

    save_disparity_maps(
        paths["test_left_images"],
        paths["test_right_images"],
        paths["test_disparities"]
    )


def generate_semantic_maps(
    image_dir,
    output_dir,
    limit,
    model_name,
    device,
    frame_suffix,
    skip_existing=True,
):
    from segmentation.extract_segmentation import generate_segmentation

    generate_segmentation(
        image_dir=image_dir,
        output_dir=output_dir,
        limit=limit,
        model_name=model_name,
        device=device,
        frame_suffix=frame_suffix,
        skip_existing=skip_existing,
    )

def build_corpus(paths, limit, frame_suffix):
    from corpus.build_corpus import build_corpus_manifest

    if frame_suffix == "none":
        frame_suffix = None

    build_corpus_manifest(
        left_image_dir=paths["test_left_images"],
        right_image_dir=paths["test_right_images"],
        disparity_dir=paths["test_disparities"],
        semantic_map_dir=paths["semantic_maps"],
        semantic_overlay_dir=paths["semantic_overlays"],
        output_path=paths["corpus_manifest"],
        limit=limit,
        require_all=True,
        frame_suffix=frame_suffix,
    )

def main():
    parser = argparse.ArgumentParser(
        description="KITTI depth/disparity estimation project entry point."
    )

    parser.add_argument(
        "command",
        choices=[
            "train",
            "generate-test-disp",
            "evaluate",
            "generate-segmentation",
            "build-corpus",
            "all"
        ],
        help="Pipeline step to run."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size."
    )

    parser.add_argument(
        "--image-dir",
        default=None,
        help="Directory containing input images for semantic segmentation."
    )

    parser.add_argument(
        "--frame-suffix",
        default="_10",
        help="Only use images whose filename stem ends with this suffix. Use 'none' to disable."
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where semantic maps will be saved."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke testing."
    )

    parser.add_argument(
        "--seg-model",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="Semantic segmentation model name."
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Device for segmentation: cuda, cpu, or auto if omitted."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate semantic outputs even if files already exist."
    )

    args = parser.parse_args()
    paths = get_paths()

    image_dir = args.image_dir or paths["test_left_images"]
    output_dir = args.output_dir or paths["semantic_maps"]

    if args.command == "train":
        train_model(paths, args.epochs, args.batch_size)

    elif args.command == "generate-test-disp":
        generate_test_disparities(paths)

    elif args.command == "evaluate":
        run_evaluation()

    elif args.command == "generate-segmentation":
        generate_semantic_maps(
            image_dir=image_dir,
            output_dir=output_dir,
            limit=args.limit,
            model_name=args.seg_model,
            device=args.device,
            frame_suffix=None if args.frame_suffix == "none" else args.frame_suffix,
            skip_existing=not args.overwrite,
        )

    elif args.command == "build-corpus":
        build_corpus(
            paths=paths,
            limit=args.limit,
            frame_suffix=args.frame_suffix,
        )

    elif args.command == "all":
        train_model(paths, args.epochs, args.batch_size)
        generate_test_disparities(paths)
        run_evaluation()



if __name__ == "__main__":
    main()