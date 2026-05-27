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


def main():
    parser = argparse.ArgumentParser(
        description="KITTI depth/disparity estimation project entry point."
    )

    parser.add_argument(
        "command",
        choices=["train", "generate-test-disp", "evaluate", "all"],
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

    args = parser.parse_args()
    paths = get_paths()

    if args.command == "train":
        train_model(paths, args.epochs, args.batch_size)

    elif args.command == "generate-test-disp":
        generate_test_disparities(paths)

    elif args.command == "evaluate":
        run_evaluation()

    elif args.command == "all":
        train_model(paths, args.epochs, args.batch_size)
        generate_test_disparities(paths)
        run_evaluation()


if __name__ == "__main__":
    main()