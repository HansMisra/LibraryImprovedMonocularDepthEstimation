import argparse
import os

from utils.runtime_utils import configure_thread_env, timed_command


def get_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    return {
        "script_dir": script_dir,

        "train_images": os.path.join(
            script_dir,
            "kitti_data",
            "data_scene_flow",
            "training",
            "image_2",
        ),
        "train_disparities": os.path.join(
            script_dir,
            "kitti_data",
            "data_scene_flow",
            "training",
            "disp_occ_0",
        ),

        "test_left_images": os.path.join(
            script_dir,
            "kitti_data",
            "data_scene_flow",
            "testing",
            "image_2",
        ),
        "test_right_images": os.path.join(
            script_dir,
            "kitti_data",
            "data_scene_flow",
            "testing",
            "image_3",
        ),
        "test_disparities": os.path.join(
            script_dir,
            "kitti_data",
            "data_scene_flow",
            "testing",
            "test_disp",
        ),

        "semantic_maps": os.path.join(
            script_dir,
            "outputs",
            "semantic_maps",
        ),
        "semantic_overlays": os.path.join(
            script_dir,
            "outputs",
            "semantic_overlays",
        ),
        "corpus_manifest": os.path.join(
            script_dir,
            "outputs",
            "corpus_manifest.jsonl",
        ),
        "corpus_visualizations": os.path.join(
            script_dir,
            "outputs",
            "corpus_visualizations",
        ),

        "semantic_baseline_model": os.path.join(
            script_dir,
            "outputs",
            "models",
            "semantic_median_baseline.json",
        ),
        "semantic_baseline_eval": os.path.join(
            script_dir,
            "outputs",
            "semantic_baseline_eval",
        ),

        "runtime_log": os.path.join(
            script_dir,
            "outputs",
            "runtime_log.csv",
        ),

        "model_weights": os.path.join(script_dir, "model_weights.pth"),
    }


def normalize_frame_suffix(frame_suffix):
    if frame_suffix == "none":
        return None

    return frame_suffix


def check_dir(path, label):
    if not os.path.exists(path):
        print(f"{label} not found: {path}")
        return False

    return True


def train_model(paths, epochs, batch_size, num_threads):
    from utils.runtime_utils import apply_library_thread_limits

    apply_library_thread_limits(num_threads)

    from train import train

    if not check_dir(paths["train_images"], "Training image directory"):
        return

    if not check_dir(paths["train_disparities"], "Training disparity directory"):
        return

    train(
        paths["train_images"],
        paths["train_disparities"],
        epochs=epochs,
        batch_size=batch_size,
        save_path=paths["model_weights"],
    )


def generate_test_disparities(paths, num_threads):
    from utils.runtime_utils import apply_library_thread_limits

    apply_library_thread_limits(num_threads)

    from create_test_disp import save_disparity_maps

    if not check_dir(paths["test_left_images"], "Test left image directory"):
        return

    if not check_dir(paths["test_right_images"], "Test right image directory"):
        return

    os.makedirs(paths["test_disparities"], exist_ok=True)

    save_disparity_maps(
        paths["test_left_images"],
        paths["test_right_images"],
        paths["test_disparities"],
    )


def run_depth_evaluation(num_threads):
    from utils.runtime_utils import apply_library_thread_limits

    apply_library_thread_limits(num_threads)

    from run_evaluation import run_evaluation

    run_evaluation()


def generate_semantic_maps(
    image_dir,
    output_dir,
    limit,
    model_name,
    device,
    frame_suffix,
    skip_existing,
    num_threads,
):
    from utils.runtime_utils import apply_library_thread_limits

    apply_library_thread_limits(num_threads)

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


def validate_corpus(paths, limit):
    from corpus.validate_corpus import validate_corpus_manifest

    validate_corpus_manifest(
        manifest_path=paths["corpus_manifest"],
        max_records=limit,
    )


def visualize_corpus(paths, limit):
    from corpus.visualize_corpus import save_corpus_visualizations

    save_corpus_visualizations(
        manifest_path=paths["corpus_manifest"],
        output_dir=paths["corpus_visualizations"],
        limit=limit,
    )


def train_semantic_baseline(paths, args):
    from utils.runtime_utils import apply_library_thread_limits

    apply_library_thread_limits(args.num_threads)

    from baselines.semantic_median import train_semantic_median_baseline

    train_semantic_median_baseline(
        manifest_path=paths["corpus_manifest"],
        output_path=paths["semantic_baseline_model"],
        max_records=args.limit,
        max_pixels_per_image=args.max_pixels_per_image,
        seed=args.seed,
    )


def evaluate_semantic_baseline(paths, args):
    from utils.runtime_utils import apply_library_thread_limits

    apply_library_thread_limits(args.num_threads)

    from baselines.semantic_median import evaluate_semantic_median_baseline

    evaluate_semantic_median_baseline(
        manifest_path=paths["corpus_manifest"],
        model_path=paths["semantic_baseline_model"],
        output_dir=paths["semantic_baseline_eval"],
        max_records=args.limit,
        risk_percentile=args.risk_percentile,
        threshold_percentile=args.threshold_percentile,
        risk_threshold=args.risk_threshold,
        save_predictions=args.save_predictions,
    )


def build_parser():
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
            "validate-corpus",
            "visualize-corpus",
            "train-semantic-baseline",
            "evaluate-semantic-baseline",
            "all",
        ],
        help="Pipeline step to run.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size.",
    )

    parser.add_argument(
        "--image-dir",
        default=None,
        help="Directory containing input images for semantic segmentation.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where semantic maps will be saved.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke testing.",
    )

    parser.add_argument(
        "--frame-suffix",
        default="_10",
        help="Only use images whose filename stem ends with this suffix. Use 'none' to disable.",
    )

    parser.add_argument(
        "--seg-model",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="Semantic segmentation model name.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Device for segmentation: cuda, cpu, or auto if omitted.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate semantic outputs even if files already exist.",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Thread limit for CPU libraries where supported.",
    )

    parser.add_argument(
        "--runtime-log",
        default=None,
        help="Optional runtime CSV path. Defaults to src_code/outputs/runtime_log.csv.",
    )

    parser.add_argument(
        "--max-pixels-per-image",
        type=int,
        default=50000,
        help="Max sampled pixels per image for semantic baseline training.",
    )

    parser.add_argument(
        "--risk-percentile",
        type=float,
        default=95.0,
        help="Percentile of ROI disparity used as collision-risk score.",
    )

    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=85.0,
        help="Dataset percentile used to estimate risk threshold when none is supplied.",
    )

    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=None,
        help="Manual pseudo-disparity threshold for collision-risk decisions.",
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save semantic baseline prediction heatmaps during evaluation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for sampled baseline training.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    configure_thread_env(args.num_threads)

    paths = get_paths()

    image_dir = args.image_dir or paths["test_left_images"]
    output_dir = args.output_dir or paths["semantic_maps"]
    frame_suffix = normalize_frame_suffix(args.frame_suffix)
    runtime_log = args.runtime_log or paths["runtime_log"]

    metadata = {
        "limit": args.limit,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "frame_suffix": frame_suffix,
        "device": args.device,
        "num_threads": args.num_threads,
    }

    with timed_command(
        command=args.command,
        log_path=runtime_log,
        num_threads=args.num_threads,
        metadata=metadata,
    ):
        if args.command == "train":
            train_model(
                paths=paths,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_threads=args.num_threads,
            )

        elif args.command == "generate-test-disp":
            generate_test_disparities(
                paths=paths,
                num_threads=args.num_threads,
            )

        elif args.command == "evaluate":
            run_depth_evaluation(num_threads=args.num_threads)

        elif args.command == "generate-segmentation":
            generate_semantic_maps(
                image_dir=image_dir,
                output_dir=output_dir,
                limit=args.limit,
                model_name=args.seg_model,
                device=args.device,
                frame_suffix=frame_suffix,
                skip_existing=not args.overwrite,
                num_threads=args.num_threads,
            )

        elif args.command == "build-corpus":
            build_corpus(
                paths=paths,
                limit=args.limit,
                frame_suffix=frame_suffix,
            )

        elif args.command == "validate-corpus":
            validate_corpus(paths, args.limit)

        elif args.command == "visualize-corpus":
            visualize_corpus(paths, args.limit)

        elif args.command == "train-semantic-baseline":
            train_semantic_baseline(paths, args)

        elif args.command == "evaluate-semantic-baseline":
            evaluate_semantic_baseline(paths, args)

        elif args.command == "all":
            train_model(
                paths=paths,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_threads=args.num_threads,
            )
            generate_test_disparities(
                paths=paths,
                num_threads=args.num_threads,
            )
            run_depth_evaluation(num_threads=args.num_threads)


if __name__ == "__main__":
    main()