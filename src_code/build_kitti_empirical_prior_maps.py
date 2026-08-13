import argparse
import json
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)

from split_utils import load_split_names


SIZE_FEATURES = [
    "area_fraction",
    "bbox_width_fraction",
    "bbox_height_fraction",
    "bbox_area_fraction",
    "bbox_aspect_ratio",
]


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def finite(value):
    return value is not None and np.isfinite(value)


def filter_records(records, min_valid_depth_fraction=0.1, exclude_border=True):
    filtered = []
    for record in records:
        target = record.get("median_disparity")
        if not finite(target) or target <= 0:
            continue
        if record.get("valid_depth_fraction", 0.0) < min_valid_depth_fraction:
            continue
        if exclude_border and record.get("touches_border", False):
            continue
        if any(not finite(record.get(name)) for name in SIZE_FEATURES):
            continue
        filtered.append(record)
    return filtered


def build_size_matrix(records):
    x = np.empty((len(records), 1 + len(SIZE_FEATURES)), dtype=object)
    y = np.empty(len(records), dtype=np.float64)

    for row, record in enumerate(records):
        x[row, 0] = record.get("instance_class_name", "unknown")
        for column, feature in enumerate(SIZE_FEATURES, start=1):
            x[row, column] = float(record[feature])
        y[row] = float(record["median_disparity"])

    return x, y


def make_size_model(records, trees, min_leaf, seed):
    x, y = build_size_matrix(records)
    preprocessing = ColumnTransformer(
        [
            ("class", OneHotEncoder(handle_unknown="ignore"), [0]),
            ("numeric", StandardScaler(), list(range(1, 1 + len(SIZE_FEATURES)))),
        ]
    )
    model = Pipeline(
        [
            ("features", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=trees,
                    min_samples_leaf=min_leaf,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def class_statistics(records, min_class_records):
    grouped = {}
    for record in records:
        class_name = record.get("instance_class_name", "unknown")
        grouped.setdefault(class_name, []).append(float(record["median_disparity"]))

    medians = {}
    counts = {}
    for class_name, values in grouped.items():
        if len(values) >= min_class_records:
            medians[class_name] = float(np.median(values))
            counts[class_name] = len(values)
    return medians, counts


def detection_features(mask, box, height, width):
    area_pixels = int(mask.sum())
    x1, y1, x2, y2 = [float(value) for value in box]
    bbox_width = max(0.0, x2 - x1)
    bbox_height = max(0.0, y2 - y1)

    return {
        "area_fraction": float(area_pixels / (height * width)),
        "bbox_width_fraction": float(bbox_width / width),
        "bbox_height_fraction": float(bbox_height / height),
        "bbox_area_fraction": float(bbox_width * bbox_height / (height * width)),
        "bbox_aspect_ratio": float(bbox_width / max(bbox_height, 1e-6)),
    }


def size_prediction(model, class_name, features):
    x = np.empty((1, 1 + len(SIZE_FEATURES)), dtype=object)
    x[0, 0] = class_name
    for column, feature in enumerate(SIZE_FEATURES, start=1):
        x[0, column] = float(features[feature])
    return float(model.predict(x)[0])


def mask_touches_border(mask):
    return bool(
        mask[0].any()
        or mask[-1].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def load_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detector_output(image, model, transform, device):
    tensor = transform(Image.fromarray(image)).to(device)
    with torch.no_grad():
        output = model([tensor])[0]
    return {key: value.detach().cpu() for key, value in output.items()}


def build_maps_for_frame(
    image,
    output,
    categories,
    class_medians,
    size_model,
    score_threshold,
    mask_threshold,
    min_mask_pixels,
    exclude_border,
):
    height, width = image.shape[:2]
    class_prior = np.zeros((height, width), dtype=np.float32)
    size_prior = np.zeros((height, width), dtype=np.float32)
    valid = np.zeros((height, width), dtype=np.uint8)
    confidence = np.zeros((height, width), dtype=np.float32)

    for index, score_tensor in enumerate(output["scores"]):
        score = float(score_tensor.item())
        if score < score_threshold:
            continue

        mask = output["masks"][index, 0].numpy() >= mask_threshold
        if int(mask.sum()) < min_mask_pixels:
            continue
        if exclude_border and mask_touches_border(mask):
            continue

        label_id = int(output["labels"][index].item())
        class_name = categories[label_id]
        if class_name not in class_medians:
            continue

        features = detection_features(
            mask,
            output["boxes"][index].numpy(),
            height,
            width,
        )
        predicted_size_disparity = max(
            0.0,
            size_prediction(size_model, class_name, features),
        )
        predicted_class_disparity = max(0.0, class_medians[class_name])

        write_mask = mask & (score >= confidence)
        class_prior[write_mask] = predicted_class_disparity
        size_prior[write_mask] = predicted_size_disparity
        confidence[write_mask] = score
        valid[write_mask] = 1

    return class_prior, size_prior, valid, confidence


def save_frame_maps(path, class_prior, size_prior, valid, confidence):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        class_prior=class_prior.astype(np.float32),
        size_prior=size_prior.astype(np.float32),
        valid=valid.astype(np.uint8),
        confidence=confidence.astype(np.float32),
    )


def make_frame_folds(frame_names, folds, seed):
    frame_names = sorted(frame_names)
    rng = random.Random(seed)
    shuffled = frame_names.copy()
    rng.shuffle(shuffled)

    result = [[] for _ in range(folds)]
    for index, frame in enumerate(shuffled):
        result[index % folds].append(frame)
    return result


def records_excluding_frames(records, excluded):
    excluded = set(excluded)
    return [record for record in records if record.get("frame") not in excluded]


def generate_frames(
    frame_names,
    image_dir,
    output_dir,
    detector,
    detector_transform,
    detector_categories,
    device,
    class_medians,
    size_model,
    args,
    desc,
):
    detections = 0
    covered_pixels = 0
    total_pixels = 0

    for frame in tqdm(frame_names, desc=desc, unit="frame"):
        image_path = os.path.join(image_dir, frame)
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        image = load_rgb(image_path)
        output = detector_output(image, detector, detector_transform, device)
        class_prior, size_prior, valid, confidence = build_maps_for_frame(
            image=image,
            output=output,
            categories=detector_categories,
            class_medians=class_medians,
            size_model=size_model,
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            min_mask_pixels=args.min_mask_pixels,
            exclude_border=args.exclude_border,
        )

        detections += int(np.unique(size_prior[size_prior > 0]).size)
        covered_pixels += int(valid.sum())
        total_pixels += int(valid.size)

        output_path = os.path.join(
            output_dir,
            os.path.splitext(frame)[0] + ".npz",
        )
        save_frame_maps(output_path, class_prior, size_prior, valid, confidence)

    coverage = covered_pixels / max(total_pixels, 1)
    print(f"Saved {len(frame_names)} prior maps to {output_dir}")
    print(f"Approximate pixel coverage: {coverage * 100:.2f}%")
    return detections, coverage


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    train_output = os.path.join(args.output_dir, "train")
    val_output = os.path.join(args.output_dir, "val")
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)

    raw_records = load_records(args.train_records)
    records = filter_records(
        raw_records,
        min_valid_depth_fraction=args.min_valid_depth_fraction,
        exclude_border=args.exclude_border,
    )
    print(f"Usable training object records: {len(records)}")

    train_frames = load_split_names(args.split_file, "train")
    val_frames = load_split_names(args.split_file, "val")
    print(f"Train frames: {len(train_frames)} | Validation frames: {len(val_frames)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running detector on device: {device}")
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    detector = maskrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
    detector_transform = weights.transforms()
    detector_categories = weights.meta["categories"]

    folds = make_frame_folds(train_frames, args.folds, args.seed)
    fold_metadata = {}

    for fold_index, heldout_frames in enumerate(folds, start=1):
        fold_records = records_excluding_frames(records, heldout_frames)
        class_medians, class_counts = class_statistics(
            fold_records,
            args.min_class_records,
        )
        eligible_records = [
            record
            for record in fold_records
            if record.get("instance_class_name") in class_medians
        ]
        if len(eligible_records) < 20:
            raise ValueError(f"Fold {fold_index} has insufficient training records.")

        size_model = make_size_model(
            eligible_records,
            trees=args.trees,
            min_leaf=args.min_leaf,
            seed=args.seed + fold_index,
        )
        print(
            f"Fold {fold_index}/{args.folds}: "
            f"fit on {len(eligible_records)} objects, "
            f"predicting {len(heldout_frames)} train frames, "
            f"classes={len(class_medians)}"
        )
        generate_frames(
            frame_names=heldout_frames,
            image_dir=args.image_dir,
            output_dir=train_output,
            detector=detector,
            detector_transform=detector_transform,
            detector_categories=detector_categories,
            device=device,
            class_medians=class_medians,
            size_model=size_model,
            args=args,
            desc=f"Cross-fit fold {fold_index}",
        )
        fold_metadata[str(fold_index)] = {
            "heldout_frames": heldout_frames,
            "training_objects": len(eligible_records),
            "class_counts": class_counts,
        }

    class_medians, class_counts = class_statistics(records, args.min_class_records)
    eligible_records = [
        record
        for record in records
        if record.get("instance_class_name") in class_medians
    ]
    size_model = make_size_model(
        eligible_records,
        trees=args.trees,
        min_leaf=args.min_leaf,
        seed=args.seed,
    )
    print(
        f"Validation prior model: fit on {len(eligible_records)} objects, "
        f"classes={len(class_medians)}"
    )
    generate_frames(
        frame_names=val_frames,
        image_dir=args.image_dir,
        output_dir=val_output,
        detector=detector,
        detector_transform=detector_transform,
        detector_categories=detector_categories,
        device=device,
        class_medians=class_medians,
        size_model=size_model,
        args=args,
        desc="Validation prior maps",
    )

    metadata = {
        "description": (
            "Cross-fitted KITTI empirical prior maps. Training maps are generated "
            "without using target disparity records from their held-out frame fold."
        ),
        "train_records": args.train_records,
        "split_file": args.split_file,
        "folds": args.folds,
        "seed": args.seed,
        "size_features": SIZE_FEATURES,
        "min_valid_depth_fraction": args.min_valid_depth_fraction,
        "exclude_border": args.exclude_border,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "min_mask_pixels": args.min_mask_pixels,
        "min_class_records": args.min_class_records,
        "full_training_class_counts": class_counts,
        "fold_metadata": fold_metadata,
    }
    with open(
        os.path.join(args.output_dir, "metadata.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nDone.")
    print(f"Training maps:   {train_output}")
    print(f"Validation maps: {val_output}")
    print(
        "Each .npz contains class_prior, size_prior, valid, and confidence maps."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-controlled KITTI semantic-class and class+apparent-size "
            "disparity-prior maps from the empirical training corpus."
        )
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--train-records", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--min-leaf", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-valid-depth-fraction", type=float, default=0.1)
    parser.add_argument("--exclude-border", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-class-records", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-mask-pixels", type=int, default=50)
    main(parser.parse_args())
