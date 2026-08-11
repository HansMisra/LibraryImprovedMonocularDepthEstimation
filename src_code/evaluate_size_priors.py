import argparse
import json

import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def predict_from_bins(feature_value, feature_prior):
    bins = feature_prior["bins"]
    if not bins:
        return None

    for i, bin_info in enumerate(bins):
        left = bin_info["min_feature"]
        right = bin_info["max_feature"]
        is_last = i == len(bins) - 1

        if left <= feature_value < right or (is_last and feature_value <= right):
            return bin_info["median_disparity"]

    if feature_value < bins[0]["min_feature"]:
        return bins[0]["median_disparity"]

    return bins[-1]["median_disparity"]


def metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = np.abs(pred - target)

    return {
        "n": int(target.size),
        "mae": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "median_abs_error": float(np.median(error)),
        "mean_relative_error": float(
            np.mean(error / np.maximum(target, 1e-6))
        ),
    }


def evaluate(
    priors_path,
    records_path,
    feature="area_fraction",
    min_valid_depth_fraction=0.1,
    exclude_border_touching=True,
):
    priors = load_json(priors_path)
    records = load_jsonl(records_path)

    target_values = []
    size_predictions = []
    class_predictions = []

    for record in records:
        class_name = record.get("instance_class_name")
        target = record.get("median_disparity")
        feature_value = record.get(feature)

        if class_name not in priors:
            continue
        if target is None or target <= 0:
            continue
        if feature_value is None or feature_value <= 0:
            continue
        if record.get("valid_depth_fraction", 0.0) < min_valid_depth_fraction:
            continue
        if exclude_border_touching and record.get("touches_border", False):
            continue

        class_prior = priors[class_name]
        feature_prior = class_prior["features"].get(feature)
        if feature_prior is None:
            continue

        size_pred = predict_from_bins(feature_value, feature_prior)
        if size_pred is None:
            continue

        target_values.append(target)
        size_predictions.append(size_pred)
        class_predictions.append(class_prior["median_disparity"])

    if not target_values:
        raise ValueError("No validation records matched the saved priors.")

    size_metrics = metrics(size_predictions, target_values)
    class_metrics = metrics(class_predictions, target_values)

    print(f"Feature: {feature}")
    print(f"Validation objects: {size_metrics['n']}")
    print("\nClass-only baseline")
    print(f"  MAE: {class_metrics['mae']:.4f} px")
    print(f"  RMSE: {class_metrics['rmse']:.4f} px")
    print(f"  Mean relative error: {class_metrics['mean_relative_error']:.4f}")

    print("\nClass + apparent-size prior")
    print(f"  MAE: {size_metrics['mae']:.4f} px")
    print(f"  RMSE: {size_metrics['rmse']:.4f} px")
    print(f"  Mean relative error: {size_metrics['mean_relative_error']:.4f}")

    mae_improvement = 1.0 - size_metrics["mae"] / class_metrics["mae"]
    print(f"\nMAE improvement over class-only: {mae_improvement * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--priors", default="semantic_corpus/train/size_priors.json")
    parser.add_argument("--records", default="semantic_corpus/val/instances.jsonl")
    parser.add_argument(
        "--feature",
        default="area_fraction",
        choices=[
            "area_fraction",
            "bbox_width_fraction",
            "bbox_height_fraction",
        ],
    )
    args = parser.parse_args()

    evaluate(
        priors_path=args.priors,
        records_path=args.records,
        feature=args.feature,
    )
