import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def read_jsonl(path):
    records = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def load_disparity(path):
    disparity = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if disparity is None:
        raise ValueError(f"Could not read disparity: {path}")

    if disparity.ndim == 3:
        disparity = cv2.cvtColor(disparity, cv2.COLOR_BGR2GRAY)

    return disparity.astype(np.float32)


def load_semantic_map(path):
    return np.load(path).astype(np.int32)


def train_semantic_median_baseline(
    manifest_path,
    output_path,
    max_records=None,
    max_pixels_per_image=50000,
    seed=7,
):
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)

    records = read_jsonl(manifest_path)

    if max_records is not None:
        records = records[:max_records]

    if len(records) == 0:
        raise ValueError(f"No records found in manifest: {manifest_path}")

    rng = np.random.default_rng(seed)

    class_chunks = {}
    global_chunks = []

    start = time.perf_counter()

    for idx, record in enumerate(records, start=1):
        sample_id = record["sample_id"]

        semantic_map = load_semantic_map(record["semantic_map"])
        disparity = load_disparity(record["pseudo_disparity"])

        if semantic_map.shape[:2] != disparity.shape[:2]:
            raise ValueError(
                f"{sample_id}: semantic shape {semantic_map.shape} "
                f"does not match disparity shape {disparity.shape}"
            )

        valid = np.isfinite(disparity) & (disparity > 0)
        valid_indices = np.flatnonzero(valid.ravel())

        if len(valid_indices) == 0:
            print(f"[{idx}/{len(records)}] skipped {sample_id}: no valid disparity")
            continue

        if max_pixels_per_image is not None and len(valid_indices) > max_pixels_per_image:
            valid_indices = rng.choice(
                valid_indices,
                size=max_pixels_per_image,
                replace=False,
            )

        flat_semantic = semantic_map.ravel()[valid_indices]
        flat_disparity = disparity.ravel()[valid_indices]

        global_chunks.append(flat_disparity)

        for class_id in np.unique(flat_semantic):
            class_values = flat_disparity[flat_semantic == class_id]
            class_key = str(int(class_id))

            if class_key not in class_chunks:
                class_chunks[class_key] = []

            class_chunks[class_key].append(class_values)

        print(f"[{idx}/{len(records)}] sampled {sample_id}")

    if not global_chunks:
        raise ValueError("No valid disparity samples found.")

    global_values = np.concatenate(global_chunks)
    global_median = float(np.median(global_values))

    class_medians = {}

    for class_key, chunks in class_chunks.items():
        values = np.concatenate(chunks)
        class_medians[class_key] = float(np.median(values))

    elapsed = time.perf_counter() - start

    model = {
        "model_type": "semantic_class_median_disparity",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest_path": str(manifest_path),
        "records_used": len(records),
        "max_pixels_per_image": max_pixels_per_image,
        "global_median": global_median,
        "class_medians": class_medians,
        "train_seconds": elapsed,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)

    print(f"Saved semantic median baseline to: {output_path}")
    print(f"Classes learned: {len(class_medians)}")
    print(f"Training runtime: {elapsed:.3f} seconds")

    return model


def load_model(model_path):
    with Path(model_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def predict_disparity_from_semantics(semantic_map, model):
    semantic_map = semantic_map.astype(np.int32)

    class_medians = {
        int(class_id): float(value)
        for class_id, value in model["class_medians"].items()
    }

    max_class_in_map = int(semantic_map.max())
    max_class_in_model = max(class_medians.keys()) if class_medians else 0
    lookup_size = max(max_class_in_map, max_class_in_model) + 1

    lookup = np.full(
        lookup_size,
        float(model["global_median"]),
        dtype=np.float32,
    )

    for class_id, value in class_medians.items():
        if class_id < lookup_size:
            lookup[class_id] = value

    return lookup[semantic_map]


def compute_regression_metrics(prediction, target):
    valid = np.isfinite(target) & (target > 0)

    if not np.any(valid):
        return None

    error = prediction[valid] - target[valid]

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))

    return {
        "mae": mae,
        "rmse": rmse,
        "valid_pixels": int(np.sum(valid)),
    }


def get_path_roi(shape, bottom_fraction=0.55, center_fraction=0.60):
    height, width = shape[:2]

    y0 = int(height * (1.0 - bottom_fraction))
    y1 = height

    x_margin = int(width * (1.0 - center_fraction) / 2.0)
    x0 = x_margin
    x1 = width - x_margin

    return y0, y1, x0, x1


def compute_risk_score(disparity, risk_percentile=95):
    valid = np.isfinite(disparity) & (disparity > 0)

    if not np.any(valid):
        return 0.0

    return float(np.percentile(disparity[valid], risk_percentile))


def estimate_risk_threshold(
    records,
    risk_percentile=95,
    threshold_percentile=85,
):
    scores = []

    for record in records:
        disparity = load_disparity(record["pseudo_disparity"])
        y0, y1, x0, x1 = get_path_roi(disparity.shape)
        roi_disparity = disparity[y0:y1, x0:x1]

        score = compute_risk_score(
            roi_disparity,
            risk_percentile=risk_percentile,
        )

        scores.append(score)

    if not scores:
        return 0.0

    return float(np.percentile(np.array(scores, dtype=np.float32), threshold_percentile))


def _time_stats(values):
    if len(values) == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
        }

    array = np.array(values, dtype=np.float64)

    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def save_prediction_heatmap(prediction, output_path):
    pred = prediction.astype(np.float32)
    min_value = float(np.min(pred))
    max_value = float(np.max(pred))

    if max_value <= min_value:
        normalized = np.zeros_like(pred, dtype=np.uint8)
    else:
        normalized = ((pred - min_value) / (max_value - min_value) * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_PLASMA)
    cv2.imwrite(str(output_path), heatmap)


def evaluate_semantic_median_baseline(
    manifest_path,
    model_path,
    output_dir,
    max_records=None,
    risk_percentile=95,
    threshold_percentile=85,
    risk_threshold=None,
    save_predictions=False,
):
    manifest_path = Path(manifest_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)

    records = read_jsonl(manifest_path)

    if max_records is not None:
        records = records[:max_records]

    if len(records) == 0:
        raise ValueError(f"No records found in manifest: {manifest_path}")

    model = load_model(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if risk_threshold is None:
        risk_threshold = estimate_risk_threshold(
            records=records,
            risk_percentile=risk_percentile,
            threshold_percentile=threshold_percentile,
        )

    rows = []
    predict_times = []

    tp = fp = tn = fn = 0
    mae_values = []
    rmse_values = []

    total_start = time.perf_counter()

    for idx, record in enumerate(records, start=1):
        sample_id = record["sample_id"]

        semantic_map = load_semantic_map(record["semantic_map"])
        target_disparity = load_disparity(record["pseudo_disparity"])

        pred_start = time.perf_counter()
        prediction = predict_disparity_from_semantics(semantic_map, model)
        pred_seconds = time.perf_counter() - pred_start

        predict_times.append(pred_seconds)

        metrics = compute_regression_metrics(prediction, target_disparity)

        y0, y1, x0, x1 = get_path_roi(target_disparity.shape)

        target_roi = target_disparity[y0:y1, x0:x1]
        prediction_roi = prediction[y0:y1, x0:x1]

        true_score = compute_risk_score(
            target_roi,
            risk_percentile=risk_percentile,
        )
        pred_score = compute_risk_score(
            prediction_roi,
            risk_percentile=risk_percentile,
        )

        true_risk = true_score >= risk_threshold
        pred_risk = pred_score >= risk_threshold

        if true_risk and pred_risk:
            tp += 1
        elif not true_risk and pred_risk:
            fp += 1
        elif not true_risk and not pred_risk:
            tn += 1
        elif true_risk and not pred_risk:
            fn += 1

        mae = metrics["mae"] if metrics else None
        rmse = metrics["rmse"] if metrics else None

        if mae is not None:
            mae_values.append(mae)

        if rmse is not None:
            rmse_values.append(rmse)

        if save_predictions:
            pred_path = output_dir / f"{sample_id}_semantic_baseline_heatmap.png"
            save_prediction_heatmap(prediction, pred_path)

        rows.append(
            {
                "sample_id": sample_id,
                "mae": mae,
                "rmse": rmse,
                "prediction_seconds": pred_seconds,
                "true_risk_score": true_score,
                "pred_risk_score": pred_score,
                "risk_threshold": risk_threshold,
                "true_risk": int(true_risk),
                "pred_risk": int(pred_risk),
            }
        )

        print(
            f"[{idx}/{len(records)}] {sample_id} "
            f"rmse={rmse:.4f} pred_time={pred_seconds:.6f}s "
            f"risk={int(pred_risk)}/{int(true_risk)}"
        )

    total_seconds = time.perf_counter() - total_start

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    summary = {
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "records_evaluated": len(records),
        "total_seconds": total_seconds,
        "prediction_time_seconds": _time_stats(predict_times),
        "mae_mean": float(np.mean(mae_values)) if mae_values else None,
        "rmse_mean": float(np.mean(rmse_values)) if rmse_values else None,
        "rmse_std": float(np.std(rmse_values)) if rmse_values else None,
        "risk_percentile": risk_percentile,
        "risk_threshold": risk_threshold,
        "risk_confusion": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "risk_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }

    summary_path = output_dir / "semantic_baseline_eval_summary.json"
    rows_path = output_dir / "semantic_baseline_eval_rows.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with rows_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id",
            "mae",
            "rmse",
            "prediction_seconds",
            "true_risk_score",
            "pred_risk_score",
            "risk_threshold",
            "true_risk",
            "pred_risk",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved eval summary to: {summary_path}")
    print(f"Saved eval rows to: {rows_path}")
    print(f"Prediction time stats: {summary['prediction_time_seconds']}")
    print(f"Risk metrics: {summary['risk_metrics']}")

    return summary