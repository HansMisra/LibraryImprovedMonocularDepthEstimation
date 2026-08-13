import argparse
import json

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SIZE_FEATURES = [
    "area_fraction",
    "bbox_width_fraction",
    "bbox_height_fraction",
    "bbox_area_fraction",
    "bbox_aspect_ratio",
]

FOCUS_FEATURES = [
    "sharp_laplacian_variance",
    "sharp_tenengrad_mean",
    "sharp_gradient_p90",
    "sharp_high_frequency_rms",
    "sharp_local_contrast",
    "sharp_boundary_gradient_p90",
    "sharp_line_tightness",
    "texture_entropy",
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


def filter_records(records, min_valid_depth_fraction, exclude_border):
    filtered = []
    for record in records:
        target = record.get("median_disparity")
        if not finite(target) or target <= 0:
            continue
        if record.get("valid_depth_fraction", 0.0) < min_valid_depth_fraction:
            continue
        if exclude_border and record.get("touches_border", False):
            continue
        filtered.append(record)
    return filtered


def feature_names(variant):
    if variant == "class":
        return []
    if variant == "class+size":
        return SIZE_FEATURES
    if variant == "class+size+focus":
        return SIZE_FEATURES + FOCUS_FEATURES
    raise ValueError(variant)


def build_matrix(records, numeric_features):
    usable = []
    for record in records:
        if any(not finite(record.get(feature)) for feature in numeric_features):
            continue
        usable.append(record)

    x = np.empty((len(usable), 1 + len(numeric_features)), dtype=object)
    y = np.empty(len(usable), dtype=np.float64)

    for row_index, record in enumerate(usable):
        x[row_index, 0] = record.get("instance_class_name", "unknown")
        for column_index, feature in enumerate(numeric_features, start=1):
            x[row_index, column_index] = float(record[feature])
        y[row_index] = float(record["median_disparity"])

    return x, y, usable


def make_model(numeric_count, args):
    transformers = [
        ("class", OneHotEncoder(handle_unknown="ignore"), [0]),
    ]

    if numeric_count:
        numeric_columns = list(range(1, numeric_count + 1))
        transformers.append(("numeric", StandardScaler(), numeric_columns))

    preprocessing = ColumnTransformer(transformers)

    return Pipeline(
        [
            ("features", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=args.trees,
                    min_samples_leaf=args.min_leaf,
                    random_state=args.seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def metrics(target, prediction):
    error = np.abs(prediction - target)
    return {
        "n": int(target.size),
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "relative": float(np.mean(error / np.maximum(target, 1e-6))),
    }


def evaluate_variant(train_records, val_records, variant, args):
    numeric_features = feature_names(variant)
    x_train, y_train, _ = build_matrix(train_records, numeric_features)
    x_val, y_val, val_used = build_matrix(val_records, numeric_features)

    if len(y_train) < 20 or len(y_val) < 5:
        return None

    model = make_model(len(numeric_features), args)
    model.fit(x_train, y_train)
    prediction = model.predict(x_val)

    result = metrics(y_val, prediction)
    result["prediction"] = prediction
    result["target"] = y_val
    result["records"] = val_used
    return result


def print_overall(name, result):
    print(
        f"{name:<18} n={result['n']:>3} | "
        f"MAE={result['mae']:.4f} px | "
        f"RMSE={result['rmse']:.4f} px | "
        f"relative={result['relative']:.4f}"
    )


def print_per_class(results, min_class_objects):
    print("\nPer-class validation MAE")
    classes = sorted(
        {
            record.get("instance_class_name", "unknown")
            for result in results.values()
            if result is not None
            for record in result["records"]
        }
    )

    for class_name in classes:
        line = [class_name]
        valid_class = False
        for variant in ("class", "class+size", "class+size+focus"):
            result = results.get(variant)
            if result is None:
                line.append("n/a")
                continue

            indices = [
                index
                for index, record in enumerate(result["records"])
                if record.get("instance_class_name") == class_name
            ]
            if len(indices) < min_class_objects:
                line.append("n/a")
                continue

            valid_class = True
            idx = np.asarray(indices, dtype=int)
            mae = mean_absolute_error(
                result["target"][idx],
                result["prediction"][idx],
            )
            line.append(f"{mae:.3f}px (n={len(indices)})")

        if valid_class:
            print(
                f"  {line[0]:<16} class={line[1]:<18} "
                f"size={line[2]:<18} focus={line[3]}"
            )


def main(args):
    train_records = filter_records(
        load_records(args.train_records),
        args.min_valid_depth_fraction,
        args.exclude_border,
    )
    val_records = filter_records(
        load_records(args.val_records),
        args.min_valid_depth_fraction,
        args.exclude_border,
    )

    print(f"Usable training objects before feature completeness checks: {len(train_records)}")
    print(f"Usable validation objects before feature completeness checks: {len(val_records)}")
    print()

    results = {}
    for variant in ("class", "class+size", "class+size+focus"):
        result = evaluate_variant(train_records, val_records, variant, args)
        results[variant] = result
        if result is None:
            print(f"{variant}: insufficient usable data")
        else:
            print_overall(variant, result)

    class_result = results.get("class")
    size_result = results.get("class+size")
    focus_result = results.get("class+size+focus")

    if class_result and size_result:
        improvement = 1.0 - size_result["mae"] / class_result["mae"]
        print(
            f"\nMAE improvement from adding size beyond class: "
            f"{improvement * 100:.2f}%"
        )

    if size_result and focus_result:
        improvement = 1.0 - focus_result["mae"] / size_result["mae"]
        print(
            f"MAE improvement from adding focus/texture beyond class+size: "
            f"{improvement * 100:.2f}%"
        )

    print_per_class(results, args.min_class_objects)

    print(
        "\nInterpretation: a positive focus/texture improvement means the measured "
        "sharpness cues carry held-out disparity information beyond semantic class "
        "and apparent size. This is still an object-level cue ablation, not yet a "
        "dense MDE-model improvement measurement."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Held-out KITTI ablation: class vs class+size vs class+size+focus/texture."
        )
    )
    parser.add_argument("--train-records", required=True)
    parser.add_argument("--val-records", required=True)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--min-leaf", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-valid-depth-fraction", type=float, default=0.1)
    parser.add_argument("--exclude-border", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-class-objects", type=int, default=5)
    main(parser.parse_args())
