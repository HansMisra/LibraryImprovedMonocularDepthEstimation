import argparse
import json

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SIZE_CANDIDATES = [
    "bbox_angular_height_rad",
    "bbox_angular_width_rad",
    "angular_area_proxy",
    "bbox_height_fraction",
    "bbox_width_fraction",
    "area_fraction",
]

SHARPNESS_FEATURES = [
    "sharp_laplacian_variance",
    "sharp_tenengrad_mean",
    "sharp_gradient_p90",
    "sharp_high_frequency_rms",
    "sharp_local_contrast",
    "sharp_boundary_gradient_p90",
    "sharp_line_tightness",
    "texture_entropy",
]


def load_records(paths):
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def finite(value):
    return value is not None and np.isfinite(value)


def choose_size_feature(train_records):
    counts = {
        feature: sum(finite(r.get(feature)) and r.get(feature, 0) > 0 for r in train_records)
        for feature in SIZE_CANDIDATES
    }
    return max(counts, key=counts.get)


def make_rows(records, size_feature, feature_set, min_valid_depth_fraction):
    rows = []
    targets = []
    for r in records:
        target = r.get("median_depth_m")
        if not finite(target) or target <= 0:
            continue
        if r.get("valid_depth_fraction", 0.0) < min_valid_depth_fraction:
            continue

        row = {"class": r.get("instance_class_name", "unknown")}
        if feature_set in ("size", "size+sharpness"):
            size = r.get(size_feature)
            if not finite(size) or size <= 0:
                continue
            row[size_feature] = float(size)

        if feature_set == "size+sharpness":
            missing = False
            for feature in SHARPNESS_FEATURES:
                value = r.get(feature)
                if not finite(value):
                    missing = True
                    break
                row[feature] = float(value)
            if missing:
                continue

        rows.append(row)
        targets.append(float(target))
    return rows, np.asarray(targets, dtype=np.float64)


def rows_to_matrix(rows, numeric_features):
    data = np.empty((len(rows), 1 + len(numeric_features)), dtype=object)
    for i, row in enumerate(rows):
        data[i, 0] = row["class"]
        for j, feature in enumerate(numeric_features, start=1):
            data[i, j] = row[feature]
    return data


def evaluate_variant(train_records, val_records, size_feature, feature_set, args):
    numeric = []
    if feature_set in ("size", "size+sharpness"):
        numeric.append(size_feature)
    if feature_set == "size+sharpness":
        numeric.extend(SHARPNESS_FEATURES)

    train_rows, y_train = make_rows(train_records, size_feature, feature_set, args.min_valid_depth_fraction)
    val_rows, y_val = make_rows(val_records, size_feature, feature_set, args.min_valid_depth_fraction)
    if len(train_rows) < 20 or len(val_rows) < 5:
        return None

    x_train = rows_to_matrix(train_rows, numeric)
    x_val = rows_to_matrix(val_rows, numeric)

    transformer = ColumnTransformer(
        [("class", OneHotEncoder(handle_unknown="ignore"), [0])],
        remainder="passthrough",
    )
    model = Pipeline(
        [
            ("features", transformer),
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
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    error = np.abs(pred - y_val)
    return {
        "n_train": len(y_train),
        "n_val": len(y_val),
        "mae": float(mean_absolute_error(y_val, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_val, pred))),
        "mean_relative_error": float(np.mean(error / np.maximum(y_val, 1e-6))),
    }


def main(args):
    train_records = load_records(args.train_records)
    val_records = load_records(args.val_records)
    size_feature = choose_size_feature(train_records)
    print(f"Selected size feature: {size_feature}")

    results = {}
    for variant in ("class", "size", "size+sharpness"):
        result = evaluate_variant(train_records, val_records, size_feature, variant, args)
        results[variant] = result
        if result is None:
            print(f"{variant}: insufficient usable data")
            continue
        print(
            f"{variant}: n={result['n_val']} | MAE={result['mae']:.4f} m | "
            f"RMSE={result['rmse']:.4f} m | rel={result['mean_relative_error']:.4f}"
        )

    if results.get("size") and results.get("size+sharpness"):
        improvement = 1.0 - results["size+sharpness"]["mae"] / results["size"]["mae"]
        print(f"Sharpness MAE improvement beyond class+size: {improvement * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-records", nargs="+", required=True)
    parser.add_argument("--val-records", nargs="+", required=True)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--min-leaf", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-valid-depth-fraction", type=float, default=0.1)
    main(parser.parse_args())
