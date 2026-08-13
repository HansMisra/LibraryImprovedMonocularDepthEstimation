import argparse
import json

import numpy as np


SIZE_FEATURES = [
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


def finite_positive(value):
    return value is not None and np.isfinite(value) and value > 0


def robust_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {
        "count": int(values.size),
        "median": median,
        "mad": mad,
        "q10": float(np.quantile(values, 0.10)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
    }


def build(args):
    records = load_records(args.records)
    grouped = {}
    for r in records:
        if not finite_positive(r.get("median_depth_m")):
            continue
        if r.get("valid_depth_fraction", 0.0) < args.min_valid_depth_fraction:
            continue
        if args.exclude_border and r.get("touches_border", False):
            continue
        grouped.setdefault(r["instance_class_name"], []).append(r)

    priors = {}
    for class_name, class_records in sorted(grouped.items()):
        if len(class_records) < args.min_examples:
            continue

        size_feature = next(
            (
                feature
                for feature in SIZE_FEATURES
                if sum(finite_positive(r.get(feature)) for r in class_records) >= args.min_examples
            ),
            None,
        )
        if size_feature is None:
            continue

        usable = [r for r in class_records if finite_positive(r.get(size_feature))]
        size_values = np.array([r[size_feature] for r in usable], dtype=np.float64)
        edges = np.unique(np.quantile(size_values, np.linspace(0, 1, args.bins + 1)))
        if len(edges) < 3:
            continue

        class_prior = {
            "count": len(usable),
            "size_feature": size_feature,
            "depth_m": robust_stats([r["median_depth_m"] for r in usable]),
            "size_bins": [],
        }

        for i in range(len(edges) - 1):
            left, right = float(edges[i]), float(edges[i + 1])
            if i == len(edges) - 2:
                selected = [r for r in usable if left <= r[size_feature] <= right]
            else:
                selected = [r for r in usable if left <= r[size_feature] < right]
            if not selected:
                continue

            sharpness = {}
            for feature in SHARPNESS_FEATURES:
                values = [r.get(feature) for r in selected if finite_positive(r.get(feature))]
                stats = robust_stats(values)
                if stats is not None:
                    sharpness[feature] = stats

            class_prior["size_bins"].append(
                {
                    "min_size": left,
                    "max_size": right,
                    "count": len(selected),
                    "depth_m": robust_stats([r["median_depth_m"] for r in selected]),
                    "sharpness": sharpness,
                }
            )

        priors[class_name] = class_prior

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(priors, f, indent=2)

    print(f"Saved cue priors for {len(priors)} classes to {args.output}")
    for class_name, prior in priors.items():
        print(f"  {class_name}: n={prior['count']}, size={prior['size_feature']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--min-examples", type=int, default=20)
    parser.add_argument("--min-valid-depth-fraction", type=float, default=0.1)
    parser.add_argument("--exclude-border", action="store_true")
    build(parser.parse_args())
