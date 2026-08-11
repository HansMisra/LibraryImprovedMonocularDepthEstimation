import argparse
import json

import numpy as np


FEATURES = [
    "area_fraction",
    "bbox_width_fraction",
    "bbox_height_fraction",
]


def read_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def make_bins(values, targets, n_bins):
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)

    if len(edges) < 3:
        return None

    bins = []
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]

        if i == len(edges) - 2:
            mask = (values >= left) & (values <= right)
        else:
            mask = (values >= left) & (values < right)

        selected = targets[mask]
        if selected.size == 0:
            continue

        bins.append(
            {
                "min_feature": float(left),
                "max_feature": float(right),
                "count": int(selected.size),
                "median_disparity": float(np.median(selected)),
                "q10_disparity": float(np.quantile(selected, 0.10)),
                "q25_disparity": float(np.quantile(selected, 0.25)),
                "q75_disparity": float(np.quantile(selected, 0.75)),
                "q90_disparity": float(np.quantile(selected, 0.90)),
            }
        )

    return {
        "edges": [float(v) for v in edges],
        "bins": bins,
    }


def build_size_priors(
    records_path,
    output_path,
    n_bins=5,
    min_examples=20,
    min_valid_depth_fraction=0.1,
    exclude_border_touching=True,
):
    records = read_records(records_path)
    grouped = {}

    for record in records:
        disparity = record.get("median_disparity")
        if disparity is None or disparity <= 0:
            continue

        if record.get("valid_depth_fraction", 0.0) < min_valid_depth_fraction:
            continue

        if exclude_border_touching and record.get("touches_border", False):
            continue

        class_name = record["instance_class_name"]
        grouped.setdefault(class_name, []).append(record)

    priors = {}

    for class_name, class_records in sorted(grouped.items()):
        if len(class_records) < min_examples:
            continue

        targets = np.array(
            [r["median_disparity"] for r in class_records],
            dtype=np.float64,
        )

        class_prior = {
            "count": len(class_records),
            "median_disparity": float(np.median(targets)),
            "features": {},
        }

        for feature in FEATURES:
            values = np.array(
                [r[feature] for r in class_records],
                dtype=np.float64,
            )

            valid = (
                np.isfinite(values)
                & (values > 0)
                & np.isfinite(targets)
                & (targets > 0)
            )

            x = values[valid]
            y = targets[valid]
            if x.size < min_examples:
                continue

            log_corr = float(
                np.corrcoef(np.log(x), np.log(y))[0, 1]
            ) if np.unique(x).size > 1 and np.unique(y).size > 1 else None

            bins = make_bins(x, y, n_bins)
            if bins is None:
                continue

            class_prior["features"][feature] = {
                "log_feature_log_disparity_corr": log_corr,
                **bins,
            }

        if class_prior["features"]:
            priors[class_name] = class_prior

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(priors, f, indent=2)

    print(f"Saved priors for {len(priors)} classes to {output_path}")

    for class_name, prior in priors.items():
        print(f"\n{class_name} | n={prior['count']}")
        for feature, stats in prior["features"].items():
            corr = stats["log_feature_log_disparity_corr"]
            corr_text = "n/a" if corr is None else f"{corr:.3f}"
            print(f"  {feature}: log-log corr={corr_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="semantic_corpus/instances.jsonl")
    parser.add_argument("--output", default="semantic_corpus/size_priors.json")
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--min-examples", type=int, default=20)
    args = parser.parse_args()

    build_size_priors(
        records_path=args.records,
        output_path=args.output,
        n_bins=args.bins,
        min_examples=args.min_examples,
    )
