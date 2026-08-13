import argparse
import random
from pathlib import Path

import cv2
import h5py
import numpy as np

from depth_manifest import write_manifest


def read_h5_image(images, index):
    image = np.asarray(images[index])
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Unexpected NYU image shape: {image.shape}; expected (3, W, H) through h5py.")
    return np.transpose(image, (2, 1, 0)).astype(np.uint8)


def read_h5_map(dataset, index):
    value = np.asarray(dataset[index])
    if value.ndim != 2:
        raise ValueError(f"Unexpected NYU map shape: {value.shape}")
    return value.T


def export_nyu(args):
    output_dir = Path(args.output_dir)
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth_m"
    label_dir = output_dir / "labels"
    instance_dir = output_dir / "instances"
    for directory in (rgb_dir, depth_dir, label_dir, instance_dir):
        directory.mkdir(parents=True, exist_ok=True)

    samples = []
    with h5py.File(args.mat, "r") as f:
        images = f["images"]
        depths = f[args.depth_key]
        labels = f.get("labels")
        instances = f.get("instances")

        n = images.shape[0]
        if args.max_samples is not None:
            n = min(n, args.max_samples)

        for i in range(n):
            image = read_h5_image(images, i)
            depth = read_h5_map(depths, i).astype(np.float32)

            rgb_path = rgb_dir / f"{i:04d}.png"
            depth_path = depth_dir / f"{i:04d}.npy"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            np.save(depth_path, depth)

            sample = {
                "id": f"nyuv2:{i:04d}",
                "dataset": "nyuv2",
                "image_path": str(rgb_path.resolve()),
                "target_path": str(depth_path.resolve()),
                "target_encoding": "depth_npy_m",
                "target_source": f"nyuv2_{args.depth_key}",
                "max_depth_m": args.max_depth,
            }

            if args.fx is not None:
                sample.update({"fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy})

            if labels is not None:
                label_path = label_dir / f"{i:04d}.npy"
                np.save(label_path, read_h5_map(labels, i).astype(np.int32))
                sample["semantic_label_path"] = str(label_path.resolve())

            if instances is not None:
                instance_path = instance_dir / f"{i:04d}.npy"
                np.save(instance_path, read_h5_map(instances, i).astype(np.int32))
                sample["instance_label_path"] = str(instance_path.resolve())

            samples.append(sample)

    rng = random.Random(args.seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(samples) * args.val_fraction)))
    val_set = set(indices[:n_val])
    train_samples = [s for i, s in enumerate(samples) if i not in val_set]
    val_samples = [s for i, s in enumerate(samples) if i in val_set]

    write_manifest(train_samples, output_dir / "nyu_train.jsonl")
    write_manifest(val_samples, output_dir / "nyu_val.jsonl")
    print(f"Exported NYUv2: {len(train_samples)} train, {len(val_samples)} validation")
    print("This deterministic split is for this project; it is not claimed to be the standard NYUv2 benchmark split.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", required=True, help="Path to nyu_depth_v2_labeled.mat")
    parser.add_argument("--output-dir", default="mixed_data/nyuv2")
    parser.add_argument("--depth-key", choices=["rawDepths", "depths"], default="rawDepths")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    export_nyu(parser.parse_args())
