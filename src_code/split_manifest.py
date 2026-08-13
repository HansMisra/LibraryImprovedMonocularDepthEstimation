import argparse
import random

from depth_manifest import read_manifest, write_manifest


def split_samples(samples, val_fraction, seed, group_key=None):
    rng = random.Random(seed)

    if group_key is None:
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        n_val = max(1, int(round(len(samples) * val_fraction)))
        val_idx = set(indices[:n_val])
        train = [s for i, s in enumerate(samples) if i not in val_idx]
        val = [s for i, s in enumerate(samples) if i in val_idx]
        return train, val

    groups = {}
    for sample in samples:
        group = sample.get(group_key)
        if group is None:
            raise ValueError(f"Sample {sample.get('id')} does not contain group key {group_key!r}")
        groups.setdefault(str(group), []).append(sample)

    group_names = list(groups)
    rng.shuffle(group_names)
    target_val = len(samples) * val_fraction
    val_groups = set()
    val_count = 0
    for name in group_names:
        if val_count >= target_val and val_groups:
            break
        val_groups.add(name)
        val_count += len(groups[name])

    train = [s for s in samples if str(s[group_key]) not in val_groups]
    val = [s for s in samples if str(s[group_key]) in val_groups]
    return train, val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-key", default=None)
    args = parser.parse_args()

    samples = read_manifest(args.input)
    train, val = split_samples(samples, args.val_fraction, args.seed, args.group_key)
    write_manifest(train, args.train_output)
    write_manifest(val, args.val_output)
    print(f"Saved {len(train)} train and {len(val)} validation samples")
