import json
import os
import random


def _sample_names(dataset):
    return [os.path.basename(image_path) for image_path, _ in dataset.samples]


def load_or_create_split(dataset, split_path, val_fraction=0.2, seed=42):
    names = _sample_names(dataset)
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)

        train_names = [n for n in split["train"] if n in name_to_idx]
        val_names = [n for n in split["val"] if n in name_to_idx]

        if not train_names or not val_names:
            raise ValueError("Saved split does not match the current dataset.")

        return (
            [name_to_idx[n] for n in train_names],
            [name_to_idx[n] for n in val_names],
        )

    rng = random.Random(seed)
    shuffled = names.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_names = sorted(shuffled[:n_val])
    train_names = sorted(shuffled[n_val:])

    split = {
        "seed": seed,
        "val_fraction": val_fraction,
        "train": train_names,
        "val": val_names,
    }

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    return (
        [name_to_idx[n] for n in train_names],
        [name_to_idx[n] for n in val_names],
    )


def load_split_names(split_path, split_name="val"):
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    return split[split_name]
