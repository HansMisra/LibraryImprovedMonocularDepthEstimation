import argparse
import copy
import json
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from split_utils import load_split_names


class DepthNetChannels(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_disparity(path):
    disparity = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if disparity is None:
        raise FileNotFoundError(path)
    return disparity.astype(np.float32) / 256.0


def load_prior(path, key):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with np.load(path) as data:
        prior = data[key].astype(np.float32)
        valid = data["valid"].astype(np.float32)
    return prior, valid


def resize_and_pad_array(array, target_size, interpolation, value_scale=1.0):
    height, width = array.shape[:2]
    scale = target_size / max(width, height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(array, (new_width, new_height), interpolation=interpolation)
    resized = resized.astype(np.float32) * float(value_scale)

    pad_w = target_size - new_width
    pad_h = target_size - new_height
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    geometry = {
        "original_height": height,
        "original_width": width,
        "scale": scale,
        "new_height": new_height,
        "new_width": new_width,
        "top": top,
        "left": left,
    }
    return padded, geometry


def prepare_rgb(image, target_size):
    height, width = image.shape[:2]
    scale = target_size / max(width, height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    pad_w = target_size - new_width
    pad_h = target_size - new_height
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    tensor = TF.to_tensor(Image.fromarray(padded)).float()
    geometry = {
        "original_height": height,
        "original_width": width,
        "scale": scale,
        "new_height": new_height,
        "new_width": new_width,
        "top": top,
        "left": left,
    }
    return tensor, geometry


class KITTIPriorDataset(Dataset):
    def __init__(
        self,
        frame_names,
        image_dir,
        disparity_dir,
        prior_dir,
        prior_key,
        target_size,
        prior_scale,
    ):
        self.frame_names = list(frame_names)
        self.image_dir = image_dir
        self.disparity_dir = disparity_dir
        self.prior_dir = prior_dir
        self.prior_key = prior_key
        self.target_size = target_size
        self.prior_scale = max(float(prior_scale), 1e-6)

    def __len__(self):
        return len(self.frame_names)

    def prior_path(self, frame):
        return os.path.join(
            self.prior_dir,
            os.path.splitext(frame)[0] + ".npz",
        )

    def __getitem__(self, index):
        frame = self.frame_names[index]
        image = load_rgb(os.path.join(self.image_dir, frame))
        disparity = load_disparity(os.path.join(self.disparity_dir, frame))
        prior, prior_valid = load_prior(self.prior_path(frame), self.prior_key)

        image_tensor, geometry = prepare_rgb(image, self.target_size)

        disparity_scaled, _ = resize_and_pad_array(
            disparity,
            self.target_size,
            interpolation=cv2.INTER_NEAREST,
            value_scale=geometry["scale"],
        )
        prior_resized, _ = resize_and_pad_array(
            prior,
            self.target_size,
            interpolation=cv2.INTER_NEAREST,
            value_scale=1.0,
        )
        valid_resized, _ = resize_and_pad_array(
            prior_valid,
            self.target_size,
            interpolation=cv2.INTER_NEAREST,
            value_scale=1.0,
        )

        prior_channel = np.clip(prior_resized / self.prior_scale, 0.0, 3.0)
        prior_channel *= (valid_resized > 0.5).astype(np.float32)

        return {
            "frame": frame,
            "image": image_tensor,
            "disparity": torch.from_numpy(disparity_scaled).unsqueeze(0).float(),
            "prior": torch.from_numpy(prior_channel).unsqueeze(0).float(),
            "prior_valid": torch.from_numpy((valid_resized > 0.5).astype(np.float32)).unsqueeze(0),
        }


def compute_prior_scale(frame_names, prior_dir, prior_key):
    values = []
    for frame in frame_names:
        path = os.path.join(prior_dir, os.path.splitext(frame)[0] + ".npz")
        prior, valid = load_prior(path, prior_key)
        unique_values = np.unique(prior[(valid > 0.5) & np.isfinite(prior) & (prior > 0)])
        if unique_values.size:
            values.extend(unique_values.tolist())

    if not values:
        return 1.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.99))


def masked_mse(prediction, target):
    valid = (
        (target > 0)
        & torch.isfinite(target)
        & torch.isfinite(prediction)
    )
    if not valid.any():
        return None
    return ((prediction - target) ** 2)[valid].mean()


def model_input(batch, variant, device):
    image = batch["image"].to(device)
    if variant == "rgb":
        return image
    prior = batch["prior"].to(device)
    prior_valid = batch["prior_valid"].to(device)
    return torch.cat([image, prior, prior_valid], dim=1)


def restore_prediction(prediction, geometry):
    prediction = prediction.squeeze().detach().cpu().numpy().astype(np.float32)
    top = geometry["top"]
    left = geometry["left"]
    new_height = geometry["new_height"]
    new_width = geometry["new_width"]

    cropped = prediction[top:top + new_height, left:left + new_width]
    restored = cv2.resize(
        cropped,
        (geometry["original_width"], geometry["original_height"]),
        interpolation=cv2.INTER_LINEAR,
    )
    restored = restored / max(float(geometry["scale"]), 1e-8)
    return restored


def update_metric_accumulator(accumulator, prediction, target, mask=None):
    valid = (
        (target > 0)
        & np.isfinite(target)
        & np.isfinite(prediction)
    )
    if mask is not None:
        valid &= mask.astype(bool)
    if not np.any(valid):
        return

    error = np.abs(prediction[valid] - target[valid])
    relative = error / np.maximum(target[valid], 1e-6)
    accumulator["abs_sum"] += float(error.sum())
    accumulator["sq_sum"] += float((error ** 2).sum())
    accumulator["bad1"] += int((error > 1.0).sum())
    accumulator["bad3"] += int((error > 3.0).sum())
    accumulator["d1"] += int(((error > 3.0) & (relative > 0.05)).sum())
    accumulator["n"] += int(error.size)


def new_accumulator():
    return {
        "abs_sum": 0.0,
        "sq_sum": 0.0,
        "bad1": 0,
        "bad3": 0,
        "d1": 0,
        "n": 0,
    }


def finalize_metrics(accumulator):
    n = accumulator["n"]
    if n == 0:
        return None
    return {
        "valid_pixels": n,
        "epe": accumulator["abs_sum"] / n,
        "rmse": math.sqrt(accumulator["sq_sum"] / n),
        "bad_1": accumulator["bad1"] / n,
        "bad_3": accumulator["bad3"] / n,
        "d1": accumulator["d1"] / n,
    }


def evaluate_original_resolution(
    model,
    variant,
    frame_names,
    image_dir,
    disparity_dir,
    prior_dir,
    prior_key,
    target_size,
    prior_scale,
    device,
):
    model.eval()
    overall = new_accumulator()
    object_region = new_accumulator()
    non_object_region = new_accumulator()

    with torch.no_grad():
        for frame in frame_names:
            image = load_rgb(os.path.join(image_dir, frame))
            target = load_disparity(os.path.join(disparity_dir, frame))
            prior_path = os.path.join(prior_dir, os.path.splitext(frame)[0] + ".npz")
            prior, prior_valid = load_prior(prior_path, prior_key)

            image_tensor, geometry = prepare_rgb(image, target_size)
            prior_resized, _ = resize_and_pad_array(
                prior,
                target_size,
                interpolation=cv2.INTER_NEAREST,
                value_scale=1.0,
            )
            valid_resized, _ = resize_and_pad_array(
                prior_valid,
                target_size,
                interpolation=cv2.INTER_NEAREST,
                value_scale=1.0,
            )
            prior_channel = np.clip(prior_resized / max(prior_scale, 1e-6), 0.0, 3.0)
            prior_channel *= (valid_resized > 0.5).astype(np.float32)

            image_batch = image_tensor.unsqueeze(0).to(device)
            if variant == "rgb":
                inputs = image_batch
            else:
                prior_tensor = torch.from_numpy(prior_channel).unsqueeze(0).unsqueeze(0).float().to(device)
                valid_tensor = torch.from_numpy((valid_resized > 0.5).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                inputs = torch.cat([image_batch, prior_tensor, valid_tensor], dim=1)

            prediction = model(inputs)
            prediction = F.interpolate(
                prediction,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
            prediction = restore_prediction(prediction[0, 0], geometry)

            update_metric_accumulator(overall, prediction, target)
            update_metric_accumulator(object_region, prediction, target, prior_valid > 0.5)
            update_metric_accumulator(non_object_region, prediction, target, prior_valid <= 0.5)

    return {
        "overall": finalize_metrics(overall),
        "prior_covered": finalize_metrics(object_region),
        "prior_uncovered": finalize_metrics(non_object_region),
    }


def format_metrics(metrics):
    if metrics is None:
        return "n/a"
    return (
        f"EPE={metrics['epe']:.4f} | "
        f"RMSE={metrics['rmse']:.4f} | "
        f"Bad-1={metrics['bad_1'] * 100:.2f}% | "
        f"Bad-3={metrics['bad_3'] * 100:.2f}% | "
        f"D1={metrics['d1'] * 100:.2f}% | "
        f"pixels={metrics['valid_pixels']}"
    )


def initialize_models(seed):
    set_seed(seed)
    baseline = DepthNetChannels(3)
    baseline_state = copy.deepcopy(baseline.state_dict())

    set_seed(seed)
    class_model = DepthNetChannels(5)
    size_model = DepthNetChannels(5)

    with torch.no_grad():
        for model in (class_model, size_model):
            model.encoder[0].weight[:, :3].copy_(baseline.encoder[0].weight)
            model.encoder[0].weight[:, 3:].zero_()
            model.encoder[0].bias.copy_(baseline.encoder[0].bias)
            model.encoder[2].load_state_dict(baseline.encoder[2].state_dict())
            model.decoder.load_state_dict(baseline.decoder.state_dict())

    return baseline, class_model, size_model, baseline_state


def make_loader(dataset, batch_size, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def train_variant(
    model,
    variant,
    train_dataset,
    val_frames,
    args,
    prior_key,
    prior_scale,
    device,
    output_path,
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_d1 = float("inf")
    best_state = None
    history = []

    train_loader = make_loader(train_dataset, args.batch_size, args.seed)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        valid_batches = 0

        progress = tqdm(
            train_loader,
            desc=f"{variant} epoch {epoch + 1}/{args.epochs}",
            unit="batch",
            leave=False,
        )

        for batch in progress:
            target = batch["disparity"].to(device)
            inputs = model_input(batch, variant, device)

            optimizer.zero_grad()
            prediction = model(inputs)
            prediction = F.interpolate(
                prediction,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = masked_mse(prediction, target)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            valid_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_mse = running_loss / max(valid_batches, 1)
        metrics = evaluate_original_resolution(
            model=model,
            variant=variant,
            frame_names=val_frames,
            image_dir=args.image_dir,
            disparity_dir=args.disparity_dir,
            prior_dir=args.val_prior_dir,
            prior_key=prior_key,
            target_size=args.target_size,
            prior_scale=prior_scale,
            device=device,
        )
        overall = metrics["overall"]
        history.append({
            "epoch": epoch + 1,
            "train_mse": train_mse,
            "metrics": metrics,
        })

        print(
            f"{variant} | epoch {epoch + 1}/{args.epochs} | "
            f"train MSE={train_mse:.4f} | {format_metrics(overall)}"
        )

        if overall is not None and overall["d1"] < best_d1:
            best_d1 = overall["d1"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output_path)
            print(f"Saved best {variant} model to {output_path}")

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate_original_resolution(
        model=model,
        variant=variant,
        frame_names=val_frames,
        image_dir=args.image_dir,
        disparity_dir=args.disparity_dir,
        prior_dir=args.val_prior_dir,
        prior_key=prior_key,
        target_size=args.target_size,
        prior_scale=prior_scale,
        device=device,
    )
    return model, final_metrics, history


def relative_improvement(baseline, candidate, metric, lower_is_better=True):
    if baseline is None or candidate is None:
        return None
    base = baseline[metric]
    cand = candidate[metric]
    if base == 0:
        return None
    if lower_is_better:
        return 1.0 - cand / base
    return cand / base - 1.0


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    train_frames = load_split_names(args.split_file, "train")
    val_frames = load_split_names(args.split_file, "val")

    class_scale = compute_prior_scale(train_frames, args.train_prior_dir, "class_prior")
    size_scale = compute_prior_scale(train_frames, args.train_prior_dir, "size_prior")
    print(f"Class-prior normalization scale: {class_scale:.4f} px")
    print(f"Size-prior normalization scale:  {size_scale:.4f} px")

    class_dataset = KITTIPriorDataset(
        train_frames,
        args.image_dir,
        args.disparity_dir,
        args.train_prior_dir,
        "class_prior",
        args.target_size,
        class_scale,
    )
    size_dataset = KITTIPriorDataset(
        train_frames,
        args.image_dir,
        args.disparity_dir,
        args.train_prior_dir,
        "size_prior",
        args.target_size,
        size_scale,
    )
    rgb_dataset = size_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running training on device: {device}")
    print(f"Train frames: {len(train_frames)} | Validation frames: {len(val_frames)}")
    print(f"Training resolution: {args.target_size}x{args.target_size}")

    baseline, class_model, size_model, _ = initialize_models(args.seed)

    variants = [
        (
            "rgb",
            baseline,
            rgb_dataset,
            "size_prior",
            size_scale,
            os.path.join(args.output_dir, "rgb_best.pth"),
        ),
        (
            "class_prior",
            class_model,
            class_dataset,
            "class_prior",
            class_scale,
            os.path.join(args.output_dir, "rgb_class_prior_best.pth"),
        ),
        (
            "size_prior",
            size_model,
            size_dataset,
            "size_prior",
            size_scale,
            os.path.join(args.output_dir, "rgb_empirical_size_prior_best.pth"),
        ),
    ]

    results = {}
    histories = {}

    for variant, model, dataset, prior_key, prior_scale, output_path in variants:
        print("\n" + "=" * 78)
        print(f"Training variant: {variant}")
        print("=" * 78)
        set_seed(args.seed)
        _, metrics, history = train_variant(
            model=model,
            variant="rgb" if variant == "rgb" else variant,
            train_dataset=dataset,
            val_frames=val_frames,
            args=args,
            prior_key=prior_key,
            prior_scale=prior_scale,
            device=device,
            output_path=output_path,
        )
        results[variant] = metrics
        histories[variant] = history

    print("\n" + "=" * 78)
    print("FINAL HELD-OUT KITTI ABLATION")
    print("=" * 78)
    for variant in ("rgb", "class_prior", "size_prior"):
        print(f"\n{variant}")
        print(f"  all valid pixels: {format_metrics(results[variant]['overall'])}")
        print(f"  prior-covered:    {format_metrics(results[variant]['prior_covered'])}")
        print(f"  prior-uncovered:  {format_metrics(results[variant]['prior_uncovered'])}")

    baseline_overall = results["rgb"]["overall"]
    class_overall = results["class_prior"]["overall"]
    size_overall = results["size_prior"]["overall"]
    class_covered = results["class_prior"]["prior_covered"]
    size_covered = results["size_prior"]["prior_covered"]

    print("\nRelative changes in held-out EPE")
    rgb_to_class = relative_improvement(baseline_overall, class_overall, "epe")
    rgb_to_size = relative_improvement(baseline_overall, size_overall, "epe")
    class_to_size = relative_improvement(class_overall, size_overall, "epe")
    covered_class_to_size = relative_improvement(class_covered, size_covered, "epe")

    if rgb_to_class is not None:
        print(f"  RGB -> RGB + semantic-class prior:       {rgb_to_class * 100:+.2f}%")
    if rgb_to_size is not None:
        print(f"  RGB -> RGB + empirical class-size prior: {rgb_to_size * 100:+.2f}%")
    if class_to_size is not None:
        print(f"  class prior -> class+size prior:          {class_to_size * 100:+.2f}%")
    if covered_class_to_size is not None:
        print(
            f"  class prior -> class+size on covered object pixels: "
            f"{covered_class_to_size * 100:+.2f}%"
        )

    summary = {
        "configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "target_size": args.target_size,
            "lr": args.lr,
            "seed": args.seed,
            "train_frames": len(train_frames),
            "val_frames": len(val_frames),
            "class_prior_scale": class_scale,
            "size_prior_scale": size_scale,
        },
        "results": results,
        "history": histories,
    }
    summary_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"\nSaved full results to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Controlled KITTI dense-MDE ablation: RGB vs RGB+semantic-class prior "
            "vs RGB+empirically learned semantic class+apparent-size prior."
        )
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--disparity-dir", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--train-prior-dir", required=True)
    parser.add_argument("--val-prior-dir", required=True)
    parser.add_argument("--output-dir", default="outputs\\empirical_prior_ablation")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
