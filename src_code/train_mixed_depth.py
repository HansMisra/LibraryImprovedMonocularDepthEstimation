import argparse
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from metric_model import TinyMetricDepthNet
from mixed_depth_dataset import DepthLetterboxTransform, MixedManifestDepthDataset


def masked_log_smooth_l1(pred, target, valid):
    valid = valid & torch.isfinite(pred) & torch.isfinite(target) & (target > 0) & (pred > 0)
    if not valid.any():
        return None
    return F.smooth_l1_loss(torch.log(pred[valid]), torch.log(target[valid]))


def depth_metric_accumulator():
    return {
        "count": 0,
        "absrel_sum": 0.0,
        "sq_error_sum": 0.0,
        "log_sq_error_sum": 0.0,
        "delta1_count": 0,
    }


def update_metrics(acc, pred, target, valid):
    valid = valid & torch.isfinite(pred) & torch.isfinite(target) & (target > 0) & (pred > 0)
    if not valid.any():
        return
    p = pred[valid]
    t = target[valid]
    n = int(t.numel())
    acc["count"] += n
    acc["absrel_sum"] += torch.sum(torch.abs(p - t) / torch.clamp(t, min=1e-6)).item()
    acc["sq_error_sum"] += torch.sum((p - t) ** 2).item()
    acc["log_sq_error_sum"] += torch.sum((torch.log(p) - torch.log(t)) ** 2).item()
    ratio = torch.maximum(p / t, t / p)
    acc["delta1_count"] += int((ratio < 1.25).sum().item())


def finalize_metrics(acc):
    n = max(acc["count"], 1)
    return {
        "valid_pixels": acc["count"],
        "absrel": acc["absrel_sum"] / n,
        "rmse": math.sqrt(acc["sq_error_sum"] / n),
        "rmse_log": math.sqrt(acc["log_sq_error_sum"] / n),
        "delta1": acc["delta1_count"] / n,
    }


def evaluate(model, loader, device):
    model.eval()
    overall = depth_metric_accumulator()
    per_source = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", unit="batch", leave=False):
            images = batch["image"].to(device)
            targets = batch["depth"].to(device)
            valid = batch["valid_mask"].to(device)
            pred = model(images)

            update_metrics(overall, pred, targets, valid)

            for i, source in enumerate(batch["dataset"]):
                per_source.setdefault(source, depth_metric_accumulator())
                update_metrics(
                    per_source[source],
                    pred[i : i + 1],
                    targets[i : i + 1],
                    valid[i : i + 1],
                )

    return finalize_metrics(overall), {
        source: finalize_metrics(acc) for source, acc in per_source.items()
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    train_transform = DepthLetterboxTransform(
        target_size=args.target_size,
        random_horizontal_flip=args.horizontal_flip,
    )
    eval_transform = DepthLetterboxTransform(target_size=args.target_size)

    train_dataset = MixedManifestDepthDataset(
        args.train_manifests,
        transform=train_transform,
        max_depth_m=args.max_depth,
    )
    val_dataset = MixedManifestDepthDataset(
        args.val_manifests,
        transform=eval_transform,
        max_depth_m=args.max_depth,
    )

    print("Training source counts:")
    for source, count in sorted(train_dataset.source_counts().items()):
        print(f"  {source}: {count}")
    print("Sampling mode: concatenated/unbalanced (probability follows dataset size)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, min(args.batch_size, 4)),
        shuffle=False,
        num_workers=args.workers,
    )

    model = TinyMetricDepthNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_absrel = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        batches = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch", leave=False)

        for batch in progress:
            images = batch["image"].to(device)
            targets = batch["depth"].to(device)
            valid = batch["valid_mask"].to(device)

            optimizer.zero_grad()
            pred = model(images)
            loss = masked_log_smooth_l1(pred, targets, valid)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()

            running += loss.item()
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        overall, per_source = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train log-Huber {running / max(batches, 1):.4f} | "
            f"val AbsRel {overall['absrel']:.4f} | "
            f"RMSE {overall['rmse']:.3f} m | "
            f"delta1 {overall['delta1']:.3f}"
        )
        for source, metrics in sorted(per_source.items()):
            print(
                f"  {source}: AbsRel={metrics['absrel']:.4f}, "
                f"RMSE={metrics['rmse']:.3f} m, delta1={metrics['delta1']:.3f}"
            )

        if overall["absrel"] < best_absrel:
            best_absrel = overall["absrel"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "target_size": args.target_size,
                "max_depth": args.max_depth,
                "train_manifests": args.train_manifests,
            }
            torch.save(checkpoint, args.output)
            print(f"Saved best mixed-depth model to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifests", nargs="+", required=True)
    parser.add_argument("--val-manifests", nargs="+", required=True)
    parser.add_argument("--output", default="mixed_metric_depth.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-size", type=int, default=384)
    parser.add_argument("--max-depth", type=float, default=100.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--horizontal-flip", action="store_true")
    train(parser.parse_args())
