import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)


from split_utils import load_split_names


SEGFORMER_ID = "nvidia/segformer-b0-finetuned-ade-512-512"


def load_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_disparity(path):
    disparity = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if disparity is None:
        raise FileNotFoundError(path)
    return disparity.astype(np.float32) / 256.0


def mask_touches_border(mask):
    return bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def make_semantic_map(image, processor, model, device):
    pil_image = Image.fromarray(image)
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    logits = F.interpolate(
        logits,
        size=image.shape[:2],
        mode="bilinear",
        align_corners=False,
    )
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint16)


def make_instance_predictions(image, model, transform, device):
    pil_image = Image.fromarray(image)
    tensor = transform(pil_image).to(device)

    with torch.no_grad():
        output = model([tensor])[0]

    return {k: v.detach().cpu() for k, v in output.items()}


def masked_mode(values):
    if values.size == 0:
        return None
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.argmax(counts)])


def build_corpus(
    image_dir,
    disparity_dir,
    output_dir,
    score_threshold=0.7,
    mask_threshold=0.5,
    min_mask_pixels=50,
    split_file=None,
    split_name="train",
):
    os.makedirs(output_dir, exist_ok=True)
    semantic_dir = os.path.join(output_dir, "semantic_maps")
    os.makedirs(semantic_dir, exist_ok=True)

    records_path = os.path.join(output_dir, "instances.jsonl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    semantic_processor = AutoImageProcessor.from_pretrained(SEGFORMER_ID)
    semantic_model = SegformerForSemanticSegmentation.from_pretrained(
        SEGFORMER_ID
    ).to(device).eval()

    instance_weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    instance_model = maskrcnn_resnet50_fpn_v2(
        weights=instance_weights
    ).to(device).eval()
    instance_transform = instance_weights.transforms()
    coco_categories = instance_weights.meta["categories"]

    ade_labels = {
        int(k): v for k, v in semantic_model.config.id2label.items()
    }
    with open(
        os.path.join(output_dir, "semantic_labels.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ade_labels, f, indent=2)

    image_files = sorted(
        f for f in os.listdir(image_dir)
        if f.endswith("_10.png")
        and os.path.exists(os.path.join(disparity_dir, f))
    )

    if split_file is not None:
        allowed = set(load_split_names(split_file, split_name))
        image_files = [f for f in image_files if f in allowed]

    with open(records_path, "w", encoding="utf-8") as records_file:
        for frame in tqdm(image_files, desc="Building semantic corpus", unit="frame"):
            image = load_rgb(os.path.join(image_dir, frame))
            disparity = load_disparity(os.path.join(disparity_dir, frame))
            height, width = image.shape[:2]

            semantic_map = make_semantic_map(
                image,
                semantic_processor,
                semantic_model,
                device,
            )
            np.save(
                os.path.join(semantic_dir, frame.replace(".png", ".npy")),
                semantic_map,
            )

            output = make_instance_predictions(
                image,
                instance_model,
                instance_transform,
                device,
            )

            instance_id = 0
            for idx, score in enumerate(output["scores"].numpy()):
                if score < score_threshold:
                    continue

                mask = output["masks"][idx, 0].numpy() >= mask_threshold
                area_pixels = int(mask.sum())
                if area_pixels < min_mask_pixels:
                    continue

                box = output["boxes"][idx].numpy()
                x1, y1, x2, y2 = [float(v) for v in box]
                label_id = int(output["labels"][idx].item())
                class_name = coco_categories[label_id]

                valid_depth = mask & np.isfinite(disparity) & (disparity > 0)
                depth_values = disparity[valid_depth]
                valid_depth_fraction = float(valid_depth.sum() / area_pixels)

                semantic_id = masked_mode(semantic_map[mask])
                semantic_name = (
                    ade_labels.get(semantic_id, "unknown")
                    if semantic_id is not None
                    else "unknown"
                )

                record = {
                    "frame": frame,
                    "instance_id": instance_id,
                    "instance_class_id": label_id,
                    "instance_class_name": class_name,
                    "instance_score": float(score),
                    "semantic_class_id": semantic_id,
                    "semantic_class_name": semantic_name,
                    "area_pixels": area_pixels,
                    "area_fraction": float(area_pixels / (height * width)),
                    "bbox_width_fraction": float((x2 - x1) / width),
                    "bbox_height_fraction": float((y2 - y1) / height),
                    "bbox_area_fraction": float(
                        max(0.0, x2 - x1) * max(0.0, y2 - y1)
                        / (height * width)
                    ),
                    "center_x_fraction": float(((x1 + x2) / 2.0) / width),
                    "center_y_fraction": float(((y1 + y2) / 2.0) / height),
                    "valid_depth_fraction": valid_depth_fraction,
                    "touches_border": mask_touches_border(mask),
                    "median_disparity": (
                        float(np.median(depth_values))
                        if depth_values.size else None
                    ),
                    "disparity_q25": (
                        float(np.quantile(depth_values, 0.25))
                        if depth_values.size else None
                    ),
                    "disparity_q75": (
                        float(np.quantile(depth_values, 0.75))
                        if depth_values.size else None
                    ),
                }

                records_file.write(json.dumps(record) + "\n")
                instance_id += 1

    print(f"Saved instance records to {records_path}")
    print(f"Saved semantic maps to {semantic_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--disparity-dir", required=True)
    parser.add_argument("--output-dir", default="semantic_corpus")
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    args = parser.parse_args()

    build_corpus(
        image_dir=args.image_dir,
        disparity_dir=args.disparity_dir,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
        split_file=args.split_file,
        split_name=args.split,
    )
