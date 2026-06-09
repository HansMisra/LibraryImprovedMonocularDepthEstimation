import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


class SegFormerSegmenter:
    def __init__(
        self,
        model_name="nvidia/segformer-b0-finetuned-ade-512-512",
        device=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(**inputs)
        logits = outputs.logits

        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        class_map = upsampled_logits.argmax(dim=1)[0]
        class_map = class_map.detach().cpu().numpy().astype(np.uint8)

        return class_map