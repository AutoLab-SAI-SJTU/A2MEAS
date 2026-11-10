"""SAM object regions from the current onboard camera frames."""

import base64
from io import BytesIO
import re
import numpy as np
from PIL import Image

CAMERAS = ("head", "left_wrist", "right_wrist")


class Grounder:
    def __init__(self, checkpoint, device):
        from a2meas.grounding import load_grounder

        self.model, self.processor = load_grounder(checkpoint, device, threshold=0.5)

    def __call__(self, images, target, step):
        query = re.sub(r"_\d+$", "", target).replace("_", " ").strip()
        evidence = []
        for camera, rgb in zip(CAMERAS, images, strict=True):
            state = self.processor.set_image(Image.fromarray(rgb))
            result = self.processor.set_text_prompt(query, state)
            boxes = result["boxes"].float().cpu().numpy()
            scores = result["scores"].float().cpu().numpy()
            segmented = result["masks"].cpu().numpy().astype(bool)
            order = np.argsort(-scores)[:2]
            height, width = rgb.shape[:2]
            for index in order:
                mask = segmented[index].squeeze()
                if not mask.any():
                    continue
                yy, xx = np.nonzero(mask)
                item = {
                    "entity": query,
                    "camera": camera,
                    "frame": int(step),
                    "box": np.round(boxes[index] / [width, height, width, height], 3)
                    .clip(0, 1)
                    .tolist(),
                    "point": [
                        round(float(xx.mean() / width), 3),
                        round(float(yy.mean() / height), 3),
                    ],
                    "confidence": round(float(scores[index]), 3),
                }
                evidence.append(item)
        evidence = sorted(evidence, key=lambda x: -x["confidence"])[:3]
        return (
            {
                "frame": "normalized_image_xy",
                "entities": evidence,
                "relation": None,
            }
            if evidence
            else None
        )


def image_url(rgb, *, quality=90):
    stream = BytesIO()
    Image.fromarray(rgb).save(stream, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()
