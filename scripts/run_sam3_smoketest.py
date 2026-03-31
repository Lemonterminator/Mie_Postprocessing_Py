#!/usr/bin/env python
"""Run the same local SAM3 inference flow used in examples/Sam3.ipynb."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
import torch
from transformers import Sam3Model, Sam3Processor


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    model_dir = repo_root / ".models" / "sam3_hf"
    image_path = repo_root / "mask.png"
    output_path = repo_root / "Results" / "sam3_smoketest.png"
    config_path = model_dir / "config.json"
    has_weights = any(
        candidate.exists()
        for candidate in (
            model_dir / "model.safetensors",
            model_dir / "pytorch_model.bin",
            model_dir / "model.safetensors.index.json",
            model_dir / "pytorch_model.bin.index.json",
        )
    )

    if not model_dir.exists() or not config_path.exists() or not has_weights:
        print(f"Error: local SAM3 model is incomplete: {model_dir}")
        print("Run scripts/download_sam3.py first.")
        return 1
    if not image_path.exists():
        print(f"Error: missing image file: {image_path}")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")
    print(f"[info] model_dir={model_dir}")
    print(f"[info] image_path={image_path}")

    model = Sam3Model.from_pretrained(model_dir).to(device)
    processor = Sam3Processor.from_pretrained(model_dir)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text="mask", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    print(f"[ok] detections={len(results['masks'])}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = np.array(image)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(img)
    num_masks = len(results["masks"])
    if num_masks == 0:
        axes[1].set_title("SAM3 Result: no detections")
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, num_masks))
        for mask, box, score, color in zip(
            results["masks"],
            results["boxes"],
            results["scores"],
            colors,
        ):
            mask_np = mask.detach().cpu().numpy().astype(bool)
            color_rgb = np.array(color[:3])
            overlay = np.zeros((*mask_np.shape, 4), dtype=np.float32)
            overlay[mask_np, :3] = color_rgb
            overlay[mask_np, 3] = 0.35
            axes[1].imshow(overlay)

            x0, y0, x1, y1 = box.detach().cpu().tolist()
            axes[1].add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=color_rgb,
                    linewidth=2,
                )
            )
            axes[1].text(
                x0,
                max(y0 - 6, 0),
                f"score={float(score):.3f}",
                color="white",
                fontsize=10,
                bbox=dict(facecolor="black", alpha=0.6, pad=2),
            )
        axes[1].set_title(f"SAM3 Result: {num_masks} detections")

    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] wrote preview: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
