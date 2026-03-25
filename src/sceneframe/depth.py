"""Depth map generation for image pairs using monocular depth estimation.

Uses Depth Anything V2 with direct model inference (AutoModelForDepthEstimation)
for maximum GPU throughput with FP16 and batched tensors.
"""

import logging
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

JPEG_QUALITY = 95

MODEL_PRESETS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def generate_depth_maps(
    input_dir: Path,
    percentage: float = 100.0,
    batch_size: int = 8,
    device: str | None = None,
    model: str = "large",
    seed: int | None = None,
) -> int:
    """Generate depth maps for _A images, saved as _C.jpg.

    Uses AutoModelForDepthEstimation with FP16 for optimal GPU throughput
    instead of the generic pipeline. Processes images in batched tensors
    for maximum utilization on high-end GPUs (RTX 5090, RTX Pro 6000).

    Parameters
    ----------
    input_dir : Path
        Directory containing image pairs (*_A.jpg, *_B.jpg).
    percentage : float
        Percentage of _A images to generate depth maps for (0-100).
    batch_size : int
        Batch size for GPU inference. Higher = faster on GPUs with more VRAM.
        Recommended: 16-32 for 24GB, 32-64 for 48GB+.
    device : str or None
        Device for inference ("cuda", "cpu"). Auto-detects if None.
    model : str
        Model preset name ("small", "base", "large") or full HuggingFace model ID.
    seed : int or None
        Random seed for reproducible subset selection.

    Returns
    -------
    int
        Number of depth maps generated.
    """
    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError:
        raise RuntimeError(
            "Depth map generation requires torch and transformers. "
            "Install with: pip install torch transformers Pillow"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_fp16 = device == "cuda"

    # Find _A images without existing _C depth maps
    a_files = sorted(input_dir.glob("*_A.jpg"))
    candidates = [
        f for f in a_files
        if not (f.parent / f.name.replace("_A.jpg", "_C.jpg")).exists()
    ]

    if not candidates:
        logger.info("No new depth maps to generate (all _A files already have _C)")
        return 0

    # Random subset selection
    if percentage < 100.0:
        if seed is not None:
            random.seed(seed)
        count = max(1, int(len(candidates) * percentage / 100.0))
        candidates = sorted(random.sample(candidates, count))

    logger.info("Generating depth maps for %d images", len(candidates))

    # Resolve model name
    model_name = MODEL_PRESETS.get(model, model)
    logger.info("Loading depth model: %s on %s (fp16=%s)", model_name, device, use_fp16)

    # Load model and processor directly for optimal batch inference
    processor = AutoImageProcessor.from_pretrained(model_name)
    dtype = torch.float16 if use_fp16 else torch.float32
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        model_name, torch_dtype=dtype,
    ).to(device)
    depth_model.eval()

    saved = 0

    for i in tqdm(range(0, len(candidates), batch_size), desc="Depth maps"):
        batch_paths = candidates[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        original_sizes = [img.size for img in images]  # (W, H)

        # Preprocess entire batch into a single tensor
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = depth_model(**inputs)

        # predicted_depth shape: [B, H, W]
        predicted_depth = outputs.predicted_depth

        for j, (path, orig_size) in enumerate(zip(batch_paths, original_sizes)):
            depth_tensor = predicted_depth[j]  # [H, W]

            # Interpolate back to original image size
            depth_resized = torch.nn.functional.interpolate(
                depth_tensor.unsqueeze(0).unsqueeze(0),
                size=(orig_size[1], orig_size[0]),  # (H, W)
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            # Normalize to 0-255
            depth_np = depth_resized.cpu().float().numpy()
            dmin, dmax = depth_np.min(), depth_np.max()
            if dmax > dmin:
                depth_norm = ((depth_np - dmin) / (dmax - dmin) * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth_np, dtype=np.uint8)

            c_path = path.parent / path.name.replace("_A.jpg", "_C.jpg")
            cv2.imwrite(str(c_path), depth_norm, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved += 1

    return saved
