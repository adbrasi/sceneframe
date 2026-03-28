"""Control image generation for _B images (depth maps and canny edge detection).

Supports:
- Depth maps via Depth Anything V2 (GPU, batched FP16 inference)
- Canny edge detection via OpenCV (CPU, very fast)

Both are saved as _C.jpg alongside the source _B.jpg.
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


def _get_candidates(input_dir: Path) -> list[Path]:
    """Find _B images without existing _C control maps."""
    b_files = sorted(input_dir.glob("*_B.jpg"))
    return [
        f for f in b_files
        if not (f.parent / f.name.replace("_B.jpg", "_C.jpg")).exists()
    ]


def _select_subset(
    candidates: list[Path], percentage: float, seed: int | None = None,
) -> list[Path]:
    """Select a random percentage-based subset of candidates."""
    if percentage >= 100.0 or not candidates:
        return candidates
    if seed is not None:
        random.seed(seed)
    count = max(1, int(len(candidates) * percentage / 100.0))
    return sorted(random.sample(candidates, count))


def generate_canny_maps(
    candidates: list[Path],
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> int:
    """Generate Canny edge maps for _B images, saved as _C.jpg.

    Pure OpenCV, no GPU required. Very fast.

    Parameters
    ----------
    candidates : list[Path]
        List of _B.jpg files to process.
    low_threshold : int
        Canny lower threshold.
    high_threshold : int
        Canny upper threshold.

    Returns
    -------
    int
        Number of canny maps generated.
    """
    if not candidates:
        return 0

    logger.info("Generating canny edge maps for %d images", len(candidates))
    saved = 0

    for path in tqdm(candidates, desc="Canny maps"):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        edges = cv2.Canny(img, low_threshold, high_threshold)
        c_path = path.parent / path.name.replace("_B.jpg", "_C.jpg")
        cv2.imwrite(str(c_path), edges, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        saved += 1

    return saved


def generate_depth_maps(
    candidates: list[Path],
    batch_size: int = 8,
    device: str | None = None,
    model: str = "large",
) -> int:
    """Generate depth maps for _B images, saved as _C.jpg.

    Uses AutoModelForDepthEstimation with FP16 for optimal GPU throughput
    instead of the generic pipeline. Processes images in batched tensors
    for maximum utilization on high-end GPUs (RTX 5090, RTX Pro 6000).

    Parameters
    ----------
    candidates : list[Path]
        List of _B.jpg files to process.
    batch_size : int
        Batch size for GPU inference. Higher = faster on GPUs with more VRAM.
        Recommended: 16-32 for 24GB, 32-64 for 48GB+.
    device : str or None
        Device for inference ("cuda", "cpu"). Auto-detects if None.
    model : str
        Model preset name ("small", "base", "large") or full HuggingFace model ID.

    Returns
    -------
    int
        Number of depth maps generated.
    """
    if not candidates:
        return 0

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

            c_path = path.parent / path.name.replace("_B.jpg", "_C.jpg")
            cv2.imwrite(str(c_path), depth_norm, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved += 1

    return saved
