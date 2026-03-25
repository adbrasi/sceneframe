"""Depth map generation for image pairs using monocular depth estimation."""

import logging
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

JPEG_QUALITY = 95

# Default model — can be overridden via CLI
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"

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

    Parameters
    ----------
    input_dir : Path
        Directory containing image pairs (*_A.jpg, *_B.jpg).
    percentage : float
        Percentage of _A images to generate depth maps for (0-100).
    batch_size : int
        Batch size for GPU inference.
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
        import torch  # noqa: F401
        from PIL import Image
        from transformers import pipeline
    except ImportError:
        raise RuntimeError(
            "Depth map generation requires torch and transformers. "
            "Install with: pip install torch transformers Pillow"
        )

    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

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

    logger.info("Loading depth model: %s on %s", model_name, device)

    pipe = pipeline(
        "depth-estimation",
        model=model_name,
        device=device,
        torch_dtype="auto",
    )

    saved = 0

    for i in tqdm(range(0, len(candidates), batch_size), desc="Depth maps"):
        batch_paths = candidates[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        results = pipe(images, batch_size=batch_size)

        for path, result in zip(batch_paths, results):
            depth_pil = result["depth"]
            depth_np = np.array(depth_pil)

            # Normalize to 0-255 grayscale
            dmin, dmax = depth_np.min(), depth_np.max()
            if dmax > dmin:
                depth_norm = ((depth_np - dmin) / (dmax - dmin) * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth_np, dtype=np.uint8)

            c_path = path.parent / path.name.replace("_A.jpg", "_C.jpg")
            cv2.imwrite(str(c_path), depth_norm, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved += 1

    return saved
