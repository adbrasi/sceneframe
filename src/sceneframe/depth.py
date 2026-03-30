"""Control image generation for _B images (depth maps and canny edge detection).

Supports:
- Depth maps via Depth Anything V2 (GPU, batched FP16 inference)
- Canny edge detection via OpenCV (CPU, multi-threaded)
- Image base copy (parallel file copy)

Both depth and canny are saved as _C.jpg alongside the source _B.jpg.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

JPEG_QUALITY = 95
DEPTH_SIZE = (518, 518)

MODEL_PRESETS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def _get_candidates(input_dir: Path) -> list[Path]:
    """Find _B images without existing _C control images.

    Note: _image_base.jpg is independent of _C.jpg — having one does not
    block generation of the other, so only _C.jpg is checked here.
    """
    b_files = sorted(input_dir.glob("*_B.jpg"))
    return [
        f for f in b_files
        if not (f.parent / f.name.replace("_B.jpg", "_C.jpg")).exists()
    ]


def generate_image_base(
    candidates: list[Path], source: str = "A", workers: int = 16,
) -> int:
    """Copy source image (A or B) as _image_base.jpg for selected _B candidates.

    Parameters
    ----------
    candidates : list[Path]
        List of _B.jpg files to process.
    source : str
        Which image to copy: "A" or "B".
    workers : int
        Number of parallel copy threads.

    Returns
    -------
    int
        Number of image_base files created.
    """
    import shutil

    if not candidates:
        return 0

    source = source.upper()
    if source not in ("A", "B"):
        raise ValueError(f"source must be 'A' or 'B', got '{source}'")

    logger.info("Generating image_base from _%s for %d images (%d workers)",
                source, len(candidates), workers)

    def _copy_one(b_path: Path) -> bool:
        src_path = b_path.parent / b_path.name.replace("_B.jpg", f"_{source}.jpg")
        if not src_path.exists():
            return False
        dest = b_path.parent / b_path.name.replace("_B.jpg", "_image_base.jpg")
        shutil.copy2(str(src_path), str(dest))
        return True

    saved = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ok in tqdm(pool.map(_copy_one, candidates), total=len(candidates), desc="Image base"):
            if ok:
                saved += 1

    return saved


def generate_canny_maps(
    candidates: list[Path],
    low_threshold: int = 100,
    high_threshold: int = 200,
    workers: int = 16,
) -> int:
    """Generate Canny edge maps for _B images, saved as _C.jpg.

    Pure OpenCV, no GPU required. Multi-threaded for maximum I/O throughput.

    Parameters
    ----------
    candidates : list[Path]
        List of _B.jpg files to process.
    low_threshold : int
        Canny lower threshold.
    high_threshold : int
        Canny upper threshold.
    workers : int
        Number of parallel processing threads.

    Returns
    -------
    int
        Number of canny maps generated.
    """
    if not candidates:
        return 0

    logger.info("Generating canny edge maps for %d images (%d workers)",
                len(candidates), workers)

    def _process_one(path: Path) -> bool:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        edges = cv2.Canny(img, low_threshold, high_threshold)
        c_path = path.parent / path.name.replace("_B.jpg", "_C.jpg")
        cv2.imwrite(str(c_path), edges, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return True

    saved = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ok in tqdm(pool.map(_process_one, candidates), total=len(candidates), desc="Canny maps"):
            if ok:
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
        model_name, dtype=dtype,
    ).to(device)
    depth_model.eval()

    saved = 0

    # Prefetch: load+resize next batch while GPU runs current batch
    def _load_batch(paths: list[Path]):
        imgs = [Image.open(p).convert("RGB") for p in paths]
        sizes = [img.size for img in imgs]
        resized = [img.resize(DEPTH_SIZE, Image.BILINEAR) for img in imgs]
        return resized, sizes

    prefetch_pool = ThreadPoolExecutor(max_workers=8)
    batches = list(range(0, len(candidates), batch_size))
    next_future = None

    for bi, i in enumerate(tqdm(batches, desc="Depth maps")):
        batch_paths = candidates[i : i + batch_size]

        # Use prefetched images if available
        if next_future is not None:
            images_resized, original_sizes = next_future.result()
        else:
            images_resized, original_sizes = _load_batch(batch_paths)

        # Prefetch next batch while GPU runs
        if bi + 1 < len(batches):
            next_i = batches[bi + 1]
            next_future = prefetch_pool.submit(_load_batch, candidates[next_i : next_i + batch_size])
        else:
            next_future = None

        inputs = processor(images=images_resized, return_tensors="pt")
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

    prefetch_pool.shutdown(wait=False)

    return saved
