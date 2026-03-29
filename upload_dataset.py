#!/usr/bin/env python3
"""Upload a folder as a HuggingFace dataset, split into ZIP files.

Creates ZIP archives of ~5GB each from the source folder, then uploads
them to a HuggingFace dataset repository using XET for maximum speed.

Requirements:
    pip install huggingface_hub hf-xet

Usage:
    export HF_TOKEN=hf_xxxxx
    python upload_dataset.py /workspace/next_scene_dataset_nsfwV2

    # Custom repo name:
    python upload_dataset.py /workspace/my_data --repo-name my-custom-dataset

    # High performance mode (64GB+ RAM):
    HF_XET_HIGH_PERFORMANCE=1 python upload_dataset.py /workspace/my_data
"""

import argparse
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 5 GB per zip (in bytes)
MAX_ZIP_SIZE = 5 * 1024 * 1024 * 1024


def get_hf_username(token: str) -> str:
    """Get HuggingFace username from token."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    info = api.whoami()
    return info["name"]


def create_zips(source_dir: Path, output_dir: Path, max_size: int = MAX_ZIP_SIZE) -> list[Path]:
    """Split source directory into ZIP files of approximately max_size bytes."""
    files = sorted(f for f in source_dir.rglob("*") if f.is_file())
    if not files:
        logger.error("No files found in %s", source_dir)
        sys.exit(1)

    total_size = sum(f.stat().st_size for f in files)
    logger.info("Found %d files (%.2f GB total)", len(files), total_size / 1e9)

    zips: list[Path] = []
    part = 1
    current_size = 0
    current_zip = None
    base_name = source_dir.name

    for f in files:
        file_size = f.stat().st_size

        # Start new zip if needed
        if current_zip is None or (current_size + file_size > max_size and current_size > 0):
            if current_zip is not None:
                current_zip.close()
                logger.info("  Created %s (%.2f GB)", zips[-1].name, current_size / 1e9)

            zip_path = output_dir / f"{base_name}_part{part:03d}.zip"
            zips.append(zip_path)
            current_zip = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED)
            current_size = 0
            part += 1

        # Use relative path inside zip
        arcname = f.relative_to(source_dir)
        current_zip.write(f, arcname)
        current_size += file_size

    if current_zip is not None:
        current_zip.close()
        logger.info("  Created %s (%.2f GB)", zips[-1].name, current_size / 1e9)

    return zips


def upload_to_hf(zip_files: list[Path], repo_id: str, token: str):
    """Upload ZIP files to HuggingFace dataset repo using XET."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Create dataset repo (no error if exists)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    logger.info("Repository: https://huggingface.co/datasets/%s", repo_id)

    for i, zip_path in enumerate(zip_files, 1):
        size_gb = zip_path.stat().st_size / 1e9
        logger.info("Uploading [%d/%d] %s (%.2f GB)...", i, len(zip_files), zip_path.name, size_gb)

        api.upload_file(
            path_or_fileobj=str(zip_path),
            path_in_repo=zip_path.name,
            repo_id=repo_id,
            repo_type="dataset",
        )
        logger.info("  Done: %s", zip_path.name)

    logger.info("All uploads complete! https://huggingface.co/datasets/%s", repo_id)


def main():
    parser = argparse.ArgumentParser(description="Upload folder as HuggingFace dataset (ZIP + XET)")
    parser.add_argument("source", type=Path, help="Source folder to upload")
    parser.add_argument("--repo-name", type=str, default=None, help="Dataset repo name (default: folder name)")
    parser.add_argument("--max-zip-gb", type=float, default=5.0, help="Max ZIP size in GB (default: 5)")
    parser.add_argument("--keep-zips", action="store_true", help="Keep ZIP files after upload")
    parser.add_argument("--zip-dir", type=Path, default=None, help="Directory for ZIP files (default: temp dir)")
    args = parser.parse_args()

    # Validate source
    source = args.source.resolve()
    if not source.is_dir():
        logger.error("Source is not a directory: %s", source)
        sys.exit(1)

    # Get token
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN environment variable not set")
        sys.exit(1)

    # Get username and build repo ID
    username = get_hf_username(token)
    repo_name = args.repo_name or source.name
    repo_id = f"{username}/{repo_name}"
    logger.info("Will upload to: %s", repo_id)

    max_size = int(args.max_zip_gb * 1024 * 1024 * 1024)

    # Create zips
    if args.zip_dir:
        zip_dir = args.zip_dir.resolve()
        zip_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        zip_dir = Path(tempfile.mkdtemp(prefix="hf_upload_"))
        cleanup = not args.keep_zips

    logger.info("Creating ZIP files in %s...", zip_dir)
    zip_files = create_zips(source, zip_dir, max_size=max_size)
    logger.info("Created %d ZIP file(s)", len(zip_files))

    # Upload
    try:
        upload_to_hf(zip_files, repo_id, token)
    finally:
        if cleanup:
            for zf in zip_files:
                zf.unlink(missing_ok=True)
            zip_dir.rmdir()
            logger.info("Cleaned up temporary ZIP files")
        elif args.keep_zips:
            logger.info("ZIP files kept at: %s", zip_dir)


if __name__ == "__main__":
    main()
