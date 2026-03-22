#!/usr/bin/env python3
"""Upload a folder as a .zip to HuggingFace Hub.

Creates a dataset repo named <username>/<folder-name> and uploads the zip.
Uses Xet storage for large files. Token from HF_TOKEN env var.

Usage:
    python hf_upload_folder.py "E:\\download\\output_pairs_scene"
    python hf_upload_folder.py "E:\\download\\output_pairs_scene" --repo-name my-dataset
    python hf_upload_folder.py "E:\\download\\output_pairs_scene" --private
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, whoami


def normalize_name(name: str) -> str:
    """Normalize a folder name to a valid HuggingFace repo name."""
    name = name.strip().lower()
    name = re.sub(r"[^\w\-.]", "-", name)
    name = re.sub(r"-+", "-", name)
    name = name.strip("-.")
    return name or "dataset"


def create_zip(folder: Path, zip_path: Path) -> None:
    """Create a zip file from a folder with progress."""
    files = [f for f in folder.rglob("*") if f.is_file()]
    total = len(files)
    print(f"Zipping {total} files from {folder}...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for i, file_path in enumerate(files, 1):
            arcname = file_path.relative_to(folder)
            zf.write(file_path, arcname)
            if i % 500 == 0 or i == total:
                size_mb = zip_path.stat().st_size / (1024 * 1024)
                print(f"  [{i}/{total}] {size_mb:.1f} MB", flush=True)

    final_size = zip_path.stat().st_size / (1024 * 1024)
    print(f"Zip created: {zip_path.name} ({final_size:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload a folder as zip to HuggingFace Hub")
    parser.add_argument("folder", type=Path, help="Folder to upload")
    parser.add_argument("--repo-name", default=None, help="Override repo name (default: normalized folder name)")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model"], help="Repository type")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument("--no-zip", action="store_true", help="Upload folder directly without zipping")
    parser.add_argument("--token", default=None, help="HuggingFace token (default: HF_TOKEN env var)")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory", file=sys.stderr)
        return 1

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: set HF_TOKEN env var or use --token", file=sys.stderr)
        return 1

    api = HfApi(token=token)

    # Get username
    user_info = whoami(token=token)
    username = user_info["name"]

    # Determine repo name
    repo_name = args.repo_name or normalize_name(folder.name)
    repo_id = f"{username}/{repo_name}"

    print(f"Username: {username}")
    print(f"Repo: {repo_id} ({args.repo_type})")
    print(f"Folder: {folder}")

    # Create repo
    api.create_repo(
        repo_id=repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )
    print(f"Repo created/exists: https://huggingface.co/datasets/{repo_id}")

    if args.no_zip:
        # Upload folder directly
        print("Uploading folder directly...")
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type=args.repo_type,
            folder_path=str(folder),
        )
    else:
        # Zip and upload
        zip_name = f"{repo_name}.zip"
        zip_path = folder.parent / zip_name
        try:
            create_zip(folder, zip_path)

            print(f"Uploading {zip_name} to {repo_id}...")
            api.upload_file(
                path_or_fileobj=str(zip_path),
                path_in_repo=zip_name,
                repo_id=repo_id,
                repo_type=args.repo_type,
                commit_message=f"Upload {zip_name}",
            )
        finally:
            # Clean up zip
            if zip_path.exists():
                zip_path.unlink()
                print(f"Cleaned up local zip: {zip_name}")

    print(f"\nDone! https://huggingface.co/datasets/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
