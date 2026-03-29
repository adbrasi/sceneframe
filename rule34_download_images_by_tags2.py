#!/usr/bin/env python3
"""Standalone Rule34 media downloader using only a tags .txt file.

Default behavior is focused on videos/webp (webm, mp4, webp), but you can
override extensions via --allowed-exts.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import secrets
import shutil
import signal
import string
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
from urllib.parse import urlencode, urlparse

import urllib3

API_BASE = "https://api.rule34.xxx/index.php"
DEFAULT_RULE34_API_KEY = os.environ.get("RULE34_API_KEY") or os.environ.get("DEFAULT_RULE34_API_KEY", "")
DEFAULT_RULE34_USER_ID = os.environ.get("RULE34_USER_ID") or os.environ.get("DEFAULT_RULE34_USER_ID", "")
VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".m4v"}
# Connection pools with keep-alive for maximum throughput on high-bandwidth links.
_API_POOL = urllib3.PoolManager(
    num_pools=4,
    maxsize=64,
    retries=False,
    headers={"User-Agent": "rule34-media-downloader/1.0"},
)
_CDN_POOL = urllib3.PoolManager(
    num_pools=8,
    maxsize=128,
    retries=False,
    headers={"User-Agent": "rule34-image-downloader/1.0"},
)
STOP_EVENT = threading.Event()
_SIGINT_COUNT = 0
_SIGINT_LOCK = threading.Lock()


def handle_sigint(_signum, _frame) -> None:
    global _SIGINT_COUNT
    with _SIGINT_LOCK:
        _SIGINT_COUNT += 1
        count = _SIGINT_COUNT
    STOP_EVENT.set()
    if count == 1:
        print("\nCtrl+C recebido: cancelando downloads em andamento... (Ctrl+C novamente força saída)", file=sys.stderr, flush=True)
        return
    print("\nForçando saída imediata.", file=sys.stderr, flush=True)
    os._exit(130)

@dataclass
class ApiCreds:
    api_key: str
    user_id: str


@dataclass
class FetchConfig:
    api_key: str
    user_id: str
    limit: int = 1000
    max_posts: int = 2000
    retries: int = 5
    backoff_base: float = 0.8
    backoff_jitter: float = 0.4
    timeout: float = 30.0
    max_backoff: float = 30.0


def log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


def load_creds(args: argparse.Namespace, quiet: bool) -> ApiCreds:
    if args.api_key and args.user_id:
        log("Using API credentials from CLI args.", quiet)
        return ApiCreds(api_key=args.api_key, user_id=args.user_id)
    if DEFAULT_RULE34_API_KEY and DEFAULT_RULE34_USER_ID:
        log("Using Rule34 API credentials from environment.", quiet)
        return ApiCreds(api_key=DEFAULT_RULE34_API_KEY, user_id=DEFAULT_RULE34_USER_ID)
    print("Error: credentials required. Set RULE34_API_KEY and RULE34_USER_ID env vars, or use --api-key and --user-id.", file=sys.stderr)
    raise SystemExit(2)


def http_get_raw(
    url: str,
    retries: int,
    backoff_base: float,
    backoff_jitter: float,
    timeout: float,
    max_backoff: float,
) -> str:
    last_err: Optional[Exception] = None
    last_body: Optional[str] = None
    resp = None
    for attempt in range(1, retries + 1):
        try:
            resp = _API_POOL.request("GET", url, timeout=timeout, preload_content=True)
            raw = resp.data.decode("utf-8")
            last_body = raw[:2000] if raw else None
            if not raw:
                raise RuntimeError("Empty response body")
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status}")
            if raw.lstrip().startswith("{"):
                data = json.loads(raw)
                if isinstance(data, dict) and data.get("success") is False:
                    raise RuntimeError(f"API search down: {data.get('message', 'unknown')}")
            return raw
        except Exception as exc:  # pragma: no cover - network handling
            last_err = exc
            if attempt >= retries:
                break
            # Respect Retry-After header on rate limit (429) or server overload (503).
            retry_after = None
            if resp is not None and resp.status in (429, 503):
                ra = resp.headers.get("Retry-After")
                if ra:
                    try:
                        retry_after = float(ra)
                    except ValueError:
                        pass
            if retry_after is not None:
                sleep_for = min(retry_after, max_backoff)
            else:
                sleep_for = backoff_base * (2 ** (attempt - 1))
                sleep_for = min(sleep_for, max_backoff)
            sleep_for += random.uniform(0, backoff_jitter)
            time.sleep(sleep_for)
    detail = f"Failed request after {retries} attempts: {last_err}"
    if last_body:
        detail += f" | last body: {last_body[:200]}"
    raise RuntimeError(detail)


def parse_posts_response(raw: str) -> List[dict]:
    raw = raw.strip()
    if not raw:
        return []
    if raw[0] == "<":
        items: List[dict] = []
        root = ET.fromstring(raw)
        if root.tag == "posts":
            nodes = root.findall("post")
        elif root.tag == "tags":
            nodes = root.findall("tag")
        else:
            nodes = root.findall(".//post")
            if not nodes:
                nodes = root.findall(".//tag")
        for node in nodes:
            items.append(dict(node.attrib))
        return items
    data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("posts") or data.get("post") or data.get("tags") or data.get("tag") or []
    if isinstance(data, list):
        return data
    return []


def build_url(params: dict) -> str:
    return f"{API_BASE}?{urlencode(params)}"


def iter_posts_by_tags(tags: str, cfg: FetchConfig, deleted: bool = False) -> Iterable[dict]:
    fetched = 0
    page = 0
    while fetched < cfg.max_posts:
        remaining = cfg.max_posts - fetched
        limit = min(cfg.limit, remaining)
        params = {
            "page": "dapi",
            "s": "post",
            "q": "index",
            "json": "1",
            "limit": str(limit),
            "pid": str(page),
            "tags": tags,
            "api_key": cfg.api_key,
            "user_id": cfg.user_id,
        }
        if deleted:
            params["deleted"] = "show"
        url = build_url(params)
        raw = http_get_raw(
            url,
            cfg.retries,
            cfg.backoff_base,
            cfg.backoff_jitter,
            cfg.timeout,
            cfg.max_backoff,
        )
        data = parse_posts_response(raw)
        if not data:
            break
        if len(data) > remaining:
            data = data[:remaining]
        for post in data:
            yield post
        fetched += len(data)
        if len(data) < limit:
            break
        page += 1
        time.sleep(0.3)


def load_tag_lines(path: Path) -> List[str]:
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        lines.append(line)
    return lines


def parse_line_selection(spec: str) -> Set[int]:
    selected: Set[int] = set()
    if not spec:
        return selected
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                continue
            if start <= 0 or end <= 0:
                continue
            if start > end:
                start, end = end, start
            selected.update(range(start, end + 1))
        else:
            try:
                idx = int(part)
            except ValueError:
                continue
            if idx > 0:
                selected.add(idx)
    return selected


def resolve_download_workers(requested: int, quiet: bool) -> int:
    if requested > 0:
        return requested
    cpu = os.cpu_count() or 8
    # High but respectful default — enough to saturate fast links without abusing the CDN.
    auto = min(64, max(16, cpu * 4))
    log(f"Auto max-workers: {auto} (cpu={cpu})", quiet)
    return auto


def resolve_line_workers(requested: int, line_count: int, download_workers: int, quiet: bool) -> int:
    if line_count <= 1:
        return 1
    if requested > 0:
        workers = min(requested, line_count)
        return max(1, workers)
    # Keep per-line pools large enough while still parallelizing multiple tag lines.
    auto = max(1, min(line_count, 16, max(1, download_workers // 24)))
    log(f"Auto line-workers: {auto}", quiet)
    return auto


def random_basename(length: int) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def reserve_basename(
    output_dir: Path,
    length: int,
    reserved: Set[str],
    global_reserved: Optional[Set[str]] = None,
    global_lock: Optional[threading.Lock] = None,
) -> str:
    for _ in range(1000):
        name = random_basename(length)
        if name in reserved:
            continue
        key = f"{output_dir}|{name}"
        if global_reserved is not None and global_lock is not None:
            with global_lock:
                if key in global_reserved:
                    continue
                if not any(output_dir.glob(f"{name}.*")):
                    reserved.add(name)
                    global_reserved.add(key)
                    return name
        else:
            if not any(output_dir.glob(f"{name}.*")):
                reserved.add(name)
                return name
    raise RuntimeError("Unable to generate a unique file name.")


def load_ids_cache(path: Optional[Path]) -> Set[str]:
    if not path:
        return set()
    if not path.exists():
        return set()
    ids: Set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        val = raw.strip()
        if val:
            ids.add(val)
    return ids


def append_id_cache(path: Optional[Path], value: str) -> None:
    if not path or not value:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(value + "\n")


def parse_required_tags(tag_line: str) -> Set[str]:
    required: Set[str] = set()
    for raw in tag_line.split():
        token = raw.strip()
        if not token:
            continue
        if token.startswith("-"):
            continue
        if ":" in token:
            continue
        required.add(token.lower())
    return required


def parse_tags_from_file(path: Path) -> Set[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    line = text.splitlines()[0]
    if "," in line:
        parts = [p.strip().lower() for p in line.split(",") if p.strip()]
        return set(parts)
    return {p.strip().lower() for p in line.split() if p.strip()}


def count_tags_in_dir(output_dir: Path, required: Set[str], quiet: bool, ids_cache_path: Optional[Path]) -> int:
    if not output_dir.exists():
        return 0
    txt_files = list(output_dir.glob("*.txt"))
    if ids_cache_path:
        try:
            txt_files = [p for p in txt_files if p.resolve() != ids_cache_path.resolve()]
        except FileNotFoundError:
            txt_files = [p for p in txt_files if p != ids_cache_path]
    if not txt_files:
        return 0
    if not required:
        log(f"Resume scan: {len(txt_files)} tag files counted in {output_dir.name}.", quiet)
        return len(txt_files)
    count = 0
    for txt_path in txt_files:
        try:
            tags = parse_tags_from_file(txt_path)
        except Exception:
            continue
        if not tags:
            continue
        if required and required.issubset(tags):
            count += 1
    log(f"Resume scan: {len(txt_files)} tag files scanned in {output_dir.name}.", quiet)
    return count


def pick_post_url(post: dict, use_sample: bool) -> Optional[str]:
    if use_sample:
        return post.get("sample_url") or post.get("file_url")
    return post.get("file_url") or post.get("sample_url")


def infer_extension(url: str) -> str:
    path = urlparse(url).path
    ext = Path(path).suffix.lower()
    if not ext or len(ext) > 6:
        return ".jpg"
    return ext


def parse_allowed_exts(spec: str) -> Optional[Set[str]]:
    if not spec:
        return None
    allowed: Set[str] = set()
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if not token.startswith("."):
            token = "." + token
        allowed.add(token)
    return allowed or None


def get_video_duration_seconds(path: Path, ffprobe_bin: Optional[str]) -> Optional[float]:
    if not ffprobe_bin:
        return None
    try:
        proc = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    raw = (proc.stdout or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def cleanup_videos_by_constraints(
    output_dir: Path,
    max_size_mb: float,
    max_duration_sec: float,
    ffprobe_bin: Optional[str],
    quiet: bool,
) -> Tuple[int, int, int, int]:
    removed_total = 0
    removed_by_size = 0
    removed_by_duration = 0
    scanned_videos = 0
    if not output_dir.exists():
        return removed_total, removed_by_size, removed_by_duration, scanned_videos

    for media_path in output_dir.rglob("*"):
        if not media_path.is_file():
            continue
        if media_path.suffix.lower() not in VIDEO_EXTS:
            continue
        scanned_videos += 1
        too_big = False
        too_long = False

        try:
            size_mb = media_path.stat().st_size / (1024 * 1024)
            too_big = max_size_mb > 0 and size_mb > max_size_mb
        except Exception:
            continue

        if max_duration_sec > 0 and ffprobe_bin:
            duration = get_video_duration_seconds(media_path, ffprobe_bin)
            if duration is not None and duration > max_duration_sec:
                too_long = True

        if not too_big and not too_long:
            continue

        try:
            media_path.unlink()
        except Exception as exc:
            log(f"Cleanup: failed removing {media_path.name}: {exc}", quiet)
            continue

        sidecar = media_path.with_suffix(".txt")
        if sidecar.exists():
            try:
                sidecar.unlink()
            except Exception:
                pass

        removed_total += 1
        if too_big:
            removed_by_size += 1
        if too_long:
            removed_by_duration += 1

    return removed_total, removed_by_size, removed_by_duration, scanned_videos


def download_job(
    url: str,
    image_path: Path,
    tags_path: Path,
    tag_text: str,
    timeout: float,
    download_timeout: Optional[float],
    max_bytes: Optional[int],
    min_bytes: Optional[int],
    chunk_bytes: int,
) -> Tuple[bool, int, float, str]:
    if STOP_EVENT.is_set():
        return False, 0, 0.0, "Interrupted"
    tmp_path = image_path.with_suffix(image_path.suffix + ".part")
    start = time.time()
    total_bytes = 0
    resp = None
    try:
        resp = _CDN_POOL.request("GET", url, timeout=timeout, preload_content=False)
        if resp.status >= 400:
            raise RuntimeError(f"HTTP {resp.status}")
        content_length = resp.headers.get("Content-Length")
        if content_length:
            try:
                size = int(content_length)
            except Exception:
                size = 0
            if max_bytes and size > max_bytes:
                return False, 0, time.time() - start, f"Skipped: size {size} bytes exceeds max limit"
            if min_bytes and size > 0 and size < min_bytes:
                return False, 0, time.time() - start, f"Skipped: size {size} bytes below min limit"
        with tmp_path.open("wb") as f:
            for chunk in resp.stream(chunk_bytes):
                if STOP_EVENT.is_set():
                    raise RuntimeError("Interrupted")
                f.write(chunk)
                total_bytes += len(chunk)
                if max_bytes and total_bytes > max_bytes:
                    raise RuntimeError("Size limit exceeded")
                if download_timeout and (time.time() - start) > download_timeout:
                    raise TimeoutError(f"Download exceeded {download_timeout:.0f}s")
        if min_bytes and total_bytes < min_bytes:
            tmp_path.unlink(missing_ok=True)
            return False, total_bytes, time.time() - start, f"Skipped: size {total_bytes} bytes below min limit"
        tmp_path.replace(image_path)
        tags_path.write_text(tag_text + "\n", encoding="utf-8")
        return True, total_bytes, time.time() - start, ""
    except Exception as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        if image_path.exists():
            try:
                image_path.unlink()
            except Exception:
                pass
        if tags_path.exists():
            try:
                tags_path.unlink()
            except Exception:
                pass
        return False, total_bytes, time.time() - start, str(exc)
    finally:
        if resp is not None:
            resp.release_conn()


def iter_candidate_posts(tags: str, cfg: FetchConfig, max_posts: int) -> Iterable[dict]:
    cfg_line = FetchConfig(
        api_key=cfg.api_key,
        user_id=cfg.user_id,
        limit=cfg.limit,
        max_posts=max_posts,
        retries=cfg.retries,
        backoff_base=cfg.backoff_base,
        backoff_jitter=cfg.backoff_jitter,
        timeout=cfg.timeout,
        max_backoff=cfg.max_backoff,
    )
    return iter_posts_by_tags(tags, cfg_line, deleted=False)


def download_for_line(
    tag_line: str,
    cfg: FetchConfig,
    output_dir: Path,
    per_line: int,
    existing_count: int,
    candidate_factor: int,
    sort: str,
    name_length: int,
    use_sample: bool,
    timeout: float,
    download_timeout: Optional[float],
    max_bytes: Optional[int],
    min_bytes: Optional[int],
    max_workers: int,
    allowed_exts: Optional[Set[str]],
    downloaded_ids: Set[str],
    inflight_ids: Set[str],
    ids_cache_path: Optional[Path],
    ids_lock: Optional[threading.Lock],
    global_reserved_names: Optional[Set[str]],
    name_lock: Optional[threading.Lock],
    chunk_bytes: int,
    quiet: bool,
) -> Tuple[int, int]:
    if STOP_EVENT.is_set():
        return 0, 0
    if existing_count >= per_line:
        log(f"Skipping line (already {existing_count}/{per_line}).", quiet)
        return 0, 0
    remaining = per_line - existing_count
    tags_query = tag_line
    if sort:
        tags_query = f"{tags_query} {sort}".strip()

    max_posts = max(remaining * candidate_factor, remaining)

    # Pipeline: producer thread fetches metadata while consumer threads download.
    job_queue: queue.Queue = queue.Queue(maxsize=max_workers * 2)
    producer_done = threading.Event()

    reserved: set = set()

    def _producer():
        try:
            for post in iter_candidate_posts(tags_query, cfg, max_posts):
                if STOP_EVENT.is_set():
                    break
                url = pick_post_url(post, use_sample)
                if not url:
                    continue
                candidate_id = str(post.get("id") or url)
                if ids_lock:
                    with ids_lock:
                        if candidate_id in downloaded_ids:
                            continue
                elif candidate_id in downloaded_ids:
                    continue
                ext = infer_extension(url)
                if allowed_exts and ext.lower() not in allowed_exts:
                    continue
                post_tags = (post.get("tags") or "").strip()
                tags_to_write = format_tags_csv(post_tags) if post_tags else format_tags_csv(tag_line)
                name = reserve_basename(output_dir, name_length, reserved, global_reserved_names, name_lock)
                image_path = output_dir / f"{name}{ext}"
                tags_path = output_dir / f"{name}.txt"
                job_queue.put((candidate_id, url, image_path, tags_path, tags_to_write))
        finally:
            producer_done.set()

    producer_thread = threading.Thread(target=_producer, daemon=True)
    producer_thread.start()

    success = 0
    attempted = 0
    pending = {}
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        while success < remaining and not STOP_EVENT.is_set():
            # Fill pending slots from queue.
            while len(pending) < max_workers and (success + len(pending)) < remaining and not STOP_EVENT.is_set():
                try:
                    item = job_queue.get(timeout=0.05)
                except queue.Empty:
                    if producer_done.is_set() and job_queue.empty():
                        break
                    continue
                candidate_id, url, image_path, tags_path, post_tags = item
                if ids_lock:
                    with ids_lock:
                        if candidate_id in downloaded_ids or candidate_id in inflight_ids:
                            continue
                        inflight_ids.add(candidate_id)
                else:
                    if candidate_id in downloaded_ids or candidate_id in inflight_ids:
                        continue
                    inflight_ids.add(candidate_id)
                future = executor.submit(
                    download_job,
                    url,
                    image_path,
                    tags_path,
                    post_tags,
                    timeout,
                    download_timeout,
                    max_bytes,
                    min_bytes,
                    chunk_bytes,
                )
                pending[future] = (candidate_id, url, image_path)
                attempted += 1
            if not pending:
                if producer_done.is_set() and job_queue.empty():
                    break
                continue
            done, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=0.5)
            for future in done:
                try:
                    ok, size, elapsed, err = future.result()
                except Exception as exc:
                    ok, size, elapsed, err = False, 0, 0.0, str(exc)
                candidate_id, _url, _image_path = pending.pop(future, ("", "", Path(".")))
                if ids_lock:
                    with ids_lock:
                        inflight_ids.discard(candidate_id)
                else:
                    inflight_ids.discard(candidate_id)
                if ok:
                    success += 1
                    if candidate_id:
                        if ids_lock:
                            with ids_lock:
                                if candidate_id not in downloaded_ids:
                                    downloaded_ids.add(candidate_id)
                                    append_id_cache(ids_cache_path, candidate_id)
                        elif candidate_id not in downloaded_ids:
                            downloaded_ids.add(candidate_id)
                            append_id_cache(ids_cache_path, candidate_id)
                    log(
                        f"Saved {size / (1024 * 1024):.1f} MB in {elapsed:.1f}s",
                        quiet,
                    )
                    if success >= remaining:
                        break
                else:
                    log(
                        f"Download failed after {elapsed:.1f}s ({size / (1024 * 1024):.1f} MB): {err}",
                        quiet,
                    )
            if success >= remaining or STOP_EVENT.is_set():
                break
    except KeyboardInterrupt:
        STOP_EVENT.set()
        log("Interrupted while downloading line; canceling pending jobs...", quiet)
    finally:
        producer_thread.join(timeout=2)
        executor.shutdown(wait=not STOP_EVENT.is_set(), cancel_futures=True)

    return success, attempted


def format_tags_csv(raw_tags: str) -> str:
    tags = [t for t in raw_tags.split() if t]
    return ", ".join(tags)


def main() -> int:
    global _SIGINT_COUNT
    STOP_EVENT.clear()
    with _SIGINT_LOCK:
        _SIGINT_COUNT = 0
    signal.signal(signal.SIGINT, handle_sigint)

    parser = argparse.ArgumentParser(description="Download Rule34 media for each tag line (standalone)")
    parser.add_argument("--input", default="tags.txt", help="Text file with one tag query per line")
    parser.add_argument("--output-dir", default="downloads", help="Directory to store media and tag files")
    parser.add_argument("--per-line", type=int, default=1, help="Media files to download per line")
    parser.add_argument("--candidate-factor", type=int, default=25, help="How many candidates to scan per target")
    parser.add_argument("--limit", type=int, default=1000, help="API limit per page (max 1000)")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sort", default="sort:score:desc", help="Sort meta tag (default: sort:score:desc)")
    parser.add_argument("--name-length", type=int, default=10, help="Random base name length (<=10)")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Total parallel download workers (0 = auto, tuned for high bandwidth)",
    )
    parser.add_argument(
        "--line-workers",
        type=int,
        default=0,
        help="How many tag lines to process in parallel (0 = auto)",
    )
    parser.add_argument(
        "--chunk-mb",
        type=float,
        default=8.0,
        help="Read chunk size per download stream in MB",
    )
    parser.add_argument("--use-sample", action="store_true", help="Prefer sample_url instead of file_url")
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=0.0,
        help="Per-file download timeout in seconds (0 to disable)",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=0.0,
        help="Skip downloads larger than this size in MB (0 to disable)",
    )
    parser.add_argument(
        "--min-size-mb",
        type=float,
        default=0.0,
        help="Skip downloads smaller than this size in MB (0 to disable)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume by counting existing tag files")
    parser.add_argument(
        "--per-line-dir",
        action="store_true",
        help="Save each line into its own subfolder (linha_1, linha_2, ...)",
    )
    parser.add_argument(
        "--lines",
        default="",
        help="Comma-separated line numbers or ranges to process (e.g. 3 or 2-4,7)",
    )
    parser.add_argument("--api-key", default="", help="Rule34 API key")
    parser.add_argument("--user-id", default="", help="Rule34 user id")
    parser.add_argument(
        "--ids-cache",
        default="downloaded_ids.txt",
        help="Optional file to persist downloaded post IDs (relative to output dir).",
    )
    parser.add_argument(
        "--allowed-exts",
        default="webm,mp4,webp",
        help="Comma-separated extensions filter (default: webm,mp4,webp). Empty = allow all.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce logging")
    args = parser.parse_args()

    if args.name_length < 1 or args.name_length > 10:
        print("--name-length must be between 1 and 10", file=sys.stderr)
        return 2
    if args.per_line < 1:
        print("--per-line must be >= 1", file=sys.stderr)
        return 2
    if args.limit < 1:
        print("--limit must be >= 1", file=sys.stderr)
        return 2
    if args.max_workers < 0:
        print("--max-workers must be >= 0", file=sys.stderr)
        return 2
    if args.line_workers < 0:
        print("--line-workers must be >= 0", file=sys.stderr)
        return 2
    if args.chunk_mb <= 0:
        print("--chunk-mb must be > 0", file=sys.stderr)
        return 2

    creds = load_creds(args, args.quiet)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag_file = Path(args.input)
    if not tag_file.exists():
        print(f"Input file not found: {tag_file}", file=sys.stderr)
        return 2

    lines = load_tag_lines(tag_file)
    if not lines:
        print("No tag lines found.", file=sys.stderr)
        return 2
    selection = parse_line_selection(args.lines)
    indexed_lines = [(idx, line) for idx, line in enumerate(lines, start=1) if not selection or idx in selection]
    if not indexed_lines:
        print("No matching lines found for selection.", file=sys.stderr)
        return 2
    ids_cache_path: Optional[Path] = None
    if args.ids_cache:
        ids_path = Path(args.ids_cache)
        if not ids_path.is_absolute():
            ids_path = output_dir / ids_path
        ids_cache_path = ids_path
    downloaded_ids: Set[str] = load_ids_cache(ids_cache_path)
    if downloaded_ids:
        log(f"Loaded {len(downloaded_ids)} cached IDs.", args.quiet)

    if args.resume:
        existing_counts = []
        for idx, line in indexed_lines:
            req = parse_required_tags(line)
            line_dir = output_dir / f"linha_{idx}" if args.per_line_dir else output_dir
            existing_counts.append(count_tags_in_dir(line_dir, req, args.quiet, ids_cache_path))
    else:
        existing_counts = [0 for _ in indexed_lines]

    cfg = FetchConfig(
        api_key=creds.api_key,
        user_id=creds.user_id,
        limit=min(args.limit, 1000),
        max_posts=args.per_line * args.candidate_factor,
        retries=args.retries,
        timeout=args.timeout,
    )

    total_success = 0
    total_attempts = 0
    log(f"Processing {len(indexed_lines)} selected tag lines...", args.quiet)

    max_bytes = None
    if args.max_size_mb and args.max_size_mb > 0:
        max_bytes = int(args.max_size_mb * 1024 * 1024)
    min_bytes = None
    if args.min_size_mb and args.min_size_mb > 0:
        min_bytes = int(args.min_size_mb * 1024 * 1024)
    allowed_exts = parse_allowed_exts(args.allowed_exts)
    if allowed_exts:
        log(f"Filtering extensions: {', '.join(sorted(allowed_exts))}", args.quiet)
    chunk_bytes = int(args.chunk_mb * 1024 * 1024)
    download_workers = resolve_download_workers(args.max_workers, args.quiet)
    line_workers = resolve_line_workers(args.line_workers, len(indexed_lines), download_workers, args.quiet)
    per_line_download_workers = max(1, download_workers // line_workers)
    log(
        (
            "Parallelism plan: "
            f"total_download_workers={download_workers}, "
            f"line_workers={line_workers}, "
            f"per_line_download_workers={per_line_download_workers}, "
            f"chunk_mb={args.chunk_mb:g}"
        ),
        args.quiet,
    )
    ids_lock = threading.Lock()
    name_lock = threading.Lock()
    global_reserved_names: Set[str] = set()
    inflight_ids: Set[str] = set()

    line_items = []
    for pos, (idx, tag_line) in enumerate(indexed_lines):
        existing = existing_counts[pos] if pos < len(existing_counts) else 0
        line_output_dir = output_dir / f"linha_{idx}" if args.per_line_dir else output_dir
        line_output_dir.mkdir(parents=True, exist_ok=True)
        line_items.append((idx, tag_line, existing, line_output_dir))

    def run_one_line(item: Tuple[int, str, int, Path]) -> Tuple[int, int, int, int]:
        idx, tag_line, existing, line_output_dir = item
        if STOP_EVENT.is_set():
            return idx, existing, 0, 0
        log(f"[{idx}/{len(lines)}] {tag_line}", args.quiet)
        success, attempted = download_for_line(
            tag_line,
            cfg,
            line_output_dir,
            args.per_line,
            existing,
            args.candidate_factor,
            args.sort,
            args.name_length,
            args.use_sample,
            args.timeout,
            args.download_timeout if args.download_timeout > 0 else None,
            max_bytes,
            min_bytes,
            per_line_download_workers,
            allowed_exts,
            downloaded_ids,
            inflight_ids,
            ids_cache_path,
            ids_lock,
            global_reserved_names,
            name_lock,
            chunk_bytes,
            args.quiet,
        )
        return idx, existing, success, attempted

    if line_workers == 1:
        for item in line_items:
            if STOP_EVENT.is_set():
                break
            idx, existing, success, attempted = run_one_line(item)
            total_success += success
            total_attempts += attempted
            total_line = existing + success
            log(f"Downloaded {total_line}/{args.per_line} for line {idx}", args.quiet)
            time.sleep(0.1)
    else:
        line_pool = ThreadPoolExecutor(max_workers=line_workers)
        futures = []
        try:
            futures = [line_pool.submit(run_one_line, item) for item in line_items]
            for future in as_completed(futures):
                if STOP_EVENT.is_set():
                    break
                try:
                    idx, existing, success, attempted = future.result()
                except Exception as exc:
                    log(f"Line worker failed: {exc}", args.quiet)
                    continue
                total_success += success
                total_attempts += attempted
                total_line = existing + success
                log(f"Downloaded {total_line}/{args.per_line} for line {idx}", args.quiet)
        except KeyboardInterrupt:
            STOP_EVENT.set()
            log("Interrupted while waiting line workers; canceling pending lines...", args.quiet)
        finally:
            for future in futures:
                if not future.done():
                    future.cancel()
            line_pool.shutdown(wait=not STOP_EVENT.is_set(), cancel_futures=True)

    if STOP_EVENT.is_set():
        log(f"Interrupted. Partial result: downloads={total_success}, attempts={total_attempts}", args.quiet)
        os._exit(130)
    log(f"Done. Downloads: {total_success}. Attempts: {total_attempts}.", args.quiet)
    return 0 if total_success > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
