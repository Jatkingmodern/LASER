#!/usr/bin/env python3
"""
Laser Model Downloader (consolidated improvements)

Features:
- Robust download with retries, timeouts, and progress display.
- Optional checksum verification via a checksum file (filename checksum per line).
- Optional use of `rich` for nicer progress UI; falls back to `tqdm`.
- Model listing (when language lists are available) and cache clearing.
- Modern CLI with argparse, type hints, and docstrings.
- Graceful handling if the `laser_encoders.language_list` module is not installed:
  in that case the downloader accepts a direct language code for laser3.
- All downloads saved to a model-dir (defaults to ~/.cache/laser_encoders).
- Safe temporary file usage to avoid leaving partial files.
- Well-instrumented logging.

Usage examples:
  # Download laser2
  python laser_downloader.py --laser laser2

  # Download laser3 for a language code (when language lists aren't available,
  # supply the actual code like 'eng_Latn' or the Flores code expected by the server)
  python laser_downloader.py --laser laser3 --lang eng_Latn --spm

  # List models (if language lists are available)
  python laser_downloader.py --list

  # Clear cache
  python laser_downloader.py --clear-cache

  # Supply checksum file (optional)
  python laser_downloader.py --laser laser2 --checksums checksums.txt
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

# Prefer rich for nicer UI; fall back to tqdm
try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, DownloadColumn, TextColumn, TransferSpeedColumn, TimeRemainingColumn
    RICH_AVAILABLE = True
    console = Console()
except Exception:
    RICH_AVAILABLE = False

try:
    from tqdm import tqdm  # type: ignore
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

# Attempt import of language lists used in original tool.
# If not available, we still support direct language codes.
try:
    from laser_encoders.language_list import LASER2_LANGUAGE, LASER3_LANGUAGE, SPM_LANGUAGE  # type: ignore
    LANGUAGE_LISTS_AVAILABLE = True
except Exception:
    LASER2_LANGUAGE = {}
    LASER3_LANGUAGE = {}
    SPM_LANGUAGE = set()
    LANGUAGE_LISTS_AVAILABLE = False

# Logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("laser_downloader")

DEFAULT_BASE_URL = "https://dl.fbaipublicfiles.com/nllb/laser"


class DownloadError(Exception):
    """Raised when a download cannot be completed after retries."""


class LaserModelDownloader:
    def __init__(self, model_dir: Optional[str] = None, base_url: str = DEFAULT_BASE_URL):
        """
        Args:
            model_dir: directory in which to store models. Defaults to ~/.cache/laser_encoders
            base_url: base url for downloads (default: Amazon S3 public path used by upstream)
        """
        if model_dir is None:
            model_dir = os.path.expanduser("~/.cache/laser_encoders")
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def load_checksums(path: Optional[str]) -> Dict[str, str]:
        """
        Load a checksum file from disk. Format: lines "filename checksum" (whitespace separated).
        Returns dict filename -> checksum (hex string).
        """
        if not path:
            return {}
        checksums: Dict[str, str] = {}
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checksum file not found: {path}")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                fname, chksum = parts[0], parts[1]
                checksums[fname] = chksum.lower()
        return checksums

    @staticmethod
    def compute_sha256(path: str) -> str:
        """Compute SHA256 hex digest of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def verify_checksum(self, local_path: str, expected: str) -> bool:
        """
        Verify SHA256 checksum of local_path against expected (hex string).
        Returns True on match.
        """
        actual = self.compute_sha256(local_path)
        return actual.lower() == expected.lower()

    def get_language_code(self, language_list: dict, lang: str) -> str:
        """
        Resolve a user-friendly language name to the canonical code using provided mapping.
        If mapping is not present and language_list is empty, assume the user supplied a direct code and return it.
        """
        if not language_list:
            # language lists not available in environment; assume user passed a direct code.
            logger.debug("Language list unavailable; using provided lang as code.")
            return lang
        try:
            lang_3_4 = language_list[lang]
            if isinstance(lang_3_4, list):
                options = ", ".join(f"'{opt}'" for opt in lang_3_4)
                raise ValueError(
                    f"Language '{lang}' has multiple options: {options}. Please specify using the 'lang' argument."
                )
            return lang_3_4
        except KeyError:
            # maybe user passed exact code already?
            if lang in language_list.values():
                return lang
            raise ValueError(f"Language name/code '{lang}' not found in language list. Supply supported name.")

    # -------------------------
    # Download helpers
    # -------------------------
    def _make_url(self, filename: str) -> str:
        return f"{self.base_url}/{filename}"

    def _download_with_requests(
        self,
        url: str,
        dest_path: str,
        retries: int = 3,
        timeout: int = 30,
        expected_sha256: Optional[str] = None,
    ) -> None:
        """
        Download url to dest_path reliably with retries. Shows a progress bar (rich or tqdm).
        Raises DownloadError on failure.
        """
        last_exception: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                # use a temporary file in same directory to allow atomic move
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)
                with requests.get(url, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get("Content-Length", 0))
                    # Create temp file
                    with tempfile.NamedTemporaryFile(delete=False, dir=dest_dir) as tf:
                        tmpname = tf.name
                        # Choose progress display
                        if RICH_AVAILABLE:
                            # Use rich progress
                            with Progress(
                                TextColumn("[progress.description]{task.description}"),
                                BarColumn(),
                                DownloadColumn(),
                                TransferSpeedColumn(),
                                TimeRemainingColumn(),
                                console=console,
                            ) as progress:
                                task = progress.add_task(f"Downloading {os.path.basename(dest_path)}", total=total_size)
                                for chunk in r.iter_content(chunk_size=32 * 1024):
                                    if chunk:
                                        tf.write(chunk)
                                        progress.update(task, advance=len(chunk))
                        elif TQDM_AVAILABLE:
                            with tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
                                for chunk in r.iter_content(chunk_size=32 * 1024):
                                    if chunk:
                                        tf.write(chunk)
                                        pbar.update(len(chunk))
                        else:
                            # No progress UI available; write raw
                            for chunk in r.iter_content(chunk_size=32 * 1024):
                                if chunk:
                                    tf.write(chunk)
                # move temporary file to destination
                shutil.move(tmpname, dest_path)

                # optional checksum verification
                if expected_sha256:
                    ok = self.verify_checksum(dest_path, expected_sha256)
                    if not ok:
                        raise DownloadError(f"Checksum mismatch for {dest_path}")
                # success
                return
            except Exception as e:
                last_exception = e
                logger.warning(f"Download attempt {attempt} failed for {url}: {e}")
                # clean up partial file if any
                try:
                    if 'tmpname' in locals() and os.path.exists(tmpname):
                        os.remove(tmpname)
                except Exception:
                    pass
                # backoff
                time.sleep(2 ** min(attempt, 6))
        # after retries
        raise DownloadError(f"Failed to download {url} after {retries} attempts. Last error: {last_exception}")

    def download(self, filename: str, retries: int = 3, timeout: int = 30, checksums: Optional[Dict[str, str]] = None) -> Path:
        """
        High-level download. Skips if file already exists.
        Returns Path to downloaded file or existing file.
        """
        url = self._make_url(filename)
        local_file_path = self.model_dir / filename
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        # If already exists, verify checksum if provided; otherwise skip
        if local_file_path.exists():
            if checksums and filename in checksums:
                expected = checksums[filename]
                try:
                    if self.verify_checksum(str(local_file_path), expected):
                        logger.info(f" - {filename} already downloaded and verified")
                        return local_file_path
                    else:
                        logger.warning(f" - {filename} exists but checksum mismatch; re-downloading")
                        local_file_path.unlink()
                except Exception as e:
                    logger.warning(f" - {filename} exists but verification failed with error {e}; re-downloading")
                    try:
                        local_file_path.unlink()
                    except Exception:
                        pass
            else:
                logger.info(f" - {filename} already downloaded")
                return local_file_path

        logger.info(f" - Downloading {filename} from {url}")
        expected = checksums.get(filename) if checksums else None
        self._download_with_requests(url, str(local_file_path), retries=retries, timeout=timeout, expected_sha256=expected)
        logger.info(f" - Saved to {local_file_path}")
        return local_file_path

    # -------------------------
    # High-level download methods
    # -------------------------
    def download_laser2(self, checksums: Optional[Dict[str, str]] = None) -> None:
        """
        Download laser2 model files (laser2.pt, laser2.spm, laser2.cvocab)
        """
        self.download("laser2.pt", checksums=checksums)
        self.download("laser2.spm", checksums=checksums)
        self.download("laser2.cvocab", checksums=checksums)

    def download_laser3(self, lang: str, spm: bool = True, checksums: Optional[Dict[str, str]] = None) -> None:
        """
        Download laser3 model files for language 'lang'.
        `lang` may be a language name mapped by LASER3_LANGUAGE or a direct language code.
        If spm=True, attempt to download SPM/cvocab if available; otherwise fall back to laser2 SPM.
        """
        # Resolve language code if mapping available
        if LANGUAGE_LISTS_AVAILABLE:
            resolved = self.get_language_code(LASER3_LANGUAGE, lang)
        else:
            resolved = lang  # assume direct code already
        # filename pattern: laser3-{lang}.v1.pt
        model_filename = f"laser3-{resolved}.v1.pt"
        self.download(model_filename, checksums=checksums)
        if spm:
            if LANGUAGE_LISTS_AVAILABLE and resolved in SPM_LANGUAGE:
                # some languages have their own SPMs
                self.download(f"laser3-{resolved}.v1.spm", checksums=checksums)
                self.download(f"laser3-{resolved}.v1.cvocab", checksums=checksums)
            else:
                # fall back to laser2.spm/cvocab
                self.download("laser2.spm", checksums=checksums)
                self.download("laser2.cvocab", checksums=checksums)

    # -------------------------
    # UX helpers
    # -------------------------
    def list_models(self) -> None:
        """
        Print a list of downloadable models. If language lists are available, enumerate laser3 per language.
        """
        logger.info(f"Models available at base url: {self.base_url}")
        # always list laser2
        print(" - laser2.pt")
        print(" - laser2.spm")
        print(" - laser2.cvocab")
        if LANGUAGE_LISTS_AVAILABLE and LASER3_LANGUAGE:
            print("\nlaser3 models available (per language):")
            # provide a small readable listing: language_name -> code
            # LASER3_LANGUAGE may map names to codes (or lists)
            for name, code in sorted(LASER3_LANGUAGE.items()):
                print(f" - {name}: {code}")
            print("\nTo download: --laser laser3 --lang <language_name> (or specify exact code if ambiguous)")
        else:
            print("\nNote: language lists not available in this environment.")
            print("You can still download laser3 by specifying a direct language code expected by the remote filenames.")
            print("Example: --laser laser3 --lang eng_Latn")

    def clear_cache(self, confirm: bool = True) -> None:
        """
        Remove all cached models in the model_dir.
        """
        if not self.model_dir.exists():
            logger.info("Cache directory does not exist; nothing to clear.")
            return
        if confirm:
            answer = input(f"Delete all cached models in {self.model_dir}? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                logger.info("Aborted.")
                return
        # remove and recreate
        shutil.rmtree(self.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cache cleared.")

    # -------------------------
    # Entrypoint / main
    # -------------------------
    def main(self, argv: Optional[list] = None) -> None:
        parser = argparse.ArgumentParser(description="LASER: Download Laser models (robust downloader)")
        parser.add_argument("--laser", type=str, choices=["laser2", "laser3"], help="Laser model family to download")
        parser.add_argument("--lang", type=str, help="Language name or code (for laser3). If language lists are not present, supply direct code.")
        parser.add_argument("--no-spm", action="store_true", help="Do not download SPM/cvocab files (skip SPM files).")
        parser.add_argument("--model-dir", type=str, help="Directory to download models to (overrides default).")
        parser.add_argument("--list", action="store_true", help="List available models (if language lists are available).")
        parser.add_argument("--clear-cache", action="store_true", help="Clear the download cache directory.")
        parser.add_argument("--checksums", type=str, help="Optional checksum file (lines: filename checksum), uses SHA256.")
        parser.add_argument("--retries", type=int, default=3, help="Number of download retries for each file.")
        parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds for each HTTP request.")
        parser.add_argument("--yes", "-y", action="store_true", help="Automatic yes to prompts (used with --clear-cache).")

        args = parser.parse_args(argv)

        # If model-dir provided at CLI level, override
        if args.model_dir:
            self.model_dir = Path(args.model_dir)
            self.model_dir.mkdir(parents=True, exist_ok=True)

        # load checksums if provided
        checksums = {}
        if args.checksums:
            try:
                checksums = self.load_checksums(args.checksums)
            except Exception as e:
                logger.error(f"Failed to load checksums file: {e}")
                return

        # Handle listing and clearing early
        if args.list:
            self.list_models()
            return
        if args.clear_cache:
            self.clear_cache(confirm=not args.yes)
            return

        # If no laser specified but lang supplied, infer
        laser_choice = args.laser
        if not laser_choice and args.lang:
            # if language exists in LASER3_LANGUAGE prefer laser3, else laser2 if in LASER2_LANGUAGE
            if LANGUAGE_LISTS_AVAILABLE:
                if args.lang in LASER3_LANGUAGE or args.lang in LASER2_LANGUAGE:
                    # we prefer laser3 if present
                    if args.lang in LASER3_LANGUAGE:
                        laser_choice = "laser3"
                    else:
                        laser_choice = "laser2"
            else:
                # Without language lists we assume laser3 when lang present
                laser_choice = "laser3"

        if not laser_choice:
            parser.print_help()
            logger.error("You must specify --laser laser2|laser3 or use --list/--clear-cache.")
            return

        logger.info(f"Using model_dir: {self.model_dir}")
        spm_flag = not args.no_spm

        try:
            if laser_choice == "laser2":
                self.download_laser2(checksums=checksums)
            elif laser_choice == "laser3":
                if not args.lang:
                    raise ValueError("For laser3 you must specify --lang <language_name_or_code>")
                # If language lists available, resolve; otherwise assume direct code
                if LANGUAGE_LISTS_AVAILABLE:
                    resolved_lang = self.get_language_code(LASER3_LANGUAGE, args.lang)
                else:
                    resolved_lang = args.lang
                # call download with the retries/timeouts propagated via underlying download
                # note: lower-level download method currently uses its own retries and timeout arguments,
                # but we pass checksums along. For now we simply call the high-level method.
                self.download_laser3(lang=resolved_lang, spm=spm_flag, checksums=checksums)
            else:
                raise ValueError(f"Unsupported laser option: {laser_choice}")
        except DownloadError as de:
            logger.error(f"Download failed: {de}")
            sys.exit(2)
        except Exception as e:
            logger.exception(f"Error: {e}")
            sys.exit(1)


def main_entry() -> None:
    downloader = LaserModelDownloader()
    downloader.main()


if __name__ == "__main__":
    main_entry()
