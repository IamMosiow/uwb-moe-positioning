"""Dataset download and archive extraction module."""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path
from uwb_moe.utils.logging import get_logger

logger = get_logger("uwb_moe.download")

DEFAULT_UWB_URL = (
    "https://www.iis.fraunhofer.de/content/dam/iis/de/doc/lv/dataanalytics/uwb_dataset.zip"
)


def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
    """Console progress reporter for urllib download."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(
            f"\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{total_mb:.1f} MB)"
        )
    else:
        mb_downloaded = downloaded / (1024 * 1024)
        sys.stdout.write(f"\rDownloading: {mb_downloaded:.1f} MB")
    sys.stdout.flush()


def download_dataset(
    url: str = DEFAULT_UWB_URL,
    dest_dir: str | Path = "data/raw",
    extract: bool = True,
    force: bool = False,
) -> Path:
    """Download and extract the UWB dataset zip file.

    Args:
        url: Remote URL of the dataset zip archive.
        dest_dir: Directory where data will be stored.
        extract: Whether to unzip the downloaded archive.
        force: Whether to overwrite existing files.

    Returns:
        Path to the extracted directory containing HDF5 files.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    zip_filename = dest_path / "uwb_dataset.zip"

    if zip_filename.exists() and not force:
        logger.info("Dataset archive already exists at: %s", zip_filename)
    else:
        logger.info("Starting download from: %s", url)
        urllib.request.urlretrieve(url, str(zip_filename), reporthook=_reporthook)
        print()  # newline after progress bar
        logger.info("Download completed: %s", zip_filename)

    if extract:
        logger.info("Extracting archive into: %s", dest_path)
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(dest_path)
        logger.info("Extraction complete.")

    return dest_path


def main() -> None:
    """CLI entry point for dataset download."""
    parser = argparse.ArgumentParser(description="Download Fraunhofer UWB Dataset")
    parser.add_argument("--url", type=str, default=DEFAULT_UWB_URL, help="Dataset URL")
    parser.add_argument(
        "--dest-dir", type=str, default="data/raw", help="Destination folder"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force redownload if file exists"
    )
    args = parser.parse_args()

    download_dataset(url=args.url, dest_dir=args.dest_dir, extract=True, force=args.force)


if __name__ == "__main__":
    main()
