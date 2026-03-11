"""Download helper for Dewesoft DWDataReader binary packages.

This script downloads the official DWDataReader ZIP archive next to the
``DeweFileLibrary`` folder, then asks the user to extract the required
``.dll``/``.so`` files into ``DeweFileLibrary`` manually.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
from pathlib import Path

DOWNLOAD_URL = "https://downloads.dewesoft.com/developers/dwdatareader/DWDataReader_v5_0_4.zip"
PACKAGE_DIR = Path(__file__).resolve().parent
LIB_DIR = PACKAGE_DIR / "DeweFileLibrary"
DEFAULT_ARCHIVE = PACKAGE_DIR / "DWDataReader_v5_0_4.zip"


def download_file(url: str, destination: Path) -> None:
    """Download ``url`` to ``destination`` using only the Python stdlib."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the official Dewesoft DWDataReader ZIP archive. "
            "After download, manually extract the native binaries into "
            "OSCC_postprocessing/dewe/DeweFileLibrary."
        )
    )
    parser.add_argument(
        "--url",
        default=DOWNLOAD_URL,
        help="Override the download URL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Where to save the downloaded ZIP archive.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the target ZIP if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_path = args.output.resolve()

    if archive_path.exists() and not args.force:
        print(f"Archive already exists: {archive_path}")
    else:
        print(f"Downloading DWDataReader archive from: {args.url}")
        print(f"Saving to: {archive_path}")
        download_file(args.url, archive_path)
        print("Download completed.")

    LIB_DIR.mkdir(parents=True, exist_ok=True)
    print()
    print("Next step:")
    print(f"1. Extract the ZIP manually: {archive_path}")
    print(f"2. Copy the native reader files into: {LIB_DIR}")
    print("3. Keep only the binaries you need for your platform.")
    print()
    print("Expected Windows files include:")
    print("  - DWDataReaderLib64.dll")
    print("  - DWDataReaderLib.dll")
    print()
    print("The package intentionally does not bundle these binaries for PyPI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
