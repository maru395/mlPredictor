"""Download the public Kaggle and Hugging Face MLBB datasets."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"

HF_FILES = {
    "hf_mpl_id_s14.csv": "https://huggingface.co/datasets/z4fL/mpl_s14_dataset/resolve/main/mpl_id_s14.csv?download=true",
    "hf_mpl_ph_s14.csv": "https://huggingface.co/datasets/z4fL/mpl_s14_dataset/resolve/main/mpl_ph_s14.csv?download=true",
}
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/bcakra/mobile-legend-tournament-match-mpl-philippines"
KAGGLE_MEMBER = "MPL Philippines Season 13 - BoxMatch.csv"
KAGGLE_DESTINATION = "kaggle_mpl_ph_s13_boxmatch.csv"


def path_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def download(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        print(f"Already present: {path_label(destination)}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "mlbb-predictor/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response, tempfile.NamedTemporaryFile(
        delete=False, dir=destination.parent, suffix=".part"
    ) as temporary:
        shutil.copyfileobj(response, temporary)
        temporary_path = Path(temporary.name)
    temporary_path.replace(destination)
    print(f"Downloaded: {path_label(destination)}", flush=True)


def extract_kaggle_archive(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zipped:
        if KAGGLE_MEMBER not in zipped.namelist():
            raise FileNotFoundError(f"{KAGGLE_MEMBER!r} is missing from the Kaggle archive.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipped.open(KAGGLE_MEMBER) as source, tempfile.NamedTemporaryFile(
            delete=False, dir=destination.parent, suffix=".part"
        ) as temporary:
            shutil.copyfileobj(source, temporary)
            temporary_path = Path(temporary.name)
        temporary_path.replace(destination)


def download_kaggle(force: bool = False) -> None:
    destination = RAW_DIR / KAGGLE_DESTINATION
    if destination.exists() and not force:
        print(f"Already present: {path_label(destination)}")
        return
    cached_archive = RAW_DIR / "kaggle_mpl_ph_s13.zip"
    if cached_archive.exists() and not force:
        extract_kaggle_archive(cached_archive, destination)
    else:
        with tempfile.TemporaryDirectory() as temporary_dir:
            archive = Path(temporary_dir) / "kaggle.zip"
            download(KAGGLE_URL, archive, force=True)
            extract_kaggle_archive(archive, destination)
    print(f"Extracted: {path_label(destination)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace existing downloads")
    args = parser.parse_args()

    for filename, url in HF_FILES.items():
        download(url, RAW_DIR / filename, force=args.force)
    download_kaggle(force=args.force)


if __name__ == "__main__":
    main()
