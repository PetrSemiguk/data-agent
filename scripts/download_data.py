"""Download test datasets from Kaggle and copy their data files into data/."""

import shutil
from pathlib import Path

import kagglehub

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_EXTENSIONS = {".csv", ".xlsx"}


def copy_data_files(source_dir: Path) -> None:
    # Search subfolders too, but copy files flat into data/.
    for file in Path(source_dir).rglob("*"):
        if file.is_file() and file.suffix.lower() in DATA_EXTENSIONS:
            shutil.copy2(file, DATA_DIR / file.name)
            print(f"Copied: {file.name}")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    path = kagglehub.dataset_download("vivek468/superstore-dataset-final")
    print("Path to dataset files:", path)
    copy_data_files(path)

    path = kagglehub.dataset_download("yasserh/titanic-dataset")
    print("Path to dataset files:", path)
    copy_data_files(path)

    path = kagglehub.dataset_download("mashlyn/online-retail-ii-uci")
    print("Path to dataset files:", path)
    copy_data_files(path)

    # Hold-out datasets from other domains: used only to evaluate the agent,
    # never when writing prompts or skills.
    path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
    print("Path to dataset files:", path)
    copy_data_files(path)

    path = kagglehub.dataset_download("sumanthvrao/daily-climate-time-series-data")
    print("Path to dataset files:", path)
    copy_data_files(path)

    # Show what ended up in data/.
    print("\nFiles in data/:")
    for file in sorted(DATA_DIR.iterdir()):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name:40} {size_mb:8.2f} MB")


if __name__ == "__main__":
    main()
