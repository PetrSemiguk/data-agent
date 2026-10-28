"""Print a quick overview of every CSV/Excel file in data/."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_EXTENSIONS = {".csv", ".xlsx"}
CSV_ENCODINGS = ["utf-8", "latin-1"]


def read_file(file: Path) -> pd.DataFrame:
    if file.suffix.lower() == ".xlsx":
        return pd.read_excel(file)

    # Try encodings in order; latin-1 never fails, so it is the fallback.
    for encoding in CSV_ENCODINGS:
        try:
            df = pd.read_csv(file, encoding=encoding)
            print(f"Encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            print(f"Encoding {encoding} failed, trying next")
    raise ValueError(f"Could not decode {file.name}")


def inspect(file: Path) -> None:
    print("=" * 80)
    print(f"File: {file.name}")
    df = read_file(file)

    rows, cols = df.shape
    print(f"Shape: {rows} rows x {cols} columns")

    print("\nColumns and types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print(missing.to_string() if not missing.empty else "none")

    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    print()


def main() -> None:
    # Show all columns side by side instead of truncating them.
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    for file in sorted(DATA_DIR.iterdir()):
        if file.is_file() and file.suffix.lower() in DATA_EXTENSIONS:
            inspect(file)


if __name__ == "__main__":
    main()
