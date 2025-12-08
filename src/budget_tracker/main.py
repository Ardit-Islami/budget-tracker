from typing import List
from pathlib import Path

from budget_tracker.config import settings, logger
from budget_tracker import etl
from budget_tracker.categorise import process_staged_files


def _is_tabular_file(path: Path) -> bool:
    return path.suffix.lower() in {".csv", ".xls", ".xlsx"}


def run_etl_on_raw_files() -> None:
    raw_dir: Path = settings.raw_dir

    raw_files: List[Path] = sorted(
        f
        for f in raw_dir.iterdir()
        if f.is_file() and _is_tabular_file(f) and not f.name.startswith("~$")
    )

    if not raw_files:
        logger.warning(f"No raw CSV/Excel files found in {raw_dir}. Nothing to ETL.")
        return

    logger.info(f"Found {len(raw_files)} raw files in {raw_dir}")

    for f in raw_files:
        logger.info(f"Running ETL on: {f.name}")
        etl.process_file(f)


def main() -> None:
    # 0. Setup
    settings.ensure_directories()
    settings.setup_logging()

    # 1. ETL: raw -> staged
    run_etl_on_raw_files()

    # 2. Categorization: staged -> categorized + master.xlsx
    process_staged_files()


if __name__ == "__main__":
    main()
