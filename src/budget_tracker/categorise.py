import pandas as pd
from pathlib import Path
from typing import List

from budget_tracker.config import settings, logger
from budget_tracker.engine import engine

# PER-FILE PROCESSOR
def process_staged_files() -> None:
    staged_files: List[Path] = sorted(settings.staged_dir.glob("Staged_*.xlsx"))

    if not staged_files:
        logger.warning(f"No staged files found in {settings.staged_dir}")
        return

    new_categorized_dfs: List[pd.DataFrame] = []

    for file_path in staged_files:
        filename = file_path.name
        logger.info(f"Categorizing: {filename}")

        try:
            df = pd.read_excel(file_path)

            if df.empty:
                logger.warning(f"Skipping {filename} - file is empty.")
                continue

            original_columns = list(df.columns)
            df.columns = [str(c).strip().lower() for c in df.columns]

            if "description" not in df.columns:
                logger.warning(
                    f"Skipping {filename} - Column 'description' not found. "
                    f"Columns present: {original_columns}"
                )
                continue

            # --- APPLY CATEGORIZATION ---
            def apply_categorization(desc) -> pd.Series:
                if pd.isna(desc) or not str(desc).strip():
                    return pd.Series(
                        ["Uncategorized", "", "none", 0],
                        index=["category", "subcategory", "rule_match", "confidence"],
                    )

                result = engine.categorize(str(desc))
                return pd.Series(
                    [result.category, result.subcategory, result.matched_by, result.confidence],
                    index=["category", "subcategory", "rule_match", "confidence"],
                )

            logger.info(f"   -> Running engine on {len(df)} transactions...")

            df[["category", "subcategory", "rule_match", "confidence"]] = (
                df["description"].apply(apply_categorization)
            )

            new_categorized_dfs.append(df)

            output_name = filename.replace("Staged_", "Categorized_")
            output_path = settings.staged_dir / output_name
            df.to_excel(output_path, index=False)
            logger.info(f"   -> Saved {output_name}")

        except Exception:
            logger.exception(f"Failed to process {filename}")

    # --- UPDATE MASTER ---
    if new_categorized_dfs:
        update_master_file(new_categorized_dfs)
    else:
        logger.warning("No categorized dataframes produced. Master file not updated.")


# ==================
# MASTER FILE MANAGER
def update_master_file(new_dfs: List[pd.DataFrame]) -> None:
    logger.info("Updating Master File...")

    try:
        # 1. Load Existing Master (if it exists)
        if settings.master_file.exists():
            old_master = pd.read_excel(settings.master_file)
            logger.info(f"   -> Loaded {len(old_master)} existing rows.")
        else:
            old_master = pd.DataFrame()
            logger.info("   -> No existing master file found. Creating new one.")

        # 2. Combine with New Data
        combined_df = pd.concat([old_master] + new_dfs, ignore_index=True)

        # 3. Deduplicate
        dedupe_cols = ["date", "description", "paid_out", "paid_in", "source"]
        missing = [c for c in dedupe_cols if c not in combined_df.columns]

        if missing:
            logger.warning(
                "   -> Skipping full dedupe: master is missing expected columns "
                f"{missing}. Available columns: {list(combined_df.columns)}"
            )
        else:
            before_count = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=dedupe_cols, keep="first")
            removed_count = before_count - len(combined_df)
            if removed_count > 0:
                logger.info(f"   -> Removed {removed_count} duplicates.")

        # 4. Sort (Newest Date First) with robust parsing
        if "date" in combined_df.columns:
            combined_df["date"] = pd.to_datetime(
                combined_df["date"], errors="coerce"
            )

            bad_dates = combined_df["date"].isna().sum()
            if bad_dates > 0:
                logger.warning(
                    f"   -> {bad_dates} rows have invalid dates (set to NaT). "
                    "Check source files if this is unexpected."
                )

            combined_df = combined_df.sort_values(by="date", ascending=False)

        # 5. Save
        combined_df.to_excel(settings.master_file, index=False)
        logger.info(f"Master File saved with {len(combined_df)} total transactions.")

    except Exception:
        logger.exception("Failed to update Master File")