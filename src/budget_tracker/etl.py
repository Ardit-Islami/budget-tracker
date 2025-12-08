import re
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
import datetime
from pydantic import BaseModel, field_validator, ValidationError

from budget_tracker.config import settings, logger


class TransactionSchema(BaseModel):
    date: datetime.date
    description: str
    paid_out: float = 0.0
    paid_in: float = 0.0
    source: str
    account_type: str
    category: str = "Uncategorized"
    subcategory: str = ""

    # Standardise Date Columns
    @field_validator("date", mode="before")
    def parse_date(cls, v) -> datetime.date:
        if isinstance(v, datetime.datetime):
            return v.date()
        if isinstance(v, datetime.date):
            return v

        date_str = str(v).strip()
        if not date_str:
            raise ValueError("Empty date string")

        try:
            parsed_timestamp = pd.to_datetime(date_str, dayfirst=True, errors="raise")
            return parsed_timestamp.date()
        except Exception as e:
            raise ValueError(f"Unparseable date: {date_str}") from e

    # Clean Currency Strings
    @field_validator("paid_out", "paid_in", mode="before")
    def clean_currency(cls, v):
        if pd.isna(v) or str(v).strip() == "":
            return 0.0

        try:
            currency_str = str(v)
            cleaned_currency = re.sub(r"[^\d.-]", "", currency_str)

            if not cleaned_currency:
                return 0.0

            return abs(float(cleaned_currency))

        except Exception as e:
            logger.warning(f"Failed to parse currency: '{v}' -> Error: {e}")
            return 0.0


# HELPER FUNCTIONS
def _normalize_column_name(name: object) -> str:
    column_name_str = str(name)
    column_name_normalised = column_name_str.replace("\ufeff", "").strip()
    return column_name_normalised


def _find_header_index(df_preview: pd.DataFrame, file_path: Path) -> int:
    for i, row in enumerate(df_preview.itertuples(index=False)):
        row_text = " ".join(str(v).lower() for v in row)
        if "date" in row_text:
            logger.debug(f"Found header at row {i} in {file_path.name}")
            return i

    logger.debug(f"No 'Date' header found in {file_path.name}. Defaulting to row 0.")
    return 0


def find_header_row(file_path: Path, sheet_name: int = 0) -> Optional[pd.DataFrame]:
    scan_limit = settings.header_scan_depth
    suffix = file_path.suffix.lower()

    try:
        if suffix in {".xls", ".xlsx"}:
            logger.debug(
                f"Scanning first {scan_limit} rows in {file_path.name} (Excel)..."
            )

            df_preview = pd.read_excel(
                file_path, sheet_name=sheet_name, header=None, nrows=scan_limit
            )
            header_idx = _find_header_index(df_preview, file_path)

            return pd.read_excel(file_path, sheet_name=sheet_name, header=header_idx)

        if suffix == ".csv":
            logger.debug(
                f"Scanning first {scan_limit} rows in {file_path.name} (CSV)..."
            )

            def _log_bad_line(bad_line: list[str]) -> None:
                logger.warning(
                    f"Skipping malformed line in {file_path.name}: {bad_line}"
                )

            df_preview = pd.read_csv(
                file_path,
                header=None,
                nrows=scan_limit,
                sep=None,
                engine="python",
                encoding="cp1252",
                on_bad_lines=_log_bad_line,
            )
            header_idx = _find_header_index(df_preview, file_path)

            return pd.read_csv(
                file_path,
                header=header_idx,
                sep=None,
                engine="python",
                encoding="cp1252",
                on_bad_lines=_log_bad_line,
            )

        logger.warning(f"Unsupported file type for header scan: {file_path.name}")
        return None

    except Exception:
        logger.exception(f"Failed to read file {file_path}")
        return None


# TRANSFORMATIONS
def transform_aqua(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Description"] = df["Description"].str.cat(df["Note"], sep=" ", na_rep="")
    df["Description"] = df["Description"].str.strip()

    # Aqua Logic: Positive = Spend (Paid Out), Negative = Repayment (Paid In)
    amount = df["Amount(GBP)"]
    df["paid_out"] = np.where(amount > 0, amount, 0)
    df["paid_in"] = np.where(amount < 0, amount, 0)
    return df


def transform_natwest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    val = df["Value"]

    df["Type"] = df["Type"].fillna("").astype(str).str.strip()
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()

    enrich_types = {"CASH"}
    condition = df["Type"].isin(enrich_types)

    df.loc[condition, "Description"] = (
        df.loc[condition, "Type"].astype("str")
        + ": "
        + df.loc[condition, "Description"]
    )

    # Natwest Logic: Positive = Spend (Paid Out), Negative = Repayment (Paid In)
    df["paid_out"] = np.where(val > 0, val, 0)
    df["paid_in"] = np.where(val < 0, val, 0)
    return df


def transform_nationwide(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Transaction type"] = df["Transaction type"].fillna("").astype(str).str.strip()
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()

    enrich_types = {
        "Foreign currency transaction fee",
        "ATM Withdrawal LINK",
        "ATM Withdrawal LLOYDS",
        "ATM Withdrawal",
        "ATM Withdrawal NOTEMACHINE LTD",
    }
    condition = df["Transaction type"].isin(enrich_types)

    df.loc[condition, "Description"] = (
        df.loc[condition, "Transaction type"].astype("str")
        + ": "
        + df.loc[condition, "Description"]
    )

    df["paid_out"] = df["Paid out"]
    df["paid_in"] = df["Paid in"]
    return df


# MAPPING
BANK_CONFIGS = [
    {
        "cols": settings.cols_aqua,
        "source": "Aqua",
        "type": "Credit",
        "func": transform_aqua,
    },
    {
        "cols": settings.cols_natwest,
        "source": "Natwest",
        "type": "Credit",
        "func": transform_natwest,
    },
    {
        "cols": settings.cols_nationwide,
        "source": "Nationwide",
        "type": "Debit",
        "func": transform_nationwide,
    },
]


# MAIN PROCESSING
def process_file(file_path):
    filename = file_path.name
    logger.info(f"Processing file: {filename}")

    df = find_header_row(file_path)
    if df is None:
        return
    logger.info(f"Columns for {filename}: {list(df.columns)}")

    # Normalize column names
    df.rename(columns=_normalize_column_name, inplace=True)
    cols = set(df.columns)

    # Identify
    transformed_df = None
    source = None
    acct_type = None

    for config in BANK_CONFIGS:
        if config["cols"].issubset(cols):
            logger.info(f"-> Identified as: {config['source'].upper()}")

            transformed_df = config["func"](df)
            source = config["source"]
            acct_type = config["type"]
            break

    if transformed_df is None or source is None or acct_type is None:
        logger.warning(
            f"Skipping {filename}: Could not match columns to any known bank."
        )
        return

    # Validate Rows
    valid_rows = []

    for idx, row in transformed_df.iterrows():
        try:
            txn = TransactionSchema(
                date=row["Date"],
                description=str(row["Description"]),
                paid_out=row["paid_out"],
                paid_in=row["paid_in"],
                source=source,
                account_type=acct_type,
            )
            valid_rows.append(txn.model_dump())

        except ValidationError as e:
            logger.warning(f"Row {idx} rejected in {filename}: {e}")
            continue
        except Exception as e:
            logger.exception(f"Unexpected crash on row {idx}: {e}")
            continue

    if not valid_rows:
        logger.warning(f"No valid rows found in {filename}")
        return

    # Save
    df_final = pd.DataFrame(valid_rows)
    stem = file_path.stem
    output_path = settings.staged_dir / f"Staged_{stem}.xlsx"
    df_final.to_excel(output_path, index=False)

    logger.info(f"Saved {len(df_final)} cleaned rows to {output_path.name}")
