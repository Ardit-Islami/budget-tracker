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

    @field_validator("date", mode="before")
    def parse_date(cls, v) -> datetime.date:
        if isinstance(v, datetime.datetime):
            return v.date()
        if isinstance(v, datetime.date):
            return v

        s = str(v).strip()
        if not s:
            raise ValueError("Empty date string")

        try:
            ts = pd.to_datetime(s, dayfirst=True, errors="raise")
            return ts.date()
        except Exception as e:
            raise ValueError(f"Unparseable date: {s}") from e

    # Fix messy currency strings before validation
    @field_validator('paid_out', 'paid_in', mode='before')
    def clean_currency(cls, v):
        if pd.isna(v) or v == "":
            return 0.0
        try:
            # Might add a debug log here.
            return abs(float(str(v).replace('£', '').replace(',', '')))
        except ValueError:
            return 0.0

# ==================
# HELPER FUNCTIONS
def _normalize_column_name(name: object) -> str:
    # Turn anything into a clean, comparable column name
    s = str(name)
    s = s.replace("\ufeff", "").strip()
    return s

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
            logger.debug(f"Scanning first {scan_limit} rows in {file_path.name} (Excel)...")

            df_preview = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=scan_limit)
            header_idx = _find_header_index(df_preview, file_path)
            return pd.read_excel(file_path, sheet_name=sheet_name, header=header_idx,)

        if suffix == ".csv":
            logger.debug(f"Scanning first {scan_limit} rows in {file_path.name} (CSV)...")

            def _log_bad_line(bad_line: list[str]) -> None:
                logger.warning(f"Skipping malformed line in {file_path.name}: {bad_line}")

            df_preview = pd.read_csv(file_path, header=None, nrows=scan_limit, sep=None, engine="python", encoding="cp1252",on_bad_lines=_log_bad_line)
            header_idx = _find_header_index(df_preview, file_path)

            return pd.read_csv(file_path, header=header_idx, sep=None, engine="python", encoding="cp1252", on_bad_lines=_log_bad_line)

        logger.warning(f"Unsupported file type for header scan: {file_path.name}")
        return None

    except Exception:
        logger.exception(f"Failed to read file {file_path}")
        return None

# ==================
# TRANSFORMATIONS
def transform_aqua(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Merge Desc + Note
    df['Description'] = df['Description'].fillna('') + ' ' + df['Note'].fillna('')
    df['Description'] = df['Description'].str.strip()
    
    # Logic: Positive = Spend (Paid Out), Negative = Repayment (Paid In)
    amount = df['Amount(GBP)'] 
    df['paid_out'] = np.where(amount > 0, amount, 0)
    df['paid_in'] = np.where(amount < 0, amount, 0) # Negative numbers
    return df

def transform_natwest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    val = df['Value']
    df['paid_out'] = np.where(val > 0, val, 0)
    df['paid_in'] = np.where(val < 0, val, 0)
    return df

def transform_nationwide(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['paid_out'] = df['Paid out']
    df['paid_in'] = df['Paid in']
    return df

# ==================
# MAPPING
BANK_CONFIGS = [
    {"cols": settings.cols_aqua, "source": "Aqua", "type": "Credit", "func": transform_aqua},
    {"cols": settings.cols_natwest, "source": "Natwest", "type": "Credit", "func": transform_natwest},
    {"cols": settings.cols_nationwide, "source": "Nationwide", "type": "Debit","func": transform_nationwide}
]

# ==================
# MAIN PROCESSING
def process_file(file_path):
    filename = file_path.name
    logger.info(f"Processing file: {filename}")

    # 1. Load
    df = find_header_row(file_path)
    if df is None: 
        return
    
    logger.info(f"Columns for {filename}: {list(df.columns)}")

    # Normalize all column names once
    df.rename(columns=_normalize_column_name, inplace=True)
    cols = set(df.columns)
    
    # 2. Identify
    transformed_df = None
    source = None
    acct_type = None

    for config in BANK_CONFIGS:
        # Check if this bank's columns exist in the file
        if config["cols"].issubset(cols):
            logger.info(f"-> Identified as: {config['source'].upper()}")
            
            # Execute the specific function for this bank
            transformed_df = config["func"](df) 
            source = config["source"]
            acct_type = config["type"]
            break # Stop looking after the first match

    # If loop finished and we found nothing
    if transformed_df is None or source is None or acct_type is None:
        logger.warning(f"Skipping {filename}: Could not match columns to any known bank.")
        return

    # 3. Validate Rows
    valid_rows = []
    
    for idx, row in transformed_df.iterrows():
        try:
            txn = TransactionSchema(
                date=row['Date'],
                description=str(row['Description']),
                paid_out=row['paid_out'],
                paid_in=row['paid_in'],
                source=source,
                account_type=acct_type
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

    # 4. Save
    df_final = pd.DataFrame(valid_rows)
    stem = file_path.stem
    output_path = settings.staged_dir / f"Staged_{stem}.xlsx"
    df_final.to_excel(output_path, index=False)
    
    logger.info(f"Saved {len(df_final)} cleaned rows to {output_path.name}")