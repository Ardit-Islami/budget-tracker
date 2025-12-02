import sys
import logging
from pathlib import Path
from typing import Set, ClassVar
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='BUDGET_')
    app_name: str = "Budget Tracker"
    debug_mode: bool = False
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def staged_dir(self) -> Path:
        return self.data_dir / "staged"

    @property
    def master_dir(self) -> Path:
        return self.data_dir / "master"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    # --- Critical Files ---
    @property
    def master_file(self) -> Path:
        return self.master_dir / "Master_Transactions.xlsx"
    
    @property
    def merchants_file(self) -> Path:
        return self.master_dir / "merchants.json"
    
    @property
    def patterns_file(self) -> Path:
        return self.master_dir / "patterns.json"

    # Bank Identifiers (Constants)
    cols_aqua: ClassVar[Set[str]] = {"Amount(GBP)", "Note"}
    cols_natwest: ClassVar[Set[str]] = {"Value", "Account Name", "Account Number"}
    cols_nationwide: ClassVar[Set[str]] = {"Transaction type", "Paid out", "Paid in"}

    # Scan for header
    header_scan_depth: int = 20

    # Setup Logic
    def ensure_directories(self) -> None:
        dirs = [self.raw_dir, self.staged_dir, self.master_dir, self.logs_dir]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def setup_logging(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_level = logging.DEBUG if self.debug_mode else logging.INFO

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(self.logs_dir / "app.log", mode="a")], force=True)

settings = Settings()
logger = logging.getLogger(settings.app_name)