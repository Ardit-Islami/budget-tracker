import json
from typing import List
from pydantic import BaseModel, field_validator, ValidationError

from budget_tracker.config import settings, logger

# DATA MODELS
class MerchantRule(BaseModel):
    id: str
    name: str
    category: str
    subcategory: str = ""
    aliases: List[str]

    @field_validator("aliases", mode="before")
    def normalize_aliases(cls, v):
        if not isinstance(v, list):
            return []
        normalized = []
        for alias in v:
            s = str(alias).strip().lower()
            if s:
                normalized.append(s)
        return normalized


class PatternRule(BaseModel):
    pattern: str
    category: str
    subcategory: str = ""
    weight: float = 1.0

    @field_validator("pattern", mode="before")
    def normalize_pattern(cls, v):
        return str(v).strip().lower()


class CategorizationResult(BaseModel):
    category: str
    subcategory: str
    confidence: int
    matched_by: str

class Candidate(BaseModel):
    category: str
    subcategory: str
    score: int
    source: str

# ENGINE
class CategorizationEngine:
    def __init__(self) -> None:
        self.merchants: List[MerchantRule] = []
        self.patterns: List[PatternRule] = []
        self.load_data()

    def load_data(self) -> None:
        # Load Merchants
        if settings.merchants_file.exists():
            try:
                with open(settings.merchants_file, "r") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    logger.error(
                        f"Merchants file {settings.merchants_file} is not a list. Got {type(data)}."
                    )
                    data = []

                valid_merchants: List[MerchantRule] = []
                for item in data:
                    try:
                        valid_merchants.append(MerchantRule(**item))
                    except ValidationError as e:
                        logger.warning(f"Skipping invalid merchant rule {item}: {e}")

                self.merchants = valid_merchants
                logger.info(f"Loaded {len(self.merchants)} merchants.")

            except Exception as e:
                logger.error(f"Failed to load merchants: {e}")
        else:
            logger.warning(f"Merchants file missing at {settings.merchants_file}")

        # Load Patterns
        if settings.patterns_file.exists():
            try:
                with open(settings.patterns_file, "r") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    logger.error(
                        f"Patterns file {settings.patterns_file} is not a list. Got {type(data)}."
                    )
                    data = []

                valid_patterns: List[PatternRule] = []
                for item in data:
                    try:
                        valid_patterns.append(PatternRule(**item))
                    except ValidationError as e:
                        logger.warning(f"Skipping invalid pattern rule {item}: {e}")

                self.patterns = valid_patterns
                logger.info(f"Loaded {len(self.patterns)} patterns.")

            except Exception as e:
                logger.error(f"Failed to load patterns: {e}")
        else:
            logger.warning(f"Patterns file missing at {settings.patterns_file}")

    def _score_match(self, text: str, pattern: str) -> int:
        if len(pattern) < 2:
            return 0
        if pattern in text:
            return len(pattern) * 10
        return 0

    def categorize(self, description: str) -> CategorizationResult:
        candidates: List[Candidate] = []
        desc_clean = str(description).strip().lower()

        # --- STAGE 1: MERCHANTS (+50 Bonus) ---
        for m in self.merchants:
            for alias in m.aliases:
                base_score = self._score_match(desc_clean, alias)
                if base_score > 0:
                    score = base_score + 50  # Merchant specificity bonus
                    candidates.append(
                        Candidate(
                            category=m.category,
                            subcategory=m.subcategory,
                            score=score,
                            source=f"merchant:{m.name}",
                        )
                    )

        # --- STAGE 2: PATTERNS (Weighted) ---
        for p in self.patterns:
            base_score = self._score_match(desc_clean, p.pattern)
            if base_score > 0:
                final_score = int(base_score * p.weight)
                candidates.append(
                    Candidate(
                        category=p.category,
                        subcategory=p.subcategory,
                        score=final_score,
                        source=f"pattern:{p.pattern}",
                    )
                )

        # --- STAGE 3: SELECTION ---
        if not candidates:
            return CategorizationResult(
                category="Uncategorized",
                subcategory="",
                confidence=0,
                matched_by="none",
            )

        # Highest score wins. If ties, first max encountered wins.
        winner = max(candidates, key=lambda c: c.score)

        return CategorizationResult(
            category=winner.category,
            subcategory=winner.subcategory,
            confidence=winner.score,
            matched_by=winner.source,
        )


# Singleton Instance
engine = CategorizationEngine()