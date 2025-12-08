from __future__ import annotations

import json
import re
import shutil
from typing import List, Dict, Any, Tuple

import pandas as pd
from rapidfuzz import process, fuzz

from budget_tracker.config import settings, logger
from budget_tracker.engine import engine, MerchantRule, PatternRule


existing_merchants: List[MerchantRule] = list(engine.merchants)

MERCHANT_BY_ID: Dict[str, MerchantRule] = {m.id: m for m in existing_merchants}
MERCHANT_BY_NAME: Dict[str, MerchantRule] = {
    m.name.lower(): m for m in existing_merchants
}
ALIAS_TO_MERCHANT: Dict[str, MerchantRule] = {
    alias.lower(): m for m in existing_merchants for alias in m.aliases
}


# LOAD UNCATEGORIZED TRANSACTIONS
def load_uncategorized() -> pd.DataFrame:
    """Loads uncategorized rows from Master file."""
    if not settings.master_file.exists():
        logger.error("No master file found.")
        return pd.DataFrame()

    df = pd.read_excel(settings.master_file)

    # Checks
    required_cols = {"category", "description"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        logger.error(f"Master file missing required columns: {missing}")
        return pd.DataFrame()

    uncats = df[df["category"] == "Uncategorized"].copy()
    uncats = uncats.dropna(subset=["description"])
    uncats["description"] = uncats["description"].astype(str).str.strip().str.lower()

    return uncats


def print_group_sample(df: pd.DataFrame, aliases: List[str], max_rows: int = 5) -> None:
    if df.empty:
        return

    mask = df["description"].isin(aliases)

    preferred_cols = [
        "date",
        "description",
        "paid_out",
        "paid_in",
        "source",
        "account_type",
    ]
    cols = [c for c in preferred_cols if c in df.columns]

    sample = df.loc[mask, cols].head(max_rows)

    if sample.empty:
        return

    print("   Sample rows:")
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(sample.to_string(index=False))

# REPORT DUMP
def generate_uncategorized_report(
    df: pd.DataFrame, clusters: List[Dict[str, Any]]
) -> None:
    report_path = settings.logs_dir / "uncategorized_report.txt"
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"   Generating report at {report_path}...")

    preferred_cols = ["date", "description", "paid_out", "paid_in", "source"]
    cols = [c for c in preferred_cols if c in df.columns]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*10 +"\n")
        f.write("UNCATEGORIZED TRANSACTIONS REPORT\n")
        f.write(f"Total Groups: {len(clusters)}\n")
        f.write("="*10 +"\n\n")

        for i, group in enumerate(clusters, 1):
            anchor = group["anchor"]
            aliases = group["aliases"]
            count = group["count"]

            f.write(f"#{i} GROUP: '{anchor}' (Total Occurrences: {count})\n")
            f.write(f"   Aliases: {', '.join(aliases)}\n")

            # Sample rows
            mask = df["description"].isin(aliases)
            sample = df.loc[mask, cols].head(5)

            if not sample.empty:
                if "date" in sample.columns:
                    sample["date"] = pd.to_datetime(
                        sample["date"], errors="coerce"
                    ).map(lambda d: d.strftime("%Y-%m-%d") if not pd.isna(d) else "")

                # Convert to string table
                table_str = sample.to_string(index=False, justify="left")
                indented_table = "\n".join(
                    "   " + line for line in table_str.split("\n")
                )

                f.write(f"{indented_table}\n")

            f.write("\n" + "-" * 60 + "\n\n")

    print(
        "   Report generated. Check 'data/logs/uncategorized_report.txt' for overview.\n"
    )


# CLUSTER SIMILAR DESCRIPTIONS
def cluster_descriptions(
    descriptions: List[str], threshold: int = 85
) -> List[Dict[str, Any]]:
    if not descriptions:
        return []

    # Frequency - prioritise big clusters
    freq = pd.Series(descriptions).value_counts()
    pool = set(freq.index)

    clusters: List[Dict[str, Any]] = []

    while pool:
        remaining_sorted = [x for x in freq.index if x in pool]
        if not remaining_sorted:
            break

        anchor = remaining_sorted[0]

        matches = process.extract(
            anchor,
            pool,
            scorer=fuzz.token_sort_ratio,
            limit=None,
            score_cutoff=threshold,
        )

        matched_strings = [m[0] for m in matches] or [anchor]

        clusters.append(
            {
                "anchor": anchor,
                "aliases": matched_strings,
                "count": int(sum(freq[s] for s in matched_strings)),
            }
        )

        for s in matched_strings:
            pool.discard(s)

    # Sort clusters by total count (highest impact first)
    clusters.sort(key=lambda x: x["count"], reverse=True)

    return clusters


# EXISTING MERCHANT SUGGESTIONS
def suggest_existing_merchants(
    anchor: str, aliases: List[str]
) -> List[Tuple[str, str, MerchantRule]]:
    
    candidates: List[Tuple[str, str, MerchantRule]] = []

    # Exact alias matches
    for a in aliases:
        key = a.lower()
        if key in ALIAS_TO_MERCHANT:
            m = ALIAS_TO_MERCHANT[key]
            candidates.append(("alias_exact", a, m))

    # Fuzzy name matching
    merchant_names = [m.name for m in existing_merchants]
    fuzzy_names = process.extract(
        anchor,
        merchant_names,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=85,
        limit=3,
    )
    for candidate_name, score, _ in fuzzy_names:
        found_merchant = MERCHANT_BY_NAME.get(candidate_name.lower())
        if found_merchant:
            candidates.append(
                ("name_fuzzy", f"{candidate_name} ({int(score)}%)", found_merchant)
            )

    # Fuzzy alias matching (If anchor resembles existing alias)
    for m in existing_merchants:
        match = process.extractOne(
            anchor,
            m.aliases,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=88,
        )
        if match:
            matched_alias, score, _ = match
            candidates.append(
                ("alias_fuzzy", f"alias '{matched_alias}' ({int(score)}%)", m)
            )

    # De-duplicate candidates by merchant ID
    seen_ids: set[str] = set()
    unique_candidates: List[Tuple[str, str, MerchantRule]] = []
    for kind, info, m in candidates:
        if m.id in seen_ids:
            continue
        seen_ids.add(m.id)
        unique_candidates.append((kind, info, m))

    return unique_candidates


# INTERACTIVE TEACHING SESSION


def interactive_session() -> None:
    print("Loading uncategorized data...")

    df = load_uncategorized()
    if df.empty:
        print("No uncategorized items found! Great job.")
        return

    descriptions = df["description"].tolist()
    print(f"Found {len(descriptions)} uncategorized rows.")
    print("Clustering similar descriptions...")

    clusters = cluster_descriptions(descriptions)
    print(f"Identified {len(clusters)} groups to review.\n")

    generate_uncategorized_report(df, clusters)

    new_merchant_rules: List[Dict[str, Any]] = []
    new_pattern_rules: List[Dict[str, Any]] = []

    try:
        for i, group in enumerate(clusters, start=1):
            anchor: str = group["anchor"]
            aliases: List[str] = group["aliases"]
            count: int = group["count"]

            # ensure choice is always bound
            choice: str = ""

            print(f"\n[{i}/{len(clusters)}] Group anchor: '{anchor}' (Total: {count})")

            shown = aliases[:5]
            extra = len(aliases) - len(shown)
            tail = f", +{extra} more..." if extra > 0 else ""
            print(f"   Includes: {', '.join(shown)}{tail}")

            # Show some context rows for this group
            print_group_sample(df, aliases)

            # --- CHECK EXISTING ---
            candidates = suggest_existing_merchants(anchor, aliases)
            if candidates:
                print("   ⚠ Possible existing merchants:")
                for kind, info, m in candidates:
                    print(f"     - [{kind}] {info}  →  {m.name} (id={m.id})")

                choice = (
                    input(
                        "   [a]ttach to existing, [n]ew merchant, [p]attern, [s]kip, [q]uit: "
                    )
                    .lower()
                    .strip()
                )

                if choice == "q":
                    print("Quitting...")
                    break

                if choice == "s":
                    continue

                if choice == "a":
                    # Attach aliases to first candidate for speed
                    _, _, target = candidates[0]
                    print(f"   → Adding aliases to: {target.name}")

                    existing_aliases_lower = {a.lower() for a in target.aliases}
                    new_aliases = [
                        a for a in aliases if a.lower() not in existing_aliases_lower
                    ]

                    if not new_aliases:
                        print("   (No new aliases to add.)")
                        continue

                    target.aliases.extend(new_aliases)
                    new_merchant_rules.append(target.model_dump())
                    print(f"   Queued update for {target.name}.")
                    continue

                # if choice is 'n' or 'p' we fall through and handle below

            else:
                choice = (
                    input("   [c]reate merchant, [p]attern, [s]kip, [q]uit: ")
                    .lower()
                    .strip()
                )

                if choice == "q":
                    print("Quitting...")
                    break

                if choice == "s":
                    continue

            # --- PATTERN BRANCH ---
            if choice == "p":
                print("   --- Define New Pattern ---")
                pat_str = input(f"   Pattern (default: {anchor}): ").strip() or anchor
                p_cat = input("   Category: ").strip()

                if not p_cat:
                    print("   Category is required; skipping pattern.")
                    continue

                p_sub = input("   Subcategory (optional): ").strip()
                sub_value = p_sub if p_sub else "Unspecified"

                print("   --- Cashflow Type ---")
                print("   [o]perating (Income/Spend) - DEFAULT")
                print("   [t]ransfer  (Internal/Savings/P2P)")
                print("   [f]inancing (Credit Card Payoff/Loan)")

                type_map = {"o": "operating", "t": "transfer", "f": "financing"}
                t_choice = input("   Type [o/t/f]: ").lower().strip()
                r_type = type_map.get(t_choice, "operating")

                try:
                    pat_obj = PatternRule(
                        pattern=pat_str.lower(),
                        category=p_cat,
                        subcategory=sub_value,
                        type=r_type,
                        weight=1.0,
                    )
                    new_pattern_rules.append(pat_obj.model_dump())
                    print(f"   Queued pattern '{pat_str}' ({r_type})")
                except Exception as e:
                    print(f"   Invalid pattern: {e}")
                continue

            # --- MERCHANT BRANCH ---
            if choice in ("c", "n"):
                print("   --- Define New Merchant ---")
                default_name = anchor.title()
                r_name = (
                    input(f"   Merchant Name (default: {default_name}): ").strip()
                    or default_name
                )

                sanitized_id = re.sub(r"[^a-z0-9_]+", "_", r_name.lower()).strip("_")

                r_cat = input("   Category: ").strip()
                if not r_cat:
                    print("   Category is required; skipping merchant.")
                    continue

                r_sub = input("   Subcategory (optional): ").strip()
                sub_value = r_sub if r_sub else "General"

                print("   --- Cashflow Type ---")
                print("   [o]perating (Income/Spend) - DEFAULT")
                print("   [t]ransfer  (Internal/Savings/P2P)")
                print("   [f]inancing (Credit Card Payoff/Loan)")

                type_map = {"o": "operating", "t": "transfer", "f": "financing"}
                t_choice = input("   Type [o/t/f]: ").lower().strip()
                r_type = type_map.get(t_choice, "operating")

                clean_aliases = [a.strip().lower() for a in aliases]

                try:
                    rule = MerchantRule(
                        id=sanitized_id,
                        name=r_name,
                        category=r_cat,
                        subcategory=sub_value,
                        type=r_type,  # Pass the type
                        aliases=clean_aliases,
                    )
                    new_merchant_rules.append(rule.model_dump())
                    print(
                        f"   Queued new merchant '{r_name}' (id={sanitized_id}, type={r_type})"
                    )
                except Exception as e:
                    print(f"   Invalid rule: {e}")

    except KeyboardInterrupt:
        print("\nSession interrupted.")

    # SAVE (WITH BACKUPS)

    # --- SAVE MERCHANTS ---
    if new_merchant_rules:
        print(f"\n--- Saving {len(new_merchant_rules)} Merchant changes ---")
        save_m = input("Confirm save to merchants.json? [y/n]: ").lower().strip()

        if save_m == "y":
            path_m = settings.merchants_file

            current_data: List[Dict[str, Any]] = []
            if path_m.exists():
                backup_path = path_m.with_suffix(".json.bak")
                shutil.copy(path_m, backup_path)
                print(f"   Backup created: {backup_path.name}")

                with open(path_m, "r") as f:
                    current_data = json.load(f)

            # 2. Merge Logic
            by_id: Dict[str, Dict[str, Any]] = {
                m["id"]: m for m in current_data if "id" in m
            }
            others = [m for m in current_data if "id" not in m]

            for r in new_merchant_rules:
                rid = r["id"]
                if rid in by_id:
                    existing = by_id[rid]

                    old_aliases = {a.lower() for a in existing.get("aliases", [])}
                    new_incoming = [
                        a for a in r["aliases"] if a.lower() not in old_aliases
                    ]
                    if new_incoming:
                        existing.setdefault("aliases", []).extend(new_incoming)
                    by_id[rid] = existing
                else:
                    by_id[rid] = r

            final_merchants = others + list(by_id.values())

            # 3. Write
            with open(path_m, "w") as f:
                json.dump(final_merchants, f, indent=2)
            print("   merchants.json updated.")

    # --- SAVE PATTERNS ---
    if new_pattern_rules:
        print(f"\n--- Saving {len(new_pattern_rules)} Pattern changes ---")
        save_p = input("Confirm save to patterns.json? [y/n]: ").lower().strip()

        if save_p == "y":
            path_p = settings.patterns_file

            current_patterns: List[Dict[str, Any]] = []
            if path_p.exists():
                backup_path = path_p.with_suffix(".json.bak")
                shutil.copy(path_p, backup_path)
                print(f"   Backup created: {backup_path.name}")

                with open(path_p, "r") as f:
                    current_patterns = json.load(f)

            current_patterns.extend(new_pattern_rules)

            with open(path_p, "w") as f:
                json.dump(current_patterns, f, indent=2)
            print("   patterns.json updated.")

    print("\nSession Complete. Remember to run your pipeline to apply changes.")


if __name__ == "__main__":
    interactive_session()
