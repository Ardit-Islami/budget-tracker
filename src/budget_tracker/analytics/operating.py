from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# DATA MODELS (The Interface)
@dataclass(frozen=True)
class OperatingMetrics:
    total_spend: float  # Total operating outflows (all time in view)
    avg_monthly_burn: float  # Important number for financial independence
    income_coverage: float  # % of Income spent (Burn Rate)
    top_category: str  # Where most money goes


# CORE LOGIC
def prepare_operating_data(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        # Return an empty frame with the expected columns so downstream code can still rely on the schema.
        return pd.DataFrame(
            columns=["date", "type", "category", "paid_out", "paid_in", "month_str"]
        )

    required_cols = {"type", "category", "paid_out", "paid_in", "date"}
    missing = required_cols.difference(ledger.columns)
    if missing:
        return pd.DataFrame(columns=list(required_cols) + ["month_str"])

    # Ensure datetime for date operations
    df = ledger.copy()
    df["date"] = pd.to_datetime(df["date"])

    mask = (
        (df["type"] == "operating") & (df["category"] != "P2P") & (df["paid_out"] > 0)
    )
    op_df = df.loc[mask].copy()

    if op_df.empty:
        return pd.DataFrame(columns=list(required_cols) + ["month_str"])

    # Month label for grouping & filtering later
    op_df["month_str"] = op_df["date"].map(
        lambda d: d.strftime("%Y-%m") if not pd.isna(d) else ""
    )

    return op_df


def calculate_kpis(
    operating_df: pd.DataFrame,
    total_income_df: pd.DataFrame,
    all_operating_df: pd.DataFrame | None = None,
) -> OperatingMetrics:
    if operating_df.empty:
        # Historical average (if we have history at all)
        if all_operating_df is not None and not all_operating_df.empty:
            monthly_totals = all_operating_df.groupby("month_str", observed=True)[
                "paid_out"
            ].sum()
            avg_burn = float(monthly_totals.mean())
        else:
            avg_burn = 0.0

        return OperatingMetrics(
            total_spend=0.0,
            avg_monthly_burn=avg_burn,
            income_coverage=0.0,
            top_category="N/A",
        )

    # Total spend in the selected month
    total_spend = float(operating_df["paid_out"].sum())

    # Historical average monthly burn
    if all_operating_df is not None and not all_operating_df.empty:
        monthly_totals = all_operating_df.groupby("month_str", observed=True)[
            "paid_out"
        ].sum()
        avg_burn = float(monthly_totals.mean())
    else:
        # Fallback: if no history passed in
        avg_burn = total_spend

    # Income coverage for the selected month
    month_labels = operating_df["month_str"].unique()

    income_df = total_income_df.copy()
    income_df["date"] = pd.to_datetime(income_df["date"])
    income_df["month_str"] = income_df["date"].map(
        lambda d: d.strftime("%Y-%m") if not pd.isna(d) else ""
    )

    income_mask = (
        (income_df["type"] == "operating")
        & (income_df["paid_in"] > 0)
        & (income_df["month_str"].isin(month_labels))
    )
    income_this_period = float(income_df.loc[income_mask, "paid_in"].sum())

    if income_this_period > 0:
        income_coverage = float(total_spend / income_this_period * 100.0)
    else:
        # No operating income recorded for this month
        income_coverage = 0.0

    # Top category this month
    top_category = (
        operating_df.groupby("category", observed=True)["paid_out"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    return OperatingMetrics(
        total_spend=total_spend,
        avg_monthly_burn=avg_burn,
        income_coverage=income_coverage,
        top_category=top_category,
    )


# VISUALISATION LAYER
def plot_monthly_category_spend(
    operating_df: pd.DataFrame,
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    if operating_df.empty:
        return go.Figure()

    # Aggregate spend by category
    cat_totals = (
        operating_df.groupby("category", observed=True)["paid_out"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    month_label = str(operating_df["month_str"].iloc[0])

    fig = px.bar(
        cat_totals,
        x="paid_out",
        y="category",
        color="category",
        title=f"Category Spend – {month_label}",
        color_discrete_map=color_map or {},
        labels={"category": "Category", "paid_out": "Total Spend"},
    )

    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Total Spend (£)",
        xaxis_tickangle=0, 
        bargap=0.1, 
        bargroupgap=0.0,
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=80),
    )

    return fig


def plot_monthly_spend_trend(
    all_operating_df: pd.DataFrame,
    selected_month: str,
    avg_monthly_burn: float,
    months_back: int = 6,
) -> go.Figure:
    if all_operating_df.empty:
        return go.Figure()

    df = all_operating_df.copy()

    # Aggregate spend by month
    monthly = (
        df.groupby("month_str", observed=True)["paid_out"]
        .sum()
        .reset_index()
        .rename(columns={"paid_out": "total_spend"})
    )

    if monthly.empty:
        return go.Figure()

    # Convert to Period - proper chronological sorting
    monthly["month_period"] = pd.PeriodIndex(monthly["month_str"], freq="M")
    monthly = monthly.sort_values("month_period")

    monthly = monthly.tail(months_back)

    fig = px.bar(
        monthly,
        x="month_str",
        y="total_spend",
        title=f"Monthly Spend vs Baseline (Last {len(monthly)} Months)",
        labels={"month_str": "Month", "total_spend": "Total Spend (£)"},
    )

    if avg_monthly_burn > 0:
        fig.add_hline(
            y=avg_monthly_burn,
            line_dash="dash",
            annotation_text="Avg Monthly Burn",
            annotation_position="top left",
        )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Total Spend (£)",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    return fig


def plot_subcategory_spend(operating_df: pd.DataFrame, category: str) -> go.Figure:
    if operating_df.empty:
        return go.Figure()

    df_cat = operating_df[operating_df["category"] == category].copy()
    if df_cat.empty:
        return go.Figure()

    df_cat["subcategory"] = df_cat["subcategory"].fillna("Uncategorised")

    sub_totals = (
        df_cat.groupby("subcategory", observed=True)["paid_out"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    # Group small slices into "Other" if too subcategories
    max_slices = 8
    if len(sub_totals) > max_slices:
        head = sub_totals.iloc[: max_slices - 1].copy()
        tail = sub_totals.iloc[max_slices - 1 :]
        other_value = tail["paid_out"].sum()
        other_row = pd.DataFrame({"subcategory": ["Other"], "paid_out": [other_value]})
        sub_totals = pd.concat([head, other_row], ignore_index=True)

    month_label = str(df_cat["month_str"].iloc[0])

    fig = px.pie(
        sub_totals,
        names="subcategory",
        values="paid_out",
        title=f"{category} – Subcategory Spend ({month_label})",
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hole=0.3,
    )

    fig.update_layout(
        legend_title_text="Subcategory",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def build_operating_transactions_view(
    operating_df: pd.DataFrame,
    category: str | None = None,
) -> pd.DataFrame:
    if operating_df.empty:
        return pd.DataFrame()

    df = operating_df.copy()

    if category is not None:
        df = df[df["category"] == category]

    if df.empty:
        return pd.DataFrame()

    if "date" in df.columns:
        df = df.sort_values("date", ascending=False)

    preferred_cols = [
        "date",
        "account",
        "account_id",
        "description",
        "category",
        "subcategory",
        "paid_out",
    ]
    columns_to_show = [c for c in preferred_cols if c in df.columns]

    if not columns_to_show:
        return df  # fallback: return everything if matched none

    return df[columns_to_show]
