from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import plotly.graph_objects as go


# DATA MODELS
@dataclass(frozen=True)
class DebtMetrics:
    total_repaid: float  # Total financing outflows (Effort)
    total_interest: float  # Total interest costs (Waste)
    principal_ratio: float  # % of payment that actually reduced debt (Efficiency)
    active_months: int  # Count of months with debt activity

# CORE LOGIC
def prepare_debt_data(ledger: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if ledger.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Money leaving your pocket categorized as 'financing'
    repay_mask = (ledger["type"] == "financing") & (ledger["paid_out"] > 0)
    repayments = ledger[repay_mask].copy()

    # Interest: Category 'Bills' or specific subcategories containing 'Interest'
    interest_mask = (ledger["type"] == "operating") & (
        ledger["subcategory"].astype(str).str.contains("interest", case=False, na=False)
    )
    interest = ledger[interest_mask].copy()

    # Align dates - Plotting
    for df_part in [repayments, interest]:
        if not df_part.empty:
            df_part["month_str"] = df_part["date"].map(
                lambda d: d.strftime("%Y-%m") if not pd.isna(d) else ""
            )

    return repayments, interest


def calculate_metrics(repayments: pd.DataFrame, interest: pd.DataFrame) -> DebtMetrics:
    total_repaid = repayments["paid_out"].sum() if not repayments.empty else 0.0
    total_interest = interest["paid_out"].sum() if not interest.empty else 0.0

    # Efficiency: (Repaid - Interest) / Repaid
    if total_repaid > 0:
        principal_paid = total_repaid - total_interest
        ratio = (principal_paid / total_repaid) * 100
    else:
        ratio = 0.0

    # Count active months - duration of struggle
    all_months: list[str]

    if not repayments.empty or not interest.empty:
        month_series = pd.concat([repayments, interest])["month_str"].dropna()
        all_months = [str(m) for m in month_series.unique().tolist()]
    else:
        all_months = []

    return DebtMetrics(
        total_repaid=total_repaid,
        total_interest=total_interest,
        principal_ratio=max(
            ratio, 0.0
        ),
        active_months=len(all_months),
    )


# VISUALISATION
def plot_debt_snowball(repayments: pd.DataFrame, interest: pd.DataFrame) -> go.Figure:
    if repayments.empty and interest.empty:
        return go.Figure()

    # Aggregation
    repay_monthly = (
        repayments.groupby("month_str")["paid_out"]
        .sum()
        .reset_index()
        .rename(columns={"paid_out": "payment"})
    )

    interest_monthly = (
        interest.groupby("month_str")["paid_out"]
        .sum()
        .reset_index()
        .rename(columns={"paid_out": "cost"})
    )

    # Single timeline
    df_plot = pd.merge(
        repay_monthly, interest_monthly, on="month_str", how="outer"
    ).fillna(0)
    df_plot = df_plot.sort_values("month_str")

    fig = go.Figure()

    # Payments
    fig.add_trace(
        go.Bar(
            x=df_plot["month_str"],
            y=df_plot["payment"],
            name="Debt Service (Payment)",
            marker_color="#636EFA",
        )
    )

    # Interest Cost
    fig.add_trace(
        go.Scatter(
            x=df_plot["month_str"],
            y=df_plot["cost"],
            name="Interest Cost",
            line=dict(color="#EF553B", width=3),
            mode="lines+markers",
        )
    )

    fig.update_layout(
        title="Debt Repayment vs. Interest Cost",
        xaxis_title=None,
        yaxis_title="Amount (£)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
    )

    return fig
