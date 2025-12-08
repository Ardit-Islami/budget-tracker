from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
import plotly.graph_objects as go


# DATA MODELS
@dataclass(frozen=True)
class P2PEntity:
    name: str
    total_lent: float  # Money leaving you (paid_out)
    total_repaid: float  # Money coming back (paid_in)
    net_balance: float  # Negative = They owe you. Positive = You owe them.


@dataclass(frozen=True)
class P2PSummary:
    entities: List[P2PEntity]
    total_outstanding: float  # How much money is currently "out" in the world
    lending_volume: float  # Total churn (sum of all movements)


# CORE LOGIC
def calculate_p2p_metrics(ledger: pd.DataFrame) -> P2PSummary:
    required_cols = {"category", "subcategory", "paid_in", "paid_out"}
    if ledger.empty or not required_cols.issubset(ledger.columns):
        return P2PSummary(entities=[], total_outstanding=0.0, lending_volume=0.0)

    # Filter specifically for P2P transactions - type: transfer covers more than P2P
    p2p_df = ledger[ledger["category"] == "P2P"].copy()

    if p2p_df.empty:
        return P2PSummary(entities=[], total_outstanding=0.0, lending_volume=0.0)

    # Group by Friend (Subcategory)
    p2p_df["subcategory"] = (
        p2p_df["subcategory"].fillna("Unknown").astype(str).str.strip()
    )

    grouped = p2p_df.groupby("subcategory")[["paid_in", "paid_out"]].sum().reset_index()

    # 3. Build Entities
    entities = []
    total_balance = 0.0
    total_volume = 0.0

    for _, row in grouped.iterrows():
        name = row["subcategory"]
        lent = float(row["paid_out"])
        repaid = float(row["paid_in"])
        net = repaid - lent

        entities.append(P2PEntity(name, lent, repaid, net))

        total_balance += net
        total_volume += lent + repaid

    # Sort by who owes you the most
    entities.sort(key=lambda x: x.net_balance)

    return P2PSummary(
        entities=entities, total_outstanding=total_balance, lending_volume=total_volume
    )


# VISUALIZATION LAYER
def plot_p2p_diverging_bars(summary: P2PSummary) -> go.Figure:
    if not summary.entities:
        return go.Figure()

    names = [e.name for e in summary.entities]
    lent = [-e.total_lent for e in summary.entities]
    repaid = [e.total_repaid for e in summary.entities]

    fig = go.Figure()

    # Red Bars (Money Out)
    fig.add_trace(
        go.Bar(
            y=names,
            x=lent,
            base=0,
            orientation="h",
            name="Lent (Out)",
            marker_color="#ef553b",
            text=[f"£{abs(x):,.0f}" for x in lent],
            textposition="auto",
        )
    )

    # Green Bars (Money In)
    fig.add_trace(
        go.Bar(
            y=names,
            x=repaid,
            base=0,
            orientation="h",
            name="Repaid (In)",
            marker_color="#00cc96",
            text=[f"£{x:,.0f}" for x in repaid],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Social Ledger Balance (Who owes who?)",
        barmode="overlay",
        xaxis_title="Net Position (£)",
        yaxis_title="Contact",
        xaxis=dict(
            tickformat="£s",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="white",
        ),
        legend=dict(x=0, y=1.0, orientation="h"),
        height=max(400, len(names) * 40),
    )

    return fig
