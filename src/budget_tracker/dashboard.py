from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import cast
from budget_tracker.config import settings, logger
from .analytics import p2p, operating, debt

st.set_page_config(
    page_title="Financial Command Center", layout="wide"
)


# DATA LOADING
@st.cache_data
def load_ledger() -> pd.DataFrame:
    if not settings.master_file.exists():
        st.error(f"Master file missing at {settings.master_file}")
        return pd.DataFrame()

    try:
        df = pd.read_excel(settings.master_file)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        cols_to_fix = ["paid_in", "paid_out"]
        for c in cols_to_fix:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        df["month_year"] = df["date"].map(
            lambda d: d.strftime("%Y-%m") if not pd.isna(d) else ""
        )

        return df
    except Exception as e:
        logger.error(f"Critical failure loading ledger: {e}", exc_info=True)
        st.error("Failed to load data. Check logs.")
        return pd.DataFrame()


# MAIN APP
def main():
    st.title("Financial Command Center")

    df_raw = load_ledger()

    if df_raw.empty:
        st.warning("No data found. Run pipeline first.")
        st.stop()

    tab_p2p, tab_operating, tab_debt = st.tabs(
        ["Social Ledger (P2P)", "Burn Rate (Operating)", "Deleveraging (Debt)"]
    )

    # SOCIAL LEDGER
    with tab_p2p:
        st.markdown("### Peer-to-Peer Balance")
        st.info("Tracking who owes who based on 'P2P' category.")

        p2p_summary = p2p.calculate_p2p_metrics(df_raw)

        c1, c2 = st.columns(2)
        c1.metric(
            "Total Outstanding (Net)",
            f"£{p2p_summary.total_outstanding:,.2f}",
            help="Negative = You are owed money. Positive = You owe others.",
        )
        c2.metric(
            "Lifetime Volume",
            f"£{p2p_summary.lending_volume:,.2f}",
            help="Total value of money swapped back and forth.",
        )

        if p2p_summary.entities:
            fig = p2p.plot_p2p_diverging_bars(p2p_summary)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No P2P transactions found. Check your 'Category' rules.")

    # OPERATING
    with tab_operating:
        st.markdown("### Monthly Burn Rate")

        op_all = operating.prepare_operating_data(df_raw)

        if op_all.empty:
            st.warning("No 'Operating' transactions found. Check your 'type' labels.")
        else:
            categories = sorted(op_all["category"].dropna().unique())
            base_palette = px.colors.qualitative.Set3
            category_color_map = {
                cat: base_palette[i % len(base_palette)]
                for i, cat in enumerate(categories)
            }

            available_months: list[str] = sorted(
                op_all["month_str"].dropna().astype(str).unique(),
                reverse=True,
            )
            if not available_months:
                st.info("No months available for Operating analysis.")
                st.stop()

            op_month_label = cast(
                str,
                st.selectbox(
                    "Month",
                    options=available_months,
                    index=0,
                    help="Operating analysis is always shown for a single month.",
                ),
            )

            op_month = op_all[op_all["month_str"] == op_month_label]

            monthly_spend = op_all.groupby("month_str")["paid_out"].sum().sort_index()

            spend_delta_str = ""
            coverage_delta_str = ""
            prev_month_label: str | None = None

            if op_month_label in monthly_spend.index:
                idx = list(monthly_spend.index).index(op_month_label)
                if idx > 0:
                    prev_label = list(monthly_spend.index)[idx - 1]
                    prev_month_label = prev_label

                    prev_spend = float(monthly_spend.loc[prev_label])
                    curr_spend = float(monthly_spend.loc[op_month_label])

                    if prev_spend > 0:
                        delta_spend = curr_spend - prev_spend
                        delta_spend_pct = (delta_spend / prev_spend) * 100.0
                        sign = "+" if delta_spend >= 0 else ""
                        spend_delta_str = (
                            f"{sign}£{delta_spend:,.0f} "
                            f"({delta_spend_pct:+.1f}%) vs {prev_month_label}"
                        )

            income_df = df_raw.copy()
            income_df["date"] = pd.to_datetime(income_df["date"])
            income_df["month_str"] = income_df["date"].map(
                lambda d: d.strftime("%Y-%m") if not pd.isna(d) else ""
            )

            income_mask = (income_df["type"] == "operating") & (
                income_df["paid_in"] > 0
            )
            monthly_income = (
                income_df.loc[income_mask]
                .groupby("month_str")["paid_in"]
                .sum()
                .sort_index()
            )

            coverage_delta_str = ""
            if prev_month_label is not None:
                income_curr = float(monthly_income.get(op_month_label, 0.0))
                income_prev = float(monthly_income.get(prev_month_label, 0.0))

                curr_spend = float(monthly_spend.get(op_month_label, 0.0))
                if income_curr > 0 and curr_spend > 0:
                    coverage_curr: float = (curr_spend / income_curr) * 100.0
                else:
                    coverage_curr = 0.0

                if income_prev > 0:
                    coverage_prev = (
                        monthly_spend.loc[prev_month_label] / income_prev
                    ) * 100.0
                    delta_cov = coverage_curr - coverage_prev
                    coverage_delta_str = f"{delta_cov:+.1f}pp vs {prev_month_label}"

            kpis = operating.calculate_kpis(
                operating_df=op_month,
                total_income_df=df_raw,
                all_operating_df=op_all,
            )

            k1, k2, k3, k4 = st.columns(4)

            k1.metric(
                "Total Spend (Period)",
                f"£{kpis.total_spend:,.0f}",
                delta=spend_delta_str or None,
                delta_color="inverse",
            )

            k2.metric(
                "Avg Monthly Burn",
                f"£{kpis.avg_monthly_burn:,.0f}",
            )

            k3.metric(
                "Income Spent",
                f"{kpis.income_coverage:,.1f}%",
                delta=coverage_delta_str or None,
                delta_color="inverse",
            )

            k4.metric(
                "Top Category",
                kpis.top_category or "N/A",
            )

            fig_categories = operating.plot_monthly_category_spend(
                op_month,
                color_map=category_color_map,
            )
            if fig_categories.data:
                st.plotly_chart(
                    fig_categories,
                    use_container_width=True,
                    key="operating_category_chart",
                )
            else:
                st.info("No operating spend for this month.")

            cat_totals = (
                op_month.groupby("category")["paid_out"]
                .sum()
                .sort_values(ascending=False)
            )

            if not cat_totals.empty:
                options: list[str] = list(cat_totals.index.astype(str))
                selected_category_for_view = cast(
                    str,
                    st.selectbox(
                        "Category for subcategory breakdown",
                        options=options,
                        index=0,
                        help="Choose a category to see its subcategories for this month.",
                    ),
                )

                selected_category = selected_category_for_view

                fig_subcats = operating.plot_subcategory_spend(
                    operating_df=op_month,
                    category=selected_category_for_view,
                )

                if fig_subcats.data:
                    st.plotly_chart(
                        fig_subcats,
                        use_container_width=True,
                        key="operating_subcategory_chart",
                    )
                else:
                    st.info(
                        "No subcategory data for this category in the selected month."
                    )
            else:
                selected_category = None
                st.info("No categories to break down for this month.")

            fig_trend = operating.plot_monthly_spend_trend(
                all_operating_df=op_all,
                selected_month=op_month_label,
                avg_monthly_burn=kpis.avg_monthly_burn,
                months_back=6,
            )
            if fig_trend.data:
                st.plotly_chart(
                    fig_trend,
                    use_container_width=True,
                    key="operating_spend_trend",
                )

            st.markdown("#### Transactions")
            tx_view = operating.build_operating_transactions_view(
                operating_df=op_month,
                category=selected_category,
            )

            if tx_view.empty:
                st.info("No transactions for this selection.")
            else:
                st.dataframe(
                    tx_view,
                    use_container_width=True,
                    hide_index=True,
                )

    # DEBT
    with tab_debt:
        st.markdown("### Debt Deleveraging")

        repay_df, interest_df = debt.prepare_debt_data(df_raw)

        if repay_df.empty:
            st.success("No debt repayment transactions found.")
        else:
            d_metrics = debt.calculate_metrics(repay_df, interest_df)

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Total Repaid", f"£{d_metrics.total_repaid:,.0f}")
            d2.metric(
                "Interest Cost",
                f"£{d_metrics.total_interest:,.0f}",
                help="Money lost to fees/interest.",
            )
            d3.metric(
                "Principal Efficiency",
                f"{d_metrics.principal_ratio:.1f}%",
                help="% of your payment that actually reduced the debt balance.",
            )
            d4.metric("Active Months", d_metrics.active_months)

            st.divider()

            st.caption(
                "The Blue Bars show your effort. The Red Line shows the bank's profit."
            )
            fig_snowball = debt.plot_debt_snowball(repay_df, interest_df)
            st.plotly_chart(fig_snowball, use_container_width=True)


if __name__ == "__main__":
    main()
