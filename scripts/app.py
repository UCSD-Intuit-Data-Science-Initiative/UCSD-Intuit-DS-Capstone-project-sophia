from math import ceil, factorial, floor

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize

# ============================
# Erlang-C helper functions
# ============================


def erlang_c_prob_wait(N, A):
    """
    Erlang C: probability a caller has to wait.
    N: number of agents (float; we floor to int)
    A: offered load in Erlangs
    """
    N_int = int(np.floor(N))
    if N_int <= 0:
        return 1.0
    if A >= N_int:
        return 1.0  # overloaded system

    numer = (A**N_int / factorial(N_int)) * (N_int / (N_int - A))
    denom = sum(A**k / factorial(k) for k in range(N_int)) + numer
    return numer / denom


def erlang_c_asa(N, A, AHT):
    """
    ASA (Average Speed of Answer) in seconds.
    """
    N_int = int(np.floor(N))
    if N_int <= 0 or A >= N_int:
        return 1e9  # effectively infeasible

    P_wait = erlang_c_prob_wait(N_int, A)
    return (P_wait * AHT) / (N_int - A)


def occupancy(N, A):
    """
    Occupancy = Erlangs / # agents.
    """
    N_int = int(np.floor(N))
    if N_int <= 0:
        return 1.0
    return A / N_int


def find_min_staff_scipy(
    A,
    AHT,
    patience_sec=20,
    max_occ=0.85,
    N_min=1,
    N_max=200,
):
    """
    Find minimum integer N such that:
        ASA(N) <= patience_sec
        occupancy(N) <= max_occ
    Using SciPy (continuous) + local integer search.
    """

    def obj(x):
        return x[0]

    def asa_constraint(x):
        return patience_sec - erlang_c_asa(x[0], A, AHT)

    def occ_constraint(x):
        return max_occ - occupancy(x[0], A)

    cons = [
        {"type": "ineq", "fun": asa_constraint},
        {"type": "ineq", "fun": occ_constraint},
    ]
    bounds = [(N_min, N_max)]
    x0 = np.array([max(A + 1, N_min + 1)])  # start above load

    res = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
    )

    # Fallback: brute-force search if SciPy fails
    if not res.success:
        for N in range(N_min, N_max + 1):
            asa_val = erlang_c_asa(N, A, AHT)
            occ_val = occupancy(N, A)
            if asa_val <= patience_sec and occ_val <= max_occ:
                return (N, asa_val, occ_val)
        return None

    center = res.x[0]
    lo = max(N_min, int(floor(center)) - 5)
    hi = min(N_max, int(ceil(center)) + 5)

    for N in range(lo, hi + 1):
        asa_val = erlang_c_asa(N, A, AHT)
        occ_val = occupancy(N, A)
        if asa_val <= patience_sec and occ_val <= max_occ:
            return (N, asa_val, occ_val)

    return None


# ============================
# Data merging
# ============================


@st.cache_data
def build_interval_demand(
    daily_path,
    intraday_path,
    interval_length_seconds=1800,
):
    """
    Expand daily + intraday data to interval-level traffic for:
    all Dates × all Product Groups × all 30-minute intervals.
    """
    daily = pd.read_csv(daily_path)
    intraday = pd.read_csv(intraday_path)

    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["Day of Week"] = daily["Date"].dt.day_name()

    # Normalize intraday arrival to sum to 1 per PG+DOW
    intraday["Intraday Arrival"] = intraday.groupby(
        ["Product Group", "Day of Week"]
    )["Intraday Arrival"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else x
    )

    merged = intraday.merge(
        daily,
        on=["Product Group", "Day of Week"],
        how="inner",
    )

    merged["Interval_Length_sec"] = interval_length_seconds
    merged["Interval_Calls"] = (
        merged["Incoming Calls"] * merged["Intraday Arrival"]
    )
    merged["Traffic_Erlangs"] = (
        merged["Interval_Calls"] / interval_length_seconds
    ) * merged["Talk Duration (AVG)"]

    return merged


def optimize_staffing_for_date(
    merged,
    date,
    product_group=None,
    patience_sec=20,
    max_occ=0.85,
    N_min=1,
    N_max=200,
    zero_load_staff=0,
):
    """
    Optimize staffing for all intervals on a single date.
    Keeps all intervals (including zero-traffic) and returns:
    - Optimal_Staff
    - ASA_sec
    - Occupancy
    - Interval_Length_min (for cost)
    """
    date_ts = pd.to_datetime(date)
    data = merged[merged["Date"] == date_ts].copy()
    if product_group:
        data = data[data["Product Group"] == product_group]

    if data.empty:
        return pd.DataFrame()

    results = []

    for _, row in data.iterrows():
        A = row["Traffic_Erlangs"]
        AHT = row["Talk Duration (AVG)"]

        if A <= 0:
            N_opt, asa_opt, occ_opt = zero_load_staff, 0.0, 0.0
        else:
            sol = find_min_staff_scipy(
                A,
                AHT,
                patience_sec=patience_sec,
                max_occ=max_occ,
                N_min=N_min,
                N_max=N_max,
            )
            if sol:
                N_opt, asa_opt, occ_opt = sol
            else:
                N_opt, asa_opt, occ_opt = None, None, None

        results.append(
            {
                "Product Group": row["Product Group"],
                "Date": row["Date"].date(),
                "Day of Week": row["Day of Week"],
                "Interval Start": row["Interval Start"],
                "Interval_Length_min": row["Interval_Length_sec"] / 60,
                "Interval_Calls": round(row["Interval_Calls"], 3),
                "Traffic_Erlangs": round(A, 3),
                "AHT_sec": row["Talk Duration (AVG)"],
                "Optimal_Staff": N_opt,
                "ASA_sec": asa_opt,
                "Occupancy": occ_opt,
            }
        )

    return pd.DataFrame(results)


def optimize_staffing_for_date_range(
    merged,
    start,
    end,
    product_group=None,
    patience_sec=20,
    max_occ=0.85,
    N_min=1,
    N_max=200,
    zero_load_staff=0,
):
    """
    Optimize staffing for all dates in [start, end] (inclusive).
    """
    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)
    subset = merged[(merged["Date"] >= start_ts) & (merged["Date"] <= end_ts)]
    if product_group:
        subset = subset[subset["Product Group"] == product_group]

    if subset.empty:
        return pd.DataFrame()

    all_dates = sorted(subset["Date"].unique())
    frames = []
    for d in all_dates:
        out = optimize_staffing_for_date(
            subset,
            d,
            product_group=product_group,
            patience_sec=patience_sec,
            max_occ=max_occ,
            N_min=N_min,
            N_max=N_max,
            zero_load_staff=zero_load_staff,
        )
        if not out.empty:
            frames.append(out)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ============================
# Streamlit App (Top Tabs)
# ============================


def main():
    st.set_page_config(page_title="Erlang-C Staffing Dashboard", layout="wide")
    st.title("📊 Erlang-C Workforce Optimization Dashboard")

    # --- Sidebar: data paths + cost ---
    st.sidebar.header("Data & Cost Settings")
    daily_path = st.sidebar.text_input(
        "Daily dataset path",
        value="call-center-data-v3-daily.csv",
        key="sb_daily_path",
    )
    intraday_path = st.sidebar.text_input(
        "Intraday dataset path",
        value="intraday-profiles.csv",
        key="sb_intraday_path",
    )

    hourly_wage = st.sidebar.number_input(
        "Hourly wage per agent ($)",
        min_value=0.0,
        max_value=1000.0,
        value=20.0,
        step=1.0,
        key="sb_hourly_wage",
    )

    # Load merged interval-level data
    try:
        merged = build_interval_demand(daily_path, intraday_path)
    except Exception as e:
        st.error(f"Failed to load/merge data: {e}")
        return

    all_pgs = sorted(merged["Product Group"].unique())
    all_dates = sorted(merged["Date"].dt.date.unique())

    # --- Top navigation tabs ---
    tab1, tab2, tab3 = st.tabs(
        ["Single-Day Optimizer", "Multi-Day Summary", "Heatmap"]
    )

    # ==========================================================
    # TAB 1 — Single-Day Optimizer
    # ==========================================================
    with tab1:
        st.header("Single-Day Staffing Optimizer")

        selected_date = st.selectbox(
            "Select Date",
            all_dates,
            key="sd_date",
        )
        selected_pg = st.selectbox(
            "Product Group",
            all_pgs,
            key="sd_pg",
        )

        c1, c2, c3 = st.columns(3)
        patience_sec = c1.slider(
            "ASA threshold (sec)",
            min_value=1,
            max_value=60,
            value=20,
            step=1,
            key="sd_patience",
        )
        max_occ = c2.slider(
            "Max occupancy",
            min_value=0.50,
            max_value=0.99,
            value=0.85,
            step=0.01,
            key="sd_max_occ",
        )
        zero_staff = c3.number_input(
            "Staff for zero-traffic intervals",
            min_value=0,
            max_value=30,
            value=0,
            key="sd_zero_staff",
        )

        N_min = st.number_input(
            "Min staff (search lower bound)",
            min_value=0,
            max_value=200,
            value=1,
            key="sd_N_min",
        )
        N_max = st.number_input(
            "Max staff (search upper bound)",
            min_value=1,
            max_value=200,
            value=100,
            key="sd_N_max",
        )

        if st.button("Run Single-Day Optimization", key="sd_run"):
            df = optimize_staffing_for_date(
                merged,
                selected_date,
                product_group=selected_pg,
                patience_sec=patience_sec,
                max_occ=max_occ,
                N_min=N_min,
                N_max=N_max,
                zero_load_staff=zero_staff,
            )

            if df.empty:
                st.warning("No data for this date/product group combination.")
            else:
                # Cost calculations
                df["Staff_Hours"] = df["Optimal_Staff"] * (
                    df["Interval_Length_min"] / 60.0
                )
                df["Interval_Cost"] = df["Staff_Hours"] * hourly_wage

                total_staff_hours = df["Staff_Hours"].sum()
                total_cost = df["Interval_Cost"].sum()

                m1, m2 = st.columns(2)
                m1.metric(
                    "Total Staff-Hours (day)", f"{total_staff_hours:.2f}"
                )
                m2.metric("Total Labor Cost (day)", f"${total_cost:,.2f}")

                st.subheader("Interval-Level Results")
                st.dataframe(df, use_container_width=True)

                # Charts
                df["Interval_dt"] = pd.to_datetime(
                    df["Interval Start"], format="%H:%M:%S"
                )
                df = df.sort_values("Interval_dt")

                st.markdown("### Optimal Staff by Interval")
                st.line_chart(df.set_index("Interval_dt")[["Optimal_Staff"]])

                st.markdown("### ASA and Occupancy by Interval")
                st.line_chart(
                    df.set_index("Interval_dt")[
                        ["ASA_sec", "Occupancy"]
                    ].dropna(how="all")
                )

    # ==========================================================
    # TAB 2 — Multi-Day Summary
    # ==========================================================
    with tab2:
        st.header("Multi-Day Summary")

        c1, c2 = st.columns(2)
        start = c1.selectbox(
            "Start Date",
            all_dates,
            index=0,
            key="md_start",
        )
        end = c2.selectbox(
            "End Date",
            all_dates,
            index=len(all_dates) - 1,
            key="md_end",
        )

        if start > end:
            st.error("Start date must be <= End date.")
        else:
            pg2 = st.selectbox(
                "Product Group (optional)",
                ["ALL"] + all_pgs,
                key="md_pg",
            )
            pg_filter = None if pg2 == "ALL" else pg2

            patience_multi = st.slider(
                "ASA threshold (sec)",
                min_value=1,
                max_value=60,
                value=20,
                step=1,
                key="md_patience",
            )
            max_occ_multi = st.slider(
                "Max occupancy",
                min_value=0.50,
                max_value=0.99,
                value=0.85,
                step=0.01,
                key="md_max_occ",
            )
            zero_staff_multi = st.number_input(
                "Staff for zero-traffic intervals",
                min_value=0,
                max_value=30,
                value=0,
                key="md_zero_staff",
            )

            N_min_multi = st.number_input(
                "Min staff (search lower bound)",
                min_value=0,
                max_value=200,
                value=1,
                key="md_N_min",
            )
            N_max_multi = st.number_input(
                "Max staff (search upper bound)",
                min_value=1,
                max_value=200,
                value=100,
                key="md_N_max",
            )

            if st.button("Run Multi-Day Optimization", key="md_run"):
                df_multi = optimize_staffing_for_date_range(
                    merged,
                    start,
                    end,
                    product_group=pg_filter,
                    patience_sec=patience_multi,
                    max_occ=max_occ_multi,
                    N_min=N_min_multi,
                    N_max=N_max_multi,
                    zero_load_staff=zero_staff_multi,
                )

                if df_multi.empty:
                    st.warning("No data for this date range / product group.")
                else:
                    df_multi["Staff_Hours"] = df_multi["Optimal_Staff"] * (
                        df_multi["Interval_Length_min"] / 60.0
                    )
                    df_multi["Interval_Cost"] = (
                        df_multi["Staff_Hours"] * hourly_wage
                    )

                    daily = (
                        df_multi.groupby("Date")
                        .agg(
                            Staff_Hours=("Staff_Hours", "sum"),
                            Cost=("Interval_Cost", "sum"),
                        )
                        .reset_index()
                    )

                    total_hours = daily["Staff_Hours"].sum()
                    total_cost = daily["Cost"].sum()

                    m1, m2 = st.columns(2)
                    m1.metric(
                        "Total Staff-Hours (range)",
                        f"{total_hours:.2f}",
                    )
                    m2.metric(
                        "Total Labor Cost (range)",
                        f"${total_cost:,.2f}",
                    )

                    st.subheader("Daily Staff-Hours & Cost")
                    st.dataframe(daily, use_container_width=True)

                    st.markdown("### Daily Cost Over Time")
                    st.line_chart(daily.set_index("Date")[["Cost"]])

    # ==========================================================
    # TAB 3 — Heatmap (dataframe form)
    # ==========================================================
    with tab3:
        st.header("Heatmap (Interval × Date)")

        selected_pg_heat = st.selectbox(
            "Product Group for Heatmap",
            all_pgs,
            key="hm_pg",
        )

        c1, c2 = st.columns(2)
        start_h = c1.selectbox(
            "Start Date (Heatmap)",
            all_dates,
            index=0,
            key="hm_start",
        )
        end_h = c2.selectbox(
            "End Date (Heatmap)",
            all_dates,
            index=len(all_dates) - 1,
            key="hm_end",
        )

        if start_h > end_h:
            st.error("Start date must be <= End date.")
        else:
            patience_h = st.slider(
                "ASA threshold (sec)",
                min_value=1,
                max_value=60,
                value=20,
                step=1,
                key="hm_patience",
            )
            max_occ_h = st.slider(
                "Max occupancy",
                min_value=0.50,
                max_value=0.99,
                value=0.85,
                step=0.01,
                key="hm_max_occ",
            )
            zero_staff_h = st.number_input(
                "Staff for zero-traffic intervals",
                min_value=0,
                max_value=30,
                value=0,
                key="hm_zero_staff",
            )
            N_min_h = st.number_input(
                "Min staff (search lower bound)",
                min_value=0,
                max_value=200,
                value=1,
                key="hm_N_min",
            )
            N_max_h = st.number_input(
                "Max staff (search upper bound)",
                min_value=1,
                max_value=200,
                value=100,
                key="hm_N_max",
            )

            if st.button("Generate Heatmap Data", key="hm_run"):
                df_heat = optimize_staffing_for_date_range(
                    merged,
                    start_h,
                    end_h,
                    product_group=selected_pg_heat,
                    patience_sec=patience_h,
                    max_occ=max_occ_h,
                    N_min=N_min_h,
                    N_max=N_max_h,
                    zero_load_staff=zero_staff_h,
                )

                if df_heat.empty:
                    st.warning("No data for this range / product group.")
                else:
                    df_heat["Date"] = pd.to_datetime(df_heat["Date"])
                    df_heat["Interval_dt"] = pd.to_datetime(
                        df_heat["Interval Start"], format="%H:%M:%S"
                    )
                    df_heat["Interval_Label"] = df_heat[
                        "Interval_dt"
                    ].dt.strftime("%H:%M")

                    pivot = (
                        df_heat.pivot_table(
                            index="Date",
                            columns="Interval_Label",
                            values="Optimal_Staff",
                            aggfunc="mean",
                        )
                        .sort_index(axis=0)
                        .sort_index(axis=1)
                    )

                    st.subheader("Optimal Staff Heatmap (as table)")
                    st.dataframe(pivot, use_container_width=True)


if __name__ == "__main__":
    main()
