"""
Brazil BESS Behind-the-Meter Viability Simulator

Interactive dashboard to assess battery storage economics
across Brazilian electricity distributors using ANEEL tariff data.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from data.load_data import (
    load_processed_data,
    get_energy_spreads,
    get_demand_charges,
)
from simulator.bess_model import (
    BESSParams,
    energy_revenue,
    demand_savings,
    payback_years,
    cashflow_table,
    calculate_npv,
    sensitivity_payback,
    total_annual_revenue,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Brazil BESS Viability | Modo Energy",
    page_icon="\u26a1",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load & cache data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading ANEEL tariff data...")
def load_data():
    return load_processed_data()


df = load_data()

# ---------------------------------------------------------------------------
# Sidebar — Parameters
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Parameters")

st.sidebar.markdown("#### Tariff Filter")
available_subgrupos = sorted(df["DscSubGrupo"].unique())
subgrupo = st.sidebar.selectbox(
    "Voltage subgroup",
    available_subgrupos,
    index=available_subgrupos.index("A4") if "A4" in available_subgrupos else 0,
    help="A4 = 2.3-25 kV (most commercial/industrial). A3a = 30-44 kV. A3 = 69 kV. A2 = 88-138 kV. A1 = 230+ kV.",
)
modalidade = st.sidebar.selectbox(
    "Tariff modality",
    ["Verde", "Azul"],
    help="Verde: single demand rate, energy arbitrage only. Azul: separate peak/off-peak demand, enables demand savings.",
)

st.sidebar.markdown("#### Battery System")
power_mw = st.sidebar.slider("Power (MW)", 0.1, 10.0, 1.0, 0.1)
duration_h = st.sidebar.select_slider("Duration (hours)", options=[1, 2, 4], value=2)
efficiency = st.sidebar.slider("Round-trip efficiency (%)", 80, 95, 87) / 100
capex_kwh = st.sidebar.slider("CAPEX (R$/kWh)", 1000, 6000, 3500, 100)
operating_days = st.sidebar.slider("Operating days/year", 200, 365, 252)
lifetime = st.sidebar.slider("Lifetime (years)", 10, 25, 15)

params = BESSParams(
    power_mw=power_mw,
    duration_h=duration_h,
    efficiency=efficiency,
    capex_per_kwh=capex_kwh,
    operating_days=operating_days,
    lifetime_years=lifetime,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**System:** {params.power_mw} MW / {params.capacity_kwh / 1000:.1f} MWh  \n"
    f"**Total CAPEX:** R$ {params.total_capex / 1e6:.2f} M"
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# Brazil BESS Behind-the-Meter Viability")
st.markdown(
    "_Where in Brazil does the tariff structure make battery storage economically viable?_"
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Calculate spreads & ranking
# ---------------------------------------------------------------------------
spreads_df = get_energy_spreads(df, subgrupo=subgrupo, modalidade=modalidade)
demand_df = (
    get_demand_charges(df, subgrupo=subgrupo)
    if modalidade == "Azul"
    else pd.DataFrame()
)

if spreads_df.empty:
    st.warning("No tariff data found for the selected filters. Try a different subgroup or modality.")
    st.stop()

# Build ranking: best spread per distributor
ranking = (
    spreads_df.groupby("SigAgente")
    .agg(
        SpreadEnergia=("SpreadEnergia", "max"),
        Total_Ponta=("Total_Ponta", "mean"),
        Total_FP=("Total_FP", "mean"),
        TUSD_Ponta=("TUSD_Ponta", "mean"),
        TE_Ponta=("TE_Ponta", "mean"),
        TUSD_FP=("TUSD_FP", "mean"),
        TE_FP=("TE_FP", "mean"),
    )
    .reset_index()
    .sort_values("SpreadEnergia", ascending=False)
)

# Merge demand charges (Azul only)
ranking["Demanda_Ponta"] = 0.0
if not demand_df.empty and "Demanda_Ponta" in demand_df.columns:
    demand_map = demand_df.groupby("SigAgente")["Demanda_Ponta"].mean()
    ranking["Demanda_Ponta"] = ranking["SigAgente"].map(demand_map).fillna(0)

# Financial metrics per distributor
ranking["AnnualRevenue"] = ranking.apply(
    lambda r: total_annual_revenue(params, r["SpreadEnergia"], r["Demanda_Ponta"], year=0),
    axis=1,
)
ranking["Payback"] = ranking.apply(
    lambda r: payback_years(params, r["SpreadEnergia"], r["Demanda_Ponta"]),
    axis=1,
)
ranking["NPV"] = ranking.apply(
    lambda r: calculate_npv(params, r["SpreadEnergia"], r["Demanda_Ponta"]),
    axis=1,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Opportunity Map",
        "Distributor Deep Dive",
        "Financial Simulation",
        "Market Context",
    ]
)

# ===================== TAB 1: OPPORTUNITY MAP =====================
with tab1:
    viable = ranking[ranking["Payback"] <= 7]

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        best = ranking.iloc[0]
        st.metric(
            "Highest Spread",
            f"R$ {best['SpreadEnergia']:.1f}/MWh",
            best["SigAgente"],
        )
    with k2:
        st.metric("Average Spread", f"R$ {ranking['SpreadEnergia'].mean():.1f}/MWh")
    with k3:
        st.metric("Distributors Analyzed", f"{len(ranking)}")
    with k4:
        pct = f"{len(viable) / len(ranking) * 100:.0f}%" if len(ranking) else "0%"
        st.metric("Payback < 7 years", f"{len(viable)}", pct + " of total")

    st.markdown("---")

    # Bar chart — top distributors by spread
    top_n = min(25, len(ranking))
    top = ranking.head(top_n).copy()
    top["PaybackCapped"] = top["Payback"].clip(upper=20)

    fig_bar = px.bar(
        top.sort_values("SpreadEnergia"),
        x="SpreadEnergia",
        y="SigAgente",
        orientation="h",
        color="PaybackCapped",
        color_continuous_scale=["#2D6A4F", "#95D5B2", "#FCA311", "#E63946"],
        labels={
            "SpreadEnergia": "Energy Spread (R$/MWh)",
            "SigAgente": "Distributor",
            "PaybackCapped": "Payback (yrs)",
        },
        title=f"Top {top_n} Distributors by Peak/Off-Peak Spread — {subgrupo} {modalidade}",
    )
    fig_bar.update_layout(
        height=max(450, top_n * 28),
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_colorbar=dict(title="Payback<br>(years)"),
        margin=dict(l=0, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter — spread vs payback
    st.subheader("Spread vs. Payback")
    plot_data = ranking[ranking["Payback"] < 30].copy()
    if not plot_data.empty:
        fig_scatter = px.scatter(
            plot_data,
            x="SpreadEnergia",
            y="Payback",
            size="AnnualRevenue",
            hover_name="SigAgente",
            color="Payback",
            color_continuous_scale=["#2D6A4F", "#FCA311", "#E63946"],
            labels={
                "SpreadEnergia": "Energy Spread (R$/MWh)",
                "Payback": "Payback (years)",
                "AnnualRevenue": "Annual Revenue (R$)",
            },
        )
        fig_scatter.add_hline(
            y=7,
            line_dash="dash",
            line_color="gray",
            annotation_text="7-year target",
        )
        fig_scatter.update_layout(height=420, margin=dict(l=0, r=20, t=20, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Full ranking table
    with st.expander("Full ranking table"):
        display_cols = ["SigAgente", "SpreadEnergia", "Total_Ponta", "Total_FP", "AnnualRevenue", "Payback", "NPV"]
        display_df = ranking[display_cols].copy()
        display_df.columns = ["Distributor", "Spread (R$/MWh)", "Peak Tariff", "Off-Peak Tariff", "Annual Revenue (R$)", "Payback (yrs)", "NPV (R$)"]
        st.dataframe(
            display_df.style.format({
                "Spread (R$/MWh)": "{:.1f}",
                "Peak Tariff": "R$ {:.2f}",
                "Off-Peak Tariff": "R$ {:.2f}",
                "Annual Revenue (R$)": "R$ {:,.0f}",
                "Payback (yrs)": "{:.1f}",
                "NPV (R$)": "R$ {:,.0f}",
            }),
            hide_index=True,
            use_container_width=True,
        )


# ===================== TAB 2: DISTRIBUTOR DEEP DIVE =====================
with tab2:
    distributors = ranking["SigAgente"].tolist()
    selected = st.selectbox("Select distributor", distributors, key="dd_select")

    row = ranking[ranking["SigAgente"] == selected].iloc[0]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Energy Spread", f"R$ {row['SpreadEnergia']:.1f}/MWh")
    with c2:
        st.metric("Annual Revenue", f"R$ {row['AnnualRevenue']:,.0f}")
    with c3:
        pb_str = f"{row['Payback']:.1f} yrs" if row["Payback"] < 30 else "> 30 yrs"
        st.metric("Simple Payback", pb_str)
    with c4:
        st.metric("NPV (10%)", f"R$ {row['NPV']:,.0f}")

    st.markdown("---")

    # Tariff breakdown — grouped bar
    st.subheader("Tariff Breakdown: Peak vs. Off-Peak (R$/MWh)")

    fig_tariff = go.Figure()
    fig_tariff.add_trace(
        go.Bar(
            name="Off-Peak",
            x=["TUSD", "TE", "Total"],
            y=[row["TUSD_FP"], row["TE_FP"], row["Total_FP"]],
            marker_color="#2D6A4F",
            text=[f"R$ {v:.1f}" for v in [row["TUSD_FP"], row["TE_FP"], row["Total_FP"]]],
            textposition="auto",
        )
    )
    fig_tariff.add_trace(
        go.Bar(
            name="Peak",
            x=["TUSD", "TE", "Total"],
            y=[row["TUSD_Ponta"], row["TE_Ponta"], row["Total_Ponta"]],
            marker_color="#E63946",
            text=[f"R$ {v:.1f}" for v in [row["TUSD_Ponta"], row["TE_Ponta"], row["Total_Ponta"]]],
            textposition="auto",
        )
    )
    fig_tariff.update_layout(
        barmode="group",
        yaxis_title="R$/MWh",
        height=380,
        margin=dict(l=0, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_tariff, use_container_width=True)

    # Demand charges (Azul only)
    if modalidade == "Azul" and row["Demanda_Ponta"] > 0:
        st.subheader("Demand Charges (R$/kW per month)")
        demand_row = demand_df[demand_df["SigAgente"] == selected]
        if not demand_row.empty:
            dr = demand_row.iloc[0]
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                st.metric("Peak Demand", f"R$ {dr['Demanda_Ponta']:.2f}/kW")
            with dc2:
                if "Demanda_FP" in dr.index:
                    st.metric("Off-Peak Demand", f"R$ {dr['Demanda_FP']:.2f}/kW")
            with dc3:
                annual = demand_savings(params, dr["Demanda_Ponta"])
                st.metric("Annual Demand Savings", f"R$ {annual:,.0f}")

    # Revenue breakdown
    st.subheader("Revenue Composition (Year 1)")
    rev_energy = energy_revenue(params, row["SpreadEnergia"], year=0)
    rev_demand = demand_savings(params, row["Demanda_Ponta"])
    rev_total = rev_energy + rev_demand

    if rev_total > 0:
        fig_pie = go.Figure(
            go.Pie(
                labels=["Energy Arbitrage", "Demand Reduction"],
                values=[rev_energy, rev_demand],
                marker_colors=["#2D6A4F", "#457B9D"],
                hole=0.4,
                textinfo="label+percent+value",
                texttemplate="%{label}<br>%{percent:.1%}<br>R$ %{value:,.0f}",
            )
        )
        fig_pie.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)


# ===================== TAB 3: FINANCIAL SIMULATION =====================
with tab3:
    sel_fin = st.selectbox(
        "Select distributor", distributors, key="fin_select"
    )
    row_fin = ranking[ranking["SigAgente"] == sel_fin].iloc[0]
    spread_fin = row_fin["SpreadEnergia"]
    demand_fin = row_fin["Demanda_Ponta"]

    # Cash flow chart
    st.subheader("Cumulative Cash Flow")

    cf = cashflow_table(params, spread_fin, demand_fin)

    fig_cf = go.Figure()
    colors = ["#E63946" if v < 0 else "#2D6A4F" for v in cf["Cumulative"]]
    fig_cf.add_trace(
        go.Scatter(
            x=cf["Year"],
            y=cf["Cumulative"],
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color="#2D6A4F", width=3),
            fillcolor="rgba(45, 106, 79, 0.1)",
            hovertemplate="Year %{x}<br>R$ %{y:,.0f}<extra></extra>",
        )
    )
    fig_cf.add_hline(y=0, line_dash="dash", line_color="gray")

    # Mark breakeven point
    breakeven = cf[cf["Cumulative"] >= 0]
    if not breakeven.empty:
        be_year = breakeven.iloc[0]["Year"]
        fig_cf.add_vline(
            x=be_year,
            line_dash="dot",
            line_color="#FCA311",
            annotation_text=f"Breakeven: Year {be_year:.0f}",
        )

    fig_cf.update_layout(
        xaxis_title="Year",
        yaxis_title="Cumulative Cash Flow (R$)",
        yaxis_tickformat=",.0f",
        height=420,
        margin=dict(l=0, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_cf, use_container_width=True)

    # Sensitivity heatmap
    st.subheader("Sensitivity: Payback vs. CAPEX and Spread")

    spread_min = max(50, ranking["SpreadEnergia"].quantile(0.1))
    spread_max = ranking["SpreadEnergia"].quantile(0.95) * 1.2
    spread_range = np.linspace(spread_min, spread_max, 12).tolist()
    capex_range = np.linspace(1500, 5500, 9).tolist()

    sens = sensitivity_payback(params, spread_range, capex_range, demand_fin)
    sens_pivot = sens.pivot_table(
        index="CAPEX (R$/kWh)", columns="Spread (R$/MWh)", values="Payback"
    )

    fig_heat = px.imshow(
        sens_pivot.values,
        x=[f"{v:.0f}" for v in sens_pivot.columns],
        y=[f"{v:.0f}" for v in sens_pivot.index],
        color_continuous_scale=["#2D6A4F", "#95D5B2", "#FCA311", "#E63946"],
        labels=dict(x="Spread (R$/MWh)", y="CAPEX (R$/kWh)", color="Payback (yrs)"),
        aspect="auto",
        text_auto=".1f",
    )
    fig_heat.update_layout(height=420, margin=dict(l=0, r=20, t=20, b=20))
    st.plotly_chart(fig_heat, use_container_width=True)

    # Assumptions
    with st.expander("Model assumptions"):
        st.markdown(
            f"""
            | Parameter | Value |
            |-----------|-------|
            | Battery Power | {params.power_mw} MW |
            | Duration | {params.duration_h}h |
            | Capacity | {params.capacity_kwh / 1000:.1f} MWh |
            | Round-trip Efficiency | {params.efficiency * 100:.0f}% |
            | Cycles/day | {params.cycles_per_day} |
            | Operating days/year | {params.operating_days} |
            | CAPEX | R$ {params.capex_per_kwh:,.0f}/kWh |
            | Total CAPEX | R$ {params.total_capex / 1e6:.2f}M |
            | Lifetime | {params.lifetime_years} years |
            | Annual degradation | {params.degradation * 100:.0f}% |
            | Discount rate (NPV) | {params.discount_rate * 100:.0f}% |

            **Energy arbitrage:** battery charges off-peak and discharges at peak.
            Revenue = usable energy (MWh) x spread (R$/MWh) x operating days.

            **Demand reduction (Azul only):** battery reduces contracted peak demand.
            Savings = power (kW) x peak demand charge (R$/kW/month) x 12 months.

            Degradation is applied linearly to energy capacity each year.
            Simple payback uses Year 1 revenue. NPV uses the specified discount rate.
            """
        )


# ===================== TAB 4: MARKET CONTEXT =====================
with tab4:
    st.subheader("The Brazilian Electricity Tariff System")

    st.markdown(
        """
### Consumer Groups

Brazil's electricity consumers are divided into two main groups:

- **Group A** (medium/high voltage, 2.3 kV and above): Industries, large commercial
  buildings, hospitals, data centers. Billed with **time-of-use** tariffs that differentiate
  peak and off-peak periods.
- **Group B** (low voltage, below 1 kV): Residential, small commercial.

### Tariff Modalities (Group A)

Group A consumers choose between two time-of-use modalities:

| Feature | **Green (Verde)** | **Blue (Azul)** |
|---------|-------------------|-----------------|
| Energy (R$/MWh) | Peak + Off-peak rates | Peak + Off-peak rates |
| Demand (R$/kW) | **Single rate** | **Peak + Off-peak rates** |
| BESS value | Energy arbitrage only | Energy arbitrage + demand savings |

### Tariff Components

Each tariff has two components that are summed:
- **TUSD** (Tarifa de Uso do Sistema de Distribuicao): Distribution network usage charge
- **TE** (Tarifa de Energia): Energy commodity charge

### Peak Hours

Typically 3 consecutive hours set by each distributor (usually 17:30-20:30 or 18:00-21:00).
The ratio between peak and off-peak tariffs varies **significantly** across Brazil's 90+
distributors, creating geographic pockets of opportunity for BESS.

### How BESS Captures Value

1. **Energy arbitrage:** Charge the battery during off-peak hours (low tariff) and discharge
   during peak hours (high tariff). The **spread** (peak minus off-peak total tariff in R$/MWh)
   is the gross revenue per MWh shifted.

2. **Peak demand reduction (Azul only):** In the Azul modality, peak demand has a separate,
   higher monthly charge (R$/kW). The battery can reduce the consumer's contracted peak demand,
   generating monthly savings.

### Why Now?

- **LRCAP Storage:** Brazil's first dedicated battery storage auction is scheduled for
  **April 2026**, signaling regulatory readiness for grid-scale storage.
- **Behind-the-meter first:** BTM BESS is the fastest path to market — no auction required,
  driven purely by tariff economics.
- **Falling CAPEX:** Global battery costs continue to decline, making more locations viable
  each year.
- **90+ distributors:** Brazil's fragmented distribution market creates a data-rich landscape
  where geographic analysis reveals hidden opportunities.

### Data Source

All tariff data in this tool comes from
[ANEEL Open Data](https://dadosabertos.aneel.gov.br/dataset/tarifas-distribuidoras-energia-eletrica) —
the Brazilian electricity regulator's public dataset of homologated distribution tariffs,
updated as new tariff resolutions are published.
    """
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "Built by **Filip Osanai** | Energy Engineer | "
    "[brasilbess.com](https://brasilbess.com) | "
    "Data: [ANEEL Open Data](https://dadosabertos.aneel.gov.br) | "
    "Modo Energy Take-Home Task"
)
