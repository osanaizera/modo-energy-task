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
# Brand palette (aligned with brasilbess.com)
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#000000",
    "surface": "#18181b",
    "surface2": "#27272a",
    "border": "rgba(255,255,255,0.1)",
    "text": "#ededed",
    "text_muted": "#a1a1aa",
    "green": "#4ade80",
    "green_dark": "#22c55e",
    "cyan": "#22d3ee",
    "cyan_dark": "#06b6d4",
    "yellow": "#facc15",
    "red": "#ef4444",
    "teal": "#003447",
    "teal_light": "#155e75",
}

def brand_layout(fig, **overrides):
    """Apply brand dark-theme layout to a Plotly figure with optional overrides."""
    base = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        xaxis=dict(gridcolor=COLORS["surface2"], zerolinecolor=COLORS["surface2"]),
        yaxis=dict(gridcolor=COLORS["surface2"], zerolinecolor=COLORS["surface2"]),
        margin=dict(l=0, r=20, t=40, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text"])),
        coloraxis_colorbar=dict(
            bgcolor="rgba(0,0,0,0)",
            tickfont=dict(color=COLORS["text_muted"]),
            title=dict(font=dict(color=COLORS["text_muted"])),
        ),
    )
    # Deep-merge dicts so overrides extend rather than conflict
    for key, val in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            base[key] = {**base[key], **val}
        else:
            base[key] = val
    fig.update_layout(**base)

# Color scales
SCALE_PAYBACK = [
    [0, COLORS["green"]],
    [0.4, COLORS["cyan"]],
    [0.7, COLORS["yellow"]],
    [1, COLORS["red"]],
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BRASIL BESS Viability Simulator",
    page_icon="\u26a1",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — brasilbess.com design system
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&display=swap');

    /* Global */
    .stApp {{
        background-color: {COLORS["bg"]};
        color: {COLORS["text"]};
        font-family: 'Inter', sans-serif;
    }}

    /* Header brand */
    .brand-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }}
    .brand-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.1;
        background: linear-gradient(to right, {COLORS["green"]}, {COLORS["cyan"]});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }}
    .brand-subtitle {{
        font-size: 1.15rem;
        color: {COLORS["text_muted"]};
        margin-top: 2px;
        margin-bottom: 16px;
    }}

    /* KPI cards */
    .kpi-container {{
        background: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }}
    .kpi-value {{
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: {COLORS["green"]};
    }}
    .kpi-label {{
        font-size: 0.8rem;
        color: {COLORS["text_muted"]};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }}
    .kpi-detail {{
        font-size: 0.78rem;
        color: {COLORS["cyan"]};
        margin-top: 2px;
    }}

    /* Divider */
    .brand-divider {{
        height: 1px;
        background: linear-gradient(to right, {COLORS["green"]}, {COLORS["cyan"]}, transparent);
        border: none;
        margin: 20px 0;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {COLORS["surface"]};
        border-radius: 8px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {COLORS["text_muted"]};
        border-radius: 6px;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLORS["surface2"]} !important;
        color: {COLORS["green"]} !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {COLORS["surface"]};
        border-right: 1px solid {COLORS["border"]};
    }}
    section[data-testid="stSidebar"] .stMarkdown h2 {{
        font-family: 'Outfit', sans-serif;
        color: {COLORS["green"]};
    }}
    section[data-testid="stSidebar"] .stMarkdown h4 {{
        color: {COLORS["cyan"]};
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Metric overrides */
    [data-testid="stMetric"] {{
        background: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 10px;
        padding: 14px 18px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {COLORS["text_muted"]} !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    [data-testid="stMetricValue"] {{
        color: {COLORS["green"]} !important;
        font-family: 'Outfit', sans-serif !important;
    }}
    [data-testid="stMetricDelta"] {{
        color: {COLORS["cyan"]} !important;
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        background: {COLORS["surface"]};
        border-radius: 8px;
        color: {COLORS["text_muted"]};
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 8px; }}
    ::-webkit-scrollbar-track {{ background: {COLORS["bg"]}; }}
    ::-webkit-scrollbar-thumb {{ background: {COLORS["teal_light"]}; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {COLORS["cyan_dark"]}; }}

    /* Footer */
    .footer {{
        text-align: center;
        color: {COLORS["text_muted"]};
        font-size: 0.85rem;
        padding: 24px 0 12px;
    }}
    .footer a {{
        color: {COLORS["green"]};
        text-decoration: none;
    }}
    .footer a:hover {{
        color: {COLORS["cyan"]};
    }}
</style>
""",
    unsafe_allow_html=True,
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
st.markdown(
    '<div class="brand-header">'
    '<div class="brand-title">BRASIL BESS</div>'
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="brand-subtitle">'
    "Where in Brazil does the tariff structure make behind-the-meter battery storage economically viable?"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown('<div class="brand-divider"></div>', unsafe_allow_html=True)

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
    best = ranking.iloc[0]
    with k1:
        st.metric("Highest Spread", f"R$ {best['SpreadEnergia']:.1f}/MWh", best["SigAgente"])
    with k2:
        st.metric("Average Spread", f"R$ {ranking['SpreadEnergia'].mean():.1f}/MWh")
    with k3:
        st.metric("Distributors Analyzed", f"{len(ranking)}")
    with k4:
        pct = f"{len(viable) / len(ranking) * 100:.0f}%" if len(ranking) else "0%"
        st.metric("Payback < 7 years", f"{len(viable)}", pct + " of total")

    st.markdown('<div class="brand-divider"></div>', unsafe_allow_html=True)

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
        color_continuous_scale=SCALE_PAYBACK,
        labels={
            "SpreadEnergia": "Energy Spread (R$/MWh)",
            "SigAgente": "Distributor",
            "PaybackCapped": "Payback (yrs)",
        },
        title=f"Top {top_n} Distributors by Peak / Off-Peak Spread  —  {subgrupo} {modalidade}",
    )
    brand_layout(fig_bar, height=max(450, top_n * 28), yaxis=dict(categoryorder="total ascending"))
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
            color_continuous_scale=SCALE_PAYBACK,
            labels={
                "SpreadEnergia": "Energy Spread (R$/MWh)",
                "Payback": "Payback (years)",
                "AnnualRevenue": "Annual Revenue (R$)",
            },
        )
        fig_scatter.add_hline(
            y=7, line_dash="dash", line_color=COLORS["yellow"],
            annotation_text="7-year target",
            annotation_font_color=COLORS["yellow"],
        )
        brand_layout(fig_scatter, height=420)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Full ranking table (fixed — no .style)
    with st.expander("Full ranking table"):
        display_df = ranking[
            ["SigAgente", "SpreadEnergia", "Total_Ponta", "Total_FP", "AnnualRevenue", "Payback", "NPV"]
        ].copy()
        display_df.columns = [
            "Distributor", "Spread (R$/MWh)", "Peak Tariff", "Off-Peak Tariff",
            "Annual Revenue (R$)", "Payback (yrs)", "NPV (R$)",
        ]
        # Format numbers manually to avoid jinja2 dependency
        display_df["Spread (R$/MWh)"] = display_df["Spread (R$/MWh)"].map("{:.1f}".format)
        display_df["Peak Tariff"] = display_df["Peak Tariff"].map("R$ {:.2f}".format)
        display_df["Off-Peak Tariff"] = display_df["Off-Peak Tariff"].map("R$ {:.2f}".format)
        display_df["Annual Revenue (R$)"] = display_df["Annual Revenue (R$)"].map("R$ {:,.0f}".format)
        display_df["Payback (yrs)"] = display_df["Payback (yrs)"].map("{:.1f}".format)
        display_df["NPV (R$)"] = display_df["NPV (R$)"].map("R$ {:,.0f}".format)
        st.dataframe(display_df, hide_index=True, use_container_width=True)


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

    st.markdown('<div class="brand-divider"></div>', unsafe_allow_html=True)

    # Tariff breakdown — grouped bar
    st.subheader("Tariff Breakdown: Peak vs. Off-Peak (R$/MWh)")

    fig_tariff = go.Figure()
    fig_tariff.add_trace(
        go.Bar(
            name="Off-Peak",
            x=["TUSD", "TE", "Total"],
            y=[row["TUSD_FP"], row["TE_FP"], row["Total_FP"]],
            marker_color=COLORS["cyan"],
            text=[f"R$ {v:.1f}" for v in [row["TUSD_FP"], row["TE_FP"], row["Total_FP"]]],
            textposition="auto",
            textfont=dict(color=COLORS["text"]),
        )
    )
    fig_tariff.add_trace(
        go.Bar(
            name="Peak",
            x=["TUSD", "TE", "Total"],
            y=[row["TUSD_Ponta"], row["TE_Ponta"], row["Total_Ponta"]],
            marker_color=COLORS["red"],
            text=[f"R$ {v:.1f}" for v in [row["TUSD_Ponta"], row["TE_Ponta"], row["Total_Ponta"]]],
            textposition="auto",
            textfont=dict(color=COLORS["text"]),
        )
    )
    brand_layout(
        fig_tariff,
        barmode="group",
        yaxis=dict(title="R$/MWh"),
        height=380,
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
                marker_colors=[COLORS["green"], COLORS["cyan"]],
                hole=0.45,
                textinfo="label+percent+value",
                texttemplate="%{label}<br>%{percent:.1%}<br>R$ %{value:,.0f}",
                textfont=dict(color=COLORS["text"]),
            )
        )
        brand_layout(fig_pie, height=350, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)


# ===================== TAB 3: FINANCIAL SIMULATION =====================
with tab3:
    sel_fin = st.selectbox("Select distributor", distributors, key="fin_select")
    row_fin = ranking[ranking["SigAgente"] == sel_fin].iloc[0]
    spread_fin = row_fin["SpreadEnergia"]
    demand_fin = row_fin["Demanda_Ponta"]

    # Cash flow chart
    st.subheader("Cumulative Cash Flow")

    cf = cashflow_table(params, spread_fin, demand_fin)

    fig_cf = go.Figure()
    fig_cf.add_trace(
        go.Scatter(
            x=cf["Year"],
            y=cf["Cumulative"],
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color=COLORS["green"], width=3),
            marker=dict(color=COLORS["green"], size=7),
            fillcolor="rgba(74, 222, 128, 0.08)",
            hovertemplate="Year %{x}<br>R$ %{y:,.0f}<extra></extra>",
        )
    )
    fig_cf.add_hline(y=0, line_dash="dash", line_color=COLORS["surface2"])

    # Mark breakeven point
    breakeven = cf[cf["Cumulative"] >= 0]
    if not breakeven.empty:
        be_year = breakeven.iloc[0]["Year"]
        fig_cf.add_vline(
            x=be_year, line_dash="dot", line_color=COLORS["yellow"],
            annotation_text=f"Breakeven: Year {be_year:.0f}",
            annotation_font_color=COLORS["yellow"],
        )

    brand_layout(
        fig_cf,
        xaxis=dict(title="Year"),
        yaxis=dict(title="Cumulative Cash Flow (R$)", tickformat=",.0f"),
        height=420,
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
        color_continuous_scale=SCALE_PAYBACK,
        labels=dict(x="Spread (R$/MWh)", y="CAPEX (R$/kWh)", color="Payback (yrs)"),
        aspect="auto",
        text_auto=".1f",
    )
    brand_layout(fig_heat, height=420)
    fig_heat.update_traces(textfont=dict(color=COLORS["bg"], size=11))
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

Degradation is applied to energy capacity each year.
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

    # --- Interactive insight: CAPEX decline → more viable distributors ---
    st.markdown('<div class="brand-divider"></div>', unsafe_allow_html=True)
    st.subheader("How Falling CAPEX Unlocks New Markets")
    st.markdown(
        "As battery costs decline, distributors that were previously uneconomical cross "
        "the viability threshold. The chart below shows how many distributors achieve a "
        "payback under 7 years at different CAPEX levels."
    )

    capex_scenarios = list(range(1000, 6001, 250))
    viable_counts = []
    for capex_val in capex_scenarios:
        p = BESSParams(
            power_mw=params.power_mw,
            duration_h=params.duration_h,
            efficiency=params.efficiency,
            capex_per_kwh=capex_val,
            operating_days=params.operating_days,
            lifetime_years=params.lifetime_years,
        )
        count = sum(
            1
            for _, r in ranking.iterrows()
            if payback_years(p, r["SpreadEnergia"], r["Demanda_Ponta"]) <= 7
        )
        viable_counts.append(count)

    fig_capex = go.Figure()
    fig_capex.add_trace(
        go.Scatter(
            x=capex_scenarios,
            y=viable_counts,
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color=COLORS["green"], width=3),
            marker=dict(color=COLORS["green"], size=6),
            fillcolor="rgba(74, 222, 128, 0.08)",
            hovertemplate="CAPEX: R$ %{x:,.0f}/kWh<br>Viable: %{y} distributors<extra></extra>",
        )
    )
    # Mark current CAPEX
    current_viable = sum(
        1
        for _, r in ranking.iterrows()
        if payback_years(params, r["SpreadEnergia"], r["Demanda_Ponta"]) <= 7
    )
    fig_capex.add_trace(
        go.Scatter(
            x=[params.capex_per_kwh],
            y=[current_viable],
            mode="markers",
            marker=dict(color=COLORS["yellow"], size=14, symbol="star"),
            name="Current CAPEX",
            hovertemplate="Current: R$ %{x:,.0f}/kWh<br>%{y} distributors<extra></extra>",
        )
    )
    brand_layout(
        fig_capex,
        xaxis=dict(title="CAPEX (R$/kWh)", autorange="reversed"),
        yaxis=dict(title="Distributors with Payback < 7 years"),
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_capex, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown('<div class="brand-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="footer">'
    "Built by <strong>Filipe Osanai</strong> &middot; Energy Engineer &middot; "
    '<a href="https://brasilbess.com" target="_blank">brasilbess.com</a> &middot; '
    '<a href="https://dadosabertos.aneel.gov.br" target="_blank">ANEEL Open Data</a> &middot; '
    "Modo Energy Take-Home Task"
    "</div>",
    unsafe_allow_html=True,
)
