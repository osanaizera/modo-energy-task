# Brazil BESS Behind-the-Meter Viability Simulator

**Modo Energy Take-Home Task** — Filipe Osanai

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-54%20passed-4ade80)

<!-- Screenshot placeholder — replace with actual dashboard screenshot -->
<!-- ![Dashboard Screenshot](docs/screenshot.png) -->

## Problem Statement

Brazil has **90+ electricity distributors**, each with different tariff structures set by the regulator (ANEEL). For Group A consumers (medium/high voltage — industries, commercial buildings, hospitals), the difference between **peak** and **off-peak** tariffs determines whether a behind-the-meter battery (BESS) is economically viable.

This spread varies enormously across distributors and regions. **No public tool exists to compare BESS viability across all Brazilian distributors.**

This simulator answers: **Where in Brazil does the tariff structure make behind-the-meter battery storage economically viable?**

## Why Now?

- **LRCAP Storage**: Brazil's first dedicated battery storage auction is scheduled for **April 2026**
- **BTM is the entry point**: Behind-the-meter BESS requires no auction — viability is driven purely by tariff economics
- **Falling CAPEX**: Global battery costs continue to decline, making more locations viable each year
- **Fragmented market**: 90+ distributors create geographic pockets of opportunity that only systematic analysis can reveal

## Key Findings

Running the simulator with default parameters (1 MW / 2 MWh, R$ 3,500/kWh CAPEX, Verde modality, A4 subgroup):

| Metric | Value |
|--------|-------|
| Distributors analyzed | 116 |
| Median energy spread | R$ 1,576/MWh |
| Highest spread | R$ 5,009/MWh (CERAL ARARUAMA) |
| Best mainstream spread | R$ 2,630/MWh (COELBA — Bahia) |
| Simple payback (best case) | 3.2 years |
| Payback with Azul demand savings | 1.5 years |

The **Northeast region** (COELBA, EQUATORIAL MA/PA/PI) consistently shows the highest spreads among mainstream distributors, making it the most attractive region for BTM BESS deployment.

## How to Run

```bash
# Clone the repository
git clone https://github.com/osanaizera/modo-energy-task.git
cd modo-energy-task

# Quick start
make install
make run          # starts Streamlit dashboard

# Run tests
make test         # 54 tests, ~0.4s
```

Or manually:

```bash
pip install -r requirements.txt
streamlit run app.py
```

A **sample dataset** (14 distributors, 252 rows) is bundled for instant offline demo. The full ANEEL dataset (~78 MB, 116 distributors) downloads automatically on first run.

Requirements: Python 3.10+

## Dashboard Sections

### 1. Opportunity Map
Top distributors ranked by peak/off-peak spread, with payback-colored bar chart and scatter plot showing the relationship between spread and payback period.

### 2. Distributor Deep Dive
Detailed analysis of a selected distributor: tariff breakdown (TUSD vs. TE, peak vs. off-peak), revenue composition (energy arbitrage vs. demand savings), and key financial metrics.

### 3. Financial Simulation
Cumulative cash flow over battery lifetime with breakeven visualization, plus sensitivity heatmap showing how payback varies with CAPEX and spread.

### 4. Market Context
Explanation of the Brazilian electricity tariff system for non-Brazilian audiences — Group A/B consumers, Verde vs. Azul modalities, TUSD/TE components, and why BESS captures value. Includes an interactive chart showing how falling CAPEX unlocks new distributor markets.

## How It Works

1. Downloads and processes **homologated tariff data** from ANEEL Open Data (all 90+ distributors)
2. Calculates the **peak vs. off-peak spread** (TUSD + TE) for each distributor
3. Simulates BESS revenue from two sources:
   - **Energy arbitrage**: charge off-peak, discharge at peak (R$/MWh spread)
   - **Peak demand reduction**: lower contracted peak demand (Azul tariff only, R$/kW)
4. Ranks all distributors by economic viability (payback, NPV)
5. Provides interactive financial simulation with adjustable parameters

## Data Source

**ANEEL Open Data** — Homologated tariffs for all Brazilian electricity distributors.

- URL: https://dadosabertos.aneel.gov.br/dataset/tarifas-distribuidoras-energia-eletrica
- ~311,000 records covering all tariff resolutions since 2010
- Updated as new tariff resolutions are published
- Filtered to: Tarifa de Aplicacao, Group A (A1-A4), Horossazonal modalities (Azul/Verde)

## Methodology

### Energy Arbitrage (Verde + Azul)
```
energy_per_cycle = power_MW × duration_h × efficiency × degradation_factor
annual_revenue = energy_per_cycle × spread_R$/MWh × operating_days
```

### Demand Reduction (Azul only)
```
annual_savings = power_kW × peak_demand_charge_R$/kW/month × 12
```

### Key Assumptions (adjustable in dashboard)
| Parameter | Default | Range |
|-----------|---------|-------|
| Power | 1 MW | 0.1 - 10 MW |
| Duration | 2 hours | 1, 2, 4 hours |
| Round-trip efficiency | 87% | 80 - 95% |
| CAPEX | R$ 3,500/kWh | R$ 1,000 - 6,000 |
| Lifetime | 15 years | 10 - 25 years |
| Degradation | 2%/year | Fixed |
| Operating days | 252/year | 200 - 365 |

## Project Structure

```
modo-energy-task/
├── app.py                       # Streamlit dashboard (4 tabs)
├── data/
│   ├── load_data.py             # ANEEL data download & processing
│   ├── sample-tarifas.csv       # Bundled sample (14 distributors)
│   └── py.typed
├── simulator/
│   ├── bess_model.py            # BESS financial model
│   └── py.typed
├── tests/
│   └── test_bess_model.py       # 54 tests covering all model functions
├── Makefile                     # make install / run / test
├── requirements.txt
├── .streamlit/config.toml       # Dark theme (brasilbess.com design)
└── README.md
```

## AI Workflow

This project was built using **Claude Code (Claude Opus 4.6)** as a collaborative development tool:

- **Problem definition**: Refined the problem statement from broad energy market analysis to a specific, actionable question about BTM BESS viability across Brazilian distributors
- **Data engineering**: Identified the ANEEL Open Data source, analyzed the data dictionary, designed the processing pipeline (encoding, decimal format, seasonal normalization, latest-resolution filtering)
- **Financial modeling**: Structured the BESS revenue model with two value streams (energy arbitrage + demand reduction), including degradation curves and sensitivity analysis
- **Dashboard development**: Full Streamlit application with Plotly charts, interactive parameters, and market context
- **Testing**: 54 pytest tests with hand-verified arithmetic covering all model functions
- **Quality assurance**: Validated outputs against known distributor tariffs and industry benchmarks

AI was used to accelerate development while domain expertise (energy engineering, Brazilian regulatory knowledge) guided every architectural and analytical decision.

## Author

**Filipe Osanai** — Energy Engineer | [brasilbess.com](https://brasilbess.com)

## Built With

Python, Streamlit, Pandas, Plotly, ANEEL Open Data
