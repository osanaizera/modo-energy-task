"""
BESS Behind-the-Meter Financial Model

Simulates revenue from two sources:
1. Energy arbitrage: charge off-peak, discharge at peak (R$/MWh spread)
2. Peak demand reduction: lower contracted peak demand (R$/kW) — Azul tariff only
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BESSParams:
    """Battery Energy Storage System parameters."""

    power_mw: float = 1.0  # Rated power
    duration_h: float = 2.0  # Hours at rated power
    efficiency: float = 0.87  # Round-trip efficiency
    cycles_per_day: int = 1  # Charge/discharge cycles per day
    operating_days: int = 252  # Business days per year
    capex_per_kwh: float = 3500.0  # R$/kWh installed cost
    lifetime_years: int = 15  # Project lifetime
    degradation: float = 0.02  # Annual capacity degradation
    discount_rate: float = 0.10  # Discount rate for NPV

    @property
    def capacity_kwh(self) -> float:
        return self.power_mw * 1000 * self.duration_h

    @property
    def total_capex(self) -> float:
        return self.capacity_kwh * self.capex_per_kwh


def energy_revenue(params: BESSParams, spread: float, year: int = 0) -> float:
    """Annual revenue from peak/off-peak energy arbitrage (R$).

    The battery charges at off-peak tariff and discharges at peak tariff.
    Revenue = energy_shifted × spread × operating_days.

    Args:
        params: Battery system parameters.
        spread: Peak minus off-peak total tariff (R$/MWh).
        year: Year of operation (0-indexed, for degradation).
    """
    degradation_factor = (1 - params.degradation) ** year
    energy_per_cycle = (
        params.power_mw * params.duration_h * params.efficiency * degradation_factor
    )
    return energy_per_cycle * spread * params.operating_days * params.cycles_per_day


def demand_savings(params: BESSParams, demand_ponta: float) -> float:
    """Annual savings from peak demand reduction — Azul tariff only (R$).

    In Azul modality, peak demand has a separate (higher) monthly charge.
    The battery discharges during peak hours, reducing the contracted
    peak demand by up to its power rating.

    Args:
        params: Battery system parameters.
        demand_ponta: Peak demand TUSD charge (R$/kW per month).
    """
    reduction_kw = params.power_mw * 1000
    return reduction_kw * demand_ponta * 12  # monthly × 12


def total_annual_revenue(
    params: BESSParams, spread: float, demand_ponta: float = 0, year: int = 0
) -> float:
    """Total annual revenue: energy arbitrage + demand savings."""
    return energy_revenue(params, spread, year) + demand_savings(params, demand_ponta)


def payback_years(params: BESSParams, spread: float, demand_ponta: float = 0) -> float:
    """Simple payback period in years."""
    year1 = total_annual_revenue(params, spread, demand_ponta, year=0)
    if year1 <= 0:
        return float("inf")
    return params.total_capex / year1


def cashflow_table(
    params: BESSParams, spread: float, demand_ponta: float = 0
) -> pd.DataFrame:
    """Generate yearly cash flow over battery lifetime."""
    rows = []
    cumulative = -params.total_capex

    for y in range(params.lifetime_years + 1):
        if y == 0:
            rows.append({"Year": 0, "Annual Revenue": 0, "Cumulative": cumulative})
        else:
            rev = total_annual_revenue(params, spread, demand_ponta, year=y)
            cumulative += rev
            rows.append({"Year": y, "Annual Revenue": rev, "Cumulative": cumulative})

    return pd.DataFrame(rows)


def calculate_npv(
    params: BESSParams, spread: float, demand_ponta: float = 0
) -> float:
    """Net Present Value of the BESS investment."""
    npv = -params.total_capex
    for y in range(1, params.lifetime_years + 1):
        rev = total_annual_revenue(params, spread, demand_ponta, year=y)
        npv += rev / (1 + params.discount_rate) ** y
    return npv


def sensitivity_payback(
    params: BESSParams,
    spreads: list[float],
    capexes: list[float],
    demand_ponta: float = 0,
) -> pd.DataFrame:
    """Payback sensitivity matrix across CAPEX and spread values."""
    rows = []
    for capex in capexes:
        p = BESSParams(
            power_mw=params.power_mw,
            duration_h=params.duration_h,
            efficiency=params.efficiency,
            cycles_per_day=params.cycles_per_day,
            operating_days=params.operating_days,
            capex_per_kwh=capex,
            lifetime_years=params.lifetime_years,
            degradation=params.degradation,
        )
        for spread in spreads:
            pb = payback_years(p, spread, demand_ponta)
            rows.append(
                {
                    "CAPEX (R$/kWh)": capex,
                    "Spread (R$/MWh)": round(spread, 1),
                    "Payback": min(pb, 30),  # cap for display
                }
            )
    return pd.DataFrame(rows)
