"""
Tests for simulator.bess_model — BESS Behind-the-Meter Financial Model.

Covers:
  - BESSParams defaults and computed properties
  - energy_revenue (basic, degradation, zero spread)
  - demand_savings (basic, zero demand)
  - total_annual_revenue (combines both sources)
  - payback_years (normal case, zero revenue)
  - cashflow_table (shape, year-0 negative, cumulative progression)
  - calculate_npv (positive and negative scenarios)
  - sensitivity_payback (shape, monotonic payback behaviour)
"""

import pandas as pd
import pytest

from simulator.bess_model import (
    BESSParams,
    calculate_npv,
    cashflow_table,
    demand_savings,
    energy_revenue,
    payback_years,
    sensitivity_payback,
    total_annual_revenue,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_params() -> BESSParams:
    """Return a BESSParams instance with all default values."""
    return BESSParams()


@pytest.fixture
def small_battery() -> BESSParams:
    """A small, easy-to-reason-about battery for hand-verified tests."""
    return BESSParams(
        power_mw=0.5,
        duration_h=2.0,
        efficiency=1.0,       # perfect round-trip for simpler maths
        cycles_per_day=1,
        operating_days=100,
        capex_per_kwh=1000.0,
        lifetime_years=5,
        degradation=0.0,      # no degradation for simpler maths
        discount_rate=0.10,
    )


# ===================================================================
# 1. BESSParams defaults and computed properties
# ===================================================================

class TestBESSParamsDefaults:
    """Verify dataclass defaults and derived properties."""

    def test_default_power(self, default_params: BESSParams):
        assert default_params.power_mw == 1.0

    def test_default_duration(self, default_params: BESSParams):
        assert default_params.duration_h == 2.0

    def test_default_efficiency(self, default_params: BESSParams):
        assert default_params.efficiency == 0.87

    def test_default_cycles_per_day(self, default_params: BESSParams):
        assert default_params.cycles_per_day == 1

    def test_default_operating_days(self, default_params: BESSParams):
        assert default_params.operating_days == 252

    def test_default_capex_per_kwh(self, default_params: BESSParams):
        assert default_params.capex_per_kwh == 3500.0

    def test_default_lifetime(self, default_params: BESSParams):
        assert default_params.lifetime_years == 15

    def test_default_degradation(self, default_params: BESSParams):
        assert default_params.degradation == 0.02

    def test_default_discount_rate(self, default_params: BESSParams):
        assert default_params.discount_rate == 0.10


class TestBESSParamsComputedProperties:
    """Verify computed properties capacity_kwh and total_capex."""

    def test_capacity_kwh_defaults(self, default_params: BESSParams):
        # 1.0 MW * 1000 * 2.0 h = 2000 kWh
        assert default_params.capacity_kwh == 2000.0

    def test_total_capex_defaults(self, default_params: BESSParams):
        # 2000 kWh * 3500 R$/kWh = 7_000_000
        assert default_params.total_capex == 7_000_000.0

    def test_capacity_kwh_custom(self):
        p = BESSParams(power_mw=2.5, duration_h=4.0)
        assert p.capacity_kwh == 10_000.0

    def test_total_capex_custom(self):
        p = BESSParams(power_mw=2.5, duration_h=4.0, capex_per_kwh=2000.0)
        # 10_000 kWh * 2000 = 20_000_000
        assert p.total_capex == 20_000_000.0


# ===================================================================
# 2. energy_revenue
# ===================================================================

class TestEnergyRevenue:
    """Test energy arbitrage revenue calculation."""

    def test_basic_calculation(self, small_battery: BESSParams):
        """Hand-verified: 0.5 MW * 2 h * 1.0 eff * 100 spread * 100 days * 1 cycle."""
        # energy_per_cycle = 0.5 * 2.0 * 1.0 * 1.0 = 1.0 MWh
        # revenue = 1.0 * 100 * 100 * 1 = 10_000
        result = energy_revenue(small_battery, spread=100.0, year=0)
        assert result == pytest.approx(10_000.0)

    def test_degradation_reduces_revenue(self, default_params: BESSParams):
        """Year-5 revenue should be lower than year-0 revenue."""
        rev_y0 = energy_revenue(default_params, spread=150.0, year=0)
        rev_y5 = energy_revenue(default_params, spread=150.0, year=5)
        assert rev_y5 < rev_y0

    def test_degradation_factor_correct(self, default_params: BESSParams):
        """Check the exact degradation multiplier for year 3."""
        rev_y0 = energy_revenue(default_params, spread=200.0, year=0)
        rev_y3 = energy_revenue(default_params, spread=200.0, year=3)
        expected_factor = (1 - 0.02) ** 3
        assert rev_y3 == pytest.approx(rev_y0 * expected_factor)

    def test_zero_spread_gives_zero_revenue(self, default_params: BESSParams):
        result = energy_revenue(default_params, spread=0.0, year=0)
        assert result == 0.0

    def test_negative_spread(self, default_params: BESSParams):
        """A negative spread yields negative revenue (losing money)."""
        result = energy_revenue(default_params, spread=-50.0, year=0)
        assert result < 0.0

    def test_no_degradation_year_0(self, default_params: BESSParams):
        """At year 0 the degradation factor is (1 - d)^0 = 1.0."""
        # energy_per_cycle = 1.0 * 2.0 * 0.87 = 1.74 MWh
        # revenue = 1.74 * 100 * 252 * 1 = 43_848.0
        result = energy_revenue(default_params, spread=100.0, year=0)
        assert result == pytest.approx(1.0 * 2.0 * 0.87 * 100.0 * 252 * 1)

    def test_multiple_cycles_per_day(self):
        p = BESSParams(
            power_mw=1.0,
            duration_h=1.0,
            efficiency=1.0,
            cycles_per_day=2,
            operating_days=100,
            degradation=0.0,
        )
        # energy_per_cycle = 1.0 * 1.0 * 1.0 = 1.0 MWh
        # revenue = 1.0 * 50 * 100 * 2 = 10_000
        assert energy_revenue(p, spread=50.0, year=0) == pytest.approx(10_000.0)


# ===================================================================
# 3. demand_savings
# ===================================================================

class TestDemandSavings:
    """Test peak demand reduction savings."""

    def test_basic_calculation(self, default_params: BESSParams):
        """1 MW = 1000 kW, demand_ponta = 50 R$/kW/month => 1000 * 50 * 12 = 600_000."""
        result = demand_savings(default_params, demand_ponta=50.0)
        assert result == pytest.approx(600_000.0)

    def test_zero_demand_ponta(self, default_params: BESSParams):
        """No peak demand charge => no savings."""
        result = demand_savings(default_params, demand_ponta=0.0)
        assert result == 0.0

    def test_proportional_to_power(self):
        """Doubling the battery power should double the savings."""
        p1 = BESSParams(power_mw=1.0)
        p2 = BESSParams(power_mw=2.0)
        assert demand_savings(p2, 30.0) == pytest.approx(
            2.0 * demand_savings(p1, 30.0)
        )

    def test_small_battery(self, small_battery: BESSParams):
        # 0.5 MW = 500 kW, demand_ponta = 20 => 500 * 20 * 12 = 120_000
        result = demand_savings(small_battery, demand_ponta=20.0)
        assert result == pytest.approx(120_000.0)


# ===================================================================
# 4. total_annual_revenue
# ===================================================================

class TestTotalAnnualRevenue:
    """Verify that total annual revenue combines both sources correctly."""

    def test_combines_energy_and_demand(self, default_params: BESSParams):
        spread = 150.0
        dp = 40.0
        year = 0
        expected = energy_revenue(default_params, spread, year) + demand_savings(
            default_params, dp
        )
        result = total_annual_revenue(default_params, spread, dp, year)
        assert result == pytest.approx(expected)

    def test_demand_ponta_defaults_to_zero(self, default_params: BESSParams):
        """When demand_ponta is omitted it defaults to 0."""
        result = total_annual_revenue(default_params, spread=100.0, year=0)
        expected = energy_revenue(default_params, spread=100.0, year=0)
        assert result == pytest.approx(expected)

    def test_zero_spread_only_demand(self, default_params: BESSParams):
        """With zero spread, revenue comes purely from demand savings."""
        dp = 25.0
        result = total_annual_revenue(default_params, spread=0.0, demand_ponta=dp, year=0)
        assert result == pytest.approx(demand_savings(default_params, dp))

    def test_degradation_affects_energy_not_demand(self, default_params: BESSParams):
        """Demand savings are constant; energy revenue degrades."""
        dp = 30.0
        spread = 100.0
        rev_y0 = total_annual_revenue(default_params, spread, dp, year=0)
        rev_y5 = total_annual_revenue(default_params, spread, dp, year=5)
        # Demand savings are the same in both years
        ds = demand_savings(default_params, dp)
        assert rev_y0 - ds > rev_y5 - ds


# ===================================================================
# 5. payback_years
# ===================================================================

class TestPaybackYears:
    """Test simple payback period calculation."""

    def test_normal_case(self, small_battery: BESSParams):
        # total_capex = 0.5 * 1000 * 2.0 * 1000 = 1_000_000
        # year1 revenue (year=0): energy_revenue + demand_savings(0)
        #   energy = 0.5 * 2 * 1.0 * 100 * 100 = 10_000 (spread=100)
        # payback = 1_000_000 / 10_000 = 100 years
        result = payback_years(small_battery, spread=100.0)
        assert result == pytest.approx(100.0)

    def test_with_demand_savings(self, small_battery: BESSParams):
        # energy = 10_000 (spread=100), demand = 500 * 10 * 12 = 60_000
        # total year1 = 70_000, capex = 1_000_000
        # payback = 1_000_000 / 70_000 ~ 14.2857
        result = payback_years(small_battery, spread=100.0, demand_ponta=10.0)
        assert result == pytest.approx(1_000_000.0 / 70_000.0)

    def test_zero_revenue_returns_inf(self, default_params: BESSParams):
        """When revenue is zero, payback should be infinity."""
        result = payback_years(default_params, spread=0.0, demand_ponta=0.0)
        assert result == float("inf")

    def test_negative_revenue_returns_inf(self, default_params: BESSParams):
        """Negative spread with no demand savings => negative revenue => inf payback."""
        result = payback_years(default_params, spread=-100.0, demand_ponta=0.0)
        assert result == float("inf")

    def test_higher_spread_lower_payback(self, default_params: BESSParams):
        """A higher spread should lead to a shorter payback period."""
        pb_low = payback_years(default_params, spread=100.0)
        pb_high = payback_years(default_params, spread=200.0)
        assert pb_high < pb_low


# ===================================================================
# 6. cashflow_table
# ===================================================================

class TestCashflowTable:
    """Test the yearly cash-flow DataFrame generation."""

    def test_correct_shape(self, small_battery: BESSParams):
        """Should have lifetime_years + 1 rows (year 0 through lifetime_years)."""
        df = cashflow_table(small_battery, spread=100.0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == small_battery.lifetime_years + 1  # 6 rows for lifetime=5

    def test_correct_columns(self, small_battery: BESSParams):
        df = cashflow_table(small_battery, spread=100.0)
        assert list(df.columns) == ["Year", "Annual Revenue", "Cumulative"]

    def test_year_0_is_negative_capex(self, small_battery: BESSParams):
        """Year 0 should have zero revenue and cumulative = -total_capex."""
        df = cashflow_table(small_battery, spread=100.0)
        row0 = df.iloc[0]
        assert row0["Year"] == 0
        assert row0["Annual Revenue"] == 0.0
        assert row0["Cumulative"] == pytest.approx(-small_battery.total_capex)

    def test_cumulative_increases_with_positive_spread(self, small_battery: BESSParams):
        """With positive revenue each year, cumulative should strictly increase."""
        df = cashflow_table(small_battery, spread=100.0)
        cumulative_values = df["Cumulative"].tolist()
        for i in range(1, len(cumulative_values)):
            assert cumulative_values[i] > cumulative_values[i - 1]

    def test_annual_revenue_positive_after_year_0(self, small_battery: BESSParams):
        """All years after year 0 should have positive annual revenue."""
        df = cashflow_table(small_battery, spread=100.0)
        for _, row in df.iloc[1:].iterrows():
            assert row["Annual Revenue"] > 0

    def test_year_column_sequential(self, default_params: BESSParams):
        df = cashflow_table(default_params, spread=100.0)
        expected_years = list(range(default_params.lifetime_years + 1))
        assert df["Year"].tolist() == expected_years

    def test_cumulative_last_row_matches_sum(self, small_battery: BESSParams):
        """The final cumulative value should equal -capex + sum of all annual revenues."""
        df = cashflow_table(small_battery, spread=100.0)
        total_rev = df["Annual Revenue"].sum()
        expected_cumulative = -small_battery.total_capex + total_rev
        assert df.iloc[-1]["Cumulative"] == pytest.approx(expected_cumulative)

    def test_zero_spread_zero_demand(self, small_battery: BESSParams):
        """With no revenue, cumulative stays at -capex throughout."""
        df = cashflow_table(small_battery, spread=0.0, demand_ponta=0.0)
        for _, row in df.iterrows():
            assert row["Cumulative"] == pytest.approx(-small_battery.total_capex)


# ===================================================================
# 7. calculate_npv
# ===================================================================

class TestCalculateNPV:
    """Test Net Present Value calculation."""

    def test_positive_npv_with_good_spread(self):
        """A high spread and low capex should produce positive NPV."""
        p = BESSParams(
            power_mw=1.0,
            duration_h=2.0,
            efficiency=1.0,
            cycles_per_day=1,
            operating_days=252,
            capex_per_kwh=500.0,   # very cheap
            lifetime_years=15,
            degradation=0.0,
            discount_rate=0.10,
        )
        # capex = 2000 * 500 = 1_000_000
        # annual revenue = 1 * 2 * 1.0 * 200 * 252 = 100_800
        # PV of annuity @ 10% for 15 years ~ 100_800 * 7.606 ~ 766_685
        # NPV ~ 766_685 - 1_000_000 is negative with those numbers.
        # Use a higher spread to ensure positive.
        npv = calculate_npv(p, spread=500.0)
        assert npv > 0

    def test_negative_npv_with_bad_spread(self, default_params: BESSParams):
        """A very low spread with expensive battery should yield negative NPV."""
        npv = calculate_npv(default_params, spread=10.0, demand_ponta=0.0)
        assert npv < 0

    def test_npv_accounts_for_discounting(self, small_battery: BESSParams):
        """NPV should be less than the simple sum of revenues minus capex
        because future cash flows are discounted."""
        df = cashflow_table(small_battery, spread=100.0)
        simple_total = df["Cumulative"].iloc[-1]  # undiscounted
        npv = calculate_npv(small_battery, spread=100.0)
        # With a positive discount rate and positive revenues,
        # NPV < undiscounted cumulative (both can be negative, NPV more negative)
        assert npv < simple_total

    def test_zero_discount_rate_equals_undiscounted(self):
        """With discount_rate=0, NPV should equal -capex + sum of revenues."""
        p = BESSParams(
            power_mw=1.0,
            duration_h=2.0,
            efficiency=1.0,
            cycles_per_day=1,
            operating_days=100,
            capex_per_kwh=1000.0,
            lifetime_years=5,
            degradation=0.0,
            discount_rate=0.0,
        )
        spread = 200.0
        npv = calculate_npv(p, spread=spread)
        # annual rev = 1 * 2 * 1.0 * 200 * 100 = 40_000 per year
        # total undiscounted = -2_000_000 + 5 * 40_000 = -1_800_000
        expected = -p.total_capex + 5 * 40_000.0
        assert npv == pytest.approx(expected)

    def test_demand_ponta_improves_npv(self, default_params: BESSParams):
        """Adding demand savings should increase NPV."""
        npv_no_demand = calculate_npv(default_params, spread=100.0, demand_ponta=0.0)
        npv_with_demand = calculate_npv(default_params, spread=100.0, demand_ponta=30.0)
        assert npv_with_demand > npv_no_demand


# ===================================================================
# 8. sensitivity_payback
# ===================================================================

class TestSensitivityPayback:
    """Test the payback sensitivity matrix generation."""

    def test_correct_shape(self, default_params: BESSParams):
        """Output should have len(spreads) * len(capexes) rows."""
        spreads = [100.0, 150.0, 200.0]
        capexes = [2000.0, 3000.0, 4000.0]
        df = sensitivity_payback(default_params, spreads, capexes)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(spreads) * len(capexes)

    def test_correct_columns(self, default_params: BESSParams):
        spreads = [100.0]
        capexes = [2000.0]
        df = sensitivity_payback(default_params, spreads, capexes)
        assert list(df.columns) == ["CAPEX (R$/kWh)", "Spread (R$/MWh)", "Payback"]

    def test_payback_decreases_with_higher_spread(self, default_params: BESSParams):
        """For a fixed capex, higher spread should give lower (or equal) payback."""
        spreads = [50.0, 100.0, 200.0, 400.0]
        capexes = [3000.0]
        df = sensitivity_payback(default_params, spreads, capexes)
        paybacks = df["Payback"].tolist()
        for i in range(1, len(paybacks)):
            assert paybacks[i] <= paybacks[i - 1]

    def test_payback_increases_with_higher_capex(self, default_params: BESSParams):
        """For a fixed spread, higher capex should give higher payback."""
        spreads = [150.0]
        capexes = [1000.0, 2000.0, 3000.0, 4000.0]
        df = sensitivity_payback(default_params, spreads, capexes)
        paybacks = df["Payback"].tolist()
        for i in range(1, len(paybacks)):
            assert paybacks[i] >= paybacks[i - 1]

    def test_payback_capped_at_30(self, default_params: BESSParams):
        """Payback values should be capped at 30 for display purposes."""
        spreads = [0.01]  # tiny spread => huge payback
        capexes = [5000.0]
        df = sensitivity_payback(default_params, spreads, capexes)
        assert df["Payback"].iloc[0] <= 30.0

    def test_with_demand_ponta(self, default_params: BESSParams):
        """Demand savings should reduce payback across the board."""
        spreads = [100.0, 200.0]
        capexes = [3000.0]
        df_no_dp = sensitivity_payback(default_params, spreads, capexes, demand_ponta=0.0)
        df_with_dp = sensitivity_payback(
            default_params, spreads, capexes, demand_ponta=20.0
        )
        for pb_no, pb_with in zip(
            df_no_dp["Payback"].tolist(), df_with_dp["Payback"].tolist()
        ):
            assert pb_with <= pb_no

    def test_zero_spread_zero_demand_capped(self, default_params: BESSParams):
        """Zero spread and zero demand => inf payback, capped to 30."""
        spreads = [0.0]
        capexes = [3000.0]
        df = sensitivity_payback(default_params, spreads, capexes, demand_ponta=0.0)
        assert df["Payback"].iloc[0] == 30.0

    def test_params_override_capex_only(self, default_params: BESSParams):
        """The sensitivity function should only vary capex, keeping other params."""
        spreads = [100.0]
        capexes = [2500.0]
        df = sensitivity_payback(default_params, spreads, capexes)
        # Verify the CAPEX column reflects the input, not the default
        assert df["CAPEX (R$/kWh)"].iloc[0] == 2500.0
