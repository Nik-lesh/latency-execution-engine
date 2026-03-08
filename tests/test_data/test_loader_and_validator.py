"""
Tests for data loading and validation.

PRODUCTION PATTERN: Test with synthetic data, not real data.
Why?
  1. Tests must run anywhere (CI/CD) without downloading 100MB files
  2. You control the test data — so you know the expected answer
  3. Tests run in milliseconds, not minutes
  4. You can create edge cases that rarely appear in real data

Testing philosophy:
  - Test the happy path (normal data works)
  - Test edge cases (empty data, single row, duplicates)
  - Test error cases (missing columns, wrong types)
  - Test contracts (output schema matches what's promised)
"""

import numpy as np
import pandas as pd
import pytest

from src.data.schemas import KlineSchema
from src.data.validator import validate_klines


def make_synthetic_klines(
    n_bars: int = 1000,
    start_price: float = 40000.0,
    volatility: float = 0.001,
    start_time: str = "2023-01-01",
    freq_minutes: int = 1,
    symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    """Generate realistic synthetic kline data for testing.

    Uses a geometric Brownian motion model to produce realistic OHLCV bars.
    This is a utility used across many tests.

    Args:
        n_bars: Number of bars to generate.
        start_price: Starting price.
        volatility: Per-bar volatility (std of log returns).
        start_time: Start timestamp.
        freq_minutes: Bar size in minutes.
        symbol: Symbol name.

    Returns:
        DataFrame matching KlineSchema.
    """
    rng = np.random.default_rng(seed=42)  # Reproducible!

    # Generate price path via geometric Brownian motion
    log_returns = rng.normal(0, volatility, n_bars)
    close_prices = start_price * np.exp(np.cumsum(log_returns))

    # Generate OHLC from close (realistic: H > C > L, with noise)
    noise = rng.uniform(0.0001, 0.001, n_bars)
    highs = close_prices * (1 + noise)
    lows = close_prices * (1 - noise)
    opens = np.roll(close_prices, 1)
    opens[0] = start_price

    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, close_prices))
    lows = np.minimum(lows, np.minimum(opens, close_prices))

    # Volume with realistic intraday pattern (U-shape)
    base_volume = rng.exponential(10.0, n_bars)
    intraday_pattern = 1 + 0.5 * np.cos(np.linspace(0, 2 * np.pi, min(n_bars, 1440)))
    if n_bars > 1440:
        intraday_pattern = np.tile(intraday_pattern, n_bars // 1440 + 1)[:n_bars]
    volume = base_volume * intraday_pattern[:n_bars]

    timestamps = pd.date_range(
        start=start_time,
        periods=n_bars,
        freq=f"{freq_minutes}min",
        tz="UTC",
    )

    df = pd.DataFrame({
        KlineSchema.TIMESTAMP: timestamps,
        KlineSchema.OPEN: opens,
        KlineSchema.HIGH: highs,
        KlineSchema.LOW: lows,
        KlineSchema.CLOSE: close_prices,
        KlineSchema.VOLUME: volume,
        KlineSchema.QUOTE_VOLUME: volume * close_prices,
        KlineSchema.TRADES: rng.integers(50, 500, n_bars),
        KlineSchema.SYMBOL: symbol,
    })

    return df


# =====================================================================
# TESTS
# =====================================================================

class TestSyntheticDataGeneration:
    """Test that our test data generator works correctly."""

    def test_basic_generation(self):
        df = make_synthetic_klines(n_bars=100)
        assert len(df) == 100
        assert set(KlineSchema.required_columns()).issubset(df.columns)

    def test_reproducibility(self):
        """Same seed → same data. Critical for debugging."""
        df1 = make_synthetic_klines(n_bars=50)
        df2 = make_synthetic_klines(n_bars=50)
        pd.testing.assert_frame_equal(df1, df2)

    def test_ohlc_consistency(self):
        """High >= max(Open, Close) and Low <= min(Open, Close)."""
        df = make_synthetic_klines(n_bars=5000)
        assert (df[KlineSchema.HIGH] >= df[KlineSchema.OPEN]).all()
        assert (df[KlineSchema.HIGH] >= df[KlineSchema.CLOSE]).all()
        assert (df[KlineSchema.LOW] <= df[KlineSchema.OPEN]).all()
        assert (df[KlineSchema.LOW] <= df[KlineSchema.CLOSE]).all()

    def test_positive_prices(self):
        df = make_synthetic_klines(n_bars=1000)
        for col in [KlineSchema.OPEN, KlineSchema.HIGH, KlineSchema.LOW, KlineSchema.CLOSE]:
            assert (df[col] > 0).all()

    def test_positive_volume(self):
        df = make_synthetic_klines(n_bars=1000)
        assert (df[KlineSchema.VOLUME] > 0).all()


class TestValidation:
    """Test the data validation pipeline."""

    def test_clean_data_passes(self):
        """Valid data should pass validation."""
        df = make_synthetic_klines(n_bars=1000)
        report = validate_klines(df)
        print(report)  # pytest -s to see this
        assert report.is_valid is True
        assert report.total_rows == 1000
        assert report.negative_prices == 0
        assert report.ohlc_violations == 0

    def test_missing_columns_fails(self):
        """Data missing required columns should fail."""
        df = make_synthetic_klines(n_bars=100)
        df = df.drop(columns=[KlineSchema.CLOSE])
        report = validate_klines(df)
        assert report.is_valid is False
        assert any("Missing required columns" in issue for issue in report.issues)

    def test_duplicate_timestamps_detected(self):
        """Duplicate timestamps should be flagged."""
        df = make_synthetic_klines(n_bars=100)
        # Duplicate the last row
        df = pd.concat([df, df.iloc[[-1]]], ignore_index=True)
        report = validate_klines(df)
        assert report.duplicate_timestamps == 1
        assert report.is_valid is False

    def test_negative_prices_detected(self):
        """Negative prices should fail validation."""
        df = make_synthetic_klines(n_bars=100)
        df.loc[5, KlineSchema.CLOSE] = -100.0
        report = validate_klines(df)
        assert report.negative_prices > 0
        assert report.is_valid is False

    def test_ohlc_violation_detected(self):
        """High < Low should be flagged."""
        df = make_synthetic_klines(n_bars=100)
        df.loc[10, KlineSchema.HIGH] = df.loc[10, KlineSchema.LOW] - 1.0
        report = validate_klines(df)
        assert report.ohlc_violations > 0
        assert report.is_valid is False

    def test_missing_bars_calculated(self):
        """Gaps in timestamps should be detected."""
        df = make_synthetic_klines(n_bars=100)
        # Remove bars 50-60 (creating a 10-minute gap)
        df = df.drop(index=range(50, 60)).reset_index(drop=True)
        report = validate_klines(df, expected_freq_minutes=1)
        assert report.missing_bars >= 9  # At least 9 missing
        assert report.missing_pct > 0

    def test_empty_dataframe(self):
        """Edge case: empty DataFrame shouldn't crash."""
        df = pd.DataFrame(columns=KlineSchema.all_columns())
        # This tests defensive coding — empty data shouldn't throw
        # It should fail validation gracefully
        report = validate_klines(df)
        assert report.total_rows == 0
