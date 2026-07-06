"""
financial_tools.py

Pure calculation utilities for the Financial Analyst MCP server.
No Streamlit, no MCP client, no agent code should live here —
this module is imported directly by mcp_server.py.
"""

import math
import numpy as np


def calculate_cagr(start_val: float, end_val: float, periods: int) -> float:
    """
    Calculates the Compound Annual Growth Rate (CAGR).

    Args:
        start_val: The starting value (e.g., initial investment or price).
        end_val: The ending value (e.g., final investment or price).
        periods: Number of periods (typically years) between start and end.

    Returns:
        CAGR as a decimal (e.g., 0.12 for 12%).
    """
    if start_val <= 0:
        raise ValueError("start_val must be greater than 0.")
    if periods <= 0:
        raise ValueError("periods must be greater than 0.")

    return (end_val / start_val) ** (1 / periods) - 1


def calculate_volatility(prices: list[float], trading_periods: int = 252) -> float:
    """
    Calculates annualized historical volatility from a list of consecutive
    closing prices, based on the standard deviation of daily log returns.

    Args:
        prices: A list of consecutive closing prices (chronological order).
        trading_periods: Number of trading periods per year used to annualize
            (default 252 for daily data; use 12 for monthly, 52 for weekly).

    Returns:
        Annualized volatility as a decimal (e.g., 0.25 for 25%).
    """
    if len(prices) < 2:
        raise ValueError("Need at least two prices to compute volatility.")

    prices_arr = np.array(prices, dtype=float)
    if np.any(prices_arr <= 0):
        raise ValueError("All prices must be positive.")

    log_returns = np.diff(np.log(prices_arr))
    daily_std = np.std(log_returns, ddof=1)

    return daily_std * math.sqrt(trading_periods)
