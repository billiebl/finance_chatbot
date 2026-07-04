import numpy as np

def calculate_cagr(start_val: float, end_val: float, periods: int) -> float:
    if start_val <= 0 or end_val <= 0 or periods <= 0:
        return 0.0
    return (end_val / start_val) ** (1 / periods) - 1

def calculate_volatility(prices: list[float]) -> float:
    if len(prices) < 2:
        return 0.0
    log_returns = np.log(np.array(prices)[1:] / np.array(prices)[:-1])
    return np.std(log_returns) * np.sqrt(252)
