import numpy as np
from data.orderbook import OrderBook
from data.volatility import HARVol

class AvellanedaStoikovEngine:
    def __init__(self, gamma: float, k: float, base_size: float):
        self.gamma = gamma
        self.k = k
        self.base_size = base_size

    def quotes(self, micro_price: float, inventory: float, sigma2: float, dt: float, deltas):
        """Compute reservation price and optimal spreads."""
        r_t = micro_price - self.gamma * inventory * sigma2 * dt
        delta_sym = 0.5 * (self.gamma * sigma2 * dt + (1 / (self.gamma * self.k)) * np.log(1 + self.gamma * self.k))
        delta_b, delta_a, mult, refresh_flag = deltas
        bid = r_t - (delta_sym + delta_b)
        ask = r_t + (delta_sym + delta_a)
        size = self.base_size * max(mult, 0.1)
        return bid, ask, size, bool(refresh_flag)
