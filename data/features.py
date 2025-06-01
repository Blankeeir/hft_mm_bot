import numpy as np
from collections import deque
from .orderbook import OrderBook

class FeatureExtractor:
    def __init__(self):
        self.ofi_1s = deque(maxlen=20)
        self.ofi_5s = deque(maxlen=100)

    def on_book_update(self, ob: OrderBook, side: str, size_change: float):
        """Compute order-flow imbalance (OFI)."""
        sign = 1 if side == 'buy' else -1
        self.ofi_1s.append(sign * size_change)
        self.ofi_5s.append(sign * size_change)

    def get_features(self, ob: OrderBook):
        micro = ob.micro_price()
        imb = ob.imbalance()
        ofi1 = sum(self.ofi_1s)
        ofi5 = sum(self.ofi_5s)
        return np.array([micro, imb, ofi1, ofi5], dtype=np.float32)
