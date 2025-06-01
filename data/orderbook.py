import numpy as np
from collections import defaultdict, deque

class OrderBook:
    """Very light L2 order book that stores top N levels."""
    def __init__(self, depth: int = 5):
        self.depth = depth
        self.bids = defaultdict(float)  # price -> size
        self.asks = defaultdict(float)
        self.bid_prices = deque(maxlen=depth)
        self.ask_prices = deque(maxlen=depth)

    def update(self, side: str, price: float, size: float):
        book = self.bids if side == 'buy' else self.asks
        if size == 0:
            if price in book:
                del book[price]
        else:
            book[price] = size
        self._trim()

    def _trim(self):
        self.bid_prices = deque(sorted(self.bids.keys(), reverse=True)[: self.depth], maxlen=self.depth)
        self.ask_prices = deque(sorted(self.asks.keys())[: self.depth], maxlen=self.depth)

    def best_bid(self):
        return self.bid_prices[0] if self.bid_prices else None

    def best_ask(self):
        return self.ask_prices[0] if self.ask_prices else None

    def micro_price(self):
        if not self.bid_prices or not self.ask_prices:
            return None
        pb, pa = self.best_bid(), self.best_ask()
        sb, sa = self.bids[pb], self.asks[pa]
        return (pb * sa + pa * sb) / (sb + sa)

    def imbalance(self):
        if not self.bid_prices or not self.ask_prices:
            return 0
        top_bids = sum(self.bids[p] for p in self.bid_prices)
        top_asks = sum(self.asks[p] for p in self.ask_prices)
        return (top_bids - top_asks) / (top_bids + top_asks + 1e-9)
