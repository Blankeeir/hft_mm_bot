from collections import deque
import numpy as np

class PnLTracker:
    def __init__(self, maxlen=1000):
        self.pnl_hist = deque(maxlen=maxlen)
        self.last_equity = None

    def update(self, equity):
        if self.last_equity is None:
            self.last_equity = equity
            return 0
        pnl = equity - self.last_equity
        self.pnl_hist.append(pnl)
        self.last_equity = equity
        return pnl

    def sharpe(self):
        arr = np.array(self.pnl_hist)
        if arr.std() == 0:
            return 0.0
        return np.sqrt(365*24*60) * arr.mean() / arr.std()
