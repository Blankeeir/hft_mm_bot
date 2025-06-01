import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class HARVol:
    """HAR(1d,1w,1m) volatility model."""
    def __init__(self):
        self.window_d = 24*60   # 1 day of minutes
        self.window_w = 7 * self.window_d
        self.window_m = 30 * self.window_d
        self.ret_series = []

    def add_return(self, r):
        self.ret_series.append(r)
        if len(self.ret_series) > self.window_m:
            self.ret_series.pop(0)

    def forecast(self):
        if len(self.ret_series) < self.window_d:
            return 0.0
        rd = np.mean(np.square(self.ret_series[-self.window_d:]))
        rw = np.mean(np.square(self.ret_series[-self.window_w:]))
        rm = np.mean(np.square(self.ret_series[-self.window_m:]))
        return 0.5 * rd + 0.3 * rw + 0.2 * rm
