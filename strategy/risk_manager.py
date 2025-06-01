class RiskManager:
    def __init__(self, inv_max, vol_max, mdd_limit):
        self.inv_max = inv_max
        self.vol_max = vol_max
        self.mdd_limit = mdd_limit
        self.high_watermark = 0

    def check(self, inventory, sigma, equity, pnl):
        actions = {}
        if abs(inventory) > self.inv_max:
            actions['liquidate'] = True
        if sigma > self.vol_max:
            actions['widen'] = True
        drawdown = (self.high_watermark - equity) / (self.high_watermark + 1e-9)
        self.high_watermark = max(self.high_watermark, equity)
        if drawdown > self.mdd_limit:
            actions['halt'] = True
        return actions
