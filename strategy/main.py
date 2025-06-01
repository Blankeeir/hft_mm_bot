import asyncio, yaml, time, uuid, numpy as np, importlib
from utils.logger import get_logger
from strategy.quote_engine import AvellanedaStoikovEngine
from strategy.risk_manager import RiskManager
from data.orderbook import OrderBook
from data.features import FeatureExtractor
from data.volatility import HARVol
from utils.metrics import PnLTracker
from stable_baselines3 import PPO

log = get_logger("main")

async def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    pair = cfg['strategy']['pair']
    # dynamic import connector, here demo with ws fallback
    from connectors.ws_fallback import BinanceWSConnector
    connector = BinanceWSConnector(cfg['exchanges']['binance'])
    await connector.connect()

    ob = OrderBook()
    feat = FeatureExtractor()
    vol = HARVol()
    pnl = PnLTracker()
    engine = AvellanedaStoikovEngine(cfg['strategy']['gamma'],
                                     cfg['strategy']['k'],
                                     cfg['strategy']['base_order_size'])
    risk = RiskManager(cfg['strategy']['inventory_max'],
                       cfg['strategy']['vol_max'],
                       cfg['strategy']['mdd_limit'])
    model = None
    try:
        model = PPO.load("ppo_mm_" + pair)
        log.info("Loaded trained policy.")
    except Exception:
        log.warning("Policy not found. Using zero deltas.")
    inventory = 0.0
    last_price = None
    async for msg in connector.stream_book(pair):
        # parse bids/asks update
        for price, size in msg.get("bids", []):
            ob.update("buy", float(price), float(size))
        for price, size in msg.get("asks", []):
            ob.update("sell", float(price), float(size))
        micro = ob.micro_price()
        if micro and last_price:
            ret = (micro - last_price) / last_price
            vol.add_return(ret)
        last_price = micro
        sigma2 = vol.forecast()
        features = feat.get_features(ob)
        dt = 1.0/20  # 50 ms
        if model:
            deltas = model.predict(np.concatenate([features, [inventory, sigma2]]), deterministic=True)[0]
        else:
            deltas = np.zeros(4)
        bid, ask, size, refresh = engine.quotes(micro, inventory, sigma2, dt, deltas)
        # risk checks
        actions = risk.check(inventory, np.sqrt(sigma2), 1.0, 0.0)
        if 'halt' in actions:
            log.error("Halting due to risk.")
            break
        # send orders (placeholder)
        client_id = str(uuid.uuid4())[:8]
        await connector.send_order("BUY", bid, size, client_id)
        await connector.send_order("SELL", ask, size, client_id)
        if refresh:
            await connector.cancel(client_id)
        equity = 1.0  # TODO: fetch
        pnl.update(equity)
        if pnl.last_equity and int(time.time()) % 60 == 0:
            log.info(f"Sharpe {pnl.sharpe():.2f}")
        await asyncio.sleep(0.05)

if __name__ == "__main__":
    asyncio.run(main())
