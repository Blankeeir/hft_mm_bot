import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
import logging
import pandas as pd
import numpy as np
from collections import defaultdict, deque

from connectors.base import OrderBook, Trade, OrderSide
from .orderbook import OrderBookProcessor, MicroPriceCalculator, OrderFlowImbalance, MicroPrice

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    symbol: str
    timestamp: int
    orderbook: Optional[OrderBook] = None
    micro_price: Optional[MicroPrice] = None
    ofi: Optional[OrderFlowImbalance] = None
    recent_trades: List[Trade] = field(default_factory=list)
    volatility: float = 0.0


class MarketDataManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.orderbook_processor = OrderBookProcessor(config)
        self.micro_price_calculator = MicroPriceCalculator(config)
        
        self.snapshots: Dict[str, MarketSnapshot] = {}
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        self.callbacks: Dict[str, List[Callable]] = {
            "snapshot_update": [],
            "price_update": [],
            "trade_update": [],
        }
        
        self.update_interval = 50
        self.running = False
        
    async def start(self) -> None:
        self.running = True
        asyncio.create_task(self._update_loop())
        logger.info("MarketDataManager started")
        
    async def stop(self) -> None:
        self.running = False
        logger.info("MarketDataManager stopped")
        
    def add_callback(self, event_type: str, callback: Callable) -> None:
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            
    def remove_callback(self, event_type: str, callback: Callable) -> None:
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
            
    async def _emit_event(self, event_type: str, data: Any) -> None:
        for callback in self.callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in callback for {event_type}: {e}")
                
    async def handle_orderbook_update(self, orderbook: OrderBook) -> None:
        self.orderbook_processor.update_orderbook(orderbook)
        await self._update_snapshot(orderbook.symbol)
        
    async def handle_trade_update(self, trade: Trade) -> None:
        self.orderbook_processor.update_trade(trade)
        self.trade_history[trade.symbol].append(trade)
        self.price_history[trade.symbol].append((trade.timestamp, trade.price))
        
        await self._emit_event("trade_update", trade)
        await self._update_snapshot(trade.symbol)
        
    async def _update_snapshot(self, symbol: str) -> None:
        current_time = int(time.time() * 1000)
        
        orderbook = self.orderbook_processor.get_level2_data(symbol)
        if not orderbook:
            return
            
        ofi = self.orderbook_processor.calculate_order_flow_imbalance(symbol, current_time)
        micro_price = self.micro_price_calculator.calculate_micro_price(orderbook, ofi)
        
        recent_trades = list(self.trade_history[symbol])[-100:]
        
        volatility = self._calculate_short_term_volatility(symbol)
        
        snapshot = MarketSnapshot(
            symbol=symbol,
            timestamp=current_time,
            orderbook=orderbook,
            micro_price=micro_price,
            ofi=ofi,
            recent_trades=recent_trades,
            volatility=volatility
        )
        
        self.snapshots[symbol] = snapshot
        await self._emit_event("snapshot_update", snapshot)
        
        if micro_price:
            await self._emit_event("price_update", {
                "symbol": symbol,
                "micro_price": micro_price.micro_price,
                "mid_price": micro_price.mid_price,
                "timestamp": current_time
            })
            
    def _calculate_short_term_volatility(self, symbol: str, window_ms: int = 60000) -> float:
        if symbol not in self.price_history:
            return 0.0
            
        current_time = int(time.time() * 1000)
        start_time = current_time - window_ms
        
        recent_prices = [price for timestamp, price in self.price_history[symbol] 
                        if timestamp >= start_time]
        
        if len(recent_prices) < 2:
            return 0.0
            
        returns = np.diff(np.log(recent_prices))
        return float(np.std(returns) * np.sqrt(len(returns)))
        
    async def _update_loop(self) -> None:
        while self.running:
            try:
                await asyncio.sleep(self.update_interval / 1000.0)
                
                for symbol in list(self.snapshots.keys()):
                    await self._update_snapshot(symbol)
                    
            except Exception as e:
                logger.error(f"Error in market data update loop: {e}")
                
    def get_snapshot(self, symbol: str) -> Optional[MarketSnapshot]:
        return self.snapshots.get(symbol)
        
    def get_all_snapshots(self) -> Dict[str, MarketSnapshot]:
        return self.snapshots.copy()
        
    def get_price_history(self, symbol: str, window_ms: int = 3600000) -> List[Tuple[int, float]]:
        if symbol not in self.price_history:
            return []
            
        current_time = int(time.time() * 1000)
        start_time = current_time - window_ms
        
        return [(timestamp, price) for timestamp, price in self.price_history[symbol] 
                if timestamp >= start_time]
                
    def get_trade_history(self, symbol: str, window_ms: int = 3600000) -> List[Trade]:
        if symbol not in self.trade_history:
            return []
            
        current_time = int(time.time() * 1000)
        start_time = current_time - window_ms
        
        return [trade for trade in self.trade_history[symbol] 
                if trade.timestamp >= start_time]
                
    def calculate_vwap(self, symbol: str, window_ms: int = 3600000) -> float:
        trades = self.get_trade_history(symbol, window_ms)
        
        if not trades:
            return 0.0
            
        total_volume = sum(trade.quantity for trade in trades)
        if total_volume == 0:
            return 0.0
            
        weighted_price = sum(trade.price * trade.quantity for trade in trades)
        return weighted_price / total_volume
        
    def calculate_twap(self, symbol: str, window_ms: int = 3600000) -> float:
        price_history = self.get_price_history(symbol, window_ms)
        
        if not price_history:
            return 0.0
            
        return sum(price for _, price in price_history) / len(price_history)
        
    def get_market_impact_estimate(self, symbol: str, side: OrderSide, quantity: float) -> float:
        snapshot = self.get_snapshot(symbol)
        if not snapshot or not snapshot.orderbook:
            return 0.0
            
        orderbook = snapshot.orderbook
        levels = orderbook.asks if side == OrderSide.BUY else orderbook.bids
        
        remaining_quantity = quantity
        total_cost = 0.0
        
        for level in levels:
            if remaining_quantity <= 0:
                break
                
            fill_quantity = min(remaining_quantity, level.quantity)
            total_cost += fill_quantity * level.price
            remaining_quantity -= fill_quantity
            
        if quantity == 0:
            return 0.0
            
        average_price = total_cost / (quantity - remaining_quantity)
        
        if side == OrderSide.BUY:
            reference_price = orderbook.asks[0].price
        else:
            reference_price = orderbook.bids[0].price
            
        return abs(average_price - reference_price) / reference_price
