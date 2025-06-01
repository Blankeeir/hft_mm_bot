import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from connectors.base import OrderBook, OrderBookLevel, Trade, OrderSide

logger = logging.getLogger(__name__)


@dataclass
class OrderFlowImbalance:
    symbol: str
    ofi_1s: float
    ofi_5s: float
    ofi_10s: float
    timestamp: int


@dataclass
class MicroPrice:
    symbol: str
    micro_price: float
    mid_price: float
    weighted_mid: float
    imbalance_ratio: float
    timestamp: int


class OrderBookProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.orderbooks: Dict[str, OrderBook] = {}
        self.trades: Dict[str, List[Trade]] = {}
        self.ofi_window_1s = 1000
        self.ofi_window_5s = 5000
        self.ofi_window_10s = 10000
        
    def update_orderbook(self, orderbook: OrderBook) -> None:
        self.orderbooks[orderbook.symbol] = orderbook
        
    def update_trade(self, trade: Trade) -> None:
        if trade.symbol not in self.trades:
            self.trades[trade.symbol] = []
        self.trades[trade.symbol].append(trade)
        
        max_trades = 10000
        if len(self.trades[trade.symbol]) > max_trades:
            self.trades[trade.symbol] = self.trades[trade.symbol][-max_trades:]
    
    def get_level2_data(self, symbol: str, levels: int = 20) -> Optional[OrderBook]:
        return self.orderbooks.get(symbol)
    
    def calculate_order_flow_imbalance(self, symbol: str, current_time: int) -> Optional[OrderFlowImbalance]:
        if symbol not in self.trades:
            return None
            
        trades = self.trades[symbol]
        
        def calculate_ofi_for_window(window_ms: int) -> float:
            start_time = current_time - window_ms
            window_trades = [t for t in trades if t.timestamp >= start_time]
            
            buy_volume = sum(t.quantity for t in window_trades if t.side == OrderSide.BUY)
            sell_volume = sum(t.quantity for t in window_trades if t.side == OrderSide.SELL)
            
            return buy_volume - sell_volume
        
        ofi_1s = calculate_ofi_for_window(self.ofi_window_1s)
        ofi_5s = calculate_ofi_for_window(self.ofi_window_5s)
        ofi_10s = calculate_ofi_for_window(self.ofi_window_10s)
        
        return OrderFlowImbalance(
            symbol=symbol,
            ofi_1s=ofi_1s,
            ofi_5s=ofi_5s,
            ofi_10s=ofi_10s,
            timestamp=current_time
        )
    
    def get_orderbook_imbalance(self, symbol: str, levels: int = 5) -> float:
        orderbook = self.orderbooks.get(symbol)
        if not orderbook or not orderbook.bids or not orderbook.asks:
            return 0.0
            
        bid_volume = sum(level.quantity for level in orderbook.bids[:levels])
        ask_volume = sum(level.quantity for level in orderbook.asks[:levels])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
            
        return (bid_volume - ask_volume) / total_volume


class MicroPriceCalculator:
    def __init__(self, config: Dict):
        self.config = config
        self.queue_weight_factor = config.get("micro_price", {}).get("queue_weight_factor", 0.5)
        self.imbalance_threshold = config.get("micro_price", {}).get("imbalance_threshold", 0.1)
        
    def calculate_micro_price(self, orderbook: OrderBook, ofi: Optional[OrderFlowImbalance] = None) -> Optional[MicroPrice]:
        if not orderbook.bids or not orderbook.asks:
            return None
            
        best_bid = orderbook.bids[0]
        best_ask = orderbook.asks[0]
        mid_price = (best_bid.price + best_ask.price) / 2
        
        bid_volume = sum(level.quantity for level in orderbook.bids[:5])
        ask_volume = sum(level.quantity for level in orderbook.asks[:5])
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return MicroPrice(
                symbol=orderbook.symbol,
                micro_price=mid_price,
                mid_price=mid_price,
                weighted_mid=mid_price,
                imbalance_ratio=0.0,
                timestamp=orderbook.timestamp
            )
        
        imbalance_ratio = (bid_volume - ask_volume) / total_volume
        
        weighted_mid = (best_ask.price * bid_volume + best_bid.price * ask_volume) / total_volume
        
        ofi_adjustment = 0.0
        if ofi and abs(ofi.ofi_1s) > 0:
            ofi_normalized = np.tanh(ofi.ofi_1s / (bid_volume + ask_volume))
            ofi_adjustment = ofi_normalized * (best_ask.price - best_bid.price) * 0.1
        
        micro_price = weighted_mid + ofi_adjustment
        
        return MicroPrice(
            symbol=orderbook.symbol,
            micro_price=micro_price,
            mid_price=mid_price,
            weighted_mid=weighted_mid,
            imbalance_ratio=imbalance_ratio,
            timestamp=orderbook.timestamp
        )
    
    def calculate_queue_position_adjustment(self, orderbook: OrderBook, our_price: float, side: OrderSide) -> float:
        if side == OrderSide.BUY:
            levels = orderbook.bids
            better_levels = [level for level in levels if level.price > our_price]
        else:
            levels = orderbook.asks
            better_levels = [level for level in levels if level.price < our_price]
        
        if not better_levels:
            return 1.0
            
        total_better_volume = sum(level.quantity for level in better_levels)
        same_price_level = next((level for level in levels if level.price == our_price), None)
        
        if same_price_level:
            queue_position = total_better_volume + (same_price_level.quantity * self.queue_weight_factor)
            total_volume = total_better_volume + same_price_level.quantity
            return 1.0 - (queue_position / max(total_volume, 1))
        
        return 0.5
    
    def estimate_fill_probability(self, orderbook: OrderBook, price: float, side: OrderSide, 
                                time_horizon_ms: int = 1000) -> float:
        if not orderbook.bids or not orderbook.asks:
            return 0.0
            
        mid_price = (orderbook.bids[0].price + orderbook.asks[0].price) / 2
        spread = orderbook.asks[0].price - orderbook.bids[0].price
        
        if side == OrderSide.BUY:
            distance_from_mid = (mid_price - price) / spread
            base_probability = max(0.1, min(0.9, 0.5 + distance_from_mid * 0.4))
        else:
            distance_from_mid = (price - mid_price) / spread
            base_probability = max(0.1, min(0.9, 0.5 + distance_from_mid * 0.4))
        
        queue_adjustment = self.calculate_queue_position_adjustment(orderbook, price, side)
        
        time_factor = min(1.0, time_horizon_ms / 5000.0)
        
        return base_probability * queue_adjustment * time_factor
