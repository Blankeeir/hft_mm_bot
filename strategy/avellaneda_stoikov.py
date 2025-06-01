import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from connectors.base import OrderSide, OrderType
from data.market_data import MarketSnapshot
from data.volatility import VolatilityForecast

logger = logging.getLogger(__name__)


@dataclass
class QuoteParameters:
    symbol: str
    reservation_price: float
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    spread: float
    timestamp: int


@dataclass
class ASParameters:
    gamma: float
    A: float
    k: float
    T: float
    sigma: float


class AvellanedaStoikovEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.as_config = config.get("avellaneda_stoikov", {})
        
        self.gamma = self.as_config.get("risk_aversion", 0.1)
        self.A = self.as_config.get("order_arrival_intensity", 1.5)
        self.k = 1.0 / self.A
        self.T = self.as_config.get("time_horizon", 86400)
        
        self.inventory: Dict[str, float] = {}
        self.last_update: Dict[str, int] = {}
        
        self.min_spread_bps = config.get("order_management", {}).get("min_spread_bps", 1)
        self.max_spread_bps = config.get("order_management", {}).get("max_spread_bps", 100)
        
    def update_inventory(self, symbol: str, inventory: float) -> None:
        self.inventory[symbol] = inventory
        
    def calculate_reservation_price(self, symbol: str, mid_price: float, 
                                  volatility: float, time_remaining: float) -> float:
        inventory = self.inventory.get(symbol, 0.0)
        
        inventory_penalty = inventory * self.gamma * (volatility ** 2) * time_remaining
        reservation_price = mid_price - inventory_penalty
        
        return reservation_price
        
    def calculate_optimal_spread(self, volatility: float, time_remaining: float) -> float:
        if time_remaining <= 0:
            time_remaining = 1.0
            
        spread_component1 = self.gamma * (volatility ** 2) * time_remaining
        spread_component2 = (1.0 / (self.gamma * self.k)) * np.log(1 + self.gamma * self.k)
        
        optimal_half_spread = 0.5 * (spread_component1 + spread_component2)
        
        return 2.0 * optimal_half_spread
        
    def calculate_optimal_quotes(self, snapshot: MarketSnapshot, 
                               volatility_forecast: Optional[VolatilityForecast] = None,
                               rl_adjustments: Optional[Dict[str, float]] = None) -> Optional[QuoteParameters]:
        if not snapshot.orderbook or not snapshot.orderbook.bids or not snapshot.orderbook.asks:
            return None
            
        symbol = snapshot.symbol
        current_time = snapshot.timestamp
        
        if snapshot.micro_price:
            reference_price = snapshot.micro_price.micro_price
        else:
            best_bid = snapshot.orderbook.bids[0].price
            best_ask = snapshot.orderbook.asks[0].price
            reference_price = (best_bid + best_ask) / 2.0
            
        if volatility_forecast:
            volatility = volatility_forecast.combined_forecast
        else:
            volatility = snapshot.volatility if snapshot.volatility > 0 else 0.01
            
        current_time_seconds = current_time / 1000.0
        time_remaining = max(1.0, self.T - (current_time_seconds % self.T))
        
        reservation_price = self.calculate_reservation_price(
            symbol, reference_price, volatility, time_remaining
        )
        
        optimal_spread = self.calculate_optimal_spread(volatility, time_remaining)
        
        min_spread = reference_price * (self.min_spread_bps / 10000.0)
        max_spread = reference_price * (self.max_spread_bps / 10000.0)
        optimal_spread = max(min_spread * 1.001, min(max_spread, optimal_spread))
        
        half_spread = optimal_spread / 2.0
        
        if rl_adjustments:
            spread_adj = rl_adjustments.get("spread_adjustment", 0.0)
            half_spread *= (1.0 + spread_adj)
            
            reservation_adj = rl_adjustments.get("reservation_adjustment", 0.0)
            reservation_price += reservation_adj * half_spread
            
        bid_price = reservation_price - half_spread
        ask_price = reservation_price + half_spread
        
        base_size = self.calculate_base_order_size(symbol, reference_price, volatility)
        
        if rl_adjustments:
            size_multiplier = rl_adjustments.get("size_multiplier", 1.0)
            base_size *= size_multiplier
            
        inventory = self.inventory.get(symbol, 0.0)
        inventory_skew = self.calculate_inventory_skew(inventory, volatility)
        
        bid_size = base_size * (1.0 - inventory_skew)
        ask_size = base_size * (1.0 + inventory_skew)
        
        bid_size = max(0.01, bid_size)
        ask_size = max(0.01, ask_size)
        
        final_spread = ask_price - bid_price
        
        return QuoteParameters(
            symbol=symbol,
            reservation_price=reservation_price,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            spread=final_spread,
            timestamp=current_time
        )
        
    def calculate_base_order_size(self, symbol: str, price: float, volatility: float) -> float:
        base_size_usd = self.config.get("order_management", {}).get("default_order_size", 1000)
        base_size = base_size_usd / price
        
        volatility_adjustment = 1.0 / (1.0 + volatility * 10.0)
        
        return base_size * volatility_adjustment
        
    def calculate_inventory_skew(self, inventory: float, volatility: float) -> float:
        max_skew = 0.3
        
        inventory_normalized = np.tanh(inventory * volatility * 10.0)
        
        return inventory_normalized * max_skew
        
    def should_refresh_quotes(self, symbol: str, current_time: int, 
                            refresh_interval_ms: int = 200) -> bool:
        last_update = self.last_update.get(symbol, 0)
        return (current_time - last_update) >= refresh_interval_ms
        
    def update_parameters_from_market_regime(self, symbol: str, 
                                           market_conditions: Dict[str, float]) -> None:
        volatility = market_conditions.get("volatility", 0.01)
        volume = market_conditions.get("volume", 1.0)
        spread = market_conditions.get("spread", 0.001)
        
        if volatility > 0.05:
            self.gamma = min(0.2, self.gamma * 1.1)
        elif volatility < 0.01:
            self.gamma = max(0.05, self.gamma * 0.9)
            
        if volume > 2.0:
            self.A = min(3.0, self.A * 1.05)
        elif volume < 0.5:
            self.A = max(0.5, self.A * 0.95)
            
        self.k = 1.0 / self.A
        
    def calculate_fair_value_adjustment(self, snapshot: MarketSnapshot) -> float:
        if not snapshot.ofi:
            return 0.0
            
        ofi_1s = snapshot.ofi.ofi_1s
        ofi_5s = snapshot.ofi.ofi_5s
        
        if not snapshot.orderbook or not snapshot.orderbook.bids or not snapshot.orderbook.asks:
            return 0.0
            
        spread = snapshot.orderbook.asks[0].price - snapshot.orderbook.bids[0].price
        
        ofi_weight_1s = 0.7
        ofi_weight_5s = 0.3
        
        combined_ofi = ofi_weight_1s * ofi_1s + ofi_weight_5s * ofi_5s
        
        max_adjustment = spread * 0.1
        
        adjustment = np.tanh(combined_ofi / 1000.0) * max_adjustment
        
        return adjustment
        
    def get_parameters(self, symbol: str) -> ASParameters:
        return ASParameters(
            gamma=self.gamma,
            A=self.A,
            k=self.k,
            T=self.T,
            sigma=0.01
        )
        
    def set_parameters(self, symbol: str, params: ASParameters) -> None:
        self.gamma = params.gamma
        self.A = params.A
        self.k = params.k
        self.T = params.T
