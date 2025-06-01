import time
import numpy as np
import asyncio
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from connectors.base import Position, Balance, OrderSide
from data.market_data import MarketSnapshot
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strategy.avellaneda_stoikov import QuoteParameters

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskLimits:
    max_inventory_ratio: float
    max_position_size: float
    volatility_threshold: float
    drawdown_limit: float
    max_daily_loss: float
    max_order_size: float
    min_spread_bps: float
    max_spread_bps: float


@dataclass
class RiskMetrics:
    symbol: str
    timestamp: int
    inventory_ratio: float
    position_size: float
    unrealized_pnl: float
    daily_pnl: float
    volatility: float
    drawdown: float
    risk_level: RiskLevel
    violations: List[str]


class RiskManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_config = config.get("risk_management", {})
        
        self.limits = RiskLimits(
            max_inventory_ratio=self.risk_config.get("max_inventory_ratio", 0.1),
            max_position_size=self.risk_config.get("max_position_size", 1000000),
            volatility_threshold=self.risk_config.get("volatility_threshold", 0.05),
            drawdown_limit=self.risk_config.get("drawdown_limit", 0.02),
            max_daily_loss=self.risk_config.get("max_daily_loss", 10000),
            max_order_size=self.risk_config.get("max_order_size", 100000),
            min_spread_bps=self.risk_config.get("min_spread_bps", 1),
            max_spread_bps=self.risk_config.get("max_spread_bps", 100)
        )
        
        self.circuit_breakers = config.get("circuit_breakers", {})
        self.volatility_multiplier = self.circuit_breakers.get("volatility_multiplier", 1.5)
        self.emergency_threshold = self.circuit_breakers.get("inventory_emergency_threshold", 0.15)
        
        self.positions: Dict[str, Position] = {}
        self.balances: Dict[str, Balance] = {}
        self.pnl_history: Dict[str, List[Tuple[int, float]]] = {}
        self.daily_pnl: Dict[str, float] = {}
        self.peak_portfolio_value: float = 0.0
        
        self.callbacks: Dict[str, List[Callable]] = {
            "risk_violation": [],
            "circuit_breaker": [],
            "emergency_liquidation": [],
        }
        
    def add_callback(self, event_type: str, callback: Callable) -> None:
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            
    async def _emit_event(self, event_type: str, data: Any) -> None:
        for callback in self.callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in callback for {event_type}: {e}")
                
    def update_positions(self, positions: List[Position]) -> None:
        for position in positions:
            self.positions[position.symbol] = position
            
    def update_balances(self, balances: List[Balance]) -> None:
        for balance in balances:
            self.balances[balance.asset] = balance
            
    def update_pnl(self, symbol: str, pnl: float) -> None:
        current_time = int(time.time() * 1000)
        
        if symbol not in self.pnl_history:
            self.pnl_history[symbol] = []
            
        self.pnl_history[symbol].append((current_time, pnl))
        
        max_history = 10000
        if len(self.pnl_history[symbol]) > max_history:
            self.pnl_history[symbol] = self.pnl_history[symbol][-max_history:]
            
        today_start = current_time - (current_time % 86400000)
        daily_pnl = sum(
            pnl for timestamp, pnl in self.pnl_history[symbol] 
            if timestamp >= today_start
        )
        self.daily_pnl[symbol] = daily_pnl
        
    def check_risk_limits(self, symbol: str, snapshot: MarketSnapshot) -> RiskMetrics:
        current_time = int(time.time() * 1000)
        violations = []
        
        position = self.positions.get(symbol)
        inventory_ratio = 0.0
        position_size = 0.0
        unrealized_pnl = 0.0
        
        if position:
            portfolio_value = self.get_portfolio_value()
            if portfolio_value > 0:
                position_value = abs(position.size * position.mark_price)
                inventory_ratio = position_value / portfolio_value
                
            position_size = abs(position.size)
            unrealized_pnl = position.unrealized_pnl
            
        daily_pnl = self.daily_pnl.get(symbol, 0.0)
        volatility = snapshot.volatility if snapshot else 0.0
        drawdown = self.calculate_drawdown()
        
        if inventory_ratio > self.limits.max_inventory_ratio:
            violations.append(f"Inventory ratio {inventory_ratio:.2%} exceeds limit {self.limits.max_inventory_ratio:.2%}")
            
        if position_size > self.limits.max_position_size:
            violations.append(f"Position size {position_size} exceeds limit {self.limits.max_position_size}")
            
        if volatility > self.limits.volatility_threshold:
            violations.append(f"Volatility {volatility:.2%} exceeds threshold {self.limits.volatility_threshold:.2%}")
            
        if drawdown > self.limits.drawdown_limit:
            violations.append(f"Drawdown {drawdown:.2%} exceeds limit {self.limits.drawdown_limit:.2%}")
            
        if daily_pnl < -self.limits.max_daily_loss:
            violations.append(f"Daily loss {daily_pnl} exceeds limit {self.limits.max_daily_loss}")
            
        risk_level = self.calculate_risk_level(violations, inventory_ratio, volatility, drawdown)
        
        metrics = RiskMetrics(
            symbol=symbol,
            timestamp=current_time,
            inventory_ratio=inventory_ratio,
            position_size=position_size,
            unrealized_pnl=unrealized_pnl,
            daily_pnl=daily_pnl,
            volatility=volatility,
            drawdown=drawdown,
            risk_level=risk_level,
            violations=violations
        )
        
        if violations:
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(self._emit_event("risk_violation", metrics))
            except RuntimeError:
                pass
            
        return metrics
        
    def calculate_risk_level(self, violations: List[str], inventory_ratio: float, 
                           volatility: float, drawdown: float) -> RiskLevel:
        if len(violations) >= 3:
            return RiskLevel.CRITICAL
        elif len(violations) >= 2:
            return RiskLevel.HIGH
        elif len(violations) >= 1:
            return RiskLevel.MEDIUM
        elif (inventory_ratio > self.limits.max_inventory_ratio * 0.8 or
              volatility > self.limits.volatility_threshold * 0.8 or
              drawdown > self.limits.drawdown_limit * 0.8):
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
            
    def validate_quotes(self, quotes: 'QuoteParameters', snapshot: MarketSnapshot) -> Tuple[bool, List[str]]:
        violations = []
        
        if not snapshot.orderbook or not snapshot.orderbook.bids or not snapshot.orderbook.asks:
            violations.append("No valid orderbook data")
            return False, violations
            
        mid_price = (snapshot.orderbook.bids[0].price + snapshot.orderbook.asks[0].price) / 2.0
        
        spread_bps = (quotes.spread / mid_price) * 10000
        if spread_bps < self.limits.min_spread_bps:
            violations.append(f"Spread {spread_bps:.1f} bps below minimum {self.limits.min_spread_bps} bps")
            
        if spread_bps > self.limits.max_spread_bps:
            violations.append(f"Spread {spread_bps:.1f} bps above maximum {self.limits.max_spread_bps} bps")
            
        bid_size_usd = quotes.bid_size * quotes.bid_price
        ask_size_usd = quotes.ask_size * quotes.ask_price
        
        if bid_size_usd > self.limits.max_order_size:
            violations.append(f"Bid size ${bid_size_usd:.0f} exceeds maximum ${self.limits.max_order_size:.0f}")
            
        if ask_size_usd > self.limits.max_order_size:
            violations.append(f"Ask size ${ask_size_usd:.0f} exceeds maximum ${self.limits.max_order_size:.0f}")
            
        if quotes.bid_price >= quotes.ask_price:
            violations.append("Bid price >= ask price (crossed quotes)")
            
        best_bid = snapshot.orderbook.bids[0].price
        best_ask = snapshot.orderbook.asks[0].price
        
        if quotes.bid_price > best_ask:
            violations.append("Bid price above best ask (aggressive)")
            
        if quotes.ask_price < best_bid:
            violations.append("Ask price below best bid (aggressive)")
            
        return len(violations) == 0, violations
        
    def should_trigger_circuit_breaker(self, symbol: str, snapshot: MarketSnapshot) -> bool:
        if not self.circuit_breakers.get("enabled", True):
            return False
            
        metrics = self.check_risk_limits(symbol, snapshot)
        
        if metrics.risk_level == RiskLevel.CRITICAL:
            return True
            
        if metrics.inventory_ratio > self.emergency_threshold:
            return True
            
        if metrics.volatility > self.limits.volatility_threshold * self.volatility_multiplier:
            return True
            
        if metrics.drawdown > self.limits.drawdown_limit:
            return True
            
        return False
        
    def get_position_adjustment(self, symbol: str, side: OrderSide) -> float:
        position = self.positions.get(symbol)
        if not position:
            return 1.0
            
        inventory_ratio = 0.0
        portfolio_value = self.get_portfolio_value()
        
        if portfolio_value > 0:
            position_value = abs(position.size * position.mark_price)
            inventory_ratio = position_value / portfolio_value
            
        if inventory_ratio > self.limits.max_inventory_ratio * 0.5:
            if ((position.side == OrderSide.BUY and side == OrderSide.BUY) or
                (position.side == OrderSide.SELL and side == OrderSide.SELL)):
                return max(0.1, 1.0 - inventory_ratio)
            else:
                return min(2.0, 1.0 + inventory_ratio)
                
        return 1.0
        
    def get_portfolio_value(self) -> float:
        total_value = 0.0
        
        for balance in self.balances.values():
            if balance.asset == "USDT" or balance.asset == "USD":
                total_value += balance.total
            else:
                total_value += balance.total * 50000
                
        return total_value
        
    def calculate_drawdown(self) -> float:
        current_value = self.get_portfolio_value()
        
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
            
        if self.peak_portfolio_value == 0:
            return 0.0
            
        return (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        
    def get_var_estimate(self, symbol: str, confidence: float = 0.95, window_hours: int = 24) -> float:
        if symbol not in self.pnl_history:
            self.pnl_history[symbol] = []
            return 0.0
            
        current_time = int(time.time() * 1000)
        start_time = current_time - (window_hours * 3600000)
        
        recent_pnl = [
            pnl for timestamp, pnl in self.pnl_history[symbol] 
            if timestamp >= start_time
        ]
        
        if len(recent_pnl) < 10:
            return 0.0
            
        return float(np.percentile(recent_pnl, (1 - confidence) * 100))
        
    def get_sharpe_ratio(self, symbol: str, window_hours: int = 24) -> float:
        if symbol not in self.pnl_history:
            self.pnl_history[symbol] = []
            return 0.0
            
        current_time = int(time.time() * 1000)
        start_time = current_time - (window_hours * 3600000)
        
        recent_pnl = [
            pnl for timestamp, pnl in self.pnl_history[symbol] 
            if timestamp >= start_time
        ]
        
        if len(recent_pnl) < 10:
            return 0.0
            
        returns = np.array(recent_pnl)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
            
        return float(mean_return / std_return * np.sqrt(len(returns)))
