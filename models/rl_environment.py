import numpy as np
import random
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import asyncio
from collections import deque

from data.market_data import MarketSnapshot
from strategy.avellaneda_stoikov import QuoteParameters, AvellanedaStoikovEngine
from risk.risk_manager import RiskManager, RiskMetrics

logger = logging.getLogger(__name__)


@dataclass
class LatencyEvent:
    timestamp: int
    delay_ms: int
    event_type: str
    data: Any


class LatencySimulator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.min_delay_ms = config.get("min_delay_ms", 30)
        self.max_delay_ms = config.get("max_delay_ms", 100)
        self.batch_matching = config.get("batch_matching", True)
        
        self.pending_events: List[LatencyEvent] = []
        
    def add_latency(self, event_type: str, data: Any) -> int:
        if not self.enabled:
            return 0
            
        delay_ms = random.randint(self.min_delay_ms, self.max_delay_ms)
        
        if self.batch_matching and event_type == "order":
            batch_delay = random.randint(10, 50)
            delay_ms += batch_delay
            
        current_time = int(time.time() * 1000)
        
        event = LatencyEvent(
            timestamp=current_time + delay_ms,
            delay_ms=delay_ms,
            event_type=event_type,
            data=data
        )
        
        self.pending_events.append(event)
        return delay_ms
        
    def process_events(self, current_time: int) -> List[LatencyEvent]:
        ready_events = []
        remaining_events = []
        
        for event in self.pending_events:
            if event.timestamp <= current_time:
                ready_events.append(event)
            else:
                remaining_events.append(event)
                
        self.pending_events = remaining_events
        return ready_events


class RLEnvironment:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rl_config = config.get("reinforcement_learning", {})
        
        self.action_space_config = self.rl_config.get("action_space", {})
        self.reward_config = self.rl_config.get("reward_function", {})
        self.latency_config = self.rl_config.get("latency_simulation", {})
        
        self.as_engine = AvellanedaStoikovEngine(config)
        self.risk_manager = RiskManager(config)
        self.latency_simulator = LatencySimulator(self.latency_config)
        
        self.observation_dim = 20
        self.action_dim = 4
        
        self.spread_adjustment_range = self.action_space_config.get("spread_adjustment", [-0.5, 0.5])
        self.size_multiplier_range = self.action_space_config.get("size_multiplier", [0.1, 2.0])
        
        self.pnl_weight = self.reward_config.get("pnl_weight", 1.0)
        self.inventory_penalty = self.reward_config.get("inventory_penalty", 0.1)
        self.order_age_penalty = self.reward_config.get("order_age_penalty", 0.05)
        
        self.state_history: deque = deque(maxlen=1000)
        self.action_history: deque = deque(maxlen=1000)
        self.reward_history: deque = deque(maxlen=1000)
        
        self.current_inventory = 0.0
        self.current_pnl = 0.0
        self.last_action_time = 0
        self.episode_start_time = 0
        
    def reset(self, initial_snapshot: MarketSnapshot) -> np.ndarray:
        self.current_inventory = 0.0
        self.current_pnl = 0.0
        self.last_action_time = int(time.time() * 1000)
        self.episode_start_time = self.last_action_time
        
        self.state_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        
        self.as_engine.update_inventory(initial_snapshot.symbol, 0.0)
        
        return self._get_observation(initial_snapshot)
        
    def step(self, action: np.ndarray, snapshot: MarketSnapshot) -> Tuple[np.ndarray, float, bool, Dict]:
        current_time = int(time.time() * 1000)
        
        rl_adjustments = self._decode_action(action)
        
        quotes = self.as_engine.calculate_optimal_quotes(
            snapshot, 
            rl_adjustments=rl_adjustments
        )
        
        if quotes is None:
            return self._get_observation(snapshot), -10.0, True, {"error": "Invalid quotes"}
            
        risk_metrics = self.risk_manager.check_risk_limits(snapshot.symbol, snapshot)
        
        valid_quotes, violations = self.risk_manager.validate_quotes(quotes, snapshot)
        if not valid_quotes:
            penalty = -5.0 * len(violations)
            info = {
                "quotes": quotes,
                "violations": violations,
                "risk_metrics": risk_metrics,
                "pnl": self.current_pnl,
                "inventory": self.current_inventory
            }
            return self._get_observation(snapshot), penalty, False, info
            
        latency_delay = self.latency_simulator.add_latency("order", quotes)
        
        simulated_fill = self._simulate_order_execution(quotes, snapshot, latency_delay)
        
        reward = self._calculate_reward(simulated_fill, risk_metrics, action, current_time)
        
        self._update_state(simulated_fill, current_time)
        
        observation = self._get_observation(snapshot)
        done = self._check_episode_termination(risk_metrics, current_time)
        
        info = {
            "quotes": quotes,
            "risk_metrics": risk_metrics,
            "latency_delay": latency_delay,
            "simulated_fill": simulated_fill,
            "pnl": self.current_pnl,
            "inventory": self.current_inventory
        }
        
        self.state_history.append(observation)
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        return observation, reward, done, info
        
    def _decode_action(self, action: np.ndarray) -> Dict[str, float]:
        spread_adj_norm = np.clip(action[0], -1.0, 1.0)
        size_mult_norm = np.clip(action[1], -1.0, 1.0)
        reservation_adj_norm = np.clip(action[2], -1.0, 1.0)
        refresh_flag = 1.0 if action[3] > 0.0 else 0.0
        
        spread_adjustment = (spread_adj_norm * 
                           (self.spread_adjustment_range[1] - self.spread_adjustment_range[0]) / 2.0)
        
        size_multiplier = (self.size_multiplier_range[0] + 
                          (size_mult_norm + 1.0) / 2.0 * 
                          (self.size_multiplier_range[1] - self.size_multiplier_range[0]))
        
        reservation_adjustment = reservation_adj_norm * 0.1
        
        return {
            "spread_adjustment": spread_adjustment,
            "size_multiplier": size_multiplier,
            "reservation_adjustment": reservation_adjustment,
            "refresh_quotes": refresh_flag > 0.5
        }
        
    def _get_observation(self, snapshot: MarketSnapshot) -> np.ndarray:
        obs = np.zeros(self.observation_dim)
        
        if snapshot.orderbook and snapshot.orderbook.bids and snapshot.orderbook.asks:
            best_bid = snapshot.orderbook.bids[0].price
            best_ask = snapshot.orderbook.asks[0].price
            mid_price = (best_bid + best_ask) / 2.0
            spread = (best_ask - best_bid) / mid_price
            
            obs[0] = mid_price / 50000.0
            obs[1] = spread * 10000.0
            
            bid_volume = sum(level.quantity for level in snapshot.orderbook.bids[:5])
            ask_volume = sum(level.quantity for level in snapshot.orderbook.asks[:5])
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                obs[2] = (bid_volume - ask_volume) / total_volume
                obs[3] = np.log(total_volume + 1) / 10.0
                
        if snapshot.micro_price:
            obs[4] = snapshot.micro_price.imbalance_ratio
            obs[5] = (snapshot.micro_price.micro_price - snapshot.micro_price.mid_price) / snapshot.micro_price.mid_price * 10000.0
            
        if snapshot.ofi:
            obs[6] = np.tanh(snapshot.ofi.ofi_1s / 1000.0)
            obs[7] = np.tanh(snapshot.ofi.ofi_5s / 5000.0)
            
        obs[8] = np.tanh(self.current_inventory / 10.0)
        obs[9] = np.tanh(self.current_pnl / 1000.0)
        obs[10] = snapshot.volatility * 100.0
        
        current_time = int(time.time() * 1000)
        time_since_last_action = (current_time - self.last_action_time) / 1000.0
        obs[11] = np.tanh(time_since_last_action / 60.0)
        
        if len(self.reward_history) > 0:
            recent_rewards = list(self.reward_history)[-10:]
            obs[12] = np.mean(recent_rewards)
            obs[13] = np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
            
        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
            obs[14:18] = last_action
            
        obs[18] = (current_time % 86400000) / 86400000.0
        obs[19] = len(snapshot.recent_trades) / 100.0
        
        return obs
        
    def _simulate_order_execution(self, quotes: QuoteParameters, snapshot: MarketSnapshot, 
                                latency_delay: int) -> Dict[str, Any]:
        if not snapshot.orderbook or not snapshot.orderbook.bids or not snapshot.orderbook.asks:
            return {"bid_filled": 0.0, "ask_filled": 0.0, "pnl": 0.0}
            
        mid_price = (snapshot.orderbook.bids[0].price + snapshot.orderbook.asks[0].price) / 2.0
        
        bid_fill_prob = self._calculate_fill_probability(
            quotes.bid_price, mid_price, "bid", latency_delay, snapshot.volatility
        )
        ask_fill_prob = self._calculate_fill_probability(
            quotes.ask_price, mid_price, "ask", latency_delay, snapshot.volatility
        )
        
        bid_filled = quotes.bid_size * bid_fill_prob if random.random() < bid_fill_prob else 0.0
        ask_filled = quotes.ask_size * ask_fill_prob if random.random() < ask_fill_prob else 0.0
        
        pnl = 0.0
        if bid_filled > 0:
            pnl -= bid_filled * quotes.bid_price
            
        if ask_filled > 0:
            pnl += ask_filled * quotes.ask_price
            
        return {
            "bid_filled": bid_filled,
            "ask_filled": ask_filled,
            "pnl": pnl,
            "bid_price": quotes.bid_price,
            "ask_price": quotes.ask_price
        }
        
    def _calculate_fill_probability(self, order_price: float, mid_price: float, 
                                  side: str, latency_delay: int, volatility: float) -> float:
        if side == "bid":
            distance = (mid_price - order_price) / mid_price
        else:
            distance = (order_price - mid_price) / mid_price
            
        base_prob = max(0.0, min(1.0, 0.8 - distance * 25.0))
        
        latency_penalty = latency_delay / 2000.0
        base_prob *= max(0.1, 1.0 - latency_penalty)
        
        volatility_boost = min(0.2, volatility * 3.0)
        base_prob += volatility_boost
        
        return min(1.0, base_prob)
        
    def _calculate_reward(self, simulated_fill: Dict[str, Any], risk_metrics: RiskMetrics, 
                        action: np.ndarray, current_time: int) -> float:
        reward = 0.0
        
        pnl_reward = simulated_fill["pnl"] * self.pnl_weight
        reward += pnl_reward
        
        inventory_penalty = abs(self.current_inventory) * self.inventory_penalty
        reward -= inventory_penalty
        
        time_since_last = (current_time - self.last_action_time) / 1000.0
        age_penalty = time_since_last * self.order_age_penalty
        reward -= age_penalty
        
        if len(risk_metrics.violations) > 0:
            violation_penalty = len(risk_metrics.violations) * 2.0
            reward -= violation_penalty
            
        if risk_metrics.risk_level.value == "CRITICAL":
            reward -= 10.0
        elif risk_metrics.risk_level.value == "HIGH":
            reward -= 5.0
        elif risk_metrics.risk_level.value == "MEDIUM":
            reward -= 2.0
            
        spread_action = action[0]
        if abs(spread_action) > 0.8:
            reward -= abs(spread_action) * 0.5
            
        return reward
        
    def _update_state(self, simulated_fill: Dict[str, Any], current_time: int) -> None:
        inventory_change = simulated_fill["bid_filled"] - simulated_fill["ask_filled"]
        self.current_inventory += inventory_change
        
        self.current_pnl += simulated_fill["pnl"]
        
        self.as_engine.update_inventory("BTC/USDT", self.current_inventory)
        
        self.last_action_time = current_time
        
    def _check_episode_termination(self, risk_metrics: RiskMetrics, current_time: int) -> bool:
        if risk_metrics.risk_level.value == "CRITICAL":
            return True
            
        episode_duration = current_time - self.episode_start_time
        max_episode_duration = 3600000
        if episode_duration > max_episode_duration:
            return True
            
        if self.current_pnl < -1000.0:
            return True
            
        if abs(self.current_inventory) > 50.0:
            return True
            
        return False
        
    def get_action_space_size(self) -> int:
        return self.action_dim
        
    def get_observation_space_size(self) -> int:
        return self.observation_dim
