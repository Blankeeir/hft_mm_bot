import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from connectors.base import BaseExchangeConnector
from connectors.binance import BinanceConnector
from connectors.coinbase import CoinbaseConnector
from connectors.okx import OKXConnector
from data.market_data import MarketDataManager, MarketSnapshot
from data.volatility import VolatilityForecaster
from strategy.avellaneda_stoikov import AvellanedaStoikovEngine, QuoteParameters
from strategy.order_manager import OrderManager
from risk.risk_manager import RiskManager, RiskLevel
from models.ppo_agent import PPOAgent, PPOConfig

logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    exchanges: Dict[str, Dict[str, Any]]
    trading_pairs: List[str]
    strategy_config: Dict[str, Any]
    risk_config: Dict[str, Any]
    rl_config: Dict[str, Any]


class TradingStrategy:
    def __init__(self, config: TradingConfig):
        self.config = config
        
        self.connectors: Dict[str, BaseExchangeConnector] = {}
        self.market_data_manager = MarketDataManager(config.strategy_config)
        self.volatility_forecaster = VolatilityForecaster(config.strategy_config)
        self.as_engine = AvellanedaStoikovEngine(config.strategy_config)
        self.order_manager = OrderManager(config.strategy_config)
        self.risk_manager = RiskManager(config.strategy_config)
        
        from models.rl_environment import RLEnvironment
        self.rl_environment = RLEnvironment(config.strategy_config)
        
        ppo_config = PPOConfig(
            learning_rate=config.rl_config.get("learning_rate", 0.0003),
            batch_size=config.rl_config.get("batch_size", 2048),
            n_epochs=config.rl_config.get("n_epochs", 10),
            clip_range=config.rl_config.get("clip_range", 0.2),
            entropy_coef=config.rl_config.get("entropy_coef", 0.01),
            value_function_coef=config.rl_config.get("value_function_coef", 0.5)
        )
        
        self.ppo_agent = PPOAgent(
            observation_dim=self.rl_environment.get_observation_space_size(),
            action_dim=self.rl_environment.get_action_space_size(),
            config=ppo_config
        )
        
        self.trading_pairs = config.trading_pairs
        self.running = False
        self.training_mode = True
        
        self.performance_metrics = {
            "total_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        }
        
    async def initialize(self) -> bool:
        try:
            await self._setup_connectors()
            await self._setup_callbacks()
            
            await self.market_data_manager.start()
            await self.order_manager.start()
            
            logger.info("Trading strategy initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize trading strategy: {e}")
            return False
            
    async def _setup_connectors(self) -> None:
        for exchange_name, exchange_config in self.config.exchanges.items():
            if exchange_name == "binance":
                connector = BinanceConnector(exchange_config)
            elif exchange_name == "coinbase":
                connector = CoinbaseConnector(exchange_config)
            elif exchange_name == "okx":
                connector = OKXConnector(exchange_config)
            else:
                logger.warning(f"Unknown exchange: {exchange_name}")
                continue
                
            success = await connector.connect()
            if success:
                self.connectors[exchange_name] = connector
                self.order_manager.add_connector(exchange_name, connector)
                logger.info(f"Connected to {exchange_name}")
            else:
                logger.error(f"Failed to connect to {exchange_name}")
                
    async def _setup_callbacks(self) -> None:
        for connector in self.connectors.values():
            connector.add_callback("orderbook", self.market_data_manager.handle_orderbook_update)
            connector.add_callback("trade", self.market_data_manager.handle_trade_update)
            
        self.market_data_manager.add_callback("snapshot_update", self._handle_market_snapshot)
        self.order_manager.add_callback("order_filled", self._handle_order_fill)
        self.risk_manager.add_callback("risk_violation", self._handle_risk_violation)
        
    async def start_trading(self) -> None:
        if not self.connectors:
            logger.error("No exchange connections available")
            return
            
        self.running = True
        
        for exchange_name in self.connectors.keys():
            for symbol in self.trading_pairs:
                asyncio.create_task(self._subscribe_to_market_data(exchange_name, symbol))
                
        asyncio.create_task(self._trading_loop())
        asyncio.create_task(self._training_loop())
        
        logger.info("Trading started")
        
    async def stop_trading(self) -> None:
        self.running = False
        
        await self.order_manager.stop()
        await self.market_data_manager.stop()
        
        for connector in self.connectors.values():
            await connector.disconnect()
            
        logger.info("Trading stopped")
        
    async def _subscribe_to_market_data(self, exchange_name: str, symbol: str) -> None:
        connector = self.connectors[exchange_name]
        
        await connector.subscribe_orderbook(symbol, self.market_data_manager.handle_orderbook_update)
        await connector.subscribe_trades(symbol, self.market_data_manager.handle_trade_update)
        
    async def _trading_loop(self) -> None:
        while self.running:
            try:
                await asyncio.sleep(0.05)
                
                for symbol in self.trading_pairs:
                    await self._process_symbol(symbol)
                    
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(1.0)
                
    async def _process_symbol(self, symbol: str) -> None:
        snapshot = self.market_data_manager.get_snapshot(symbol)
        if not snapshot or not snapshot.orderbook:
            return
            
        risk_metrics = self.risk_manager.check_risk_limits(symbol, snapshot)
        
        if self.risk_manager.should_trigger_circuit_breaker(symbol, snapshot):
            logger.warning(f"Circuit breaker triggered for {symbol}")
            await self.order_manager.cancel_existing_orders("binance", symbol)
            return
            
        if risk_metrics.risk_level == RiskLevel.CRITICAL:
            logger.error(f"Critical risk level for {symbol}, stopping trading")
            return
            
        volatility_forecast = self.volatility_forecaster.forecast_volatility(symbol)
        
        if self.training_mode:
            await self._rl_trading_step(symbol, snapshot, volatility_forecast)
        else:
            await self._traditional_trading_step(symbol, snapshot, volatility_forecast)
            
    async def _rl_trading_step(self, symbol: str, snapshot: MarketSnapshot, 
                             volatility_forecast: Optional[Any]) -> None:
        observation = self.rl_environment._get_observation(snapshot)
        action, log_prob, value = self.ppo_agent.get_action(observation)
        
        next_obs, reward, done, info = self.rl_environment.step(action, snapshot)
        
        self.ppo_agent.store_transition(observation, action, log_prob, reward, value, done)
        
        if "quotes" in info and info["quotes"]:
            quotes = info["quotes"]
            
            valid_quotes, violations = self.risk_manager.validate_quotes(quotes, snapshot)
            if valid_quotes:
                await self.order_manager.place_quote_orders("binance", quotes)
            else:
                logger.warning(f"Invalid quotes for {symbol}: {violations}")
                
        if done:
            self.rl_environment.reset(snapshot)
            
    async def _traditional_trading_step(self, symbol: str, snapshot: MarketSnapshot, 
                                      volatility_forecast: Optional[Any]) -> None:
        quotes = self.as_engine.calculate_optimal_quotes(snapshot, volatility_forecast)
        
        if not quotes:
            return
            
        valid_quotes, violations = self.risk_manager.validate_quotes(quotes, snapshot)
        if not valid_quotes:
            logger.warning(f"Invalid quotes for {symbol}: {violations}")
            return
            
        await self.order_manager.place_quote_orders("binance", quotes)
        
    async def _training_loop(self) -> None:
        while self.running and self.training_mode:
            try:
                await asyncio.sleep(10.0)
                
                if len(self.ppo_agent.observations) >= self.ppo_agent.config.batch_size:
                    update_info = self.ppo_agent.update()
                    
                    if update_info:
                        logger.info(f"PPO update: {update_info}")
                        
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(5.0)
                
    async def _handle_market_snapshot(self, snapshot: MarketSnapshot) -> None:
        self.volatility_forecaster.add_price_data(
            snapshot.symbol,
            snapshot.timestamp,
            snapshot.orderbook.bids[0].price if snapshot.orderbook and snapshot.orderbook.bids else 0.0,
            sum(trade.quantity for trade in snapshot.recent_trades[-10:]) if snapshot.recent_trades else 0.0
        )
        
    async def _handle_order_fill(self, managed_order: Any) -> None:
        self.performance_metrics["total_trades"] += 1
        
        if managed_order.fill_events:
            fill_event = managed_order.fill_events[-1]
            pnl = self._calculate_fill_pnl(managed_order, fill_event)
            
            self.performance_metrics["total_pnl"] += pnl
            
            if pnl > 0:
                self.performance_metrics["winning_trades"] += 1
            else:
                self.performance_metrics["losing_trades"] += 1
                
            self.risk_manager.update_pnl(managed_order.order.symbol, pnl)
            
        logger.info(f"Order filled: {managed_order.order.symbol} {managed_order.order.side.value} "
                   f"{managed_order.order.filled_quantity} @ {fill_event.get('price', 0)}")
                   
    def _calculate_fill_pnl(self, managed_order: Any, fill_event: Dict[str, Any]) -> float:
        return 0.0
        
    async def _handle_risk_violation(self, risk_metrics: Any) -> None:
        logger.warning(f"Risk violation for {risk_metrics.symbol}: {risk_metrics.violations}")
        
        if risk_metrics.risk_level == RiskLevel.HIGH:
            await self.order_manager.cancel_existing_orders("binance", risk_metrics.symbol)
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        metrics = self.performance_metrics.copy()
        
        if metrics["total_trades"] > 0:
            metrics["win_rate"] = metrics["winning_trades"] / metrics["total_trades"]
        else:
            metrics["win_rate"] = 0.0
            
        return metrics
        
    def set_training_mode(self, training: bool) -> None:
        self.training_mode = training
        self.ppo_agent.set_training_mode(training)
        
    async def save_model(self, filepath: str) -> None:
        self.ppo_agent.save(filepath)
        logger.info(f"Model saved to {filepath}")
        
    async def load_model(self, filepath: str) -> None:
        self.ppo_agent.load(filepath)
        logger.info(f"Model loaded from {filepath}")
