import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from strategy.main import TradingStrategy, TradingConfig
from connectors.base import OrderBook, OrderBookLevel, Trade, OrderSide
from data.market_data import MarketSnapshot


@pytest.fixture
def integration_config():
    return TradingConfig(
        exchanges={
            "binance": {
                "name": "Binance",
                "api_key": "test_key",
                "secret_key": "test_secret",
                "websocket_url": "wss://test.binance.com"
            }
        },
        trading_pairs=["BTC/USDT"],
        strategy_config={
            "avellaneda_stoikov": {
                "risk_aversion": 0.1,
                "order_arrival_intensity": 1.5,
                "time_horizon": 86400
            },
            "order_management": {
                "default_order_size": 1000,
                "min_spread_bps": 1,
                "max_spread_bps": 100
            },
            "risk_management": {
                "max_inventory_ratio": 0.1,
                "max_position_size": 1000000,
                "volatility_threshold": 0.05,
                "drawdown_limit": 0.02
            },
            "reinforcement_learning": {
                "learning_rate": 0.0003,
                "batch_size": 64,
                "n_epochs": 5
            }
        },
        risk_config={
            "max_inventory_ratio": 0.1,
            "volatility_threshold": 0.05
        },
        rl_config={
            "learning_rate": 0.0003,
            "batch_size": 64
        }
    )


@pytest.fixture
def mock_orderbook():
    bids = [
        OrderBookLevel(50000.0, 0.5),
        OrderBookLevel(49999.0, 0.3),
        OrderBookLevel(49998.0, 0.2)
    ]
    asks = [
        OrderBookLevel(50001.0, 0.4),
        OrderBookLevel(50002.0, 0.6),
        OrderBookLevel(50003.0, 0.3)
    ]
    
    return OrderBook(
        symbol="BTC/USDT",
        bids=bids,
        asks=asks,
        timestamp=int(time.time() * 1000)
    )


@pytest.fixture
def mock_trades():
    current_time = int(time.time() * 1000)
    return [
        Trade("BTC/USDT", 50000.0, 0.1, OrderSide.BUY, current_time - 1000, "1"),
        Trade("BTC/USDT", 50001.0, 0.2, OrderSide.SELL, current_time - 500, "2")
    ]


class TestTradingStrategyIntegration:
    @pytest.mark.asyncio
    async def test_strategy_initialization(self, integration_config):
        strategy = TradingStrategy(integration_config)
        
        assert strategy.config == integration_config
        assert strategy.trading_pairs == ["BTC/USDT"]
        assert strategy.running is False
        
    @pytest.mark.asyncio
    async def test_strategy_components_integration(self, integration_config):
        strategy = TradingStrategy(integration_config)
        
        assert strategy.market_data_manager is not None
        assert strategy.volatility_forecaster is not None
        assert strategy.as_engine is not None
        assert strategy.order_manager is not None
        assert strategy.risk_manager is not None
        assert strategy.rl_environment is not None
        assert strategy.ppo_agent is not None
        
    @pytest.mark.asyncio
    async def test_market_data_flow(self, integration_config, mock_orderbook, mock_trades):
        strategy = TradingStrategy(integration_config)
        
        await strategy.market_data_manager.start()
        
        await strategy.market_data_manager.handle_orderbook_update(mock_orderbook)
        
        for trade in mock_trades:
            await strategy.market_data_manager.handle_trade_update(trade)
            
        await asyncio.sleep(0.1)
        
        snapshot = strategy.market_data_manager.get_snapshot("BTC/USDT")
        assert snapshot is not None
        assert snapshot.orderbook is not None
        assert len(snapshot.recent_trades) > 0
        
        await strategy.market_data_manager.stop()
        
    @pytest.mark.asyncio
    async def test_volatility_forecasting_integration(self, integration_config):
        strategy = TradingStrategy(integration_config)
        
        current_time = int(time.time() * 1000)
        for i in range(120):
            timestamp = current_time - (119 - i) * 3600000
            price = 50000.0 * (1 + np.random.normal(0, 0.02))
            strategy.volatility_forecaster.add_price_data("BTC/USDT", timestamp, price, 1000.0)
            
        success = strategy.volatility_forecaster.train_models("BTC/USDT")
        assert success is True
        
        forecast = strategy.volatility_forecaster.forecast_volatility("BTC/USDT")
        assert forecast is not None
        assert forecast.combined_forecast >= 0
        
    @pytest.mark.asyncio
    async def test_avellaneda_stoikov_integration(self, integration_config, mock_orderbook):
        strategy = TradingStrategy(integration_config)
        
        snapshot = MarketSnapshot(
            symbol="BTC/USDT",
            timestamp=int(time.time() * 1000),
            orderbook=mock_orderbook,
            volatility=0.02
        )
        
        quotes = strategy.as_engine.calculate_optimal_quotes(snapshot)
        
        assert quotes is not None
        assert quotes.symbol == "BTC/USDT"
        assert quotes.bid_price < quotes.ask_price
        assert quotes.bid_size > 0
        assert quotes.ask_size > 0
        
    @pytest.mark.asyncio
    async def test_risk_management_integration(self, integration_config, mock_orderbook):
        strategy = TradingStrategy(integration_config)
        
        snapshot = MarketSnapshot(
            symbol="BTC/USDT",
            timestamp=int(time.time() * 1000),
            orderbook=mock_orderbook,
            volatility=0.02
        )
        
        risk_metrics = strategy.risk_manager.check_risk_limits("BTC/USDT", snapshot)
        
        assert risk_metrics is not None
        assert risk_metrics.symbol == "BTC/USDT"
        assert len(risk_metrics.violations) == 0
        
    @pytest.mark.asyncio
    async def test_rl_environment_integration(self, integration_config, mock_orderbook):
        strategy = TradingStrategy(integration_config)
        
        snapshot = MarketSnapshot(
            symbol="BTC/USDT",
            timestamp=int(time.time() * 1000),
            orderbook=mock_orderbook,
            volatility=0.02
        )
        
        observation = strategy.rl_environment.reset(snapshot)
        assert observation is not None
        assert len(observation) == strategy.rl_environment.get_observation_space_size()
        
        action = np.random.uniform(-1, 1, strategy.rl_environment.get_action_space_size())
        next_obs, reward, done, info = strategy.rl_environment.step(action, snapshot)
        
        assert len(next_obs) == len(observation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
    @pytest.mark.asyncio
    async def test_ppo_agent_integration(self, integration_config):
        strategy = TradingStrategy(integration_config)
        
        observation = np.random.uniform(-1, 1, strategy.rl_environment.get_observation_space_size())
        
        action, log_prob, value = strategy.ppo_agent.get_action(observation)
        
        assert len(action) == strategy.rl_environment.get_action_space_size()
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        
        strategy.ppo_agent.store_transition(observation, action, log_prob, 1.0, value, False)
        
        assert len(strategy.ppo_agent.observations) == 1
        assert len(strategy.ppo_agent.actions) == 1
        
    @pytest.mark.asyncio
    async def test_end_to_end_trading_step(self, integration_config, mock_orderbook):
        strategy = TradingStrategy(integration_config)
        strategy.set_training_mode(False)
        
        snapshot = MarketSnapshot(
            symbol="BTC/USDT",
            timestamp=int(time.time() * 1000),
            orderbook=mock_orderbook,
            volatility=0.02
        )
        
        strategy.market_data_manager.snapshots["BTC/USDT"] = snapshot
        
        with patch.object(strategy.order_manager, 'place_quote_orders', new_callable=AsyncMock) as mock_place:
            mock_place.return_value = {"bid": "order1", "ask": "order2"}
            
            await strategy._process_symbol("BTC/USDT")
            
            mock_place.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, integration_config):
        strategy = TradingStrategy(integration_config)
        
        initial_metrics = strategy.get_performance_metrics()
        assert initial_metrics["total_pnl"] == 0.0
        assert initial_metrics["total_trades"] == 0
        assert initial_metrics["win_rate"] == 0.0
        
        strategy.performance_metrics["total_trades"] = 10
        strategy.performance_metrics["winning_trades"] = 6
        strategy.performance_metrics["total_pnl"] = 1500.0
        
        updated_metrics = strategy.get_performance_metrics()
        assert updated_metrics["total_trades"] == 10
        assert updated_metrics["win_rate"] == 0.6
        assert updated_metrics["total_pnl"] == 1500.0


@pytest.mark.asyncio
async def test_full_system_simulation(integration_config, mock_orderbook, mock_trades):
    strategy = TradingStrategy(integration_config)
    strategy.set_training_mode(True)
    
    await strategy.market_data_manager.start()
    
    await strategy.market_data_manager.handle_orderbook_update(mock_orderbook)
    
    for trade in mock_trades:
        await strategy.market_data_manager.handle_trade_update(trade)
        
    await asyncio.sleep(0.1)
    
    snapshot = strategy.market_data_manager.get_snapshot("BTC/USDT")
    assert snapshot is not None
    
    observation = strategy.rl_environment.reset(snapshot)
    
    for _ in range(5):
        action, log_prob, value = strategy.ppo_agent.get_action(observation)
        next_obs, reward, done, info = strategy.rl_environment.step(action, snapshot)
        
        strategy.ppo_agent.store_transition(observation, action, log_prob, reward, value, done)
        
        observation = next_obs
        
        if done:
            observation = strategy.rl_environment.reset(snapshot)
            
    assert len(strategy.ppo_agent.observations) > 0
    assert len(strategy.ppo_agent.actions) > 0
    
    await strategy.market_data_manager.stop()


@pytest.mark.asyncio
async def test_error_handling_and_recovery(integration_config, mock_orderbook):
    strategy = TradingStrategy(integration_config)
    
    invalid_snapshot = MarketSnapshot(
        symbol="BTC/USDT",
        timestamp=int(time.time() * 1000),
        orderbook=None,
        volatility=0.02
    )
    
    quotes = strategy.as_engine.calculate_optimal_quotes(invalid_snapshot)
    assert quotes is None
    
    risk_metrics = strategy.risk_manager.check_risk_limits("BTC/USDT", invalid_snapshot)
    assert risk_metrics is not None
    assert len(risk_metrics.violations) == 0
    
    observation = strategy.rl_environment._get_observation(invalid_snapshot)
    assert observation is not None
    assert len(observation) == strategy.rl_environment.get_observation_space_size()


@pytest.mark.asyncio
async def test_configuration_validation(integration_config):
    strategy = TradingStrategy(integration_config)
    
    assert strategy.config.exchanges is not None
    assert strategy.config.trading_pairs is not None
    assert strategy.config.strategy_config is not None
    assert strategy.config.risk_config is not None
    assert strategy.config.rl_config is not None
    
    assert "BTC/USDT" in strategy.trading_pairs
    assert strategy.config.strategy_config["avellaneda_stoikov"]["risk_aversion"] == 0.1
