import pytest
import time
from unittest.mock import Mock, AsyncMock
import numpy as np

from risk.risk_manager import RiskManager, RiskLimits, RiskMetrics, RiskLevel
from connectors.base import Position, Balance, OrderSide
from data.market_data import MarketSnapshot
from data.orderbook import OrderBook, OrderBookLevel
from strategy.avellaneda_stoikov import QuoteParameters


@pytest.fixture
def risk_config():
    return {
        "risk_management": {
            "max_inventory_ratio": 0.1,
            "max_position_size": 1000000,
            "volatility_threshold": 0.05,
            "drawdown_limit": 0.02,
            "max_daily_loss": 10000,
            "max_order_size": 100000,
            "min_spread_bps": 1,
            "max_spread_bps": 100
        },
        "circuit_breakers": {
            "enabled": True,
            "volatility_multiplier": 1.5,
            "inventory_emergency_threshold": 0.15
        }
    }


@pytest.fixture
def sample_position():
    return Position(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        size=0.5,
        entry_price=50000.0,
        mark_price=51000.0,
        unrealized_pnl=500.0,
        realized_pnl=0.0,
        timestamp=int(time.time() * 1000)
    )


@pytest.fixture
def sample_balances():
    return [
        Balance(asset="BTC", free=1.0, locked=0.1, total=1.1),
        Balance(asset="USDT", free=50000.0, locked=5000.0, total=55000.0)
    ]


@pytest.fixture
def sample_snapshot():
    bids = [OrderBookLevel(50000.0, 0.5)]
    asks = [OrderBookLevel(50001.0, 0.4)]
    
    orderbook = OrderBook(
        symbol="BTC/USDT",
        bids=bids,
        asks=asks,
        timestamp=int(time.time() * 1000)
    )
    
    return MarketSnapshot(
        symbol="BTC/USDT",
        timestamp=int(time.time() * 1000),
        orderbook=orderbook,
        volatility=0.02
    )


@pytest.fixture
def sample_quotes():
    return QuoteParameters(
        symbol="BTC/USDT",
        reservation_price=50000.0,
        bid_price=49995.0,
        ask_price=50005.0,
        bid_size=0.02,
        ask_size=0.02,
        spread=10.0,
        timestamp=int(time.time() * 1000)
    )


class TestRiskManager:
    def test_initialization(self, risk_config):
        risk_manager = RiskManager(risk_config)
        
        assert risk_manager.limits.max_inventory_ratio == 0.1
        assert risk_manager.limits.max_position_size == 1000000
        assert risk_manager.limits.volatility_threshold == 0.05
        assert risk_manager.circuit_breakers.get("enabled", True) is True
        
    def test_update_positions(self, risk_config, sample_position):
        risk_manager = RiskManager(risk_config)
        
        risk_manager.update_positions([sample_position])
        
        assert "BTC/USDT" in risk_manager.positions
        assert risk_manager.positions["BTC/USDT"] == sample_position
        
    def test_update_balances(self, risk_config, sample_balances):
        risk_manager = RiskManager(risk_config)
        
        risk_manager.update_balances(sample_balances)
        
        assert "BTC" in risk_manager.balances
        assert "USDT" in risk_manager.balances
        assert risk_manager.balances["BTC"].total == 1.1
        
    def test_update_pnl(self, risk_config):
        risk_manager = RiskManager(risk_config)
        
        risk_manager.update_pnl("BTC/USDT", 100.0)
        risk_manager.update_pnl("BTC/USDT", -50.0)
        
        assert "BTC/USDT" in risk_manager.pnl_history
        assert len(risk_manager.pnl_history["BTC/USDT"]) == 2
        assert "BTC/USDT" in risk_manager.daily_pnl
        
    def test_check_risk_limits_no_violations(self, risk_config, sample_snapshot):
        risk_manager = RiskManager(risk_config)
        
        metrics = risk_manager.check_risk_limits("BTC/USDT", sample_snapshot)
        
        assert isinstance(metrics, RiskMetrics)
        assert metrics.symbol == "BTC/USDT"
        assert metrics.risk_level == RiskLevel.LOW
        assert len(metrics.violations) == 0
        
    def test_check_risk_limits_with_violations(self, risk_config, sample_snapshot, sample_position):
        risk_manager = RiskManager(risk_config)
        
        large_position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            size=100.0,
            entry_price=50000.0,
            mark_price=51000.0,
            unrealized_pnl=100000.0,
            realized_pnl=0.0,
            timestamp=int(time.time() * 1000)
        )
        
        risk_manager.update_positions([large_position])
        risk_manager.balances["USDT"] = Balance("USDT", 100000.0, 0.0, 100000.0)
        
        sample_snapshot.volatility = 0.1
        
        metrics = risk_manager.check_risk_limits("BTC/USDT", sample_snapshot)
        
        assert len(metrics.violations) > 0
        assert metrics.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        
    def test_validate_quotes_success(self, risk_config, sample_quotes, sample_snapshot):
        risk_manager = RiskManager(risk_config)
        
        valid, violations = risk_manager.validate_quotes(sample_quotes, sample_snapshot)
        
        assert valid is True
        assert len(violations) == 0
        
    def test_validate_quotes_spread_too_narrow(self, risk_config, sample_snapshot):
        risk_manager = RiskManager(risk_config)
        
        narrow_quotes = QuoteParameters(
            symbol="BTC/USDT",
            reservation_price=50000.0,
            bid_price=49999.9,
            ask_price=50000.1,
            bid_size=0.02,
            ask_size=0.02,
            spread=0.2,
            timestamp=int(time.time() * 1000)
        )
        
        valid, violations = risk_manager.validate_quotes(narrow_quotes, sample_snapshot)
        
        assert valid is False
        assert len(violations) > 0
        assert any("below minimum" in violation for violation in violations)
        
    def test_validate_quotes_crossed_market(self, risk_config, sample_snapshot):
        risk_manager = RiskManager(risk_config)
        
        crossed_quotes = QuoteParameters(
            symbol="BTC/USDT",
            reservation_price=50000.0,
            bid_price=50005.0,
            ask_price=49995.0,
            bid_size=0.02,
            ask_size=0.02,
            spread=-10.0,
            timestamp=int(time.time() * 1000)
        )
        
        valid, violations = risk_manager.validate_quotes(crossed_quotes, sample_snapshot)
        
        assert valid is False
        assert len(violations) > 0
        assert any("crossed quotes" in violation for violation in violations)
        
    def test_should_trigger_circuit_breaker_disabled(self, risk_config, sample_snapshot):
        risk_config["circuit_breakers"]["enabled"] = False
        risk_manager = RiskManager(risk_config)
        
        sample_snapshot.volatility = 0.2
        
        should_trigger = risk_manager.should_trigger_circuit_breaker("BTC/USDT", sample_snapshot)
        
        assert should_trigger is False
        
    def test_should_trigger_circuit_breaker_high_volatility(self, risk_config, sample_snapshot):
        risk_manager = RiskManager(risk_config)
        
        sample_snapshot.volatility = 0.1
        
        should_trigger = risk_manager.should_trigger_circuit_breaker("BTC/USDT", sample_snapshot)
        
        assert should_trigger is True
        
    def test_get_position_adjustment_no_position(self, risk_config):
        risk_manager = RiskManager(risk_config)
        
        adjustment = risk_manager.get_position_adjustment("BTC/USDT", OrderSide.BUY)
        
        assert adjustment == 1.0
        
    def test_get_position_adjustment_with_position(self, risk_config, sample_position):
        risk_manager = RiskManager(risk_config)
        
        risk_manager.update_positions([sample_position])
        risk_manager.balances["USDT"] = Balance("USDT", 100000.0, 0.0, 100000.0)
        
        adjustment_same_side = risk_manager.get_position_adjustment("BTC/USDT", OrderSide.BUY)
        adjustment_opposite_side = risk_manager.get_position_adjustment("BTC/USDT", OrderSide.SELL)
        
        assert adjustment_same_side <= 1.0
        assert adjustment_opposite_side >= 1.0
        
    def test_get_portfolio_value(self, risk_config, sample_balances):
        risk_manager = RiskManager(risk_config)
        
        risk_manager.update_balances(sample_balances)
        
        portfolio_value = risk_manager.get_portfolio_value()
        
        assert portfolio_value > 0
        assert isinstance(portfolio_value, float)
        
    def test_calculate_drawdown_no_peak(self, risk_config):
        risk_manager = RiskManager(risk_config)
        
        drawdown = risk_manager.calculate_drawdown()
        
        assert drawdown == 0.0
        
    def test_calculate_drawdown_with_loss(self, risk_config, sample_balances):
        risk_manager = RiskManager(risk_config)
        
        risk_manager.update_balances(sample_balances)
        initial_value = risk_manager.get_portfolio_value()
        risk_manager.peak_portfolio_value = initial_value * 1.2
        
        drawdown = risk_manager.calculate_drawdown()
        
        assert drawdown > 0
        assert drawdown < 1.0
        
    def test_get_var_estimate_no_data(self, risk_config):
        risk_manager = RiskManager(risk_config)
        
        var = risk_manager.get_var_estimate("BTC/USDT")
        
        assert var == 0.0
        
    def test_get_var_estimate_with_data(self, risk_config):
        risk_manager = RiskManager(risk_config)
        
        current_time = int(time.time() * 1000)
        pnl_data = [100, -50, 200, -30, 150, -80, 75, -25, 120, -60]
        
        risk_manager.pnl_history["BTC/USDT"] = []
        for i, pnl in enumerate(pnl_data):
            timestamp = current_time - (len(pnl_data) - i) * 3600000
            risk_manager.pnl_history["BTC/USDT"].append((timestamp, pnl))
            
        var = risk_manager.get_var_estimate("BTC/USDT")
        
        assert isinstance(var, float)
        
    def test_get_sharpe_ratio_no_data(self, risk_config):
        risk_manager = RiskManager(risk_config)
        
        sharpe = risk_manager.get_sharpe_ratio("BTC/USDT")
        
        assert sharpe == 0.0
        
    def test_get_sharpe_ratio_with_data(self, risk_config):
        risk_manager = RiskManager(risk_config)
        
        current_time = int(time.time() * 1000)
        pnl_data = [100, 50, 200, 30, 150, 80, 75, 25, 120, 60]
        
        risk_manager.pnl_history["BTC/USDT"] = []
        for i, pnl in enumerate(pnl_data):
            timestamp = current_time - (len(pnl_data) - i) * 3600000
            risk_manager.pnl_history["BTC/USDT"].append((timestamp, pnl))
            
        sharpe = risk_manager.get_sharpe_ratio("BTC/USDT")
        
        assert isinstance(sharpe, float)
        assert sharpe > 0


class TestRiskLimits:
    def test_risk_limits_creation(self):
        limits = RiskLimits(
            max_inventory_ratio=0.1,
            max_position_size=1000000,
            volatility_threshold=0.05,
            drawdown_limit=0.02,
            max_daily_loss=10000,
            max_order_size=100000,
            min_spread_bps=1,
            max_spread_bps=100
        )
        
        assert limits.max_inventory_ratio == 0.1
        assert limits.max_position_size == 1000000
        assert limits.volatility_threshold == 0.05


class TestRiskMetrics:
    def test_risk_metrics_creation(self):
        metrics = RiskMetrics(
            symbol="BTC/USDT",
            timestamp=int(time.time() * 1000),
            inventory_ratio=0.05,
            position_size=500000,
            unrealized_pnl=1000.0,
            daily_pnl=500.0,
            volatility=0.02,
            drawdown=0.01,
            risk_level=RiskLevel.LOW,
            violations=[]
        )
        
        assert metrics.symbol == "BTC/USDT"
        assert metrics.inventory_ratio == 0.05
        assert metrics.risk_level == RiskLevel.LOW
        assert len(metrics.violations) == 0


@pytest.mark.parametrize("risk_level", [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL])
def test_risk_levels(risk_level):
    assert isinstance(risk_level, RiskLevel)
    assert risk_level.value in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@pytest.mark.asyncio
async def test_risk_manager_callbacks(risk_config):
    risk_manager = RiskManager(risk_config)
    
    callback_called = False
    callback_data = None
    
    def test_callback(data):
        nonlocal callback_called, callback_data
        callback_called = True
        callback_data = data
        
    risk_manager.add_callback("risk_violation", test_callback)
    
    await risk_manager._emit_event("risk_violation", {"test": "data"})
    
    assert callback_called
    assert callback_data == {"test": "data"}
