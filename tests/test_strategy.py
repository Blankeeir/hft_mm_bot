import pytest
import numpy as np
from unittest.mock import Mock, patch

from strategy.avellaneda_stoikov import AvellanedaStoikovEngine, QuoteParameters
from data.market_data import MarketSnapshot
from data.orderbook import OrderBook, OrderBookLevel
from connectors.base import OrderSide


@pytest.fixture
def strategy_config():
    return {
        "avellaneda_stoikov": {
            "risk_aversion": 0.1,
            "order_arrival_intensity": 1.5,
            "time_horizon": 86400,
            "volatility_lookback": 21
        },
        "order_management": {
            "default_order_size": 1000,
            "min_spread_bps": 1,
            "max_spread_bps": 100,
            "order_refresh_interval": 200
        }
    }


@pytest.fixture
def sample_orderbook():
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
        timestamp=1640995200000
    )


@pytest.fixture
def sample_snapshot(sample_orderbook):
    return MarketSnapshot(
        symbol="BTC/USDT",
        timestamp=1640995200000,
        orderbook=sample_orderbook,
        volatility=0.02
    )


class TestAvellanedaStoikovEngine:
    def test_initialization(self, strategy_config):
        engine = AvellanedaStoikovEngine(strategy_config)
        assert engine.gamma == 0.1
        assert engine.A == 1.5
        assert engine.T == 86400
        
    def test_reservation_price_calculation(self, strategy_config):
        engine = AvellanedaStoikovEngine(strategy_config)
        engine.update_inventory("BTC/USDT", 0.5)
        
        mid_price = 50000.0
        volatility = 0.02
        time_remaining = 3600.0
        
        reservation_price = engine.calculate_reservation_price(
            "BTC/USDT", mid_price, volatility, time_remaining
        )
        
        expected_penalty = 0.5 * 0.1 * (0.02 ** 2) * 3600.0
        expected_price = mid_price - expected_penalty
        
        assert abs(reservation_price - expected_price) < 0.01
        
    def test_optimal_spread_calculation(self, strategy_config):
        engine = AvellanedaStoikovEngine(strategy_config)
        
        volatility = 0.02
        time_remaining = 3600.0
        
        spread = engine.calculate_optimal_spread(volatility, time_remaining)
        
        assert spread > 0
        assert isinstance(spread, float)
        
    def test_optimal_quotes_generation(self, strategy_config, sample_snapshot):
        engine = AvellanedaStoikovEngine(strategy_config)
        engine.update_inventory("BTC/USDT", 0.0)
        
        quotes = engine.calculate_optimal_quotes(sample_snapshot)
        
        assert quotes is not None
        assert isinstance(quotes, QuoteParameters)
        assert quotes.symbol == "BTC/USDT"
        assert quotes.bid_price < quotes.ask_price
        assert quotes.bid_size > 0
        assert quotes.ask_size > 0
        assert quotes.spread > 0
        
    def test_inventory_skew_effect(self, strategy_config, sample_snapshot):
        engine = AvellanedaStoikovEngine(strategy_config)
        
        engine.update_inventory("BTC/USDT", 1.0)
        quotes_long = engine.calculate_optimal_quotes(sample_snapshot)
        
        engine.update_inventory("BTC/USDT", -1.0)
        quotes_short = engine.calculate_optimal_quotes(sample_snapshot)
        
        engine.update_inventory("BTC/USDT", 0.0)
        quotes_neutral = engine.calculate_optimal_quotes(sample_snapshot)
        
        assert quotes_long.reservation_price < quotes_neutral.reservation_price
        assert quotes_short.reservation_price > quotes_neutral.reservation_price
        
    def test_volatility_impact_on_spread(self, strategy_config, sample_snapshot):
        engine = AvellanedaStoikovEngine(strategy_config)
        
        sample_snapshot.volatility = 0.01
        quotes_low_vol = engine.calculate_optimal_quotes(sample_snapshot)
        
        sample_snapshot.volatility = 0.05
        quotes_high_vol = engine.calculate_optimal_quotes(sample_snapshot)
        
        assert quotes_high_vol.spread > quotes_low_vol.spread
        
    def test_rl_adjustments(self, strategy_config, sample_snapshot):
        engine = AvellanedaStoikovEngine(strategy_config)
        
        rl_adjustments = {
            "spread_adjustment": 0.2,
            "size_multiplier": 1.5,
            "reservation_adjustment": 0.1
        }
        
        quotes_base = engine.calculate_optimal_quotes(sample_snapshot)
        quotes_adjusted = engine.calculate_optimal_quotes(sample_snapshot, rl_adjustments=rl_adjustments)
        
        assert quotes_adjusted.spread > quotes_base.spread
        assert quotes_adjusted.bid_size > quotes_base.bid_size
        assert quotes_adjusted.ask_size > quotes_base.ask_size
        
    def test_spread_limits(self, strategy_config, sample_snapshot):
        engine = AvellanedaStoikovEngine(strategy_config)
        
        quotes = engine.calculate_optimal_quotes(sample_snapshot)
        
        mid_price = (sample_snapshot.orderbook.bids[0].price + sample_snapshot.orderbook.asks[0].price) / 2.0
        spread_bps = (quotes.spread / mid_price) * 10000
        
        assert spread_bps >= engine.min_spread_bps
        assert spread_bps <= engine.max_spread_bps
        
    def test_base_order_size_calculation(self, strategy_config):
        engine = AvellanedaStoikovEngine(strategy_config)
        
        price = 50000.0
        volatility = 0.02
        
        base_size = engine.calculate_base_order_size("BTC/USDT", price, volatility)
        
        assert base_size > 0
        assert isinstance(base_size, float)
        
        high_vol_size = engine.calculate_base_order_size("BTC/USDT", price, 0.1)
        assert high_vol_size < base_size
        
    def test_inventory_skew_calculation(self, strategy_config):
        engine = AvellanedaStoikovEngine(strategy_config)
        
        skew_positive = engine.calculate_inventory_skew(1.0, 0.02)
        skew_negative = engine.calculate_inventory_skew(-1.0, 0.02)
        skew_zero = engine.calculate_inventory_skew(0.0, 0.02)
        
        assert skew_positive > 0
        assert skew_negative < 0
        assert skew_zero == 0
        assert abs(skew_positive) <= 0.3
        assert abs(skew_negative) <= 0.3


class TestQuoteValidation:
    def test_quote_parameters_structure(self):
        quotes = QuoteParameters(
            symbol="BTC/USDT",
            reservation_price=50000.0,
            bid_price=49995.0,
            ask_price=50005.0,
            bid_size=0.02,
            ask_size=0.02,
            spread=10.0,
            timestamp=1640995200000
        )
        
        assert quotes.symbol == "BTC/USDT"
        assert quotes.bid_price < quotes.ask_price
        assert quotes.spread == quotes.ask_price - quotes.bid_price
        assert quotes.bid_size > 0
        assert quotes.ask_size > 0


@pytest.mark.parametrize("inventory,expected_direction", [
    (1.0, "negative_skew"),
    (-1.0, "positive_skew"),
    (0.0, "no_skew")
])
def test_inventory_effects(strategy_config, sample_snapshot, inventory, expected_direction):
    engine = AvellanedaStoikovEngine(strategy_config)
    engine.update_inventory("BTC/USDT", inventory)
    
    quotes = engine.calculate_optimal_quotes(sample_snapshot)
    mid_price = (sample_snapshot.orderbook.bids[0].price + sample_snapshot.orderbook.asks[0].price) / 2.0
    
    if expected_direction == "negative_skew":
        assert quotes.reservation_price < mid_price
    elif expected_direction == "positive_skew":
        assert quotes.reservation_price > mid_price
    else:
        assert abs(quotes.reservation_price - mid_price) < 1.0
