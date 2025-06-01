import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from data.market_data import MarketDataManager, MarketSnapshot
from data.orderbook import OrderBookProcessor, MicroPriceCalculator
from connectors.base import OrderBook, OrderBookLevel, Trade, OrderSide


@pytest.fixture
def market_data_config():
    return {
        "micro_price": {
            "queue_weight_factor": 0.5,
            "imbalance_threshold": 0.1
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
        timestamp=int(time.time() * 1000)
    )


@pytest.fixture
def sample_trades():
    current_time = int(time.time() * 1000)
    return [
        Trade("BTC/USDT", 50000.0, 0.1, OrderSide.BUY, current_time - 1000, "1"),
        Trade("BTC/USDT", 50001.0, 0.2, OrderSide.SELL, current_time - 500, "2"),
        Trade("BTC/USDT", 49999.0, 0.15, OrderSide.BUY, current_time - 100, "3")
    ]


class TestOrderBookProcessor:
    def test_initialization(self, market_data_config):
        processor = OrderBookProcessor(market_data_config)
        assert processor.config == market_data_config
        assert processor.ofi_window_1s == 1000
        assert processor.ofi_window_5s == 5000
        
    def test_update_orderbook(self, market_data_config, sample_orderbook):
        processor = OrderBookProcessor(market_data_config)
        processor.update_orderbook(sample_orderbook)
        
        assert "BTC/USDT" in processor.orderbooks
        assert processor.orderbooks["BTC/USDT"] == sample_orderbook
        
    def test_update_trade(self, market_data_config, sample_trades):
        processor = OrderBookProcessor(market_data_config)
        
        for trade in sample_trades:
            processor.update_trade(trade)
            
        assert "BTC/USDT" in processor.trades
        assert len(processor.trades["BTC/USDT"]) == 3
        
    def test_get_level2_data(self, market_data_config, sample_orderbook):
        processor = OrderBookProcessor(market_data_config)
        processor.update_orderbook(sample_orderbook)
        
        level2_data = processor.get_level2_data("BTC/USDT")
        assert level2_data == sample_orderbook
        
    def test_calculate_order_flow_imbalance(self, market_data_config, sample_trades):
        processor = OrderBookProcessor(market_data_config)
        
        for trade in sample_trades:
            processor.update_trade(trade)
            
        current_time = int(time.time() * 1000)
        ofi = processor.calculate_order_flow_imbalance("BTC/USDT", current_time)
        
        assert ofi is not None
        assert ofi.symbol == "BTC/USDT"
        assert isinstance(ofi.ofi_1s, float)
        assert isinstance(ofi.ofi_5s, float)
        
    def test_get_orderbook_imbalance(self, market_data_config, sample_orderbook):
        processor = OrderBookProcessor(market_data_config)
        processor.update_orderbook(sample_orderbook)
        
        imbalance = processor.get_orderbook_imbalance("BTC/USDT")
        assert isinstance(imbalance, float)
        assert -1.0 <= imbalance <= 1.0


class TestMicroPriceCalculator:
    def test_initialization(self, market_data_config):
        calculator = MicroPriceCalculator(market_data_config)
        assert calculator.queue_weight_factor == 0.5
        assert calculator.imbalance_threshold == 0.1
        
    def test_calculate_micro_price(self, market_data_config, sample_orderbook):
        calculator = MicroPriceCalculator(market_data_config)
        
        micro_price = calculator.calculate_micro_price(sample_orderbook)
        
        assert micro_price is not None
        assert micro_price.symbol == "BTC/USDT"
        assert micro_price.micro_price > 0
        assert micro_price.mid_price > 0
        assert micro_price.weighted_mid > 0
        assert -1.0 <= micro_price.imbalance_ratio <= 1.0
        
    def test_calculate_queue_position_adjustment(self, market_data_config, sample_orderbook):
        calculator = MicroPriceCalculator(market_data_config)
        
        bid_adjustment = calculator.calculate_queue_position_adjustment(
            sample_orderbook, 49999.5, OrderSide.BUY
        )
        ask_adjustment = calculator.calculate_queue_position_adjustment(
            sample_orderbook, 50001.5, OrderSide.SELL
        )
        
        assert 0.0 <= bid_adjustment <= 1.0
        assert 0.0 <= ask_adjustment <= 1.0
        
    def test_estimate_fill_probability(self, market_data_config, sample_orderbook):
        calculator = MicroPriceCalculator(market_data_config)
        
        prob_aggressive_bid = calculator.estimate_fill_probability(
            sample_orderbook, 50000.5, OrderSide.BUY
        )
        prob_passive_bid = calculator.estimate_fill_probability(
            sample_orderbook, 49995.0, OrderSide.BUY
        )
        
        assert 0.0 <= prob_aggressive_bid <= 1.0
        assert 0.0 <= prob_passive_bid <= 1.0
        assert prob_aggressive_bid > prob_passive_bid


class TestMarketDataManager:
    @pytest.mark.asyncio
    async def test_initialization(self, market_data_config):
        manager = MarketDataManager(market_data_config)
        assert manager.config == market_data_config
        assert isinstance(manager.orderbook_processor, OrderBookProcessor)
        assert isinstance(manager.micro_price_calculator, MicroPriceCalculator)
        
    @pytest.mark.asyncio
    async def test_start_stop(self, market_data_config):
        manager = MarketDataManager(market_data_config)
        
        await manager.start()
        assert manager.running is True
        
        await manager.stop()
        assert manager.running is False
        
    @pytest.mark.asyncio
    async def test_handle_orderbook_update(self, market_data_config, sample_orderbook):
        manager = MarketDataManager(market_data_config)
        
        callback_called = False
        callback_data = None
        
        def test_callback(data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data
            
        manager.add_callback("snapshot_update", test_callback)
        
        await manager.handle_orderbook_update(sample_orderbook)
        
        await asyncio.sleep(0.1)
        
        assert "BTC/USDT" in manager.snapshots
        
    @pytest.mark.asyncio
    async def test_handle_trade_update(self, market_data_config, sample_trades):
        manager = MarketDataManager(market_data_config)
        
        for trade in sample_trades:
            await manager.handle_trade_update(trade)
            
        assert "BTC/USDT" in manager.trade_history
        assert len(manager.trade_history["BTC/USDT"]) == 3
        
    def test_get_snapshot(self, market_data_config):
        manager = MarketDataManager(market_data_config)
        
        snapshot = MarketSnapshot(
            symbol="BTC/USDT",
            timestamp=int(time.time() * 1000)
        )
        manager.snapshots["BTC/USDT"] = snapshot
        
        retrieved_snapshot = manager.get_snapshot("BTC/USDT")
        assert retrieved_snapshot == snapshot
        
    def test_calculate_vwap(self, market_data_config, sample_trades):
        manager = MarketDataManager(market_data_config)
        
        for trade in sample_trades:
            manager.trade_history["BTC/USDT"].append(trade)
            
        vwap = manager.calculate_vwap("BTC/USDT")
        
        assert vwap > 0
        assert isinstance(vwap, float)
        
    def test_calculate_twap(self, market_data_config):
        manager = MarketDataManager(market_data_config)
        
        current_time = int(time.time() * 1000)
        price_data = [
            (current_time - 3000, 50000.0),
            (current_time - 2000, 50100.0),
            (current_time - 1000, 49900.0)
        ]
        
        for timestamp, price in price_data:
            manager.price_history["BTC/USDT"].append((timestamp, price))
            
        twap = manager.calculate_twap("BTC/USDT")
        
        assert twap > 0
        assert isinstance(twap, float)
        
    def test_get_market_impact_estimate(self, market_data_config, sample_orderbook):
        manager = MarketDataManager(market_data_config)
        
        snapshot = MarketSnapshot(
            symbol="BTC/USDT",
            timestamp=int(time.time() * 1000),
            orderbook=sample_orderbook
        )
        manager.snapshots["BTC/USDT"] = snapshot
        
        impact_buy = manager.get_market_impact_estimate("BTC/USDT", OrderSide.BUY, 0.5)
        impact_sell = manager.get_market_impact_estimate("BTC/USDT", OrderSide.SELL, 0.5)
        
        assert impact_buy >= 0
        assert impact_sell >= 0
        assert isinstance(impact_buy, float)
        assert isinstance(impact_sell, float)


@pytest.mark.asyncio
async def test_market_data_integration(market_data_config, sample_orderbook, sample_trades):
    manager = MarketDataManager(market_data_config)
    await manager.start()
    
    await manager.handle_orderbook_update(sample_orderbook)
    
    for trade in sample_trades:
        await manager.handle_trade_update(trade)
        
    await asyncio.sleep(0.1)
    
    snapshot = manager.get_snapshot("BTC/USDT")
    assert snapshot is not None
    assert snapshot.orderbook is not None
    assert len(snapshot.recent_trades) > 0
    
    await manager.stop()
