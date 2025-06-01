import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from connectors.coinbase_fix_adapter import CoinbaseFIXAdapter
from connectors.base import Order, OrderSide, OrderType, OrderStatus


@pytest.fixture
def fix_config():
    return {
        "name": "Coinbase International",
        "api_key": "test_api_key",
        "secret_key": "dGVzdF9zZWNyZXRfa2V5",
        "passphrase": "test_passphrase",
        "sender_comp_id": "TEST_SENDER",
        "fix_session_type": "order_entry",
        "test_mode": True
    }


@pytest.fixture
def fix_adapter(fix_config):
    return CoinbaseFIXAdapter(fix_config)


class TestCoinbaseFIXAdapter:
    def test_init(self, fix_adapter, fix_config):
        assert fix_adapter.api_key == fix_config["api_key"]
        assert fix_adapter.secret_key == fix_config["secret_key"]
        assert fix_adapter.sender_comp_id == fix_config["sender_comp_id"]
        assert fix_adapter.session_type == fix_config["fix_session_type"]
        assert fix_adapter.test_mode == fix_config["test_mode"]
        assert not fix_adapter.authenticated
        assert not fix_adapter.is_connected

    @pytest.mark.asyncio
    async def test_connect_test_mode(self, fix_adapter):
        result = await fix_adapter.connect()
        
        assert result is True
        assert fix_adapter.is_connected
        assert fix_adapter.authenticated

    @pytest.mark.asyncio
    async def test_disconnect(self, fix_adapter):
        fix_adapter.is_connected = True
        fix_adapter.authenticated = True
        
        await fix_adapter.disconnect()
        
        assert not fix_adapter.is_connected
        assert not fix_adapter.authenticated

    @pytest.mark.asyncio
    async def test_place_order(self, fix_adapter):
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0
        )
        
        fix_adapter.authenticated = True
        
        result = await fix_adapter.place_order(order)
        
        assert result.startswith("TEST-")

    @pytest.mark.asyncio
    async def test_place_order_not_authenticated(self, fix_adapter):
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0
        )
        
        fix_adapter.authenticated = False
        
        result = await fix_adapter.place_order(order)
        
        assert result == ""

    @pytest.mark.asyncio
    async def test_cancel_order(self, fix_adapter):
        fix_adapter.authenticated = True
        
        result = await fix_adapter.cancel_order("BTC/USD", "test_order_id")
        
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_order_not_authenticated(self, fix_adapter):
        fix_adapter.authenticated = False
        
        result = await fix_adapter.cancel_order("BTC/USD", "test_order_id")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_get_order_status(self, fix_adapter):
        result = await fix_adapter.get_order_status("BTC/USD", "test_order_id")
        
        assert isinstance(result, Order)
        assert result.symbol == "BTC/USD"
        assert result.order_id == "test_order_id"
        assert result.status == OrderStatus.NEW

    @pytest.mark.asyncio
    async def test_get_open_orders(self, fix_adapter):
        result = await fix_adapter.get_open_orders()
        
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_positions(self, fix_adapter):
        result = await fix_adapter.get_positions()
        
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_balances(self, fix_adapter):
        result = await fix_adapter.get_balances()
        
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_orderbook(self, fix_adapter):
        result = await fix_adapter.get_orderbook("BTC/USD")
        
        assert result.symbol == "BTC/USD"
        assert len(result.bids) > 0
        assert len(result.asks) > 0

    @pytest.mark.asyncio
    async def test_subscribe_orderbook(self, fix_adapter):
        callback = Mock()
        
        await fix_adapter.subscribe_orderbook("BTC/USD", callback)
        
        assert callback in fix_adapter.callbacks.get('orderbook', [])

    @pytest.mark.asyncio
    async def test_subscribe_trades(self, fix_adapter):
        callback = Mock()
        
        await fix_adapter.subscribe_trades("BTC/USD", callback)
        
        assert callback in fix_adapter.callbacks.get('trade', [])

    @pytest.mark.asyncio
    async def test_subscribe_order_updates(self, fix_adapter):
        callback = Mock()
        
        await fix_adapter.subscribe_order_updates(callback)
        
        assert callback in fix_adapter.callbacks.get('order_update', [])

    def test_get_next_request_id(self, fix_adapter):
        initial_id = fix_adapter.next_request_id
        
        result = fix_adapter._get_next_request_id()
        
        assert result == initial_id + 1
        assert fix_adapter.next_request_id == initial_id + 1

    def test_get_utc_timestamp(self, fix_adapter):
        result = fix_adapter._get_utc_timestamp()
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "-" in result
        assert ":" in result

    @pytest.mark.asyncio
    async def test_add_callback(self, fix_adapter):
        callback = Mock()
        
        fix_adapter.add_callback("test_event", callback)
        
        assert "test_event" in fix_adapter.callbacks
        assert callback in fix_adapter.callbacks["test_event"]

    @pytest.mark.asyncio
    async def test_emit_event(self, fix_adapter):
        callback_called = False
        callback_data = None
        
        def test_callback(data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data
        
        fix_adapter.add_callback("test_event", test_callback)
        
        await fix_adapter._emit_event("test_event", {"test": "data"})
        
        assert callback_called
        assert callback_data == {"test": "data"}
