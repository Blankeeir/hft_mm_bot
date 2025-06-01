import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from connectors.coinbase_fix import CoinbaseFIXConnector
from connectors.base import Order, OrderSide, OrderType, OrderStatus


@pytest.fixture
def fix_config():
    return {
        "name": "Coinbase International",
        "fix_endpoint": "tcp+tls://fix.international.coinbase.com:4198",
        "api_key": "test_api_key",
        "secret_key": "dGVzdF9zZWNyZXRfa2V5",
        "passphrase": "test_passphrase",
        "sender_comp_id": "TEST_SENDER",
        "fix_session_type": "order_entry",
        "test_mode": True
    }


@pytest.fixture
def fix_connector(fix_config):
    return CoinbaseFIXConnector(fix_config)


class TestCoinbaseFIXConnector:
    def test_init(self, fix_connector, fix_config):
        assert fix_connector.fix_endpoint == fix_config["fix_endpoint"]
        assert fix_connector.api_key == fix_config["api_key"]
        assert fix_connector.secret_key == fix_config["secret_key"]
        assert fix_connector.sender_comp_id == fix_config["sender_comp_id"]
        assert fix_connector.session_type == fix_config["fix_session_type"]
        assert fix_connector.test_mode == fix_config["test_mode"]
        assert not fix_connector.authenticated
        assert not fix_connector.is_connected

    @pytest.mark.asyncio
    async def test_connect_without_asyncfix(self, fix_connector):
        with patch('connectors.coinbase_fix.FIXMessage', None):
            result = await fix_connector.connect()
            assert not result
            assert not fix_connector.is_connected

    @pytest.mark.asyncio
    async def test_disconnect(self, fix_connector):
        fix_connector.connection = Mock()
        fix_connector.connection.disconnect = AsyncMock()
        fix_connector.is_connected = True
        fix_connector.authenticated = True
        
        await fix_connector.disconnect()
        
        assert not fix_connector.is_connected
        assert not fix_connector.authenticated
        fix_connector.connection.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_order_test_mode(self, fix_connector):
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0
        )
        
        fix_connector.authenticated = True
        fix_connector.session_type = "order_entry"
        
        result = await fix_connector.place_order(order)
        
        assert result.startswith("TEST-")

    @pytest.mark.asyncio
    async def test_place_order_not_authenticated(self, fix_connector):
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0
        )
        
        fix_connector.authenticated = False
        
        result = await fix_connector.place_order(order)
        
        assert result == ""

    @pytest.mark.asyncio
    async def test_cancel_order_test_mode(self, fix_connector):
        fix_connector.authenticated = True
        fix_connector.session_type = "order_entry"
        
        result = await fix_connector.cancel_order("BTC/USD", "test_order_id")
        
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_order_not_authenticated(self, fix_connector):
        fix_connector.authenticated = False
        
        result = await fix_connector.cancel_order("BTC/USD", "test_order_id")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_get_order_status(self, fix_connector):
        result = await fix_connector.get_order_status("BTC/USD", "test_order_id")
        
        assert isinstance(result, Order)
        assert result.symbol == "BTC/USD"
        assert result.order_id == "test_order_id"
        assert result.status == OrderStatus.NEW

    @pytest.mark.asyncio
    async def test_get_open_orders(self, fix_connector):
        result = await fix_connector.get_open_orders()
        
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_positions(self, fix_connector):
        result = await fix_connector.get_positions()
        
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_balances(self, fix_connector):
        result = await fix_connector.get_balances()
        
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_orderbook(self, fix_connector):
        result = await fix_connector.get_orderbook("BTC/USD")
        
        assert result.symbol == "BTC/USD"
        assert len(result.bids) > 0
        assert len(result.asks) > 0

    @pytest.mark.asyncio
    async def test_subscribe_orderbook_not_authenticated(self, fix_connector):
        callback = Mock()
        fix_connector.authenticated = False
        
        await fix_connector.subscribe_orderbook("BTC/USD", callback)
        
        assert len(fix_connector.callbacks.get('orderbook', [])) == 0

    @pytest.mark.asyncio
    async def test_subscribe_orderbook_test_mode(self, fix_connector):
        callback = Mock()
        fix_connector.authenticated = True
        fix_connector.session_type = "market_data"
        
        await fix_connector.subscribe_orderbook("BTC/USD", callback)
        
        assert callback in fix_connector.callbacks.get('orderbook', [])

    @pytest.mark.asyncio
    async def test_subscribe_trades(self, fix_connector):
        callback = Mock()
        
        await fix_connector.subscribe_trades("BTC/USD", callback)
        
        assert callback in fix_connector.callbacks.get('trade', [])

    @pytest.mark.asyncio
    async def test_subscribe_order_updates(self, fix_connector):
        callback = Mock()
        
        await fix_connector.subscribe_order_updates(callback)
        
        assert callback in fix_connector.callbacks.get('order_update', [])

    def test_get_next_request_id(self, fix_connector):
        initial_id = fix_connector.next_request_id
        
        result = fix_connector._get_next_request_id()
        
        assert result == initial_id + 1
        assert fix_connector.next_request_id == initial_id + 1

    def test_get_utc_timestamp(self, fix_connector):
        result = fix_connector._get_utc_timestamp()
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "-" in result
        assert ":" in result

    def test_map_fix_order_status(self, fix_connector):
        assert fix_connector._map_fix_order_status("0") == OrderStatus.NEW
        assert fix_connector._map_fix_order_status("1") == OrderStatus.PARTIALLY_FILLED
        assert fix_connector._map_fix_order_status("2") == OrderStatus.FILLED
        assert fix_connector._map_fix_order_status("4") == OrderStatus.CANCELED
        assert fix_connector._map_fix_order_status("8") == OrderStatus.REJECTED
        assert fix_connector._map_fix_order_status("C") == OrderStatus.EXPIRED
        assert fix_connector._map_fix_order_status("unknown") == OrderStatus.NEW

    @pytest.mark.asyncio
    async def test_place_tpsl_order_sell(self, fix_connector):
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            order_type=OrderType.TAKE_PROFIT_STOP_LOSS,
            quantity=1.0,
            price=60000.0,
            stop_price=55000.0,
            stop_limit_price=50000.0,
            time_in_force="GTC"
        )
        
        fix_connector.authenticated = True
        fix_connector.session_type = "order_entry"
        
        result = await fix_connector.place_order(order)
        
        assert result.startswith("TEST-")

    @pytest.mark.asyncio
    async def test_place_tpsl_order_buy(self, fix_connector):
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.TAKE_PROFIT_STOP_LOSS,
            quantity=1.0,
            price=50000.0,
            stop_price=55000.0,
            stop_limit_price=60000.0,
            time_in_force="GTC"
        )
        
        fix_connector.authenticated = True
        fix_connector.session_type = "order_entry"
        
        result = await fix_connector.place_order(order)
        
        assert result.startswith("TEST-")

    @pytest.mark.asyncio
    async def test_tpsl_validation_sell_invalid(self, fix_connector):
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            order_type=OrderType.TAKE_PROFIT_STOP_LOSS,
            quantity=1.0,
            price=50000.0,
            stop_price=55000.0,
            stop_limit_price=45000.0
        )
        
        fix_connector.authenticated = True
        fix_connector.session_type = "order_entry"
        fix_connector.test_mode = False
        
        result = await fix_connector.place_order(order)
        assert result == ""

    @pytest.mark.asyncio
    async def test_tpsl_validation_buy_invalid(self, fix_connector):
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.TAKE_PROFIT_STOP_LOSS,
            quantity=1.0,
            price=60000.0,
            stop_price=55000.0,
            stop_limit_price=50000.0
        )
        
        fix_connector.authenticated = True
        fix_connector.session_type = "order_entry"
        fix_connector.test_mode = False
        
        result = await fix_connector.place_order(order)
        assert result == ""

    @pytest.mark.asyncio
    async def test_tpsl_invalid_time_in_force(self, fix_connector):
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            order_type=OrderType.TAKE_PROFIT_STOP_LOSS,
            quantity=1.0,
            price=60000.0,
            stop_price=55000.0,
            stop_limit_price=50000.0,
            time_in_force="IOC"
        )
        
        fix_connector.authenticated = True
        fix_connector.session_type = "order_entry"
        fix_connector.test_mode = False
        
        result = await fix_connector.place_order(order)
        assert result == ""

    @pytest.mark.asyncio
    async def test_modify_order(self, fix_connector):
        fix_connector.authenticated = True
        fix_connector.session_type = "order_entry"
        
        result = await fix_connector.modify_order(
            "original_order_id",
            "BTC/USD",
            quantity=2.0,
            price=55000.0
        )
        
        assert result != ""

    @pytest.mark.asyncio
    async def test_modify_order_not_authenticated(self, fix_connector):
        fix_connector.authenticated = False
        
        result = await fix_connector.modify_order(
            "original_order_id",
            "BTC/USD",
            quantity=2.0
        )
        
        assert result == ""

    @pytest.mark.asyncio
    async def test_mass_cancel_orders(self, fix_connector):
        fix_connector.authenticated = True
        fix_connector.session_type = "order_entry"
        
        result = await fix_connector.mass_cancel_orders(symbol="BTC/USD", side="BUY")
        
        assert result != ""

    @pytest.mark.asyncio
    async def test_mass_cancel_orders_not_authenticated(self, fix_connector):
        fix_connector.authenticated = False
        
        result = await fix_connector.mass_cancel_orders()
        
        assert result == ""

    @pytest.mark.asyncio
    async def test_handle_execution_report_tpsl(self, fix_connector):
        mock_message = Mock()
        mock_message.get.side_effect = lambda tag, default=None: {
            "35": "8",
            "37": "order123",
            "11": "client123",
            "55": "BTC-USD",
            "54": "2",
            "39": "0",
            "150": "0",
            "40": "O",
            "38": "1.0",
            "44": "60000.0",
            "14": "0.0",
            "99": "55000.0",
            "3040": "50000.0"
        }.get(str(tag), default)
        
        with patch('time.time', return_value=1234567890):
            await fix_connector._handle_execution_report(mock_message)

    @pytest.mark.asyncio
    async def test_handle_order_mass_cancel_report(self, fix_connector):
        mock_message = Mock()
        mock_message.get.side_effect = lambda tag, default="": {
            "11": "mass_cancel_123",
            "1369": "report_456",
            "55": "BTC-USD",
            "54": "1",
            "531": "7",
            "533": "5"
        }.get(str(tag), default)
        
        await fix_connector._handle_order_mass_cancel_report(mock_message)

    @pytest.mark.asyncio
    async def test_handle_order_mass_cancel_report_rejected(self, fix_connector):
        mock_message = Mock()
        mock_message.get.side_effect = lambda tag, default="": {
            "11": "mass_cancel_123",
            "531": "0",
            "532": "1"
        }.get(str(tag), default)
        
        await fix_connector._handle_order_mass_cancel_report(mock_message)

    @pytest.mark.asyncio
    async def test_place_order_with_fix_message(self, fix_connector):
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            time_in_force="GTC"
        )
        
        fix_connector.authenticated = True
        fix_connector.session_type = "order_entry"
        fix_connector.test_mode = False
        
        with patch('connectors.coinbase_fix.FIXMessage') as mock_fix_message, \
             patch('connectors.coinbase_fix.FMsg') as mock_fmsg, \
             patch('connectors.coinbase_fix.FTag') as mock_ftag, \
             patch.object(fix_connector, 'connection') as mock_connection:
            
            mock_connection.send_msg = AsyncMock()
            mock_fix_message.return_value = Mock()
            mock_fmsg.NEWORDERSINGLE = "D"
            mock_ftag.ClOrdID = "11"
            mock_ftag.Symbol = "55"
            mock_ftag.Side = "54"
            mock_ftag.TransactTime = "60"
            mock_ftag.OrdType = "40"
            mock_ftag.OrderQty = "38"
            mock_ftag.TimeInForce = "59"
            mock_ftag.Price = "44"
            
            result = await fix_connector.place_order(order)
            
            assert result != ""
            mock_connection.send_msg.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_order_with_fix_message(self, fix_connector):
        fix_connector.authenticated = True
        fix_connector.session_type = "order_entry"
        fix_connector.test_mode = False
        
        with patch('connectors.coinbase_fix.FIXMessage') as mock_fix_message, \
             patch('connectors.coinbase_fix.FMsg') as mock_fmsg, \
             patch('connectors.coinbase_fix.FTag') as mock_ftag, \
             patch.object(fix_connector, 'connection') as mock_connection:
            
            mock_connection.send_msg = AsyncMock()
            mock_fix_message.return_value = Mock()
            mock_fmsg.ORDERCANCELREQUEST = "F"
            mock_ftag.OrigClOrdID = "41"
            mock_ftag.ClOrdID = "11"
            mock_ftag.Symbol = "55"
            mock_ftag.TransactTime = "60"
            
            result = await fix_connector.cancel_order("BTC/USD", "order123")
            
            assert result is True
            mock_connection.send_msg.assert_called_once()

    @pytest.mark.asyncio
    async def test_modify_order_with_fix_message(self, fix_connector):
        fix_connector.authenticated = True
        fix_connector.session_type = "order_entry"
        fix_connector.test_mode = False
        
        with patch('connectors.coinbase_fix.FIXMessage') as mock_fix_message, \
             patch('connectors.coinbase_fix.FMsg') as mock_fmsg, \
             patch('connectors.coinbase_fix.FTag') as mock_ftag, \
             patch.object(fix_connector, 'connection') as mock_connection:
            
            mock_connection.send_msg = AsyncMock()
            mock_fix_message.return_value = Mock()
            mock_fmsg.ORDERCANCELREPLACEREQUEST = "G"
            mock_ftag.OrigClOrdID = "41"
            mock_ftag.ClOrdID = "11"
            mock_ftag.Symbol = "55"
            mock_ftag.TransactTime = "60"
            mock_ftag.OrderQty = "38"
            mock_ftag.Price = "44"
            mock_ftag.StopPx = "99"
            
            result = await fix_connector.modify_order(
                "original_order_id",
                "BTC/USD",
                quantity=2.0,
                price=55000.0
            )
            
            assert result != ""
            mock_connection.send_msg.assert_called_once()

    @pytest.mark.asyncio
    async def test_mass_cancel_orders_with_fix_message(self, fix_connector):
        fix_connector.authenticated = True
        fix_connector.session_type = "order_entry"
        fix_connector.test_mode = False
        
        with patch('connectors.coinbase_fix.FIXMessage') as mock_fix_message, \
             patch('connectors.coinbase_fix.FTag') as mock_ftag, \
             patch.object(fix_connector, 'connection') as mock_connection:
            
            mock_connection.send_msg = AsyncMock()
            mock_fix_message.return_value = Mock()
            mock_ftag.ClOrdID = "11"
            mock_ftag.TransactTime = "60"
            mock_ftag.Side = "54"
            mock_ftag.Symbol = "55"
            
            result = await fix_connector.mass_cancel_orders(symbol="BTC/USD", side="BUY")
            
            assert result != ""
            mock_connection.send_msg.assert_called_once()

    @pytest.mark.asyncio
    async def test_tpsl_post_only_validation(self, fix_connector):
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            order_type=OrderType.TAKE_PROFIT_STOP_LOSS,
            quantity=1.0,
            price=60000.0,
            stop_price=55000.0,
            stop_limit_price=50000.0,
            time_in_force="GTC",
            post_only=True
        )
        
        fix_connector.authenticated = True
        fix_connector.session_type = "order_entry"
        fix_connector.test_mode = False
        
        result = await fix_connector.place_order(order)
        assert result == ""

    @pytest.mark.asyncio
    async def test_order_with_portfolio_id(self, fix_connector):
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            portfolio_id="test-portfolio-uuid"
        )
        
        fix_connector.authenticated = True
        fix_connector.session_type = "order_entry"
        
        result = await fix_connector.place_order(order)
        assert result != ""
