import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import json

from connectors.base import Order, OrderSide, OrderType, OrderStatus
from connectors.binance import BinanceConnector
from connectors.coinbase import CoinbaseConnector
from connectors.okx import OKXConnector


@pytest.fixture
def binance_config():
    return {
        "name": "Binance",
        "base_url": "https://api.binance.com",
        "api_key": "test_api_key",
        "secret_key": "test_secret_key",
        "websocket_url": "wss://stream.binance.com:9443/ws"
    }


@pytest.fixture
def coinbase_config():
    return {
        "name": "Coinbase International",
        "base_url": "https://api.international.coinbase.com",
        "api_key": "test_api_key",
        "secret_key": "dGVzdF9zZWNyZXRfa2V5X2Jhc2U2NA==",
        "passphrase": "test_passphrase",
        "websocket_url": "wss://advanced-trade-ws.international.coinbase.com"
    }


@pytest.fixture
def okx_config():
    return {
        "name": "OKX",
        "base_url": "https://www.okx.com",
        "api_key": "test_api_key",
        "secret_key": "test_secret_key",
        "passphrase": "test_passphrase",
        "websocket_url": "wss://ws.okx.com:8443/ws/v5/public"
    }


@pytest.fixture
def sample_order():
    return Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.001,
        price=50000.0,
        client_order_id="test_order_123"
    )


class TestBinanceConnector:
    @pytest.mark.asyncio
    async def test_initialization(self, binance_config):
        connector = BinanceConnector(binance_config)
        assert connector.name == "Binance"
        assert connector.base_url == "https://api.binance.com"
        assert connector.api_key == "test_api_key"
        
    @pytest.mark.asyncio
    async def test_signature_generation(self, binance_config):
        connector = BinanceConnector(binance_config)
        query_string = "symbol=BTCUSDT&side=BUY&type=LIMIT&quantity=0.001"
        signature = connector._generate_signature(query_string)
        assert isinstance(signature, str)
        assert len(signature) == 64
        
    @pytest.mark.asyncio
    async def test_place_order_success(self, binance_config, sample_order):
        connector = BinanceConnector(binance_config)
        
        mock_response = {"orderId": "12345", "status": "NEW"}
        
        with patch.object(connector, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            order_id = await connector.place_order(sample_order)
            assert order_id == "12345"
            mock_request.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_cancel_order_success(self, binance_config):
        connector = BinanceConnector(binance_config)
        
        mock_response = {"status": "CANCELED"}
        
        with patch.object(connector, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await connector.cancel_order("BTC/USDT", "12345")
            assert result is True
            mock_request.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_get_orderbook(self, binance_config):
        connector = BinanceConnector(binance_config)
        
        mock_response = {
            "bids": [["50000.0", "0.1"], ["49999.0", "0.2"]],
            "asks": [["50001.0", "0.1"], ["50002.0", "0.2"]]
        }
        
        with patch.object(connector, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            orderbook = await connector.get_orderbook("BTC/USDT")
            assert orderbook.symbol == "BTC/USDT"
            assert len(orderbook.bids) == 2
            assert len(orderbook.asks) == 2
            assert orderbook.bids[0].price == 50000.0


class TestCoinbaseConnector:
    @pytest.mark.asyncio
    async def test_initialization(self, coinbase_config):
        connector = CoinbaseConnector(coinbase_config)
        assert connector.name == "Coinbase International"
        assert connector.passphrase == "test_passphrase"
        
    @pytest.mark.asyncio
    async def test_signature_generation(self, coinbase_config):
        connector = CoinbaseConnector(coinbase_config)
        timestamp = "1234567890"
        method = "POST"
        path = "/api/v3/brokerage/orders"
        body = '{"product_id":"BTC-USD"}'
        
        signature = connector._generate_signature(timestamp, method, path, body)
        assert isinstance(signature, str)
        assert len(signature) > 0
        
    @pytest.mark.asyncio
    async def test_place_order_success(self, coinbase_config, sample_order):
        connector = CoinbaseConnector(coinbase_config)
        
        mock_response = {"order_id": "abc123"}
        
        with patch.object(connector, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            order_id = await connector.place_order(sample_order)
            assert order_id == "abc123"


class TestOKXConnector:
    @pytest.mark.asyncio
    async def test_initialization(self, okx_config):
        connector = OKXConnector(okx_config)
        assert connector.name == "OKX"
        assert connector.passphrase == "test_passphrase"
        
    @pytest.mark.asyncio
    async def test_place_order_success(self, okx_config, sample_order):
        connector = OKXConnector(okx_config)
        
        mock_response = {
            "code": "0",
            "data": [{"ordId": "xyz789"}]
        }
        
        with patch.object(connector, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            order_id = await connector.place_order(sample_order)
            assert order_id == "xyz789"
            
    @pytest.mark.asyncio
    async def test_cancel_order_success(self, okx_config):
        connector = OKXConnector(okx_config)
        
        mock_response = {"code": "0"}
        
        with patch.object(connector, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await connector.cancel_order("BTC/USDT", "xyz789")
            assert result is True


@pytest.mark.asyncio
async def test_connector_callbacks():
    config = {
        "name": "Test",
        "websocket_url": "wss://test.com"
    }
    
    connector = BinanceConnector(config)
    
    callback_called = False
    callback_data = None
    
    def test_callback(data):
        nonlocal callback_called, callback_data
        callback_called = True
        callback_data = data
        
    connector.add_callback("test_event", test_callback)
    
    await connector._emit_event("test_event", {"test": "data"})
    
    assert callback_called
    assert callback_data == {"test": "data"}
