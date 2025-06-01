import asyncio
import json
import hmac
import hashlib
import base64
import time
from typing import Dict, List, Optional, Any, Callable
import aiohttp
import websockets
import logging

from .base import (
    BaseExchangeConnector, BaseWSConnector,
    Order, Position, Balance, OrderBook, OrderBookLevel, Trade,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)


class CoinbaseConnector(BaseExchangeConnector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://api.international.coinbase.com")
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")
        self.passphrase = config.get("passphrase")
        self.ws_connector = CoinbaseWSConnector(config)
        self.session = None

    async def connect(self) -> bool:
        try:
            self.session = aiohttp.ClientSession()
            
            ws_connected = await self.ws_connector.connect()
            if ws_connected:
                logger.info("Coinbase WebSocket connection established")
                self.is_connected = True
                return True
                
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Coinbase: {e}")
            return False

    async def disconnect(self) -> None:
        if self.ws_connector:
            await self.ws_connector.disconnect()
        if self.session:
            await self.session.close()
        self.is_connected = False

    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = '') -> str:
        message = timestamp + method + path + body
        signature = hmac.new(
            base64.b64decode(self.secret_key),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')

    async def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        timestamp = str(time.time())
        path = endpoint
        body = json.dumps(data) if data else ''
        
        signature = self._generate_signature(timestamp, method.upper(), path, body)
        
        headers = {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': signature,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
        
        url = f"{self.base_url}{endpoint}"
        
        async with self.session.request(method, url, json=data, headers=headers) as response:
            return await response.json()

    async def place_order(self, order: Order) -> str:
        data = {
            'product_id': order.symbol.replace('/', '-'),
            'side': order.side.value.lower(),
            'order_configuration': {
                'limit_limit_gtc': {
                    'base_size': str(order.quantity),
                    'limit_price': str(order.price)
                }
            }
        }
        
        if order.client_order_id:
            data['client_order_id'] = order.client_order_id

        response = await self._make_request('POST', '/api/v3/brokerage/orders', data)
        return response.get('order_id')

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        response = await self._make_request('DELETE', f'/api/v3/brokerage/orders/{order_id}')
        return response.get('success', False)

    async def get_order_status(self, symbol: str, order_id: str) -> Order:
        response = await self._make_request('GET', f'/api/v3/brokerage/orders/historical/{order_id}')
        
        order_data = response.get('order', {})
        
        return Order(
            symbol=order_data.get('product_id', '').replace('-', '/'),
            side=OrderSide.BUY if order_data.get('side') == 'BUY' else OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=float(order_data.get('order_configuration', {}).get('limit_limit_gtc', {}).get('base_size', 0)),
            price=float(order_data.get('order_configuration', {}).get('limit_limit_gtc', {}).get('limit_price', 0)),
            order_id=order_data.get('order_id'),
            client_order_id=order_data.get('client_order_id'),
            status=self._map_order_status(order_data.get('status')),
            filled_quantity=float(order_data.get('filled_size', 0)),
            timestamp=int(time.time() * 1000)
        )

    def _map_order_status(self, status: str) -> OrderStatus:
        status_map = {
            'OPEN': OrderStatus.NEW,
            'FILLED': OrderStatus.FILLED,
            'CANCELLED': OrderStatus.CANCELED,
            'EXPIRED': OrderStatus.EXPIRED,
            'FAILED': OrderStatus.REJECTED
        }
        return status_map.get(status, OrderStatus.NEW)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        params = {}
        if symbol:
            params['product_id'] = symbol.replace('/', '-')
            
        response = await self._make_request('GET', '/api/v3/brokerage/orders/historical/batch')
        
        orders = []
        for order_data in response.get('orders', []):
            if order_data.get('status') == 'OPEN':
                orders.append(Order(
                    symbol=order_data.get('product_id', '').replace('-', '/'),
                    side=OrderSide.BUY if order_data.get('side') == 'BUY' else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=float(order_data.get('order_configuration', {}).get('limit_limit_gtc', {}).get('base_size', 0)),
                    price=float(order_data.get('order_configuration', {}).get('limit_limit_gtc', {}).get('limit_price', 0)),
                    order_id=order_data.get('order_id'),
                    client_order_id=order_data.get('client_order_id'),
                    status=self._map_order_status(order_data.get('status')),
                    filled_quantity=float(order_data.get('filled_size', 0)),
                    timestamp=int(time.time() * 1000)
                ))
        
        return orders

    async def get_positions(self) -> List[Position]:
        response = await self._make_request('GET', '/api/v3/brokerage/accounts')
        
        positions = []
        for account in response.get('accounts', []):
            balance = float(account.get('available_balance', {}).get('value', 0))
            if balance > 0:
                positions.append(Position(
                    symbol=account.get('currency'),
                    side=OrderSide.BUY,
                    size=balance,
                    entry_price=0.0,
                    mark_price=0.0,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    timestamp=int(time.time() * 1000)
                ))
        
        return positions

    async def get_balances(self) -> List[Balance]:
        response = await self._make_request('GET', '/api/v3/brokerage/accounts')
        
        balances = []
        for account in response.get('accounts', []):
            available = float(account.get('available_balance', {}).get('value', 0))
            hold = float(account.get('hold', {}).get('value', 0))
            
            if available > 0 or hold > 0:
                balances.append(Balance(
                    asset=account.get('currency'),
                    free=available,
                    locked=hold,
                    total=available + hold
                ))
        
        return balances

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        product_id = symbol.replace('/', '-')
        response = await self._make_request('GET', f'/api/v3/brokerage/product_book?product_id={product_id}&limit={depth}')
        
        pricebook = response.get('pricebook', {})
        
        bids = [OrderBookLevel(float(bid['price']), float(bid['size'])) for bid in pricebook.get('bids', [])]
        asks = [OrderBookLevel(float(ask['price']), float(ask['size'])) for ask in pricebook.get('asks', [])]
        
        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=int(time.time() * 1000)
        )

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        if self.ws_connector:
            await self.ws_connector.subscribe_orderbook(symbol, callback)

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        if self.ws_connector:
            await self.ws_connector.subscribe_trades(symbol, callback)

    async def subscribe_order_updates(self, callback: Callable) -> None:
        if self.ws_connector:
            await self.ws_connector.subscribe_order_updates(callback)


class CoinbaseWSConnector(BaseWSConnector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ws_url = config.get("websocket_url", "wss://advanced-trade-ws.international.coinbase.com")

    async def connect(self) -> bool:
        try:
            await self.connect_websocket()
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            return False

    async def disconnect(self) -> None:
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False

    async def connect_websocket(self) -> bool:
        try:
            self.websocket = await websockets.connect(self.ws_url)
            asyncio.create_task(self._listen())
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False

    async def _listen(self) -> None:
        try:
            async for message in self.websocket:
                await self.handle_ws_message(message)
        except Exception as e:
            logger.error(f"WebSocket listen error: {e}")

    async def send_ws_message(self, message: Dict[str, Any]) -> None:
        if self.websocket:
            await self.websocket.send(json.dumps(message))

    async def handle_ws_message(self, message: str) -> None:
        try:
            data = json.loads(message)
            
            if 'channel' in data:
                channel = data['channel']
                
                if channel == 'level2':
                    await self._handle_orderbook_update(data)
                elif channel == 'market_trades':
                    await self._handle_trade_update(data)
                elif channel == 'user':
                    await self._handle_order_update(data)
                    
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    async def _handle_orderbook_update(self, data: Dict) -> None:
        for event in data.get('events', []):
            if event.get('type') == 'snapshot' or event.get('type') == 'update':
                updates = event.get('updates', [])
                for update in updates:
                    product_id = update.get('product_id', '')
                    symbol = product_id.replace('-', '/')
                    
                    bids = [OrderBookLevel(float(bid['price_level']), float(bid['new_quantity'])) 
                           for bid in update.get('bids', [])]
                    asks = [OrderBookLevel(float(ask['price_level']), float(ask['new_quantity'])) 
                           for ask in update.get('asks', [])]
                    
                    orderbook = OrderBook(
                        symbol=symbol,
                        bids=bids,
                        asks=asks,
                        timestamp=int(time.time() * 1000)
                    )
                    
                    await self._emit_event('orderbook', orderbook)

    async def _handle_trade_update(self, data: Dict) -> None:
        for event in data.get('events', []):
            for trade_data in event.get('trades', []):
                trade = Trade(
                    symbol=trade_data.get('product_id', '').replace('-', '/'),
                    price=float(trade_data.get('price', 0)),
                    quantity=float(trade_data.get('size', 0)),
                    side=OrderSide.BUY if trade_data.get('side') == 'BUY' else OrderSide.SELL,
                    timestamp=int(time.time() * 1000),
                    trade_id=trade_data.get('trade_id', '')
                )
                
                await self._emit_event('trade', trade)

    async def _handle_order_update(self, data: Dict) -> None:
        for event in data.get('events', []):
            for order_data in event.get('orders', []):
                order = Order(
                    symbol=order_data.get('product_id', '').replace('-', '/'),
                    side=OrderSide.BUY if order_data.get('side') == 'BUY' else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=float(order_data.get('base_size', 0)),
                    price=float(order_data.get('limit_price', 0)),
                    order_id=order_data.get('order_id'),
                    client_order_id=order_data.get('client_order_id'),
                    status=self._map_order_status(order_data.get('status')),
                    filled_quantity=float(order_data.get('filled_size', 0)),
                    timestamp=int(time.time() * 1000)
                )
                
                await self._emit_event('order_update', order)

    def _map_order_status(self, status: str) -> OrderStatus:
        status_map = {
            'OPEN': OrderStatus.NEW,
            'FILLED': OrderStatus.FILLED,
            'CANCELLED': OrderStatus.CANCELED,
            'EXPIRED': OrderStatus.EXPIRED,
            'FAILED': OrderStatus.REJECTED
        }
        return status_map.get(status, OrderStatus.NEW)

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        product_id = symbol.replace('/', '-')
        self.add_callback('orderbook', callback)
        
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": [product_id],
            "channel": "level2"
        }
        
        await self.send_ws_message(subscribe_msg)

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        product_id = symbol.replace('/', '-')
        self.add_callback('trade', callback)
        
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": [product_id],
            "channel": "market_trades"
        }
        
        await self.send_ws_message(subscribe_msg)

    async def subscribe_order_updates(self, callback: Callable) -> None:
        self.add_callback('order_update', callback)
        
        subscribe_msg = {
            "type": "subscribe",
            "channel": "user"
        }
        
        await self.send_ws_message(subscribe_msg)

    async def place_order(self, order: Order) -> str:
        return f"ws_order_{int(time.time())}"

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        return True

    async def get_order_status(self, symbol: str, order_id: str) -> Order:
        return Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            order_id=order_id,
            status=OrderStatus.NEW,
            timestamp=int(time.time() * 1000)
        )

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        return []

    async def get_positions(self) -> List[Position]:
        return []

    async def get_balances(self) -> List[Balance]:
        return []

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        return OrderBook(
            symbol=symbol,
            bids=[OrderBookLevel(49999.0, 1.0)],
            asks=[OrderBookLevel(50001.0, 1.0)],
            timestamp=int(time.time() * 1000)
        )
