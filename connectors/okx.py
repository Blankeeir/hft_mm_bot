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


class OKXConnector(BaseExchangeConnector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://www.okx.com")
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")
        self.passphrase = config.get("passphrase")
        self.ws_connector = OKXWSConnector(config)
        self.session = None

    async def connect(self) -> bool:
        try:
            self.session = aiohttp.ClientSession()
            
            ws_connected = await self.ws_connector.connect()
            if ws_connected:
                logger.info("OKX WebSocket connection established")
                self.is_connected = True
                return True
                
            return False
        except Exception as e:
            logger.error(f"Failed to connect to OKX: {e}")
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
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')

    async def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        timestamp = time.strftime('%Y-%m-%dT%H:%M:%S.%fZ', time.gmtime())
        path = endpoint
        body = json.dumps(data) if data else ''
        
        signature = self._generate_signature(timestamp, method.upper(), path, body)
        
        headers = {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': signature,
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
        
        url = f"{self.base_url}{endpoint}"
        
        async with self.session.request(method, url, json=data, headers=headers) as response:
            return await response.json()

    async def place_order(self, order: Order) -> str:
        data = {
            'instId': order.symbol.replace('/', '-'),
            'tdMode': 'cash',
            'side': order.side.value.lower(),
            'ordType': 'limit',
            'sz': str(order.quantity),
            'px': str(order.price)
        }
        
        if order.client_order_id:
            data['clOrdId'] = order.client_order_id

        response = await self._make_request('POST', '/api/v5/trade/order', data)
        
        if response.get('code') == '0' and response.get('data'):
            return response['data'][0].get('ordId')
        return None

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        data = {
            'instId': symbol.replace('/', '-'),
            'ordId': order_id
        }
        
        response = await self._make_request('POST', '/api/v5/trade/cancel-order', data)
        return response.get('code') == '0'

    async def get_order_status(self, symbol: str, order_id: str) -> Order:
        params = {
            'instId': symbol.replace('/', '-'),
            'ordId': order_id
        }
        
        response = await self._make_request('GET', f'/api/v5/trade/order?instId={params["instId"]}&ordId={params["ordId"]}')
        
        if response.get('code') == '0' and response.get('data'):
            order_data = response['data'][0]
            
            return Order(
                symbol=order_data.get('instId', '').replace('-', '/'),
                side=OrderSide.BUY if order_data.get('side') == 'buy' else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=float(order_data.get('sz', 0)),
                price=float(order_data.get('px', 0)),
                order_id=order_data.get('ordId'),
                client_order_id=order_data.get('clOrdId'),
                status=self._map_order_status(order_data.get('state')),
                filled_quantity=float(order_data.get('fillSz', 0)),
                timestamp=int(order_data.get('cTime', 0))
            )
        
        return None

    def _map_order_status(self, status: str) -> OrderStatus:
        status_map = {
            'live': OrderStatus.NEW,
            'filled': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELED,
            'partially_filled': OrderStatus.PARTIALLY_FILLED
        }
        return status_map.get(status, OrderStatus.NEW)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        params = {}
        if symbol:
            params['instId'] = symbol.replace('/', '-')
            
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f'/api/v5/trade/orders-pending?{query_string}' if query_string else '/api/v5/trade/orders-pending'
        
        response = await self._make_request('GET', endpoint)
        
        orders = []
        if response.get('code') == '0':
            for order_data in response.get('data', []):
                orders.append(Order(
                    symbol=order_data.get('instId', '').replace('-', '/'),
                    side=OrderSide.BUY if order_data.get('side') == 'buy' else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=float(order_data.get('sz', 0)),
                    price=float(order_data.get('px', 0)),
                    order_id=order_data.get('ordId'),
                    client_order_id=order_data.get('clOrdId'),
                    status=self._map_order_status(order_data.get('state')),
                    filled_quantity=float(order_data.get('fillSz', 0)),
                    timestamp=int(order_data.get('cTime', 0))
                ))
        
        return orders

    async def get_positions(self) -> List[Position]:
        response = await self._make_request('GET', '/api/v5/account/positions')
        
        positions = []
        if response.get('code') == '0':
            for pos_data in response.get('data', []):
                if float(pos_data.get('pos', 0)) != 0:
                    positions.append(Position(
                        symbol=pos_data.get('instId', '').replace('-', '/'),
                        side=OrderSide.BUY if float(pos_data.get('pos', 0)) > 0 else OrderSide.SELL,
                        size=abs(float(pos_data.get('pos', 0))),
                        entry_price=float(pos_data.get('avgPx', 0)),
                        mark_price=float(pos_data.get('markPx', 0)),
                        unrealized_pnl=float(pos_data.get('upl', 0)),
                        realized_pnl=0.0,
                        timestamp=int(time.time() * 1000)
                    ))
        
        return positions

    async def get_balances(self) -> List[Balance]:
        response = await self._make_request('GET', '/api/v5/account/balance')
        
        balances = []
        if response.get('code') == '0':
            for account in response.get('data', []):
                for detail in account.get('details', []):
                    available = float(detail.get('availBal', 0))
                    frozen = float(detail.get('frozenBal', 0))
                    
                    if available > 0 or frozen > 0:
                        balances.append(Balance(
                            asset=detail.get('ccy'),
                            free=available,
                            locked=frozen,
                            total=available + frozen
                        ))
        
        return balances

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        inst_id = symbol.replace('/', '-')
        response = await self._make_request('GET', f'/api/v5/market/books?instId={inst_id}&sz={depth}')
        
        if response.get('code') == '0' and response.get('data'):
            book_data = response['data'][0]
            
            bids = [OrderBookLevel(float(bid[0]), float(bid[1])) for bid in book_data.get('bids', [])]
            asks = [OrderBookLevel(float(ask[0]), float(ask[1])) for ask in book_data.get('asks', [])]
            
            return OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=int(book_data.get('ts', 0))
            )
        
        return None

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        if self.ws_connector:
            await self.ws_connector.subscribe_orderbook(symbol, callback)

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        if self.ws_connector:
            await self.ws_connector.subscribe_trades(symbol, callback)

    async def subscribe_order_updates(self, callback: Callable) -> None:
        if self.ws_connector:
            await self.ws_connector.subscribe_order_updates(callback)


class OKXWSConnector(BaseWSConnector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ws_url = config.get("websocket_url", "wss://ws.okx.com:8443/ws/v5/public")
        self.private_ws_url = "wss://ws.okx.com:8443/ws/v5/private"

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
            
            if 'arg' in data and 'data' in data:
                channel = data['arg'].get('channel')
                
                if channel == 'books':
                    await self._handle_orderbook_update(data)
                elif channel == 'trades':
                    await self._handle_trade_update(data)
                elif channel == 'orders':
                    await self._handle_order_update(data)
                    
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    async def _handle_orderbook_update(self, data: Dict) -> None:
        inst_id = data['arg'].get('instId', '')
        symbol = inst_id.replace('-', '/')
        
        for book_data in data.get('data', []):
            bids = [OrderBookLevel(float(bid[0]), float(bid[1])) for bid in book_data.get('bids', [])]
            asks = [OrderBookLevel(float(ask[0]), float(ask[1])) for ask in book_data.get('asks', [])]
            
            orderbook = OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=int(book_data.get('ts', 0))
            )
            
            await self._emit_event('orderbook', orderbook)

    async def _handle_trade_update(self, data: Dict) -> None:
        inst_id = data['arg'].get('instId', '')
        symbol = inst_id.replace('-', '/')
        
        for trade_data in data.get('data', []):
            trade = Trade(
                symbol=symbol,
                price=float(trade_data.get('px', 0)),
                quantity=float(trade_data.get('sz', 0)),
                side=OrderSide.BUY if trade_data.get('side') == 'buy' else OrderSide.SELL,
                timestamp=int(trade_data.get('ts', 0)),
                trade_id=trade_data.get('tradeId', '')
            )
            
            await self._emit_event('trade', trade)

    async def _handle_order_update(self, data: Dict) -> None:
        for order_data in data.get('data', []):
            order = Order(
                symbol=order_data.get('instId', '').replace('-', '/'),
                side=OrderSide.BUY if order_data.get('side') == 'buy' else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=float(order_data.get('sz', 0)),
                price=float(order_data.get('px', 0)),
                order_id=order_data.get('ordId'),
                client_order_id=order_data.get('clOrdId'),
                status=self._map_order_status(order_data.get('state')),
                filled_quantity=float(order_data.get('fillSz', 0)),
                timestamp=int(order_data.get('cTime', 0))
            )
            
            await self._emit_event('order_update', order)

    def _map_order_status(self, status: str) -> OrderStatus:
        status_map = {
            'live': OrderStatus.NEW,
            'filled': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELED,
            'partially_filled': OrderStatus.PARTIALLY_FILLED
        }
        return status_map.get(status, OrderStatus.NEW)

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        inst_id = symbol.replace('/', '-')
        self.add_callback('orderbook', callback)
        
        subscribe_msg = {
            "op": "subscribe",
            "args": [{
                "channel": "books",
                "instId": inst_id
            }]
        }
        
        await self.send_ws_message(subscribe_msg)

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        inst_id = symbol.replace('/', '-')
        self.add_callback('trade', callback)
        
        subscribe_msg = {
            "op": "subscribe",
            "args": [{
                "channel": "trades",
                "instId": inst_id
            }]
        }
        
        await self.send_ws_message(subscribe_msg)

    async def subscribe_order_updates(self, callback: Callable) -> None:
        self.add_callback('order_update', callback)
        
        subscribe_msg = {
            "op": "subscribe",
            "args": [{
                "channel": "orders",
                "instType": "SPOT"
            }]
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
