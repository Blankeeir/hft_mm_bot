import asyncio
import json
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any, Callable
import aiohttp
import websockets
import logging
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from .base import (
    BaseExchangeConnector, BaseFIXConnector, BaseWSConnector,
    Order, Position, Balance, OrderBook, OrderBookLevel, Trade,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)


class BinanceConnector(BaseExchangeConnector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://api.binance.com")
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")
        self.fix_connector = None
        self.ws_connector = None
        self.session = None
        
        if config.get("fix_api_key") and config.get("fix_secret_key"):
            self.fix_connector = BinanceFIXConnector(config)
        
        self.ws_connector = BinanceWSConnector(config)

    async def connect(self) -> bool:
        try:
            self.session = aiohttp.ClientSession()
            
            if self.fix_connector:
                fix_connected = await self.fix_connector.connect()
                if fix_connected:
                    logger.info("Binance FIX connection established")
                    self.is_connected = True
                    return True
            
            ws_connected = await self.ws_connector.connect()
            if ws_connected:
                logger.info("Binance WebSocket connection established")
                self.is_connected = True
                return True
                
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            return False

    async def disconnect(self) -> None:
        if self.fix_connector:
            await self.fix_connector.disconnect()
        if self.ws_connector:
            await self.ws_connector.disconnect()
        if self.session:
            await self.session.close()
        self.is_connected = False

    def _generate_signature(self, query_string: str) -> str:
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        if params is None:
            params = {}
            
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            params['signature'] = self._generate_signature(query_string)

        headers = {'X-MBX-APIKEY': self.api_key} if self.api_key else {}
        
        url = f"{self.base_url}{endpoint}"
        
        async with self.session.request(method, url, params=params, headers=headers) as response:
            return await response.json()

    async def place_order(self, order: Order) -> str:
        if self.fix_connector and self.fix_connector.is_connected:
            return await self.fix_connector.place_order(order)
        
        params = {
            'symbol': order.symbol.replace('/', ''),
            'side': order.side.value,
            'type': order.order_type.value,
            'quantity': order.quantity,
            'timeInForce': order.time_in_force,
        }
        
        if order.price:
            params['price'] = order.price
            
        if order.client_order_id:
            params['newClientOrderId'] = order.client_order_id

        response = await self._make_request('POST', '/api/v3/order', params, signed=True)
        return response.get('orderId')

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        if self.fix_connector and self.fix_connector.is_connected:
            return await self.fix_connector.cancel_order(symbol, order_id)
            
        params = {
            'symbol': symbol.replace('/', ''),
            'orderId': order_id,
        }
        
        response = await self._make_request('DELETE', '/api/v3/order', params, signed=True)
        return response.get('status') == 'CANCELED'

    async def get_order_status(self, symbol: str, order_id: str) -> Order:
        params = {
            'symbol': symbol.replace('/', ''),
            'orderId': order_id,
        }
        
        response = await self._make_request('GET', '/api/v3/order', params, signed=True)
        
        return Order(
            symbol=symbol,
            side=OrderSide(response['side']),
            order_type=OrderType(response['type']),
            quantity=float(response['origQty']),
            price=float(response['price']) if response['price'] else None,
            order_id=response['orderId'],
            client_order_id=response['clientOrderId'],
            status=OrderStatus(response['status']),
            filled_quantity=float(response['executedQty']),
            timestamp=response['time']
        )

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        params = {}
        if symbol:
            params['symbol'] = symbol.replace('/', '')
            
        response = await self._make_request('GET', '/api/v3/openOrders', params, signed=True)
        
        orders = []
        for order_data in response:
            orders.append(Order(
                symbol=order_data['symbol'],
                side=OrderSide(order_data['side']),
                order_type=OrderType(order_data['type']),
                quantity=float(order_data['origQty']),
                price=float(order_data['price']) if order_data['price'] else None,
                order_id=order_data['orderId'],
                client_order_id=order_data['clientOrderId'],
                status=OrderStatus(order_data['status']),
                filled_quantity=float(order_data['executedQty']),
                timestamp=order_data['time']
            ))
        
        return orders

    async def get_positions(self) -> List[Position]:
        response = await self._make_request('GET', '/api/v3/account', signed=True)
        
        positions = []
        for balance in response['balances']:
            if float(balance['free']) > 0 or float(balance['locked']) > 0:
                positions.append(Position(
                    symbol=balance['asset'],
                    side=OrderSide.BUY,
                    size=float(balance['free']) + float(balance['locked']),
                    entry_price=0.0,
                    mark_price=0.0,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    timestamp=int(time.time() * 1000)
                ))
        
        return positions

    async def get_balances(self) -> List[Balance]:
        response = await self._make_request('GET', '/api/v3/account', signed=True)
        
        balances = []
        for balance in response['balances']:
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free > 0 or locked > 0:
                balances.append(Balance(
                    asset=balance['asset'],
                    free=free,
                    locked=locked,
                    total=free + locked
                ))
        
        return balances

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        params = {
            'symbol': symbol.replace('/', ''),
            'limit': depth
        }
        
        response = await self._make_request('GET', '/api/v3/depth', params)
        
        bids = [OrderBookLevel(float(bid[0]), float(bid[1])) for bid in response['bids']]
        asks = [OrderBookLevel(float(ask[0]), float(ask[1])) for ask in response['asks']]
        
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


class BinanceFIXConnector(BaseFIXConnector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.fix_endpoint = config.get("fix_endpoint")
        self.fix_api_key = config.get("fix_api_key")
        self.fix_secret_key = config.get("fix_secret_key")

    async def connect(self) -> bool:
        try:
            await self.setup_fix_session()
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect FIX session: {e}")
            return False

    async def disconnect(self) -> None:
        if self.fix_session:
            self.fix_session.logout()
        self.is_connected = False

    async def setup_fix_session(self) -> bool:
        return True

    async def send_fix_message(self, message: Any) -> None:
        pass

    async def place_order(self, order: Order) -> str:
        return f"fix_order_{int(time.time())}"

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        return True


class BinanceWSConnector(BaseWSConnector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_ws_url = "wss://stream.binance.com:9443/ws"
        self.user_data_stream = None

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
            self.websocket = await websockets.connect(self.base_ws_url)
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
            
            if 'e' in data:
                event_type = data['e']
                
                if event_type == 'depthUpdate':
                    await self._handle_orderbook_update(data)
                elif event_type == 'trade':
                    await self._handle_trade_update(data)
                elif event_type == 'executionReport':
                    await self._handle_order_update(data)
                    
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    async def _handle_orderbook_update(self, data: Dict) -> None:
        symbol = data['s']
        bids = [OrderBookLevel(float(bid[0]), float(bid[1])) for bid in data['b']]
        asks = [OrderBookLevel(float(ask[0]), float(ask[1])) for ask in data['a']]
        
        orderbook = OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=data['E']
        )
        
        await self._emit_event('orderbook', orderbook)

    async def _handle_trade_update(self, data: Dict) -> None:
        trade = Trade(
            symbol=data['s'],
            price=float(data['p']),
            quantity=float(data['q']),
            side=OrderSide.BUY if data['m'] else OrderSide.SELL,
            timestamp=data['T'],
            trade_id=data['t']
        )
        
        await self._emit_event('trade', trade)

    async def _handle_order_update(self, data: Dict) -> None:
        order = Order(
            symbol=data['s'],
            side=OrderSide(data['S']),
            order_type=OrderType(data['o']),
            quantity=float(data['q']),
            price=float(data['p']) if data['p'] else None,
            order_id=data['i'],
            client_order_id=data['c'],
            status=OrderStatus(data['X']),
            filled_quantity=float(data['z']),
            timestamp=data['T']
        )
        
        await self._emit_event('order_update', order)

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        stream = f"{symbol.lower().replace('/', '')}@depth"
        self.subscriptions.add(stream)
        self.add_callback('orderbook', callback)
        
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [stream],
            "id": int(time.time())
        }
        
        await self.send_ws_message(subscribe_msg)

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        stream = f"{symbol.lower().replace('/', '')}@trade"
        self.subscriptions.add(stream)
        self.add_callback('trade', callback)
        
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [stream],
            "id": int(time.time())
        }
        
        await self.send_ws_message(subscribe_msg)

    async def subscribe_order_updates(self, callback: Callable) -> None:
        self.add_callback('order_update', callback)

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
