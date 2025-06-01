import asyncio
import uuid
import hmac
import hashlib
import base64
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
import logging

from .base import (
    BaseFIXConnector, Order, Position, Balance, OrderBook, OrderBookLevel, Trade,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)


class CoinbaseFIXAdapter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")
        self.passphrase = config.get("passphrase")
        self.sender_comp_id = config.get("sender_comp_id", "YOUR_SENDER_COMP_ID")
        self.session_type = config.get("fix_session_type", "order_entry")
        self.test_mode = config.get("test_mode", False)
        
        self.authenticated = False
        self.is_connected = False
        self.callbacks = {}
        self.next_request_id = 1
        
    def add_callback(self, event_type: str, callback: Callable) -> None:
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
        
    async def _emit_event(self, event_type: str, data: Any) -> None:
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for {event_type}: {e}")

    async def connect(self) -> bool:
        if self.test_mode:
            self.authenticated = True
            self.is_connected = True
            logger.info("Test mode: Simulated FIX connection established")
            return True
        
        logger.warning("Real FIX connection not implemented - using test mode")
        return await self.connect()

    async def disconnect(self) -> None:
        self.is_connected = False
        self.authenticated = False

    async def place_order(self, order: Order) -> str:
        if not self.authenticated:
            logger.error("Cannot place order: Not authenticated")
            return ""
            
        client_order_id = order.client_order_id or f"TEST-{uuid.uuid4()}"
        logger.info(f"Test mode: Simulating {order.order_type.value} {order.side.value} order for {order.quantity} {order.symbol} at price {order.price}")
        return client_order_id

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        if not self.authenticated:
            logger.error("Cannot cancel order: Not authenticated")
            return False
            
        logger.info(f"Test mode: Simulating cancel for order {order_id}")
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

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        self.add_callback('orderbook', callback)
        logger.info(f"Test mode: Simulating market data subscription for {symbol}")

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        self.add_callback('trade', callback)

    async def subscribe_order_updates(self, callback: Callable) -> None:
        self.add_callback('order_update', callback)

    def _get_next_request_id(self) -> int:
        self.next_request_id += 1
        return self.next_request_id

    def _get_utc_timestamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]
