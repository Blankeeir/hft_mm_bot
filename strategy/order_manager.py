import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from connectors.base import (
    BaseExchangeConnector, Order, OrderSide, OrderType, OrderStatus,
    Position, Balance
)
from .avellaneda_stoikov import QuoteParameters

logger = logging.getLogger(__name__)


class OrderState(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class ManagedOrder:
    order: Order
    connector: BaseExchangeConnector
    state: OrderState
    created_time: int
    last_update_time: int
    target_price: float
    target_size: float
    fill_events: List[Dict[str, Any]]


class OrderManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connectors: Dict[str, BaseExchangeConnector] = {}
        self.active_orders: Dict[str, ManagedOrder] = {}
        self.order_history: List[ManagedOrder] = []
        
        self.refresh_interval = config.get("order_management", {}).get("order_refresh_interval", 200)
        self.max_order_age = 5000
        
        self.callbacks: Dict[str, List[Callable]] = {
            "order_filled": [],
            "order_cancelled": [],
            "order_failed": [],
        }
        
        self.running = False
        
    def add_connector(self, exchange_name: str, connector: BaseExchangeConnector) -> None:
        self.connectors[exchange_name] = connector
        connector.add_callback("order_update", self._handle_order_update)
        
    def remove_connector(self, exchange_name: str) -> None:
        if exchange_name in self.connectors:
            del self.connectors[exchange_name]
            
    async def start(self) -> None:
        self.running = True
        asyncio.create_task(self._management_loop())
        logger.info("OrderManager started")
        
    async def stop(self) -> None:
        self.running = False
        await self.cancel_all_orders()
        logger.info("OrderManager stopped")
        
    def add_callback(self, event_type: str, callback: Callable) -> None:
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            
    async def _emit_event(self, event_type: str, data: Any) -> None:
        for callback in self.callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in callback for {event_type}: {e}")
                
    async def place_quote_orders(self, exchange_name: str, quotes: QuoteParameters) -> Dict[str, str]:
        if exchange_name not in self.connectors:
            logger.error(f"Connector for {exchange_name} not found")
            return {}
            
        connector = self.connectors[exchange_name]
        current_time = int(time.time() * 1000)
        
        await self.cancel_existing_orders(exchange_name, quotes.symbol)
        
        results = {}
        
        if quotes.bid_size > 0:
            bid_order = Order(
                symbol=quotes.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=quotes.bid_size,
                price=quotes.bid_price,
                client_order_id=f"bid_{uuid.uuid4().hex[:8]}",
                time_in_force="GTC"
            )
            
            try:
                order_id = await connector.place_order(bid_order)
                if order_id:
                    bid_order.order_id = order_id
                    managed_order = ManagedOrder(
                        order=bid_order,
                        connector=connector,
                        state=OrderState.PENDING,
                        created_time=current_time,
                        last_update_time=current_time,
                        target_price=quotes.bid_price,
                        target_size=quotes.bid_size,
                        fill_events=[]
                    )
                    self.active_orders[order_id] = managed_order
                    results["bid"] = order_id
                    logger.debug(f"Placed bid order {order_id} for {quotes.symbol} at {quotes.bid_price}")
                    
            except Exception as e:
                logger.error(f"Failed to place bid order for {quotes.symbol}: {e}")
                
        if quotes.ask_size > 0:
            ask_order = Order(
                symbol=quotes.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=quotes.ask_size,
                price=quotes.ask_price,
                client_order_id=f"ask_{uuid.uuid4().hex[:8]}",
                time_in_force="GTC"
            )
            
            try:
                order_id = await connector.place_order(ask_order)
                if order_id:
                    ask_order.order_id = order_id
                    managed_order = ManagedOrder(
                        order=ask_order,
                        connector=connector,
                        state=OrderState.PENDING,
                        created_time=current_time,
                        last_update_time=current_time,
                        target_price=quotes.ask_price,
                        target_size=quotes.ask_size,
                        fill_events=[]
                    )
                    self.active_orders[order_id] = managed_order
                    results["ask"] = order_id
                    logger.debug(f"Placed ask order {order_id} for {quotes.symbol} at {quotes.ask_price}")
                    
            except Exception as e:
                logger.error(f"Failed to place ask order for {quotes.symbol}: {e}")
                
        return results
        
    async def cancel_existing_orders(self, exchange_name: str, symbol: str) -> None:
        orders_to_cancel = []
        
        for order_id, managed_order in self.active_orders.items():
            if (managed_order.order.symbol == symbol and 
                managed_order.connector == self.connectors.get(exchange_name) and
                managed_order.state in [OrderState.PENDING, OrderState.ACTIVE]):
                orders_to_cancel.append(order_id)
                
        for order_id in orders_to_cancel:
            await self.cancel_order(order_id)
            
    async def cancel_order(self, order_id: str) -> bool:
        if order_id not in self.active_orders:
            return False
            
        managed_order = self.active_orders[order_id]
        
        try:
            success = await managed_order.connector.cancel_order(
                managed_order.order.symbol, 
                order_id
            )
            
            if success:
                managed_order.state = OrderState.CANCELLED
                managed_order.last_update_time = int(time.time() * 1000)
                await self._emit_event("order_cancelled", managed_order)
                logger.debug(f"Cancelled order {order_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            managed_order.state = OrderState.FAILED
            return False
            
    async def cancel_all_orders(self) -> None:
        order_ids = list(self.active_orders.keys())
        
        for order_id in order_ids:
            await self.cancel_order(order_id)
            
    async def _handle_order_update(self, order: Order) -> None:
        if not order.order_id or order.order_id not in self.active_orders:
            return
            
        managed_order = self.active_orders[order.order_id]
        managed_order.order = order
        managed_order.last_update_time = int(time.time() * 1000)
        
        if order.status == OrderStatus.FILLED:
            managed_order.state = OrderState.FILLED
            fill_event = {
                "timestamp": managed_order.last_update_time,
                "price": order.average_price or order.price,
                "quantity": order.filled_quantity,
                "side": order.side.value
            }
            managed_order.fill_events.append(fill_event)
            await self._emit_event("order_filled", managed_order)
            
        elif order.status == OrderStatus.CANCELED:
            managed_order.state = OrderState.CANCELLED
            await self._emit_event("order_cancelled", managed_order)
            
        elif order.status == OrderStatus.REJECTED:
            managed_order.state = OrderState.FAILED
            await self._emit_event("order_failed", managed_order)
            
        elif order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
            managed_order.state = OrderState.ACTIVE
            
    async def _management_loop(self) -> None:
        while self.running:
            try:
                await asyncio.sleep(self.refresh_interval / 1000.0)
                await self._check_stale_orders()
                await self._cleanup_completed_orders()
                
            except Exception as e:
                logger.error(f"Error in order management loop: {e}")
                
    async def _check_stale_orders(self) -> None:
        current_time = int(time.time() * 1000)
        stale_orders = []
        
        for order_id, managed_order in self.active_orders.items():
            age = current_time - managed_order.created_time
            
            if (age > self.max_order_age and 
                managed_order.state in [OrderState.PENDING, OrderState.ACTIVE]):
                stale_orders.append(order_id)
                
        for order_id in stale_orders:
            logger.warning(f"Cancelling stale order {order_id}")
            await self.cancel_order(order_id)
            
    async def _cleanup_completed_orders(self) -> None:
        completed_orders = []
        
        for order_id, managed_order in self.active_orders.items():
            if managed_order.state in [OrderState.FILLED, OrderState.CANCELLED, OrderState.FAILED]:
                completed_orders.append(order_id)
                
        for order_id in completed_orders:
            managed_order = self.active_orders.pop(order_id)
            self.order_history.append(managed_order)
            
        max_history = 1000
        if len(self.order_history) > max_history:
            self.order_history = self.order_history[-max_history:]
            
    def get_active_orders(self, symbol: Optional[str] = None) -> List[ManagedOrder]:
        orders = list(self.active_orders.values())
        
        if symbol:
            orders = [order for order in orders if order.order.symbol == symbol]
            
        return orders
        
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[ManagedOrder]:
        history = self.order_history[-limit:]
        
        if symbol:
            history = [order for order in history if order.order.symbol == symbol]
            
        return history
        
    def get_fill_rate(self, symbol: Optional[str] = None, window_hours: int = 1) -> float:
        current_time = int(time.time() * 1000)
        start_time = current_time - (window_hours * 3600000)
        
        recent_orders = [
            order for order in self.order_history 
            if order.created_time >= start_time and 
            (symbol is None or order.order.symbol == symbol)
        ]
        
        if not recent_orders:
            return 0.0
            
        filled_orders = [
            order for order in recent_orders 
            if order.state == OrderState.FILLED
        ]
        
        return len(filled_orders) / len(recent_orders)
        
    def get_average_fill_time(self, symbol: Optional[str] = None, window_hours: int = 1) -> float:
        current_time = int(time.time() * 1000)
        start_time = current_time - (window_hours * 3600000)
        
        filled_orders = [
            order for order in self.order_history 
            if (order.state == OrderState.FILLED and 
                order.created_time >= start_time and 
                (symbol is None or order.order.symbol == symbol))
        ]
        
        if not filled_orders:
            return 0.0
            
        fill_times = [
            order.last_update_time - order.created_time 
            for order in filled_orders
        ]
        
        return sum(fill_times) / len(fill_times)
