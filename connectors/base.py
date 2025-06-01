from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    OCO = "OCO"
    OTO = "OTO"
    OTOCO = "OTOCO"


class OrderStatus(Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    timestamp: Optional[int] = None
    time_in_force: str = "GTC"


@dataclass
class Position:
    symbol: str
    side: OrderSide
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: int


@dataclass
class Balance:
    asset: str
    free: float
    locked: float
    total: float


@dataclass
class OrderBookLevel:
    price: float
    quantity: float
    count: int = 1


@dataclass
class OrderBook:
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: int


@dataclass
class Trade:
    symbol: str
    price: float
    quantity: float
    side: OrderSide
    timestamp: int
    trade_id: str


class BaseExchangeConnector(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "Unknown")
        self.is_connected = False
        self.callbacks: Dict[str, List[Callable]] = {
            "order_update": [],
            "trade": [],
            "orderbook": [],
            "balance_update": [],
        }

    @abstractmethod
    async def connect(self) -> bool:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def place_order(self, order: Order) -> str:
        pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        pass

    @abstractmethod
    async def get_order_status(self, symbol: str, order_id: str) -> Order:
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        pass

    @abstractmethod
    async def get_balances(self) -> List[Balance]:
        pass

    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        pass

    @abstractmethod
    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        pass

    @abstractmethod
    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        pass

    @abstractmethod
    async def subscribe_order_updates(self, callback: Callable) -> None:
        pass

    def add_callback(self, event_type: str, callback: Callable) -> None:
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)

    def remove_callback(self, event_type: str, callback: Callable) -> None:
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)

    async def _emit_event(self, event_type: str, data: Any) -> None:
        for callback in self.callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in callback for {event_type}: {e}")


class BaseFIXConnector(BaseExchangeConnector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.fix_session = None
        self.fix_config_file = None

    @abstractmethod
    async def setup_fix_session(self) -> bool:
        pass

    @abstractmethod
    async def send_fix_message(self, message: Any) -> None:
        pass


class BaseWSConnector(BaseExchangeConnector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.websocket = None
        self.ws_url = config.get("websocket_url")
        self.subscriptions = set()

    @abstractmethod
    async def connect_websocket(self) -> bool:
        pass

    @abstractmethod
    async def send_ws_message(self, message: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def handle_ws_message(self, message: str) -> None:
        pass
