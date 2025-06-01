import abc
from typing import Dict

class BaseConnector(abc.ABC):
    def __init__(self, cfg: Dict):
        self.cfg = cfg

    @abc.abstractmethod
    async def connect(self): ...

    @abc.abstractmethod
    async def send_order(self, side: str, price: float, size: float, client_id: str): ...

    @abc.abstractmethod
    async def cancel(self, order_id: str): ...

    @abc.abstractmethod
    async def positions(self): ...

    @abc.abstractmethod
    async def current_equity(self): ...

    @abc.abstractmethod
    async def stream_book(self, pair: str): ...
