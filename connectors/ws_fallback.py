import asyncio, json, websockets, gzip, zlib, base64, uuid
from .base import BaseConnector
from utils.logger import get_logger

log = get_logger("ws_fallback")

class BinanceWSConnector(BaseConnector):
    async def connect(self):
        self.ws = await websockets.connect(self.cfg["ws_url"] + "/stream")
        log.info("Connected WS")

    async def stream_book(self, pair: str):
        stream = pair.lower() + "@depth@100ms"
        await self.ws.send(json.dumps({"method": "SUBSCRIBE", "params": [stream], "id": 1}))
        async for msg in self.ws:
            data = json.loads(msg)
            if "bids" in data:
                yield data
