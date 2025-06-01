import asyncio
import uuid
import hmac
import hashlib
import base64
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import logging

try:
    from asyncfix import FIXMessage, FIXConnection, FTag, FMsg
    ASYNCFIX_AVAILABLE = True
except ImportError:
    FIXMessage = None
    FIXConnection = None
    FTag = None
    FMsg = None
    ASYNCFIX_AVAILABLE = False

from .base import (
    BaseFIXConnector, Order, Position, Balance, OrderBook, OrderBookLevel, Trade,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)


class CoinbaseFIXConnector(BaseFIXConnector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.fix_endpoint = config.get("fix_endpoint", "tcp+tls://fix.international.coinbase.com:4198")
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")
        self.passphrase = config.get("passphrase")
        self.sender_comp_id = config.get("sender_comp_id", "YOUR_SENDER_COMP_ID")
        self.session_type = config.get("fix_session_type", "order_entry")
        
        self.connection = None
        self.authenticated = False
        self.next_request_id = 1
        self.test_mode = config.get("test_mode", False)
        
        if not ASYNCFIX_AVAILABLE:
            logger.warning("asyncfix not available, FIX connector will not work")

    async def connect(self) -> bool:
        if not ASYNCFIX_AVAILABLE:
            logger.error("asyncfix library not available")
            return False
            
        try:
            await self.setup_fix_session()
            if self.connection and self.authenticated:
                logger.info("Coinbase FIX connection established")
                self.is_connected = True
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Coinbase FIX: {e}")
            return False

    async def disconnect(self) -> None:
        if self.connection:
            try:
                await self.connection.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting FIX session: {e}")
        self.is_connected = False
        self.authenticated = False

    async def setup_fix_session(self) -> bool:
        try:
            if self.fix_endpoint.startswith("tcp+tls://"):
                host_port = self.fix_endpoint[10:]
                host, port = host_port.split(":")
                port = int(port)
                use_tls = True
            else:
                raise ValueError(f"Unsupported FIX endpoint format: {self.fix_endpoint}")
            
            self.connection = FIXConnection(
                host=host,
                port=port,
                sender_comp_id=self.sender_comp_id,
                target_comp_id="Coinbase",
                use_tls=use_tls,
                on_message=self._on_message
            )
            
            await self.connection.connect()
            await self._authenticate()
            
            return self.authenticated
            
        except Exception as e:
            logger.error(f"Error setting up FIX session: {e}")
            return False

    async def send_fix_message(self, message: Any) -> None:
        if self.connection:
            await self.connection.send_msg(message)

    async def _authenticate(self) -> None:
        try:
            timestamp = self._get_utc_timestamp()
            
            message = f"{timestamp}A{self.sender_comp_id}Coinbase"
            signature = hmac.new(
                base64.b64decode(self.secret_key),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
            signature_b64 = base64.b64encode(signature).decode('utf-8')
            
            logon = FIXMessage(FMsg.LOGON)
            logon.set(FTag.EncryptMethod, "0")
            logon.set(FTag.HeartBtInt, "30")
            logon.set(FTag.Username, self.api_key)
            logon.set(FTag.Password, signature_b64)
            logon.set(FTag.SendingTime, timestamp)
            logon.set(8013, self.session_type)
            
            await self.connection.send_msg(logon)
            
            await asyncio.sleep(2)
            
            if self.authenticated:
                logger.info(f"Coinbase FIX authentication successful for {self.session_type} session")
            else:
                logger.error("Coinbase FIX authentication failed")
                
        except Exception as e:
            logger.error(f"Error during FIX authentication: {e}")

    async def _on_message(self, message: FIXMessage) -> None:
        try:
            msg_type = message.get(FTag.MsgType)
            
            if msg_type == FMsg.LOGON:
                self.authenticated = True
                logger.info(f"Logon successful for {self.session_type} session")
                
            elif msg_type == FMsg.LOGOUT:
                self.authenticated = False
                text = message.get(FTag.Text, "No reason provided")
                logger.warning(f"Received Logout: {text}")
                
            elif msg_type == FMsg.TESTREQUEST:
                await self._handle_test_request(message)
                
            elif msg_type == FMsg.REJECT:
                await self._handle_reject(message)
                
            elif msg_type == FMsg.BUSINESSMESSAGEREJECT:
                await self._handle_business_reject(message)
                
            elif msg_type == FMsg.EXECUTIONREPORT:
                await self._handle_execution_report(message)
                
            elif msg_type == FMsg.ORDERCANCELREJECT:
                await self._handle_order_cancel_reject(message)
                
            elif msg_type == FMsg.ORDERMASSCANCELREPORT:
                await self._handle_order_mass_cancel_report(message)
                
            elif msg_type == FMsg.TRADECAPTUREREPORT:
                await self._handle_trade_capture_report(message)
                
            elif msg_type == "W":
                await self._handle_market_data_snapshot(message)
                
            elif msg_type == "X":
                await self._handle_market_data_incremental(message)
                
            else:
                logger.debug(f"Received unhandled message type: {msg_type}")
                
        except Exception as e:
            logger.error(f"Error handling FIX message: {e}")

    async def _handle_test_request(self, message: FIXMessage) -> None:
        test_req_id = message.get(FTag.TestReqID)
        
        heartbeat = FIXMessage(FMsg.HEARTBEAT)
        if test_req_id:
            heartbeat.set(FTag.TestReqID, test_req_id)
        
        await self.connection.send_msg(heartbeat)

    async def _handle_reject(self, message: FIXMessage) -> None:
        ref_seq_num = message.get(FTag.RefSeqNum)
        text = message.get(FTag.Text)
        logger.error(f"FIX Reject received - RefSeqNum: {ref_seq_num}, Text: {text}")

    async def _handle_business_reject(self, message: FIXMessage) -> None:
        ref_msg_type = message.get(FTag.RefMsgType)
        business_reject_reason = message.get(FTag.BusinessRejectReason)
        text = message.get(FTag.Text)
        logger.error(f"Business Reject - RefMsgType: {ref_msg_type}, Reason: {business_reject_reason}, Text: {text}")

    async def _handle_execution_report(self, message: FIXMessage) -> None:
        try:
            order_id = message.get(FTag.OrderID)
            client_order_id = message.get(FTag.ClOrdID)
            symbol = message.get(FTag.Symbol)
            side = message.get(FTag.Side)
            order_status = message.get(FTag.OrdStatus)
            exec_type = message.get(FTag.ExecType)
            ord_type = message.get(FTag.OrdType, "")
            
            order_side = OrderSide.BUY if side == "1" else OrderSide.SELL
            status = self._map_fix_order_status(order_status)
            
            order_type = OrderType.LIMIT
            if ord_type == "1":
                order_type = OrderType.MARKET
            elif ord_type == "2":
                order_type = OrderType.LIMIT
            elif ord_type == "3":
                order_type = OrderType.STOP
            elif ord_type == "4":
                order_type = OrderType.STOP_LIMIT
            elif ord_type == "O":
                order_type = OrderType.TAKE_PROFIT_STOP_LOSS
            
            order = Order(
                symbol=symbol.replace('-', '/'),
                side=order_side,
                order_type=order_type,
                quantity=float(message.get(FTag.OrderQty, 0)),
                price=float(message.get(FTag.Price, 0)) if message.get(FTag.Price) else None,
                order_id=order_id,
                client_order_id=client_order_id,
                status=status,
                filled_quantity=float(message.get(FTag.CumQty, 0)),
                timestamp=int(time.time() * 1000),
                stop_price=float(message.get(FTag.StopPx, 0)) if message.get(FTag.StopPx) else None,
                stop_limit_price=float(message.get(3040, 0)) if message.get(3040) else None
            )
            
            ord_type_desc = "TPSL" if ord_type == "O" else ""
            exec_type_map = {
                "0": "New", "1": "Partial Fill", "2": "Fill", 
                "4": "Canceled", "8": "Rejected", "C": "Expired"
            }
            exec_desc = exec_type_map.get(exec_type, exec_type)
            logger.info(f"ExecutionReport: {client_order_id} {symbol} {side} {ord_type_desc} - {exec_desc}")
            
            await self._emit_event('order_update', order)
            
        except Exception as e:
            logger.error(f"Error handling execution report: {e}")

    async def _handle_order_cancel_reject(self, message: FIXMessage) -> None:
        client_order_id = message.get(FTag.ClOrdID)
        orig_client_order_id = message.get(FTag.OrigClOrdID)
        text = message.get(FTag.Text)
        logger.error(f"Order Cancel Reject - ClOrdID: {client_order_id}, OrigClOrdID: {orig_client_order_id}, Text: {text}")

    async def _handle_trade_capture_report(self, message: FIXMessage) -> None:
        try:
            symbol = message.get(FTag.Symbol)
            price = float(message.get(FTag.LastPx, 0))
            quantity = float(message.get(FTag.LastQty, 0))
            side = message.get(FTag.Side)
            trade_id = message.get(FTag.TradeID)
            
            trade = Trade(
                symbol=symbol.replace('-', '/'),
                price=price,
                quantity=quantity,
                side=OrderSide.BUY if side == "1" else OrderSide.SELL,
                timestamp=int(time.time() * 1000),
                trade_id=trade_id or str(uuid.uuid4())
            )
            
            await self._emit_event('trade', trade)
            
        except Exception as e:
            logger.error(f"Error handling trade capture report: {e}")

    async def _handle_market_data_snapshot(self, message: FIXMessage) -> None:
        try:
            symbol = message.get(FTag.Symbol)
            
            bids = []
            asks = []
            
            num_entries = int(message.get(FTag.NoMDEntries, 0))
            for i in range(num_entries):
                entry_type = message.get(f"{FTag.MDEntryType}_{i}")
                price = float(message.get(f"{FTag.MDEntryPx}_{i}", 0))
                size = float(message.get(f"{FTag.MDEntrySize}_{i}", 0))
                
                if entry_type == "0":
                    bids.append(OrderBookLevel(price, size))
                elif entry_type == "1":
                    asks.append(OrderBookLevel(price, size))
            
            orderbook = OrderBook(
                symbol=symbol.replace('-', '/'),
                bids=sorted(bids, key=lambda x: x.price, reverse=True),
                asks=sorted(asks, key=lambda x: x.price),
                timestamp=int(time.time() * 1000)
            )
            
            await self._emit_event('orderbook', orderbook)
            
        except Exception as e:
            logger.error(f"Error handling market data snapshot: {e}")

    async def _handle_market_data_incremental(self, message: FIXMessage) -> None:
        await self._handle_market_data_snapshot(message)

    def _map_fix_order_status(self, status: str) -> OrderStatus:
        status_map = {
            "0": OrderStatus.NEW,
            "1": OrderStatus.PARTIALLY_FILLED,
            "2": OrderStatus.FILLED,
            "4": OrderStatus.CANCELED,
            "8": OrderStatus.REJECTED,
            "C": OrderStatus.EXPIRED
        }
        return status_map.get(status, OrderStatus.NEW)

    async def place_order(self, order: Order) -> str:
        if not self.authenticated or self.session_type != "order_entry":
            logger.error("Cannot place order: Not authenticated or wrong session type")
            return ""
            
        if self.test_mode:
            client_order_id = order.client_order_id or f"TEST-{uuid.uuid4()}"
            logger.info(f"Test mode: Simulating {order.order_type.value} {order.side.value} order for {order.quantity} {order.symbol} at price {order.price}")
            return client_order_id
        
        try:
            client_order_id = order.client_order_id or str(uuid.uuid4())
            
            side_map = {"BUY": "1", "SELL": "2"}
            fix_side = side_map.get(order.side.value)
            if not fix_side:
                raise ValueError(f"Invalid side: {order.side.value}")
            
            order_type_map = {
                "MARKET": "1",
                "LIMIT": "2",
                "STOP": "3",
                "STOP_LIMIT": "4",
                "TAKE_PROFIT_STOP_LOSS": "O"
            }
            fix_order_type = order_type_map.get(order.order_type.value)
            if not fix_order_type:
                raise ValueError(f"Invalid order type: {order.order_type.value}")
            
            tif_map = {
                "DAY": "0",
                "GTC": "1",
                "IOC": "3",
                "FOK": "4",
                "GTD": "6"
            }
            fix_tif = tif_map.get(order.time_in_force.upper())
            if not fix_tif:
                raise ValueError(f"Invalid time in force: {order.time_in_force}")
            
            nos = FIXMessage(FMsg.NEWORDERSINGLE)
            nos.set(FTag.ClOrdID, client_order_id)
            nos.set(FTag.Symbol, order.symbol.replace('/', '-'))
            nos.set(FTag.Side, fix_side)
            nos.set(FTag.TransactTime, self._get_utc_timestamp())
            nos.set(FTag.OrdType, fix_order_type)
            nos.set(FTag.OrderQty, str(order.quantity))
            nos.set(FTag.TimeInForce, fix_tif)
            
            if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price is not None:
                nos.set(FTag.Price, str(order.price))
            
            if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.price is not None:
                nos.set(FTag.StopPx, str(order.price))
            
            if hasattr(order, 'portfolio_id') and order.portfolio_id:
                nos.set(453, "1")  # NoPartyIDs = 1
                nos.set(448, order.portfolio_id)  # PartyID = portfolio UUID
                nos.set(452, "24")  # PartyRole = 24 (Customer account)
            
            stp_strategy = getattr(order, 'self_trade_prevention', 'Q')
            nos.set(8000, stp_strategy)  # SelfTradePreventionStrategy
            
            if hasattr(order, 'post_only') and order.post_only:
                nos.set(18, "6")  # ExecInst = 6 (Post only)
            
            if order.order_type == OrderType.TAKE_PROFIT_STOP_LOSS:
                if order.time_in_force.upper() not in ["GTC", "GTD"]:
                    raise ValueError("TPSL orders only support GTC and GTD time in force")
                
                if hasattr(order, 'post_only') and order.post_only:
                    raise ValueError("TPSL orders do not support post_only")
                
                if order.price is None or order.stop_price is None or order.stop_limit_price is None:
                    raise ValueError("TPSL orders require price, stop_price, and stop_limit_price")
                
                nos.set(FTag.Price, str(order.price))
                nos.set(FTag.StopPx, str(order.stop_price))
                nos.set(3040, str(order.stop_limit_price))
                
                if fix_side == "2":
                    if not (order.price > order.stop_price > order.stop_limit_price):
                        raise ValueError("For Sell TPSL: Price must be > StopPx and StopPx must be > StopLimitPx")
                else:
                    if not (order.price < order.stop_price < order.stop_limit_price):
                        raise ValueError("For Buy TPSL: Price must be < StopPx and StopPx must be < StopLimitPx")
            
            await self.connection.send_msg(nos)
            logger.info(f"Placed {order.order_type.value} {order.side.value} order for {order.quantity} {order.symbol} with client order ID {client_order_id}")
            
            return client_order_id
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return ""

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        if not self.authenticated or self.session_type != "order_entry":
            logger.error("Cannot cancel order: Not authenticated or wrong session type")
            return False
            
        if self.test_mode:
            logger.info(f"Test mode: Simulating cancel for order {order_id}")
            return True
        
        try:
            cancel_client_order_id = str(uuid.uuid4())
            
            ocr = FIXMessage(FMsg.ORDERCANCELREQUEST)
            ocr.set(FTag.OrigClOrdID, order_id)
            ocr.set(FTag.ClOrdID, cancel_client_order_id)
            ocr.set(FTag.Symbol, symbol.replace('/', '-'))
            ocr.set(FTag.TransactTime, self._get_utc_timestamp())
            
            await self.connection.send_msg(ocr)
            logger.info(f"Sent cancel request for order {order_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return False

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
        if not self.authenticated or self.session_type != "market_data":
            logger.error("Cannot subscribe to market data: Not authenticated or wrong session type")
            return
            
        self.add_callback('orderbook', callback)
        
        if self.test_mode:
            logger.info(f"Test mode: Simulating market data subscription for {symbol}")
            return
        
        try:
            req_id = str(self._get_next_request_id())
            
            mdr = FIXMessage(FMsg.MARKETDATAREQUEST)
            mdr.set(FTag.MDReqID, req_id)
            mdr.set(FTag.SubscriptionRequestType, "1")
            mdr.set(FTag.MarketDepth, "0")
            mdr.set(FTag.MDUpdateType, "0")
            mdr.set(FTag.AggregatedBook, "1")
            mdr.set(FTag.NoMDEntryTypes, "2")
            mdr.set(FTag.MDEntryType, "0")
            mdr.set(FTag.MDEntryType, "1")
            mdr.set(FTag.NoRelatedSym, "1")
            mdr.set(FTag.Symbol, symbol.replace('/', '-'))
            
            await self.connection.send_msg(mdr)
            logger.info(f"Sent market data subscription request for {symbol}")
            
        except Exception as e:
            logger.error(f"Error subscribing to market data: {e}")

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        self.add_callback('trade', callback)

    async def subscribe_order_updates(self, callback: Callable) -> None:
        self.add_callback('order_update', callback)

    def _get_next_request_id(self) -> int:
        self.next_request_id += 1
        return self.next_request_id

    async def modify_order(
        self,
        original_client_order_id: str,
        symbol: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        stop_limit_price: Optional[float] = None,
    ) -> str:
        if not self.authenticated or self.session_type != "order_entry":
            logger.error("Cannot modify order: Not authenticated or wrong session type")
            return ""
            
        if self.test_mode:
            new_client_order_id = str(uuid.uuid4())
            logger.info(f"Test mode: Simulating modify for order {original_client_order_id} with new ID {new_client_order_id}")
            return new_client_order_id
        
        try:
            new_client_order_id = str(uuid.uuid4())
            
            ocrr = FIXMessage(FMsg.ORDERCANCELREPLACEREQUEST)
            ocrr.set(FTag.OrigClOrdID, original_client_order_id)
            ocrr.set(FTag.ClOrdID, new_client_order_id)
            ocrr.set(FTag.Symbol, symbol.replace('/', '-'))
            ocrr.set(FTag.TransactTime, self._get_utc_timestamp())
            
            if quantity is not None:
                ocrr.set(FTag.OrderQty, str(quantity))
                
            if price is not None:
                ocrr.set(FTag.Price, str(price))
                
            if stop_price is not None:
                ocrr.set(FTag.StopPx, str(stop_price))
                
            if stop_limit_price is not None:
                ocrr.set(3040, str(stop_limit_price))
            
            await self.connection.send_msg(ocrr)
            logger.info(f"Sent modify request for order {original_client_order_id} with new ID {new_client_order_id}")
            
            return new_client_order_id
            
        except Exception as e:
            logger.error(f"Error modifying order: {e}")
            return ""

    async def mass_cancel_orders(
        self,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
    ) -> str:
        if not self.authenticated or self.session_type != "order_entry":
            logger.error("Cannot mass cancel orders: Not authenticated or wrong session type")
            return ""
            
        if self.test_mode:
            client_order_id = str(uuid.uuid4())
            logger.info(f"Test mode: Simulating mass cancel with ID {client_order_id}")
            return client_order_id
        
        try:
            client_order_id = str(uuid.uuid4())
            
            omcr = FIXMessage("q")
            omcr.set(FTag.ClOrdID, client_order_id)
            omcr.set(FTag.TransactTime, self._get_utc_timestamp())
            
            if side:
                fix_side = "1" if side.upper() == "BUY" else "2"
                omcr.set(FTag.Side, fix_side)
            
            if symbol:
                omcr.set(FTag.Symbol, symbol.replace('/', '-'))
            
            await self.connection.send_msg(omcr)
            logger.info(f"Sent mass cancel request with ID {client_order_id}")
            
            return client_order_id
            
        except Exception as e:
            logger.error(f"Error sending mass cancel request: {e}")
            return ""

    async def _handle_order_mass_cancel_report(self, message: FIXMessage) -> None:
        try:
            clord_id = message.get(FTag.ClOrdID, "")
            mass_action_report_id = message.get(1369, "")
            symbol = message.get(FTag.Symbol, "")
            side = message.get(FTag.Side, "")
            mass_cancel_response = message.get(531, "")
            total_affected_orders = message.get(533, "0")
            
            response_map = {
                "0": "Cancel Request Rejected",
                "7": "Cancel All Orders"
            }
            
            response_desc = response_map.get(mass_cancel_response, mass_cancel_response)
            
            if mass_cancel_response == "0":
                reject_reason = message.get(532, "0")
                reject_reason_map = {
                    "0": "Mass Cancel Not Supported",
                    "1": "Invalid or unknown Security",
                    "99": "Other"
                }
                reject_desc = reject_reason_map.get(reject_reason, reject_reason)
                logger.warning(f"Mass Cancel Rejected: {clord_id} - {reject_desc}")
            else:
                logger.info(f"Mass Cancel Report: {clord_id} {symbol} {side} - {response_desc}, "
                          f"Affected: {total_affected_orders} orders")
            
        except Exception as e:
            logger.error(f"Error handling mass cancel report: {e}")

    def _get_utc_timestamp(self) -> str:
        from datetime import timezone
        return datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]
