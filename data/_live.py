import asyncio
import datetime
import logging
import threading
import queue as thread_queue
from typing import Any, Dict, List, Optional, Tuple

import ccxt

from data._base import DataHandler
from data._timeframe import (
    _ms_to_utc, _ensure_utc, parse_timeframe_to_seconds,
)
from data._arctic import _make_symbol_key, _init_arctic_lib
from data._exchange import _build_exchange_config, _fetch_all_candles
from data._ohlcv import _candles_to_dataframe
from event import BarEvent

logger = logging.getLogger(__name__)


class LiveDataHandler(DataHandler):
    """LiveDataHandler for live trading.

    Uses CCXT for REST API (backfill) and CCXT Pro for WebSocket streaming.

    Architecture:
    - WebSocket streaming runs in a background thread
    - Bars are passed to main thread via thread-safe queue
    - update_bar() polls from queue (sync, same interface as HistoricDataHandler)
    """

    def __init__(self, events_queue: thread_queue.Queue[Any], symbol_list: List[str],
                 exchange_id: str, base_timeframe: str, timeframes: Dict[str, int],
                 market_type: str, db_path: str = "arctic_data"):
        """Initialize for live trading with CCXT REST + WebSocket.

        Sets up the exchange connection, initializes ArcticDB for storage,
        and performs a warmup (backfill from DB + REST) before streaming.
        """
        super().__init__(events_queue, symbol_list, base_timeframe, timeframes)
        self.exchange_id = exchange_id
        self.db_path = db_path
        self.market_type = market_type

        # Exchange setup
        self._exchange_config = _build_exchange_config(market_type)
        self.exchange: Any = getattr(ccxt, exchange_id)(self._exchange_config)
        self.async_exchange: Optional[Any] = None

        # Threading infrastructure
        self._bar_queue: thread_queue.Queue[BarEvent] = thread_queue.Queue()
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self._is_running = False
        self._streaming_active = False

        # ArcticDB for storage/warmup
        try:
            self.store, self.lib = _init_arctic_lib(db_path)
        except Exception as e:
            logger.error("Error initializing ArcticDB: %s", e)
            raise

        # Track last processed timestamp to avoid duplicates
        self.last_timestamps: Dict[str, Optional[datetime.datetime]] = {s: None for s in self.symbol_list}
        self._ts_lock = threading.Lock()

        # Perform Warmup (uses REST API)
        self._warmup()

    def _is_candle_closed(self, candle_timestamp: datetime.datetime, buffer_seconds: int = 5) -> bool:
        """Check if a candle is closed (finished forming).

        A candle is closed if: current_time >= candle_open_time + timeframe_duration + buffer
        """
        tf_seconds = parse_timeframe_to_seconds(self.base_timeframe)
        candle_close_time = candle_timestamp + datetime.timedelta(seconds=tf_seconds + buffer_seconds)
        now = datetime.datetime.now(datetime.timezone.utc)
        return now >= candle_close_time

    # ── warmup (decomposed) ─────────────────────

    def _warmup(self) -> None:
        """Warm up the system by syncing DB and filling deques for each symbol."""
        logger.info("Starting Data Handler Warmup...")

        for symbol in self.symbol_list:
            symbol_key = _make_symbol_key(symbol, self.base_timeframe)
            try:
                last_db_ts = self._get_last_db_timestamp(symbol_key, symbol)
                since_ms = self._compute_since_ms(last_db_ts)

                closed_candles = self._fetch_and_filter_candles(symbol, since_ms)
                if closed_candles:
                    self._save_candles_to_db(symbol_key, closed_candles, last_db_ts)

                self._fill_deque_from_db(symbol, symbol_key)
                logger.info("Warmup complete for %s. Deque has %d bars.",
                            symbol, len(self._base_bar_data[symbol]))
            except Exception as e:
                logger.error("Warmup failed for %s: %s", symbol, e)

    def _get_last_db_timestamp(self, symbol_key: str,
                               symbol: str) -> Optional[datetime.datetime]:
        """Read the last stored timestamp from ArcticDB for a symbol."""
        try:
            if self.lib.has_symbol(symbol_key):
                df = self.lib.read(symbol_key).data
                if not df.empty:
                    ts = _ensure_utc(df.index[-1])
                    logger.debug("Last DB timestamp for %s: %s", symbol, ts)
                    return ts
        except Exception as e:
            logger.warning("Could not read last timestamp for %s: %s", symbol, e)
        return None

    def _compute_since_ms(self, last_db_timestamp: Optional[datetime.datetime]) -> Optional[int]:
        """Compute the 'since' parameter (ms) for CCXT candle fetching."""
        if last_db_timestamp is None:
            return None
        tf_seconds = parse_timeframe_to_seconds(self.base_timeframe)
        since_dt = last_db_timestamp + datetime.timedelta(seconds=tf_seconds)
        return int(since_dt.timestamp() * 1000)

    def _fetch_and_filter_candles(self, symbol: str,
                                  since_ms: Optional[int]) -> List[List[Any]]:
        """Fetch candles from exchange and return only closed ones."""
        logger.info("Fetching missing candles for %s...", symbol)
        base_maxlen = self.timeframes[self.base_timeframe]
        all_candles = _fetch_all_candles(
            self.exchange, symbol, self.base_timeframe,
            since_ms=since_ms, limit=base_maxlen
        )
        closed = [c for c in all_candles if self._is_candle_closed(_ms_to_utc(c[0]))]
        logger.debug("Fetched %d candles, %d are closed.", len(all_candles), len(closed))
        return closed

    def _save_candles_to_db(self, symbol_key: str, closed_candles: List[List[Any]],
                            last_db_timestamp: Optional[datetime.datetime]) -> None:
        """Save closed candles to ArcticDB (append or write)."""
        df = _candles_to_dataframe(closed_candles)
        if last_db_timestamp:
            self.lib.append(symbol_key, df)
            logger.info("Appended %d candles to DB for %s", len(df), symbol_key)
        else:
            self.lib.write(symbol_key, df)
            logger.info("Wrote %d candles to DB for %s", len(df), symbol_key)

    def _fill_deque_from_db(self, symbol: str, symbol_key: str) -> None:
        """Fill the rolling deque from the latest DB entries."""
        base_maxlen = self.timeframes[self.base_timeframe]
        df = self.lib.read(symbol_key).data
        recent_df = df.tail(base_maxlen)
        for idx, row in recent_df.iterrows():
            ts = _ensure_utc(idx)
            self._append_bar(symbol, ts, row['Open'], row['High'], row['Low'], row['Close'], row['Volume'])
            with self._ts_lock:
                self.last_timestamps[symbol] = ts

    # ── bar polling ─────────────────────────────

    def update_bar(self) -> None:
        """Poll for new bars from the WebSocket stream.

        Every emission (forming and completed) upserts the base deque's last
        entry, so ``get_latest_bars`` always exposes the latest forming bar
        at ``iloc[-1]`` and the most recent completed bar at ``iloc[-2]``.
        HTF accumulators advance only when a base bar completes — forming
        emissions carry cumulative OHLCV for the in-progress base bar and
        would double-count volume if propagated.
        """
        if not self._is_running:
            return

        while True:
            try:
                bar = self._bar_queue.get_nowait()
                if self._append_bar(bar.symbol, bar.timestamp,
                                    bar.open, bar.high, bar.low,
                                    bar.close, bar.volume,
                                    is_forming=bar.is_forming):
                    self.events_queue.put(bar)
            except thread_queue.Empty:
                break

    # ── WebSocket lifecycle ─────────────────────

    def start(self) -> None:
        """Start the WebSocket streaming in a background thread."""
        if self._is_running:
            logger.warning("Already running.")
            return

        self._is_running = True
        self._streaming_active = True

        self._ws_thread = threading.Thread(
            target=self._run_ws_thread,
            name="LiveDataHandler-WS",
            daemon=True
        )
        self._ws_thread.start()
        logger.info("Started WebSocket streaming thread for %s", self.symbol_list)

    def stop(self) -> None:
        """Stop the WebSocket streaming background thread."""
        if not self._is_running:
            return

        logger.info("Stopping WebSocket streaming...")
        self._streaming_active = False
        self._is_running = False

        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5.0)

        logger.info("WebSocket streaming stopped.")

    def _run_ws_thread(self) -> None:
        """Background thread entry point. Creates a new event loop and runs async streaming."""
        self._ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._ws_loop)

        try:
            self._ws_loop.run_until_complete(self._async_stream())
        except Exception as e:
            logger.error("WebSocket thread error: %s", e)
        finally:
            self._ws_loop.close()
            self._ws_loop = None

    # ── async streaming ─────────────────────────

    async def _async_stream(self) -> None:
        """Async streaming logic (runs in background thread)."""
        try:
            import ccxt.pro as ccxtpro
        except ImportError:
            raise ImportError(
                "CCXT Pro is required for WebSocket streaming. "
                "Install with: pip install ccxt[pro]"
            )

        if self.async_exchange is None:
            self.async_exchange = getattr(ccxtpro, self.exchange_id)(self._exchange_config)

        logger.info("WebSocket connected for %s...", self.symbol_list)

        try:
            tasks = [self._stream_symbol_loop(symbol) for symbol in self.symbol_list]
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Streaming cancelled.")
        except Exception as e:
            logger.error("Streaming error: %s", e)
        finally:
            await self._close_async_exchange()

    async def _stream_symbol_loop(self, symbol: str) -> None:
        """Continuous streaming loop for a single symbol using watch_ohlcv.

        When the current candle's timestamp changes, the previous candle is closed.
        """
        previous_candle: Optional[List[Any]] = None

        logger.debug("Starting OHLCV stream for %s, timeframe=%s", symbol, self.base_timeframe)

        # Gap-fill: Fetch last 3 candles via REST to bridge any gap between warmup and streaming
        try:
            ohlcv = await self.async_exchange.fetch_ohlcv(symbol, self.base_timeframe, limit=3)
            if ohlcv:
                for candle in ohlcv[:-1]:  # Exclude the last (current/forming) candle
                    ts = _ms_to_utc(candle[0])

                    with self._ts_lock:
                        last_ts = self.last_timestamps.get(symbol)
                        if last_ts is not None and ts <= last_ts:
                            continue

                    self._append_bar(symbol, ts, candle[1], candle[2], candle[3], candle[4], candle[5])
                    with self._ts_lock:
                        self.last_timestamps[symbol] = ts
                    logger.debug("Gap-fill: %s", ts)
        except Exception as e:
            logger.warning("Gap-fill fetch failed: %s", e)

        while self._streaming_active:
            try:
                ohlcv = await self.async_exchange.watch_ohlcv(symbol, self.base_timeframe)

                if not ohlcv:
                    continue

                current_candle = ohlcv[-1]
                current_ts_ms = current_candle[0]

                if previous_candle is not None:
                    prev_ts_ms = previous_candle[0]
                    if current_ts_ms > prev_ts_ms:
                        ts = _ms_to_utc(prev_ts_ms)
                        logger.debug("Candle closed: %s", ts)
                        await self._process_candle(symbol, previous_candle, ts, is_forming=False)

                # Emit current forming bar for real-time monitoring
                ts = _ms_to_utc(current_ts_ms)
                await self._process_candle(symbol, current_candle, ts, is_forming=True)

                previous_candle = current_candle

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error streaming %s: %s", symbol, e)
                await asyncio.sleep(1)

    async def _process_candle(self, symbol: str, candle: List[Any],
                              ts: datetime.datetime, is_forming: bool = False) -> None:
        """Process a candle from WebSocket stream into the bar queue."""
        if not is_forming:
            with self._ts_lock:
                self.last_timestamps[symbol] = ts

        bar = BarEvent(
            symbol=symbol,
            timestamp=ts,
            open=float(candle[1]),
            high=float(candle[2]),
            low=float(candle[3]),
            close=float(candle[4]),
            volume=float(candle[5]),
            period=self.base_timeframe,
            is_forming=is_forming,
        )

        self._bar_queue.put(bar)
        logger.debug("[WS] %s | %s | Close: %s | Forming: %s", symbol, ts, bar.close, is_forming)

    async def _close_async_exchange(self) -> None:
        """Close the async exchange connection."""
        if self.async_exchange:
            try:
                await self.async_exchange.close()
            except Exception as e:
                logger.warning("Error closing async exchange: %s", e)
            self.async_exchange = None
