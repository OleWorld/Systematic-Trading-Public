import heapq
import logging
import queue as thread_queue
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import pandas as pd

from data._base import DataHandler
from data._tz import ensure_utc_index
from event import BarEvent

logger = logging.getLogger(__name__)

# Default rolling-window length for alt feeds when the caller does not
# override via ``alt_maxlen`` — mirrors the ``{base: 500}`` bar default.
DEFAULT_ALT_MAXLEN = 500

# Heap tie-break priorities: alt records sort BEFORE bars at equal
# timestamps. A print at exactly T is known at T, while the bar stamped
# T (open-time convention) is processed at its close one period later —
# alt-first is causal and avoids one bar of staleness.
_KIND_ALT = 0
_KIND_BAR = 1


class HistoricDataHandler(DataHandler):
    """HistoricDataHandler is designed for backtesting.

    It is fed pre-built per-symbol OHLCV DataFrames (``data={symbol: df}``) and
    yields bars one by one in time-sorted order across all symbols.

    Each DataFrame must be indexed by a timezone-aware ``DatetimeIndex`` and
    expose ``Open``/``High``/``Low``/``Close``/``Volume`` columns. Sourcing,
    cleaning, and windowing the data is the caller's responsibility. Naive-indexed
    frames raise ``ValueError`` at construction; tz-aware non-UTC frames are converted
    to UTC. Each frame's index must be sorted ascending — unsorted input raises
    ``ValueError`` at construction (same-timestamp adjacent duplicates are
    tolerated here and handled by the stream gate: first bar wins).

    Optionally, ``alt_data={feed: {symbol: df}}`` supplies named per-symbol
    alternative-data series (funding rates, open interest, ...) that are
    merged into the same time-sorted stream and stored in rolling windows —
    no events emitted; strategies read them via ``get_latest_alt`` at bar
    cadence. At equal timestamps alt rows sort before bar rows, so the bar
    at open-time T sees exactly the records with ``ts <= T``.
    """

    def __init__(self, events_queue: thread_queue.Queue[Any], symbol_list: List[str],
                 base_timeframe: str,
                 timeframes: Dict[str, int],
                 data: Dict[str, pd.DataFrame],
                 alt_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
                 alt_maxlen: Optional[Dict[str, int]] = None):
        """Initialize for backtesting from in-memory DataFrames.

        ``data`` maps each symbol to a time-indexed OHLCV DataFrame and is the
        sole bar source. Raises ``ValueError`` if it is missing or empty.

        ``alt_data`` optionally maps ``{feed: {symbol: DataFrame}}`` —
        named per-symbol alternative-data series (funding rates, open
        interest, ...). Each frame needs a tz-aware ``DatetimeIndex``
        (naive raises; non-UTC converts) sorted ascending, and numeric,
        uniquely-named columns (column names become the record field
        names). Symbols must be a subset of ``symbol_list``. Alt record
        timestamps mean "the moment the value became known" — supplying
        correctly-stamped data is the caller's responsibility.
        ``alt_maxlen`` optionally overrides the per-feed rolling-window
        length (``{feed: maxlen}``, default ``DEFAULT_ALT_MAXLEN``);
        naming a feed absent from ``alt_data`` raises (typo guard).
        """
        alt_data = alt_data or {}
        alt_maxlen = alt_maxlen or {}
        unknown_feeds = set(alt_maxlen) - set(alt_data)
        if unknown_feeds:
            raise ValueError(
                f"alt_maxlen names feeds absent from alt_data: "
                f"{sorted(unknown_feeds)} — every maxlen override must "
                f"correspond to a supplied feed (typo guard)."
            )
        alt_feeds = {feed: alt_maxlen.get(feed, DEFAULT_ALT_MAXLEN)
                     for feed in alt_data}
        super().__init__(events_queue, symbol_list, base_timeframe,
                         timeframes, alt_feeds=alt_feeds)

        if not data:
            raise ValueError(
                "data is required: pass data={symbol: DataFrame} with a "
                "tz-aware DatetimeIndex and Open/High/Low/Close/Volume columns."
            )

        normalized: Dict[str, pd.DataFrame] = {}
        for sym, df in data.items():
            if df is None or df.empty:
                continue
            idx = ensure_utc_index(df.index, f"data[{sym!r}]")
            if not idx.is_monotonic_increasing:
                raise ValueError(
                    f"data[{sym!r}] index is not sorted ascending: bars "
                    f"must be supplied in time order. An out-of-order bar "
                    f"would silently corrupt HTF aggregates, indicator "
                    f"recursion, and the latest-price cache, so the run "
                    f"is refused up front."
                )
            # set_axis returns a new frame sharing the data — the caller's
            # frame is never mutated and nothing is copied.
            normalized[sym] = df.set_axis(idx)

        self._alt_frames = self._normalize_alt(alt_data)
        self._bar_generators = self._build_stream(normalized, self._alt_frames)

    # ── alt-frame validation ─────────────────────

    def _normalize_alt(
        self, alt_data: Dict[str, Dict[str, pd.DataFrame]],
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Validate and UTC-normalize alt-feed frames at construction.

        Per (feed, symbol) frame gates — all raise ``ValueError`` up
        front, before the run starts:

        - symbol must be registered in ``symbol_list`` (wiring typo);
        - index must be tz-aware (naive raises via ``ensure_utc_index``;
          non-UTC converts) and sorted ascending (mirrors the bar gate);
        - columns must be uniquely named and numeric (they become the
          record field names).

        Empty/``None`` frames are skipped silently (mirrors the bar
        path). Column ORDER is recorded in ``self._alt_columns`` per
        (symbol, feed) — the same key order as the alt deques — so the
        stream can zip positional ``itertuples`` values back to field
        names regardless of pandas' identifier renaming.
        """
        normalized: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._alt_columns: Dict[Tuple[str, str], List[str]] = {}
        for feed, per_symbol in alt_data.items():
            clean: Dict[str, pd.DataFrame] = {}
            for sym, df in per_symbol.items():
                if sym not in self._base_bar_data:
                    raise ValueError(
                        f"alt_data[{feed!r}] covers unregistered symbol "
                        f"{sym!r}: alt feeds must key to symbols in "
                        f"symbol_list. Registered: {self.symbol_list}"
                    )
                if df is None or df.empty:
                    continue
                idx = ensure_utc_index(df.index,
                                       f"alt_data[{feed!r}][{sym!r}]")
                if not idx.is_monotonic_increasing:
                    raise ValueError(
                        f"alt_data[{feed!r}][{sym!r}] index is not sorted "
                        f"ascending: records must be supplied in time "
                        f"order."
                    )
                cols = list(df.columns)
                if len(set(cols)) != len(cols):
                    raise ValueError(
                        f"alt_data[{feed!r}][{sym!r}] has duplicate "
                        f"column names: {cols}. Field names must be "
                        f"unique."
                    )
                non_numeric = [c for c in cols
                               if not pd.api.types.is_numeric_dtype(df[c])]
                if non_numeric:
                    raise ValueError(
                        f"alt_data[{feed!r}][{sym!r}] has non-numeric "
                        f"columns {non_numeric}: alt fields must be "
                        f"numeric."
                    )
                clean[sym] = df.set_axis(idx)
                self._alt_columns[(sym, feed)] = [str(c) for c in cols]
            if clean:
                normalized[feed] = clean
        return normalized

    # ── stream construction ─────────────────────

    def _build_stream(
        self,
        dataframes: Dict[str, pd.DataFrame],
        alt_frames: Dict[str, Dict[str, pd.DataFrame]],
    ) -> Generator[Tuple[Tuple[int, str, Optional[str]], Any], None, None]:
        """Convert bar frames + alt frames into one time-sorted stream.

        Stream items are ``(key, row)`` where ``key`` is
        ``(_KIND_BAR, symbol, None)`` for bars and
        ``(_KIND_ALT, symbol, feed)`` for alt records. Bar generators
        are inserted first so bar-vs-bar ties keep the pre-existing
        first-inserted-symbol order; the kind field in the heap entry
        puts alt records ahead of bars at equal timestamps regardless
        of insertion order.
        """
        generators: Dict[Tuple[int, str, Optional[str]], Iterator[Any]] = {}
        for sym, df in dataframes.items():
            if df is not None and not df.empty:
                generators[(_KIND_BAR, sym, None)] = df.itertuples()
        for feed, per_symbol in alt_frames.items():
            for sym, df in per_symbol.items():
                generators[(_KIND_ALT, sym, feed)] = df.itertuples()
        return self._merge_generators(generators)

    def _merge_generators(
        self,
        generators: Dict[Tuple[int, str, Optional[str]], Iterator[Any]],
    ) -> Generator[Tuple[Tuple[int, str, Optional[str]], Any], None, None]:
        """Merge bar and alt generators into a single time-sorted stream.

        A heap of ``(timestamp, kind, seq, key, row)`` heads yields the
        earliest row in O(log S) per item (S = generators with data
        remaining). ``kind`` (``_KIND_ALT=0``, ``_KIND_BAR=1``) breaks
        equal-timestamp ties in favor of alt records — the causality
        contract. ``seq`` is the generator's insertion index, so
        same-kind ties keep insertion order (for bars: byte-identical to
        the previous single-kind behavior). ``(timestamp, kind, seq)``
        is unique per heap entry (one head per generator), so
        ``key``/``row`` are never compared.
        """
        heap: List[Tuple[Any, int, int, Tuple[int, str, Optional[str]], Any]] = []
        for seq, (key, gen) in enumerate(generators.items()):
            try:
                row = next(gen)
            except StopIteration:
                continue
            heap.append((row.Index, key[0], seq, key, row))
        heapq.heapify(heap)

        while heap:
            _, _, seq, key, row = heapq.heappop(heap)

            yield (key, row)

            try:
                nxt = next(generators[key])
            except StopIteration:
                continue
            heapq.heappush(heap, (nxt.Index, key[0], seq, key, nxt))

    def update_bar(self) -> None:
        """Ingest due alt rows, then push the next bar to the queue.

        Loops the merged stream: every alt row that sorts before the
        next bar row — including rows at the SAME timestamp, which the
        heap orders alt-first — is stored via ``_append_alt`` with no
        event emitted; the first bar row is appended and emitted as a
        ``BarEvent`` exactly as before, ending the call. So when the
        bar at open-time T reaches the queue, the alt windows contain
        exactly the records with ``ts <= T``. Stream exhaustion
        (including trailing alt rows after the final bar) ends the
        backtest.
        """
        try:
            while True:
                key, row = next(self._bar_generators)
                kind, symbol, feed = key

                if kind == _KIND_ALT:
                    columns = self._alt_columns[(symbol, feed)]
                    values = {col: float(val)
                              for col, val in zip(columns, row[1:])}
                    self._append_alt(feed, symbol, row.Index, values)
                elif kind == _KIND_BAR:
                    bar = BarEvent(
                        symbol=symbol,
                        timestamp=row.Index,
                        open=float(row.Open),
                        high=float(row.High),
                        low=float(row.Low),
                        close=float(row.Close),
                        volume=float(row.Volume),
                        period=self.base_timeframe
                    )
                    if self._append_bar(symbol, bar.timestamp, bar.open,
                                        bar.high, bar.low, bar.close,
                                        bar.volume):
                        self.events_queue.put(bar)
                    return
                else:
                    raise ValueError(f"Unexpected stream kind: {kind!r}")

        except StopIteration:
            self.continue_backtest = False
