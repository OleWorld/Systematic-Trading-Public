"""CorrelationManager — walk-forward correlation estimation on a cadence.

Owns the rho pipeline (estimate -> shrink -> floor -> PSD repair), the
period-bucketed refresh cadence, and the 'constant_price' exclusion
lifecycle (pushed into the UniverseManager through its public mark/clear
interface). One policy source among many feeding the universe manager's
single source of truth. update_bar returns at most one CorrelationEvent
per bar; the engine dispatches it synchronously to the risk manager.
"""

import datetime
import logging
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

from analytics import correlation_matrix
from data import get_period_start
from event import BarEvent, CorrelationEvent

logger = logging.getLogger(__name__)

# Minimum valid return observations for a stable correlation estimate;
# below this the refresh publishes reason='insufficient_observations'.
_MIN_CORR_OBS = 30


class _DataHandlerLike(Protocol):
    """Subset of the DataHandler surface that CorrelationManager reads.

    ``timeframes`` is read at construction time to validate that the
    configured ``timeframe`` is registered (and to read its deque
    maxlen for the ``lookback`` bound). ``get_latest_bars_df`` pulls the
    trailing close window used to derive per-symbol price changes /
    returns for the correlation estimate (Task 6).
    """

    timeframes: Dict[str, int]

    def get_latest_bars_df(self, symbol: str, n: int = 1,
                           timeframe: Optional[str] = None) -> pd.DataFrame: ...


class _UniverseManagerLike(Protocol):
    """Subset of the UniverseManager surface that CorrelationManager reads
    from and writes to.

    ``min_history_bars`` / ``history_timeframe`` are read at construction
    time for the drift guard (they must be consistent with ``lookback`` /
    ``timeframe`` so every live symbol carries a complete correlation
    window). ``symbol_list`` / ``status`` enumerate refresh candidates.
    ``reassess_all`` is called at the top of every refresh so the
    correlation window is built from freshly-derived liveness.
    ``get_live_symbols`` supplies the ``CorrelationEvent.live_symbols``
    snapshot. ``mark_excluded`` / ``clear_excluded`` are the
    'constant_price' exclusion lifecycle's write surface (Task 6) — the
    only way this module influences universe state.
    """

    min_history_bars: int
    history_timeframe: str
    symbol_list: List[str]

    def reassess_all(self) -> None: ...

    def get_live_symbols(self) -> List[str]: ...

    def status(self, symbol: str) -> Any: ...

    def mark_excluded(self, symbol: str, reason: str,
                      timestamp: Any = None) -> None: ...

    def clear_excluded(self, symbol: str, reason: str,
                       timestamp: Any = None) -> None: ...


class CorrelationManager:
    """Cadenced rho estimation over the live universe (module docstring)."""

    def __init__(self, data_handler: _DataHandlerLike,
                 universe_manager: _UniverseManagerLike,
                 lookback: int = 60, step_size: int = 30,
                 timeframe: str = '1d', mode: str = 'absolute_price_chg',
                 floor: Optional[float] = None,
                 shrinkage: Optional[str] = 'ledoit_wolf'):
        """Validate estimation params + the universe drift guard.

        Drift guard: universe_manager.history_timeframe must equal
        ``timeframe`` and universe_manager.min_history_bars must be >=
        ``lookback`` — every live symbol then contributes a complete
        correlation window. Raises ValueError on any invalid param.
        """
        if lookback < _MIN_CORR_OBS + 2:
            raise ValueError(
                f"lookback must be >= {_MIN_CORR_OBS + 2}, got {lookback}. "
                f"The window yields lookback - 1 price-change observations "
                f"and cross-symbol alignment trims one more; lookback - 2 "
                f"must cover the {_MIN_CORR_OBS}-obs minimum."
            )
        if step_size < 0:
            raise ValueError(f"step_size must be >= 0, got {step_size}")
        if timeframe not in data_handler.timeframes:
            raise ValueError(
                f"timeframe '{timeframe}' not registered in "
                f"data_handler.timeframes; available: "
                f"{list(data_handler.timeframes.keys())}"
            )
        if lookback > data_handler.timeframes[timeframe]:
            raise ValueError(
                f"lookback ({lookback}) exceeds the '{timeframe}' deque "
                f"maxlen ({data_handler.timeframes[timeframe]})."
            )
        if mode not in ('simple_return', 'absolute_price_chg'):
            raise ValueError(
                f"Unknown mode: {mode!r}. Must be 'simple_return' or "
                f"'absolute_price_chg'."
            )
        if floor is not None and not (-1.0 <= floor <= 1.0):
            raise ValueError(
                f"floor must be in [-1.0, 1.0] or None, got {floor}"
            )
        if shrinkage not in (None, 'ledoit_wolf'):
            raise ValueError(
                f"shrinkage must be None or 'ledoit_wolf', got {shrinkage!r}"
            )
        if universe_manager.history_timeframe != timeframe:
            raise ValueError(
                f"Drift guard: universe history_timeframe "
                f"({universe_manager.history_timeframe!r}) must equal the "
                f"correlation timeframe ({timeframe!r})."
            )
        if universe_manager.min_history_bars < lookback:
            raise ValueError(
                f"Drift guard: universe min_history_bars "
                f"({universe_manager.min_history_bars}) must be >= lookback "
                f"({lookback}) so every live symbol carries a complete "
                f"correlation window."
            )
        self.data_handler = data_handler
        self.universe_manager = universe_manager
        self.lookback = lookback
        self.step_size = step_size
        self.timeframe = timeframe
        self.mode = mode
        self.floor = floor
        self.shrinkage = shrinkage

        self.matrix: Optional[pd.DataFrame] = None
        self.last_refresh_timestamp: Optional[datetime.datetime] = None
        self._periods_since_refresh: int = 0
        self._last_seen_period_start: Optional[datetime.datetime] = None

    def update_bar(self, event: BarEvent) -> Optional[CorrelationEvent]:
        """Tick the period-bucketed cadence; on the boundary run a refresh
        and return its CorrelationEvent (else None). Forming bars are
        skipped. step_size=0 disables auto-refresh. The counter resets
        after every refresh regardless of outcome."""
        if event.is_forming or self.step_size == 0:
            return None
        period_start = get_period_start(event.timestamp, self.timeframe)
        if period_start == self._last_seen_period_start:
            return None
        self._last_seen_period_start = period_start
        self._periods_since_refresh += 1
        if self._periods_since_refresh < self.step_size:
            return None
        self._periods_since_refresh = 0
        return self._refresh(event.timestamp)

    def _candidates(self) -> List[str]:
        """Live symbols plus re-measurement candidates: symbols excluded
        SOLELY for 'constant_price' (they must enter the window pull or
        constancy could never be re-measured)."""
        out = []
        for s in self.universe_manager.symbol_list:
            st = self.universe_manager.status(s)
            if st.live or st.reasons == ['constant_price']:
                out.append(s)
        return out

    def _refresh(self, timestamp: Any) -> CorrelationEvent:
        """Run one refresh (spec §6.3); always returns a CorrelationEvent."""
        self.universe_manager.reassess_all()
        candidates = self._candidates()
        if len(candidates) == 0:
            logger.info("correlation refresh: no candidates (universe empty)")
            return self._publish(timestamp, None, 'empty_universe')
        if len(candidates) == 1:
            return self._publish(timestamp, None, 'singleton')
        return self._refresh_full(timestamp, candidates)   # Task 6

    def _refresh_full(self, timestamp: Any,
                      candidates: List[str]) -> CorrelationEvent:
        """Window pull, mode transform, constancy re-measurement (mark new
        constants / clear moved ones — the 'constant_price' lifecycle this
        manager owns), then estimate -> shrink -> floor -> PSD repair."""
        closes = {
            s: self.data_handler.get_latest_bars_df(
                s, self.lookback, timeframe=self.timeframe)['Close']
            for s in candidates
        }
        frame = pd.DataFrame(closes)
        if self.mode == 'simple_return':
            returns = frame.pct_change(fill_method=None).dropna()
        elif self.mode == 'absolute_price_chg':
            returns = frame.diff().dropna()
        else:
            raise ValueError(f"Unexpected mode: {self.mode!r}")
        if len(returns) < _MIN_CORR_OBS:
            logger.warning(
                "correlation refresh: only %d valid return observations "
                "(need >= %d); publishing insufficient_observations",
                len(returns), _MIN_CORR_OBS,
            )
            return self._publish(timestamp, None, 'insufficient_observations')

        # Constancy re-measurement over the candidates in the window.
        variances = returns.var(ddof=0)
        constant = [c for c in returns.columns if variances[c] == 0.0]
        if constant:
            logger.warning(
                "correlation refresh: %d constant-price symbol(s) excluded "
                "this refresh: %s", len(constant), constant,
            )
        for s in constant:
            self.universe_manager.mark_excluded(s, 'constant_price', timestamp)
        for s in returns.columns:
            if s not in constant:
                self.universe_manager.clear_excluded(s, 'constant_price',
                                                     timestamp)
        returns = returns.drop(columns=constant)
        if returns.shape[1] < 2:
            logger.warning(
                "correlation refresh: fewer than 2 non-constant symbols; "
                "publishing too_few_symbols",
            )
            return self._publish(timestamp, None, 'too_few_symbols')

        corr = correlation_matrix(returns, shrinkage=self.shrinkage)
        if self.shrinkage is not None:
            logger.debug(
                "correlation refresh: %s shrinkage intensity %.4f over %d "
                "observations", self.shrinkage,
                corr.attrs.get('lw_shrinkage', float('nan')), len(returns),
            )
        if corr.isna().any().any():
            logger.warning(
                "correlation refresh: matrix contains NaN despite "
                "zero-variance filtering; publishing nan_fallback",
            )
            return self._publish(timestamp, None, 'nan_fallback')
        if self.floor is not None:
            corr = corr.clip(lower=self.floor)
        return self._publish(timestamp, self._nearest_psd_correlation(corr),
                             'ok')

    @staticmethod
    def _nearest_psd_correlation(corr: pd.DataFrame) -> pd.DataFrame:
        """Project ``corr`` back to a valid (PSD) correlation matrix.

        Cheap ``eigvalsh`` check first: PSD input is returned unchanged
        (the common case — repair only triggers when ``corr_floor``
        clipping actually broke PSD-ness). Otherwise: clip negative
        eigenvalues to zero, reconstruct, rescale to a unit diagonal
        (a congruence transform, so PSD-ness is preserved exactly), and
        re-symmetrize. The result is a small perturbation of the input,
        not a rebuild — labels and the unit diagonal are preserved.
        """
        vals = corr.to_numpy(dtype=float)
        if float(np.linalg.eigvalsh(vals)[0]) >= 0.0:
            return corr
        eigvals, eigvecs = np.linalg.eigh(vals)
        repaired = (eigvecs * np.clip(eigvals, 0.0, None)) @ eigvecs.T
        d = np.sqrt(np.diag(repaired))
        repaired = repaired / np.outer(d, d)
        repaired = 0.5 * (repaired + repaired.T)
        np.fill_diagonal(repaired, 1.0)
        return pd.DataFrame(repaired, index=corr.index, columns=corr.columns)

    def _publish(self, timestamp: Any, matrix: Optional[pd.DataFrame],
                 reason: str) -> CorrelationEvent:
        """Record + build the event; live_symbols snapshot is taken HERE
        (post any constancy marks/clears of this refresh)."""
        self.matrix = matrix
        self.last_refresh_timestamp = timestamp
        return CorrelationEvent(
            timestamp=timestamp, matrix=matrix,
            live_symbols=self.universe_manager.get_live_symbols(),
            reason=reason,
        )
