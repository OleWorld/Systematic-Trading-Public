"""Orchestrator — multi-strategy forecast aggregator ("PM over traders").

A composite *forecast oracle*: it wraps N strategies, drives each one on
every bar (so each strategy behaves exactly as it does standalone — its
own forecast logic, its own instrument universe), and exposes a single
**combined** forecast per symbol. The risk manager points at the
orchestrator instead of a single strategy.

The orchestrator satisfies the same read surface a risk manager needs
from a forecast source — ``symbol_list`` / ``get_forecast(symbol)`` /
``is_warmed_up(symbol)`` (the ``RiskManager`` ``_StrategyLike`` /
``_OrchestratorLike`` contract) — plus ``update_bar(event)`` (driven by
the engine in the strategy slot) and ``get_records(symbol)`` (per-bar
diagnostics). It is a drop-in: passing it where a ``Strategy`` used to go
needs no engine or risk-manager code change.

Combination math (Carver, applied one level up from a strategy's own
variation combination)::

    combined(symbol) = clamp( FDM × Σ wᵢ_renorm · forecastᵢ(symbol),
                              ±FORECAST_CAP )

where the sum runs over the strategies that *contribute* to ``symbol`` —
those that include it in their ``symbol_list`` AND currently return a
non-``None`` forecast for it. Their per-strategy weights ``wᵢ`` are
**renormalized to sum to 1 over that contributing subset**, so a symbol
traded by a single strategy passes that strategy's forecast through at
full strength (renormalized weight → 1), and a symbol whose contributors
are still warming up combines only the ones that are ready. ``FDM`` (the
cross-strategy Forecast Diversification Multiplier) is applied only when
there are **two or more** contributors — a lone forecast earns no
diversification uplift (mirroring ``analytics.diversification_multiplier``
returning ``1.0`` for ``N=1``); the ``±FORECAST_CAP`` clamp is a separate
safety net.

``is_warmed_up(symbol)`` flips True once any contributor has produced a
forecast for the symbol (the first time a combined forecast is cached);
``get_forecast(symbol)`` returns ``None`` until then — the same warmup
convention strategies use, so the risk manager's liveness gate treats the
orchestrator exactly as it treats a single strategy.

Strategy weighting lives **here** (the orchestrator is the sole owner):
the weights are baked into the combined forecast, so the risk manager no
longer applies a strategy weight of its own. Because vol-target sizing is
linear in the forecast, forecast-level weighting is equivalent to
position-level weighting.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analytics import equal_weight, validate_named_weights
from event import BarEvent
from strategy import Strategy

__all__ = ['Orchestrator']

logger = logging.getLogger(__name__)

# Weights summing to within this of 1.0 are accepted (matches the
# ``analytics.diversification_multiplier`` weight-sum convention).
_WEIGHT_SUM_TOL = 1e-9


class Orchestrator:
    """Aggregate multiple strategies' forecasts into one combined forecast.

    Drop-in forecast source for a ``RiskManager``: exposes ``symbol_list``
    / ``get_forecast`` / ``is_warmed_up`` and is driven by the engine via
    ``update_bar`` (passed in the strategy slot of both the risk manager
    and the ``Backtester``).

    Parameters
    ----------
    strategies
        Mapping ``{label: strategy}`` of the managed strategies. Labels
        are unique by construction (dict keys) and key the per-strategy
        weights and the per-bar diagnostic columns; using a label (rather
        than the class name) lets two instances of the same class
        (e.g. two EWMACs with different look-backs) coexist. Each value
        must expose ``symbol_list``, ``get_forecast(symbol)`` and
        ``update_bar(event)``. Must be non-empty.
    weights
        Optional ``{label: weight}`` per-strategy capital weights. Keys
        must match ``strategies`` exactly; values must be finite and
        non-negative and sum to ``1.0`` within ``1e-9``. Default
        ``None`` → equal weight ``1/M`` via ``analytics.equal_weight``.
        These are the *global* weights; per symbol they are renormalized
        over the contributing subset (see the module docstring).
    fdm
        Cross-strategy Forecast Diversification Multiplier. Applied only
        when a symbol has two or more contributors. Must be finite and
        ``> 0``. Default ``1.0`` (no adjustment). Auto-derivation from
        inter-strategy correlation is a future extension.

    Raises
    ------
    ValueError
        On an empty ``strategies``, a ``weights`` key mismatch, a
        negative / non-finite / non-sum-to-1 weight, or a non-positive
        ``fdm``.
    TypeError
        If a strategy value lacks the required ``symbol_list`` /
        ``get_forecast`` / ``update_bar`` surface.
    """

    def __init__(
        self,
        strategies: Dict[str, Strategy],
        weights: Optional[Dict[str, float]] = None,
        fdm: float = 1.0,
    ):
        if not isinstance(strategies, dict) or not strategies:
            raise ValueError(
                "strategies must be a non-empty dict of {label: strategy}"
            )
        for label, strat in strategies.items():
            if not (
                hasattr(strat, 'symbol_list')
                and callable(getattr(strat, 'get_forecast', None))
                and callable(getattr(strat, 'update_bar', None))
            ):
                raise TypeError(
                    f"strategy {label!r} must expose symbol_list, "
                    f"get_forecast(symbol) and update_bar(event)"
                )
        labels = list(strategies.keys())

        if not (np.isfinite(fdm) and fdm > 0):
            raise ValueError(f"fdm must be finite and > 0, got {fdm}")

        self.strategies: Dict[str, Strategy] = dict(strategies)
        self.fdm: float = float(fdm)
        # Per-strategy capital weights (the orchestrator owns strategy
        # weighting — see module docstring). Plural name to avoid
        # confusion with the risk manager's removed ``strategy_weight``.
        self.strategy_weights: Dict[str, float] = self._validate_weights(
            weights, labels,
        )

        # symbol_list is the sorted union of every child's universe — a
        # symbol is managed if at least one strategy trades it.
        union: set = set()
        for strat in strategies.values():
            union.update(strat.symbol_list)
        self.symbol_list: List[str] = sorted(union)

        # Per-symbol combined-forecast cache. None means "no combined
        # forecast cached yet" (warmup) — distinct from a flat 0.0, and
        # the same convention ``Strategy`` uses.
        self._forecasts: Dict[str, Optional[float]] = {
            s: None for s in self.symbol_list
        }
        # Monotone warmup flag: True once a combined forecast is cached.
        self._warmed_up: Dict[str, bool] = {s: False for s in self.symbol_list}
        # Per-symbol list of diagnostic row dicts (see get_records).
        self._records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    @staticmethod
    def _validate_weights(
        weights: Optional[Dict[str, float]], labels: List[str],
    ) -> Dict[str, float]:
        """Validate / default the per-strategy weights, keyed by ``labels``.

        ``None`` → equal weight ``1/M``. Otherwise the keys must match
        ``labels`` exactly and the values must be finite, non-negative,
        and sum to ``1.0`` within ``_WEIGHT_SUM_TOL``. Thin delegator to
        the shared ``analytics.validate_named_weights`` so strategy,
        instrument, and forecast-variation weights all validate identically.
        """
        return validate_named_weights(weights, labels, sum_tol=_WEIGHT_SUM_TOL)

    def calculate_strategy_weights(self) -> None:
        """Reset ``self.strategy_weights`` to equal weight over current labels.

        Explicit recalc hook (mirrors the risk manager's old
        ``calculate_strategy_weight``); ``self.strategy_weights`` may also
        be overwritten directly. Mutates in place; safe to re-call any time.
        """
        self.strategy_weights = equal_weight(list(self.strategies.keys()))

    def get_budget_groups(self) -> Dict[str, Tuple[float, List[str]]]:
        """Return the budget-group structure for the risk manager's budget layer.

        ``{label: (budget_weight, universe)}`` — one entry per managed
        strategy: its current ``strategy_weights`` value (the share of the
        portfolio risk budget that strategy's book consumes) and a copy of
        its declared ``symbol_list``. A stateless view over existing state
        (no new attributes): the weights are read fresh on every call, so a
        direct ``strategy_weights`` overwrite propagates at the risk
        manager's next weight recalc. Consumed — via ``hasattr`` — by
        ``VolTargetingRiskManager.calculate_instrument_weight`` to build
        strategy-budgeted instrument weights (sum-of-books); a bare
        ``Strategy`` lacks this method and gets a single implicit group.
        """
        return {
            label: (self.strategy_weights[label], list(strat.symbol_list))
            for label, strat in self.strategies.items()
        }

    def update_bar(self, event: BarEvent) -> None:
        """Drive every child, then recompute the combined forecast.

        Forwards ``event`` to each strategy's ``update_bar`` (each
        self-filters by its own ``symbol_list``), then — if the bar's
        symbol is managed — combines the children's forecasts, clamps to
        ``±FORECAST_CAP``, caches it (and flips the warmup flag) when
        non-``None``, and records one diagnostic row. Per-event and
        idempotent (children handle forming/completed idempotency), so no
        forming-bar gating is needed; the risk manager gates on
        ``is_forming`` downstream.
        """
        for strat in self.strategies.values():
            strat.update_bar(event)

        symbol = event.symbol
        if symbol not in self._forecasts:
            return

        combined, diag = self._combine(symbol)

        if combined is not None:
            cap = Strategy.FORECAST_CAP
            clamped = max(-cap, min(cap, float(combined)))
            self._forecasts[symbol] = clamped
            self._warmed_up[symbol] = True
            forecast_val: float = clamped
        else:
            forecast_val = float('nan')

        row: Dict[str, Any] = {
            'timestamp': event.timestamp,
            'open': event.open,
            'high': event.high,
            'low': event.low,
            'close': event.close,
            'volume': event.volume,
            'forecast': forecast_val,
            'fdm': diag['fdm'],
            'n_contributors': diag['n_contributors'],
        }
        row.update(diag['per_label'])
        self._records[symbol].append(row)

    def _combine(self, symbol: str) -> Tuple[Optional[float], Dict[str, Any]]:
        """Combine the children's forecasts for ``symbol`` (the core math).

        Contributors are the strategies that cover ``symbol`` and return a
        non-``None`` forecast for it. Their weights are renormalized to
        sum to 1; the combined forecast is ``effective_fdm × Σ wᵢ·fᵢ``,
        with ``effective_fdm`` the configured ``fdm`` only when there are
        ≥2 contributors (else ``1.0``). Returns ``(None, diag)`` when there
        are no contributors or their summed weight is 0 (e.g. all
        contributors carry weight 0). ``diag`` carries ``fdm`` (effective,
        ``None`` when no combine), ``n_contributors``, and a ``per_label``
        dict of ``forecast_<label>`` (raw child forecast, NaN if absent /
        warming) and ``weight_<label>`` (renormalized weight used, 0.0 if
        not contributing).
        """
        per_label: Dict[str, Any] = {}
        contributions: Dict[str, float] = {}
        for label, strat in self.strategies.items():
            covers = symbol in strat.symbol_list
            f = strat.get_forecast(symbol) if covers else None
            if covers and f is not None and not pd.isna(f):
                fval = float(f)
                contributions[label] = fval
                per_label[f'forecast_{label}'] = fval
            else:
                per_label[f'forecast_{label}'] = float('nan')
            per_label[f'weight_{label}'] = 0.0

        diag: Dict[str, Any] = {
            'fdm': None,
            'n_contributors': len(contributions),
            'per_label': per_label,
        }

        if not contributions:
            return None, diag
        total_w = sum(self.strategy_weights[label] for label in contributions)
        if total_w <= 0:
            # All contributors carry zero weight — no effective forecast.
            return None, diag

        weighted_sum = 0.0
        for label, fval in contributions.items():
            w = self.strategy_weights[label] / total_w
            per_label[f'weight_{label}'] = w
            weighted_sum += w * fval

        effective_fdm = self.fdm if len(contributions) >= 2 else 1.0
        diag['fdm'] = effective_fdm
        return effective_fdm * weighted_sum, diag

    def get_forecast(self, symbol: str) -> Optional[float]:
        """Return the cached combined forecast for ``symbol`` (or ``None``).

        ``None`` means no combined forecast has been cached yet (no
        contributor has warmed up) — the same warmup convention as
        ``Strategy.get_forecast``. Unknown symbols return ``None``.
        """
        return self._forecasts.get(symbol)

    def is_warmed_up(self, symbol: str) -> bool:
        """Return True once a combined forecast has been cached for ``symbol``.

        Flips True the first time any contributor produces a forecast for
        the symbol, and never resets (monotone). Unknown symbols return
        ``False``. Consumed by the risk manager's universe liveness gate.
        """
        return self._warmed_up.get(symbol, False)

    def get_records(self, symbol: str) -> pd.DataFrame:
        """Return per-bar combination diagnostics for ``symbol``.

        Indexed by ``timestamp``; columns are OHLCV plus ``forecast``
        (combined, clamped; NaN while warming), ``fdm`` (effective),
        ``n_contributors``, and per-strategy ``forecast_<label>`` /
        ``weight_<label>``. Empty DataFrame if nothing recorded yet.
        Each child's own records remain reachable via
        ``orchestrator.strategies[label].get_records(symbol)``.
        """
        rows = self._records.get(symbol)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.set_index('timestamp', inplace=True)
        return df
