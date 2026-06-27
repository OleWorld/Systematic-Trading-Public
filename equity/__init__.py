"""equity — Single-name equity-valuation research helpers.

One-shot, fully parameter-fed calculators for fundamental equity
analysis — called from research notebooks, not on every bar. Distinct
from ``analytics/`` (systematic-trading portfolio math): this package
reasons about a single company's valuation, not a basket's risk.

Public surface:

* ``reverse_dcf_cagr(*, name=None, market_price, shares_outstanding,
  base_revenue, fcf_margin=None, base_fcf=None, wacc, terminal_growth,
  horizon, net_debt=0.0, axis_a, axis_a_values, axis_b, axis_b_values,
  cagr_bounds=(-0.5, 1.0))`` — reverse two-stage FCFF DCF. Holds the
  market price fixed and solves for the revenue CAGR it implies, swept
  across a 2-D grid of two chosen model inputs. Rate inputs (``wacc``,
  ``terminal_growth``, ``fcf_margin``, ``cagr_bounds``) are fractions
  (0.08 = 8 %); the output heatmap is in percentage points (38.9 = 38.9 %).
  Supply the FCF level as either ``fcf_margin`` or a dollar ``base_fcf``
  (margin derived); pass an optional ``name`` to label the run. Returns a
  ``ReverseDCFResult`` (the heatmap DataFrame plus the consensus cell /
  CAGR and a ``.styled()`` Styler that highlights the consensus cell).
* ``ReverseDCFResult`` — the frozen result dataclass.
"""

from equity._reverse_dcf import ReverseDCFResult, reverse_dcf_cagr

__all__ = [
    'ReverseDCFResult',
    'reverse_dcf_cagr',
]
