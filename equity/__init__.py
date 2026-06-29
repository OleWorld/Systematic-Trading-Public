"""equity — Single-name equity-valuation research helpers.

One-shot, fully parameter-fed calculators for fundamental equity
analysis — called from research notebooks, not on every bar. Distinct
from ``analytics/`` (systematic-trading portfolio math): this package
reasons about a single company's valuation, not a basket's risk.

Public surface:

* ``reverse_dcf_cagr(*, name=None, market_price, shares_outstanding,
  base_revenue, fcf_margin=None, base_fcf=None, wacc, terminal_growth,
  horizon, net_debt=0.0, segments=None, axis_a, axis_a_values, axis_b,
  axis_b_values, cagr_bounds=(-0.5, 1.0))`` — reverse two-stage FCFF DCF.
  Holds the market price fixed and solves for the revenue CAGR it implies,
  swept across a 2-D grid of two chosen model inputs. Rate inputs (``wacc``,
  ``terminal_growth``, ``fcf_margin``, ``cagr_bounds``) are fractions
  (0.08 = 8 %); the output heatmap is in percentage points (38.9 = 38.9 %).
  Supply the FCF level as either ``fcf_margin`` or a dollar ``base_fcf``
  (margin derived); pass an optional ``name`` to label the run. Pass
  ``segments=[Segment(...), ...]`` to model the company as a mix of streams
  with different growth and margins and reverse-solve the one **swing**
  segment's growth (``growth=None``) — e.g. "what growth is the market
  pricing into Nokia's AI segment?" — addressing segment inputs on sweep
  axes via dotted ``'<segment>.<field>'`` names. Returns a
  ``ReverseDCFResult`` (the heatmap DataFrame plus the consensus cell /
  CAGR, the swing segment name, and a ``.styled()`` Styler that highlights
  the consensus cell).
* ``Segment`` — a frozen ``(name, revenue_share, fcf_margin, growth=None)``
  business-segment input for the multi-segment path.
* ``ReverseDCFResult`` — the frozen result dataclass.
"""

from equity._reverse_dcf import ReverseDCFResult, Segment, reverse_dcf_cagr

__all__ = [
    'ReverseDCFResult',
    'Segment',
    'reverse_dcf_cagr',
]
