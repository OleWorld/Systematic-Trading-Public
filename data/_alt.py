"""Internal storage type for DataHandler alt-feed rolling windows.

``AltRecord`` is the in-memory format for alternative-data records
(funding rates, open interest, liquidations, ...) held inside the
per-``(symbol, feed)`` deques maintained by ``DataHandler``. Mirrors
``Bar``: frozen/immutable; only ``DataHandler`` is expected to construct
instances. External consumers read records via
``DataHandler.get_latest_alt()`` (a ``List[AltRecord]``, oldest→newest)
or ``DataHandler.get_latest_alt_df()`` (the DataFrame counterpart).

Unlike bars, alt records carry NO forming/completed distinction — every
record is atomic and final on arrival, so ``[-1]`` of an alt window is a
finalized record (bar windows keep the forming bar at ``[-1]``).
"""
import datetime
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class AltRecord:
    """Immutable multi-field alternative-data record.

    ``values`` maps field name → float (e.g. ``{'rate': -0.0001}``). At
    construction the mapping is copied and wrapped in a read-only
    ``MappingProxyType``, so neither the dataclass fields nor the field
    values can be mutated by a consumer holding a reference — and later
    mutation of the caller's original dict cannot reach into the record.
    """
    timestamp: datetime.datetime
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        """Replace ``values`` with a read-only proxy over a private copy."""
        object.__setattr__(self, 'values', MappingProxyType(dict(self.values)))
