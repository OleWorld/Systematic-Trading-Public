"""UniverseStatus — per-symbol tradable-universe record."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class UniverseStatus:
    """Snapshot of one symbol's universe state.

    ``live`` (no liveness reason present) and ``excluded`` (an exclusion
    mark present) are DERIVED by ``UniverseManager`` — never set
    independently, so contradictory states cannot exist. ``reasons`` is
    canonically ordered: ``'warmup_forecast'``, ``'warmup_history'``,
    then exclusion marks in mark order. ``reasons == []`` ⇔ live.
    Mutate only through the manager; ``UniverseManager.status`` returns
    a defensive copy.
    """

    live: bool
    excluded: bool
    reasons: List[str] = field(default_factory=list)
