"""universe — dynamic tradable-universe state, history, and events.

Submodules (internal):
    _status    UniverseStatus dataclass
    _manager   UniverseManager (gates, exclusion marks, transition log)
"""

from universe._manager import UniverseManager
from universe._status import UniverseStatus

__all__ = ["UniverseManager", "UniverseStatus"]
