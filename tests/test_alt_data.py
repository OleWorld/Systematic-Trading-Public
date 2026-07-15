"""
Unit + integration tests for alternative-data (alt feed) ingestion.

Covers, smallest first:

  Section 1 — `AltRecord` storage type (immutability, copy semantics).
  Section 2 — `DataHandler` base: alt deques, `_append_alt` gates, accessors.
  Section 3 — `HistoricDataHandler` construction gates for `alt_data`.
  Section 4 — stream merge + `update_bar` ingestion (the causality contract).
  Section 5 — end-to-end integration through a real `Backtester`.

Run from the repo root:  pytest tests/test_alt_data.py -v
"""

import dataclasses
import datetime
import queue as thread_queue
from typing import Any, List

import pandas as pd
import pytest

from data import AltRecord
from data._base import DataHandler
from data._historic import HistoricDataHandler

UTC = datetime.timezone.utc
TS = datetime.datetime(2026, 1, 1, tzinfo=UTC)


# ──────────────────────────────────────────────
# Test doubles & helpers
# ──────────────────────────────────────────────

class FakeQueue:
    """Captures every put() for inspection."""

    def __init__(self):
        self.items: List[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


# ──────────────────────────────────────────────
# Section 1 — AltRecord storage type
# ──────────────────────────────────────────────

def test_altrecord_fields_are_frozen():
    rec = AltRecord(TS, {'rate': 0.01})
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.timestamp = TS + datetime.timedelta(days=1)


def test_altrecord_values_mapping_is_read_only():
    rec = AltRecord(TS, {'rate': 0.01})
    with pytest.raises(TypeError):
        rec.values['rate'] = 99.0


def test_altrecord_copies_the_caller_dict():
    d = {'rate': 0.01}
    rec = AltRecord(TS, d)
    d['rate'] = 99.0
    assert rec.values['rate'] == 0.01


def test_altrecord_value_access_and_equality():
    a = AltRecord(TS, {'rate': 0.01, 'oi': 5.0})
    b = AltRecord(TS, {'rate': 0.01, 'oi': 5.0})
    assert a.values['oi'] == 5.0
    assert a == b
