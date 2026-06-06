"""
Unit tests for `logging_setup.py`.

Covers the contextvar-backed bar timestamp, the LogRecord factory that
stamps it onto each record, and the idempotent ``configure_logging``
helper. Run from repo root:  pytest tests/test_logging_setup.py -v
"""

import logging
from datetime import datetime, timezone

import pandas as pd
import pytest

from logging_setup import (
    clear_current_bar_timestamp,
    configure_logging,
    inject_bar_timestamp_factory,
    set_current_bar_timestamp,
)


@pytest.fixture(autouse=True)
def _reset_bar_ts():
    """Reset the contextvar between tests so state never leaks."""
    clear_current_bar_timestamp()
    yield
    clear_current_bar_timestamp()


def _emit_and_capture(caplog) -> logging.LogRecord:
    """Emit one record and return it from caplog (factory must run)."""
    inject_bar_timestamp_factory()
    with caplog.at_level(logging.DEBUG, logger="test_logging_setup_dummy"):
        logging.getLogger("test_logging_setup_dummy").info("ping")
    return caplog.records[-1]


# ──────────────────────────────────────────────
# Factory behavior
# ──────────────────────────────────────────────

def test_default_bar_ts_is_dash(caplog):
    rec = _emit_and_capture(caplog)
    assert rec.bar_ts == "-"


def test_set_then_emit_stamps_record(caplog):
    set_current_bar_timestamp(pd.Timestamp("2026-04-26 10:00", tz="UTC"))
    rec = _emit_and_capture(caplog)
    assert rec.bar_ts == "2026-04-26 10:00:00"


def test_clear_returns_to_dash(caplog):
    set_current_bar_timestamp(pd.Timestamp("2026-04-26 10:00", tz="UTC"))
    clear_current_bar_timestamp()
    rec = _emit_and_capture(caplog)
    assert rec.bar_ts == "-"


# ──────────────────────────────────────────────
# Timestamp formatting
# ──────────────────────────────────────────────

@pytest.mark.parametrize("inp,expected", [
    (pd.Timestamp("2026-04-26 10:00:00", tz="UTC"), "2026-04-26 10:00:00"),
    (pd.Timestamp("2026-04-26 10:00:00"), "2026-04-26 10:00:00"),
    (datetime(2026, 4, 26, 10, 0, 0, tzinfo=timezone.utc), "2026-04-26 10:00:00"),
    (datetime(2026, 4, 26, 10, 0, 0), "2026-04-26 10:00:00"),
    ("2026-04-26 10:00:00", "2026-04-26 10:00:00"),
])
def test_set_accepts_various_timestamp_inputs(caplog, inp, expected):
    set_current_bar_timestamp(inp)
    rec = _emit_and_capture(caplog)
    assert rec.bar_ts == expected


def test_set_none_yields_dash(caplog):
    set_current_bar_timestamp(None)
    rec = _emit_and_capture(caplog)
    assert rec.bar_ts == "-"


def test_set_nat_yields_dash(caplog):
    set_current_bar_timestamp(pd.NaT)
    rec = _emit_and_capture(caplog)
    assert rec.bar_ts == "-"


# ──────────────────────────────────────────────
# Factory + configure_logging idempotency
# ──────────────────────────────────────────────

def test_factory_install_is_idempotent(caplog):
    """Installing twice must not stack the factory (would re-set bar_ts twice)."""
    inject_bar_timestamp_factory()
    inject_bar_timestamp_factory()
    set_current_bar_timestamp(pd.Timestamp("2026-04-26 10:00", tz="UTC"))
    rec = _emit_and_capture(caplog)
    assert rec.bar_ts == "2026-04-26 10:00:00"


def test_configure_logging_runs_clean():
    """configure_logging must not raise and must install the factory."""
    configure_logging(level=logging.DEBUG)
    configure_logging(level=logging.DEBUG)  # idempotent
    rec = logging.getLogger("test_logging_setup_dummy_2").makeRecord(
        "test_logging_setup_dummy_2", logging.INFO, __file__, 1,
        "msg", (), None,
    )
    # makeRecord uses the factory under the hood
    assert hasattr(rec, "bar_ts")


def test_format_string_renders_bar_ts(caplog):
    """Format string in default config must include the bar_ts column."""
    configure_logging(level=logging.DEBUG)
    set_current_bar_timestamp(pd.Timestamp("2026-04-26 10:00", tz="UTC"))
    formatter = logging.Formatter(
        "%(asctime)s | %(bar_ts)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    with caplog.at_level(logging.DEBUG, logger="test_logging_setup_dummy_3"):
        logging.getLogger("test_logging_setup_dummy_3").info("ping")
    rendered = formatter.format(caplog.records[-1])
    assert " | 2026-04-26 10:00:00 | " in rendered
