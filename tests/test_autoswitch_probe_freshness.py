"""A header-probe reading must be fresh before its account becomes a target.

Incident (2026-11-08): the active account crossed its line and the proactive
escape picked a setup-token account whose stored header-probe reading said
5h 36%. That reading was ~30 minutes old, still decision-trusted (a live
claim or a future ``nextPollAt`` extends trust up to ``TRUST_MAX_AGE_S``), and
the proactive/at-limit/failover path has no per-target freshness gate (only
consume-first and priority do). The next tick's probe was refused with a
Retry-After of 3 h: the account was at 112%, and every session that had just
been moved onto it hit "5-hour limit".

A setup-token account is consumed by sessions cswap does not see (its own
``cswap run`` profiles), and its reading only ever comes from the probe, so
an old reading is no evidence of headroom. The rule: a candidate whose
reading comes from the header probe (``source: headers``) is a switch target
only when that reading is no older than ``SERVE_TTL_S`` -- the escalation
refetch is what freshens it; a candidate it could not freshen (live claim,
backoff, hold) is unknown this tick. A fresh refused probe already carries a
100% ``rejected`` window and stays unselectable.

Synthetic fixtures only: ``EngineHarness`` seeds a throwaway store under
``temp_home``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from claude_swap import oauth
from claude_swap.autoswitch import NoSwitchEvent, SwitchEvent, TickOutcome
from claude_swap.poll_policy import SERVE_TTL_S
from claude_swap.usage_store import UsageEntry

from tests.test_autoswitch import EngineHarness, _usage

STALE_AGE_S = 1800.0  # the incident's ~30 min old reading


def _probe(pct5: float, *, rejected: bool = False) -> dict:
    value = _usage(pct5)
    value["source"] = oauth.USAGE_SOURCE_HEADERS
    if rejected:
        value["rejected"] = {"retry_after_s": 11286.0, "resets_at": None}
    return value


def _fresh(value: dict, now: float) -> UsageEntry:
    return UsageEntry(last_good=value, fetched_at=now, age_s=0.0)


def _stale_trusted(value: dict, now: float, age: float = STALE_AGE_S) -> UsageEntry:
    """What the store serves for a row the escalation could not refetch:
    old, yet decision-trusted (live claim / future plan -> trust_extended)."""
    return UsageEntry(
        last_good=value,
        fetched_at=now - age,
        age_s=age,
        trust_extended=True,
        next_poll_at=now + 300.0,
    )


@pytest.fixture
def h(temp_home) -> EngineHarness:
    harness = EngineHarness(temp_home)
    harness.seed(1, "a@example.com")
    harness.seed(2, "token-9@token.local")
    harness.seed(3, "c@example.com")
    harness.make_live("a@example.com", 1)
    return harness


def _reasons(h: EngineHarness) -> list[str]:
    return [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]


class TestStaleProbeReadingIsNotATarget:
    def test_stale_low_probe_loses_to_a_fresh_candidate(self, h):
        """Red arm of the incident: #2's stale 36% outranks #3's fresh 60%
        on headroom; it must not be picked."""
        now = h.clock.now
        outcome = h.tick_with_entries({
            "1": _fresh(_usage(95), now),
            "2": _stale_trusted(_probe(36), now),
            "3": _fresh(_usage(60), now),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_only_a_stale_probe_candidate_means_no_switch(self, h):
        """No fresh evidence anywhere: hold (BLOCKED, normal cadence) rather
        than move every session onto an account that may be at its limit."""
        now = h.clock.now
        outcome = h.tick_with_entries({
            "1": _fresh(_usage(95), now),
            "2": _stale_trusted(_probe(36), now),
            "3": _fresh(_usage(97), now),
        })
        assert outcome is TickOutcome.BLOCKED
        assert h.active_number() == 1
        assert not [e for e in h.events if isinstance(e, SwitchEvent)]

    @pytest.mark.parametrize("active_pct", [95, 100])
    def test_at_limit_and_proactive_both_refuse_the_stale_probe(self, h, active_pct):
        now = h.clock.now
        outcome = h.tick_with_entries({
            "1": _fresh(_usage(active_pct), now),
            "2": _stale_trusted(_probe(10), now),
            "3": _fresh(_usage(98), now),
        })
        # at-limit may legitimately escape to #3 (2% left); #2 never.
        assert h.active_number() != 2
        if active_pct < 100:
            assert outcome is TickOutcome.BLOCKED
            assert h.active_number() == 1

    def test_failover_refuses_the_stale_probe(self, h):
        now = h.clock.now
        entries = {
            "1": UsageEntry(),  # active unreadable
            "2": _stale_trusted(_probe(10), now),
            "3": UsageEntry(),
        }
        outcomes = [
            h.tick_with_entries(entries)
            for _ in range(h.settings.unhealthy_ticks)
        ]
        assert TickOutcome.SWITCHED not in outcomes
        assert h.active_number() == 1

    def test_fresh_refused_probe_is_never_a_target(self, h):
        """The probe at 01:01:53: refused with Retry-After -> 100% window."""
        now = h.clock.now
        outcome = h.tick_with_entries({
            "1": _fresh(_usage(95), now),
            "2": _fresh(_probe(0, rejected=True), now),
            "3": _fresh(_usage(98), now),
        })
        assert outcome is not TickOutcome.SWITCHED
        assert h.active_number() == 1


class TestFreshProbeReadingStillSwitches:
    def test_fresh_probe_reading_is_a_normal_target(self, h):
        now = h.clock.now
        outcome = h.tick_with_entries({
            "1": _fresh(_usage(95), now),
            "2": _fresh(_probe(36), now),
            "3": _fresh(_usage(60), now),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_probe_reading_inside_the_serve_ttl_counts_as_fresh(self, h):
        now = h.clock.now
        outcome = h.tick_with_entries({
            "1": _fresh(_usage(95), now),
            "2": _stale_trusted(_probe(36), now, age=SERVE_TTL_S),
            "3": _fresh(_usage(60), now),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_stale_endpoint_reading_is_untouched(self, h):
        """Scope: only header-probe readings. An endpoint reading keeps the
        store's decision trust exactly as before."""
        now = h.clock.now
        outcome = h.tick_with_entries({
            "1": _fresh(_usage(95), now),
            "2": _stale_trusted(_usage(36), now),
            "3": _fresh(_usage(60), now),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_stale_probe_on_the_active_account_still_decides_departure(self, h):
        """The gate is about targets. The active account's own trusted
        reading keeps driving the departure decision."""
        now = h.clock.now
        outcome = h.tick_with_entries({
            "1": _stale_trusted(_probe(95), now),
            "2": _fresh(_usage(20), now),
            "3": _fresh(_usage(60), now),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2


class TestThroughTheRealStore:
    """End to end: the escalation cannot refetch #2 because another collector
    holds a live claim on it (the store then extends trust to the old row),
    so the old 36% is what the ranking sees."""

    @pytest.fixture(autouse=True)
    def _no_profile_probe(self):
        with patch("claude_swap.oauth.fetch_oauth_profile", return_value=None):
            yield

    def test_claimed_stale_probe_row_is_not_picked(self, h, monkeypatch):
        monkeypatch.setattr("claude_swap.switcher._FETCH_STAGGER_S", 0)
        monkeypatch.setattr(h.switcher, "_live_session_pids", lambda *a: [])
        store = h.switcher._usage_store
        from claude_swap.usage_store import FetchRecord

        idents = {
            "1": ("a@example.com", ""),
            "2": ("token-9@token.local", ""),
            "3": ("c@example.com", ""),
        }
        store.record({"2": FetchRecord(usage=_probe(36))}, {"2": idents["2"]})
        h.clock.advance(STALE_AGE_S)
        # Another surface (`cswap list`, a dashboard) is mid-fetch on #2.
        store.claim(["2"], {"2": idents["2"]})

        live = {"1": _usage(95), "2": _probe(112, rejected=True), "3": _usage(60)}
        fetched: list[str] = []

        def fake(num, email, creds, is_active=False, **kwargs):
            fetched.append(num)
            return oauth.UsageOutcome(dict(live[num]))

        with patch(
            "claude_swap.oauth.try_fetch_usage_for_account", side_effect=fake
        ):
            outcome = h.engine.tick()

        assert "2" not in fetched  # the claim kept the escalation off #2
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3
