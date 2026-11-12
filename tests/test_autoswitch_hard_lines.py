"""A per-account line is hard, the global line soft, manual switching free.

- ``cswap threshold <n> <pct>`` is a hard limit. An active account that
  reaches its own line leaves at once (trigger ``hard-line``: no cooldown, no
  hysteresis margin), and no trigger ever lands on an account at or over its
  own line. With nowhere to go the engine holds, reports ``hard-line-hold``
  (BLOCKED) and logs a warning.
- ``autoswitch.threshold`` / ``weeklyThreshold`` are soft. A proactive move
  lands only on an account under all of its lines; with none there the engine
  stays and lets the active run to its limit. The at-limit escape may then land
  on an account over a soft line that still has room.
- A target needs a reading from this tick (``SERVE_TTL_S``), whatever its
  source and trigger.
- ``cswap switch <n>`` ignores all of it.
- The active account near a line is polled every ``URGENT_INTERVAL_S`` when
  its reading comes from the header probe, moving or not.

The fleet shape replays the 2026-11-12 log: per-account lines of 70, header-
probe readings, active at 93% five-hour, one candidate at 5h 0% · 7d 79%
(over its 70 line on the weekly window) and one at 5h 78%. Synthetic
fixtures only.
"""

from __future__ import annotations

import logging

import pytest

from claude_swap import oauth, poll_policy
from claude_swap.autoswitch import NoSwitchEvent, SwitchEvent, TickOutcome
from claude_swap.usage_store import FetchRecord, UsageEntry

from tests.test_autoswitch import EngineHarness, _usage, _usage7


def _probe(pct5: float, pct7: float = 0.0) -> dict:
    value = _usage7(pct5, pct7)
    value["source"] = oauth.USAGE_SOURCE_HEADERS
    return value


def _fresh(value: dict, now: float) -> UsageEntry:
    return UsageEntry(last_good=value, fetched_at=now, age_s=0.0)


def _fleet(temp_home, n: int = 3, lines: dict[str, float] | None = None, **kw):
    h = EngineHarness(temp_home, **kw)
    for num in range(1, n + 1):
        h.seed(num, f"{num}@example.com")
    h.make_live("1@example.com", 1)
    for num, line in (lines or {}).items():
        h.switcher.set_account_threshold(num, line)
    h.engine = h._make_engine()
    return h


def _switches(h: EngineHarness) -> list[SwitchEvent]:
    return [e for e in h.events if isinstance(e, SwitchEvent)]


def _reasons(h: EngineHarness) -> list[str]:
    return [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]


class TestPerAccountLineIsHard:
    def test_the_measured_fleet_never_lands_over_a_line(self, temp_home, caplog):
        """Red arm of 2026-11-12 12:17:31 (17 -> 9): every candidate is over
        its own 70 line, so the engine stays and says why."""
        h = _fleet(temp_home, lines={"1": 70.0, "2": 70.0, "3": 70.0})
        now = h.clock.now
        with caplog.at_level(logging.INFO, logger="claude-swap"):
            outcome = h.tick_with_entries({
                "1": _fresh(_probe(93, 45), now),
                "2": _fresh(_probe(0, 79), now),   # 7d over its 70 line
                "3": _fresh(_probe(78, 13), now),  # 5h over its 70 line
            })
        assert outcome is TickOutcome.BLOCKED, _reasons(h)
        assert h.active_number() == 1
        assert _switches(h) == []
        assert "hard-line-hold" in _reasons(h)
        assert any(
            r.levelno == logging.WARNING and "own line" in r.getMessage()
            for r in caplog.records
        )

    def test_the_hold_is_logged_once_per_state(self, temp_home, caplog):
        h = _fleet(temp_home, lines={"1": 70.0, "2": 70.0, "3": 70.0})
        with caplog.at_level(logging.INFO, logger="claude-swap"):
            for _ in range(3):
                now = h.clock.now
                h.tick_with_entries({
                    "1": _fresh(_probe(93, 45), now),
                    "2": _fresh(_probe(0, 79), now),
                    "3": _fresh(_probe(78, 13), now),
                })
                h.clock.advance(60.0)
        holds = [r for r in caplog.records if "own line" in r.getMessage()]
        assert len(holds) == 1

    def test_at_its_line_it_leaves_without_margin_or_cooldown(self, temp_home):
        """Active at 71 on a 70 line, candidate at 65 under its own 70 line:
        a proactive move would refuse (margin 6 < 10) and wait out the
        cooldown. A hard line leaves at once."""
        h = _fleet(temp_home, lines={"1": 70.0, "2": 70.0})
        h.engine._mutate_state(lambda s: s.update(lastSwitchAt=h.clock.now))
        now = h.clock.now
        outcome = h.tick_with_entries({
            "1": _fresh(_probe(71), now),
            "2": _fresh(_probe(65), now),
            "3": _fresh(_probe(100), now),
        })
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 2
        assert [s.trigger for s in _switches(h)] == ["hard-line"]

    def test_it_falls_back_to_an_account_on_the_soft_global_line(self, temp_home):
        """No account under its own line, but 3 has no line of its own: it is
        over the soft global 90 yet has room, so the hard-line escape takes it
        rather than keep using 1 past its limit."""
        h = _fleet(temp_home, lines={"1": 70.0, "2": 70.0})
        now = h.clock.now
        outcome = h.tick_with_entries({
            "1": _fresh(_probe(75), now),
            "2": _fresh(_probe(72), now),   # over its own line: never
            "3": _fresh(_probe(94), now),   # over the soft line, 6 left
        })
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 3
        assert [s.trigger for s in _switches(h)] == ["hard-line"]

    def test_an_at_limit_escape_never_lands_over_a_hard_line(self, temp_home):
        h = _fleet(temp_home, lines={"2": 70.0, "3": 70.0})
        outcome = h.tick_with_usage({
            "1": _usage(100), "2": _usage(72), "3": _usage(80),
        })
        assert outcome is TickOutcome.BLOCKED
        assert h.active_number() == 1

    def test_a_failover_never_lands_over_a_hard_line(self, temp_home):
        h = _fleet(temp_home, lines={"2": 70.0, "3": 70.0})
        for _ in range(3):  # unhealthy_ticks default is 3
            outcome = h.tick_with_usage({
                "1": None, "2": _usage(72), "3": _usage(80),
            })
            h.clock.advance(60.0)
        assert outcome is TickOutcome.BLOCKED
        assert h.active_number() == 1

    def test_with_a_weekly_line_the_override_governs_the_five_hour_only(
        self, temp_home
    ):
        """Weekly line 97 set: 2's 7d at 79 is judged on 97 (soft), not on
        its own 70, so it is a valid target; 3's 5h at 72 is over its 70."""
        h = _fleet(
            temp_home, lines={"1": 70.0, "2": 70.0, "3": 70.0},
            weekly_threshold=97.0,
        )
        now = h.clock.now
        outcome = h.tick_with_entries({
            "1": _fresh(_probe(80, 10), now),
            "2": _fresh(_probe(0, 79), now),
            "3": _fresh(_probe(72, 10), now),
        })
        assert outcome is TickOutcome.SWITCHED, _reasons(h)
        assert h.active_number() == 2


class TestGlobalLineIsSoft:
    def test_nothing_under_its_lines_means_stay(self, temp_home):
        """Red arm of the all-above move: active 99, peers 100 and 95, global
        90. The engine used to land on 95; now it stays and runs on."""
        h = _fleet(temp_home)
        outcome = h.tick_with_usage({
            "1": _usage(99), "2": _usage(100), "3": _usage(95),
        })
        assert outcome is TickOutcome.BLOCKED
        assert h.active_number() == 1
        assert "no-qualifying-candidate" in _reasons(h)

    @pytest.mark.parametrize("strategy", ["best", "weekly-first", "consume-first"])
    def test_every_strategy_stays(self, temp_home, strategy):
        h = _fleet(temp_home, strategy=strategy)
        outcome = h.tick_with_usage({
            "1": _usage7(95, 50), "2": _usage7(97, 50), "3": _usage7(91, 50),
        })
        assert outcome is TickOutcome.BLOCKED
        assert h.active_number() == 1

    def test_at_its_limit_it_lands_on_the_most_room_over_a_soft_line(
        self, temp_home
    ):
        """The global line never stops work: at 100% the escape takes the
        account with the most room, though it is over the global line."""
        h = _fleet(temp_home)
        outcome = h.tick_with_usage({
            "1": _usage(100), "2": _usage(97), "3": _usage(91),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3
        assert [s.trigger for s in _switches(h)] == ["at-limit"]

    def test_a_healthy_peer_still_wins_the_normal_way(self, temp_home):
        h = _fleet(temp_home)
        outcome = h.tick_with_usage({
            "1": _usage(92), "2": _usage(100), "3": _usage(40),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3
        assert [s.trigger for s in _switches(h)] == ["proactive"]


class TestATargetNeedsAFreshReading:
    @pytest.mark.parametrize("active_pct", [95, 100])
    def test_a_stale_endpoint_reading_is_skipped(self, temp_home, active_pct):
        """An endpoint reading the escalation could not refresh (here 10
        minutes old, trust extended by a live claim) is not a target; the
        fresh one is taken."""
        h = _fleet(temp_home)
        now = h.clock.now
        outcome = h.tick_with_entries({
            "1": _fresh(_usage(active_pct), now),
            "2": UsageEntry(
                last_good=_usage(10), fetched_at=now - 600.0, age_s=600.0,
                trust_extended=True,
            ),
            "3": _fresh(_usage(50), now),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_only_stale_targets_means_no_switch(self, temp_home):
        h = _fleet(temp_home, n=2)
        now = h.clock.now
        outcome = h.tick_with_entries({
            "1": _fresh(_usage(100), now),
            "2": UsageEntry(
                last_good=_usage(10), fetched_at=now - 600.0, age_s=600.0,
                trust_extended=True,
            ),
        })
        assert outcome is TickOutcome.BLOCKED
        assert h.active_number() == 1
        assert "stale-usage" in _reasons(h)


class TestManualSwitchIsUnrestricted:
    def test_switch_to_lands_on_an_account_over_its_line_and_at_its_limit(
        self, temp_home
    ):
        h = _fleet(temp_home, lines={"2": 50.0})
        h.switcher._usage_store.record(
            {"2": FetchRecord(usage=_probe(100, 99))},
            {"2": ("2@example.com", "")},
        )
        result = h.switcher.switch_to("2", json_output=True)
        assert result and result.get("switched") is True
        assert h.active_number() == 2


class TestActiveCadenceNearALine:
    def _plan(self, new, *, prev=None, recent_429=False, is_active=True):
        return poll_policy.plan_after_fetch(
            prev_interval_s=poll_policy.MIN_INTERVAL_S,
            prev_usage=prev if prev is not None else new,
            new_usage=new,
            is_active=is_active,
            threshold=70.0,
            models=(),
            recent_429=recent_429,
            now=1_000_000.0,
            rng=lambda: 0.5,
        )[1]

    def test_a_flat_probe_reading_in_the_band_polls_every_minute(self):
        """Red arm: 21 sat at 78% between two polls and was probed 285 s
        later at 95%. A flat reading used to decay to 270 s."""
        assert self._plan(_probe(60)) == poll_policy.URGENT_INTERVAL_S

    def test_an_endpoint_reading_in_the_band_holds_the_normal_floor(self):
        assert self._plan(_usage(60)) == poll_policy.MIN_INTERVAL_S

    def test_outside_the_band_it_still_decays(self):
        assert self._plan(_probe(40)) == 270.0

    def test_an_exhausted_account_keeps_the_slow_cadence(self):
        assert self._plan(_probe(100)) == poll_policy.EXHAUSTED_INTERVAL_S

    def test_a_recent_429_wins(self):
        assert self._plan(_probe(60), recent_429=True) >= (
            poll_policy.POST_429_MIN_INTERVAL_S
        )

    def test_candidates_keep_their_cadence(self):
        assert self._plan(_probe(60), is_active=False) == 270.0

    def test_a_switch_pulls_a_probe_target_in_the_band_to_one_minute(
        self, temp_home
    ):
        """Red arm: the new active's first poll waited the 180 s floor."""
        h = _fleet(temp_home, n=2)
        h.switcher.set_poll_policy_inputs(70.0, None, ())
        identities = {"2": ("2@example.com", "")}
        h.switcher._usage_store.record(
            {"2": FetchRecord(usage=_probe(60))}, identities,
            plans={"2": (h.clock.now + 600.0, 600.0)},
        )
        h.switcher._replan_new_active("2", "2@example.com", "")
        entry = h.switcher._usage_store.entries(identities)["2"]
        assert entry.poll_interval_s == poll_policy.URGENT_INTERVAL_S
        assert entry.next_poll_at == pytest.approx(
            h.clock.now + poll_policy.URGENT_INTERVAL_S
        )
