"""Auto-switch view mode survives a restart; successful header probes log.

1. The auto-switch view used to open in dry-run on every start, so a process
   restart silently stopped the rotation until someone pressed ``l`` + ``y``
   (2026-11-08 01:52). The last chosen mode is now kept in settings.json
   (``ui.autoViewMode``) and restored on open; ``cswap auto-view [live|dry-run]``
   opens the TUI straight on that view.
2. A successful header probe left no INFO line, so a wrong switch could not be
   traced back to the reading it decided on (2026-11-08 01:00:52). Each
   success now logs account, 5h/7d pct and the previous probe's age, at most
   once a minute per account unless the numbers change.
"""

from __future__ import annotations

import json
import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from claude_swap import oauth
from claude_swap.settings import load_ui_settings, set_setting, settings_path
from tests.test_oauth import _UNIFIED_HEADERS, _headers
from tests.test_tui import (
    FakeSwitcher,
    _FakeEngine,
    make_account,
    make_app,
    settle,
)


@pytest.fixture
def fake_engine(monkeypatch):
    _FakeEngine.instances = []
    monkeypatch.setattr("claude_swap.tui.autoview.AutoSwitchEngine", _FakeEngine)
    return _FakeEngine


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------


class TestAutoViewModeSetting:
    def test_defaults_to_dry_run(self, tmp_path):
        assert load_ui_settings(tmp_path).auto_view_mode == "dry-run"

    def test_reads_live(self, tmp_path):
        settings_path(tmp_path).write_text(
            json.dumps({"ui": {"autoViewMode": "live"}})
        )
        assert load_ui_settings(tmp_path).auto_view_mode == "live"

    def test_unknown_value_falls_back_to_dry_run(self, tmp_path):
        settings_path(tmp_path).write_text(
            json.dumps({"ui": {"theme": "light", "autoViewMode": "turbo"}})
        )
        ui = load_ui_settings(tmp_path)
        assert ui.auto_view_mode == "dry-run"
        assert ui.theme == "light"  # one bad key never resets the other

    def test_bad_theme_keeps_the_mode(self, tmp_path):
        settings_path(tmp_path).write_text(
            json.dumps({"ui": {"theme": "purple", "autoViewMode": "live"}})
        )
        ui = load_ui_settings(tmp_path)
        assert ui.theme == "auto"
        assert ui.auto_view_mode == "live"

    def test_config_set_accepts_both_modes(self, tmp_path):
        assert set_setting(tmp_path, "ui.autoViewMode", "live") == "live"
        assert set_setting(tmp_path, "ui.autoViewMode", "dry-run") == "dry-run"


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------


def _two_accounts(tmp_path):
    return FakeSwitcher([make_account(1, active=True), make_account(2)], tmp_path)


def _log_lines(app) -> list[str]:
    from textual.widgets import RichLog

    return [s.text for s in app.screen.query_one("#event-log", RichLog).lines]


@pytest.mark.asyncio
class TestAutoViewModePersistence:
    async def _open(self, pilot):
        await settle(pilot)
        await pilot.press("g")
        await pilot.pause()

    async def test_going_live_is_remembered(self, tmp_path, fake_engine):
        app = make_app(_two_accounts(tmp_path))
        async with app.run_test(size=(100, 40)) as pilot:
            await self._open(pilot)
            await pilot.press("l")
            await pilot.pause()
            await pilot.press("y")
            await settle(pilot)
        assert load_ui_settings(tmp_path).auto_view_mode == "live"

    async def test_back_to_dry_run_is_remembered(self, tmp_path, fake_engine):
        set_setting(tmp_path, "ui.autoViewMode", "live")
        app = make_app(_two_accounts(tmp_path))
        async with app.run_test(size=(100, 40)) as pilot:
            await self._open(pilot)
            await pilot.press("l")  # live -> dry-run needs no confirmation
            await settle(pilot)
        assert load_ui_settings(tmp_path).auto_view_mode == "dry-run"

    async def test_declined_confirmation_saves_nothing(self, tmp_path, fake_engine):
        app = make_app(_two_accounts(tmp_path))
        async with app.run_test(size=(100, 40)) as pilot:
            await self._open(pilot)
            await pilot.press("l")
            await pilot.pause()
            await pilot.press("escape")
            await settle(pilot)
        assert not settings_path(tmp_path).exists()

    async def test_restart_restores_live(self, tmp_path, fake_engine, caplog):
        set_setting(tmp_path, "ui.autoViewMode", "live")
        app = make_app(_two_accounts(tmp_path))
        with caplog.at_level(logging.INFO, logger="claude-swap"):
            async with app.run_test(size=(100, 40)) as pilot:
                await self._open(pilot)
                await settle(pilot)
                assert len(fake_engine.instances) == 1
                assert fake_engine.instances[0].dry_run is False
                badge = app.screen.query_one("#mode-badge")
                assert "LIVE" in str(badge.render())
                first = _log_lines(app)[0]
                assert "LIVE" in first and "restored" in first
        infos = [
            r.getMessage() for r in caplog.records if r.levelno == logging.INFO
        ]
        assert any("LIVE" in m and "restored" in m for m in infos)

    async def test_default_still_opens_dry_run(self, tmp_path, fake_engine):
        app = make_app(_two_accounts(tmp_path))
        async with app.run_test(size=(100, 40)) as pilot:
            await self._open(pilot)
            assert fake_engine.instances[0].dry_run is True
            assert "DRY-RUN" in _log_lines(app)[0]

    async def test_start_auto_opens_the_view_over_the_dashboard(
        self, tmp_path, fake_engine
    ):
        from claude_swap.tui.app import CswapApp
        from claude_swap.tui.autoview import AutoScreen
        from claude_swap.tui.dashboard import DashboardScreen

        set_setting(tmp_path, "ui.autoViewMode", "live")
        app = CswapApp(_two_accounts(tmp_path), start="auto")
        async with app.run_test(size=(100, 40)) as pilot:
            await settle(pilot)
            assert isinstance(app.screen, AutoScreen)
            assert fake_engine.instances[0].dry_run is False
            await pilot.press("escape")
            await settle(pilot)
            assert isinstance(app.screen, DashboardScreen)

    @pytest.mark.parametrize("mode,dry", [("live", False), ("dry-run", True)])
    async def test_explicit_start_mode_overrides_and_is_saved(
        self, tmp_path, fake_engine, mode, dry
    ):
        from claude_swap.tui.app import CswapApp

        other = "dry-run" if mode == "live" else "live"
        set_setting(tmp_path, "ui.autoViewMode", other)
        app = CswapApp(_two_accounts(tmp_path), start="auto", auto_mode=mode)
        async with app.run_test(size=(100, 40)) as pilot:
            await settle(pilot)
            assert fake_engine.instances[0].dry_run is dry
            first = _log_lines(app)[0]
            assert "command line" in first
        assert load_ui_settings(tmp_path).auto_view_mode == mode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestAutoViewCli:
    @pytest.mark.parametrize(
        "argv,mode",
        [
            (["cswap", "auto-view"], None),
            (["cswap", "auto-view", "live"], "live"),
            (["cswap", "auto-view", "dry-run"], "dry-run"),
        ],
    )
    def test_opens_tui_on_auto_view(self, monkeypatch, temp_home, argv, mode):
        import claude_swap.cli as cli
        import claude_swap.tui as tui

        launched = {}

        def fake_run(switcher, start="dashboard", auto_mode=None):
            launched["start"] = start
            launched["auto_mode"] = auto_mode
            return 0

        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(tui, "run", fake_run)
        with pytest.raises(SystemExit) as excinfo:
            cli.main()
        assert excinfo.value.code == 0
        assert launched == {"start": "auto", "auto_mode": mode}

    def test_rejects_unknown_mode(self, monkeypatch, temp_home):
        import claude_swap.cli as cli

        monkeypatch.setattr(sys, "argv", ["cswap", "auto-view", "turbo"])
        with pytest.raises(SystemExit) as excinfo:
            cli.main()
        assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# header-probe INFO log
# ---------------------------------------------------------------------------


class TestProbeSuccessLog:
    TOKEN = "sk-ant-oat01-secret-token"

    def _creds(self) -> str:
        return json.dumps({"claudeAiOauth": {
            "accessToken": self.TOKEN, "scopes": ["user:inference"],
        }})

    @pytest.fixture(autouse=True)
    def _fresh_throttle(self):
        oauth._reset_probe_log_throttle()
        yield
        oauth._reset_probe_log_throttle()

    def _probe_at(self, now, headers=None, account="3"):
        resp = MagicMock()
        resp.headers = _headers(headers or _UNIFIED_HEADERS)
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        def fake(req, timeout=None):
            assert req.full_url == oauth.USAGE_PROBE_URL
            return resp

        with (
            patch("claude_swap.oauth.time.monotonic", return_value=now),
            patch("claude_swap.oauth.urllib.request.urlopen", side_effect=fake),
        ):
            return oauth.try_fetch_usage_for_account(
                account, "a@b.c", self._creds(), is_active=False, header_probe=True
            )

    @staticmethod
    def _infos(caplog) -> list[str]:
        return [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.INFO and "header probe" in r.getMessage()
        ]

    def test_success_logs_account_and_both_windows(self, caplog):
        with caplog.at_level(logging.INFO, logger="claude-swap"):
            self._probe_at(1000.0)
        (line,) = self._infos(caplog)
        assert "account 3" in line
        assert "5h 34%" in line and "7d 60%" in line
        assert "first probe" in line
        assert self.TOKEN not in line and "a@b.c" not in line

    def test_unchanged_reading_within_a_minute_is_quiet(self, caplog):
        with caplog.at_level(logging.INFO, logger="claude-swap"):
            self._probe_at(1000.0)
            self._probe_at(1030.0)
        assert len(self._infos(caplog)) == 1

    def test_unchanged_reading_after_a_minute_logs_again_with_age(self, caplog):
        with caplog.at_level(logging.INFO, logger="claude-swap"):
            self._probe_at(1000.0)
            self._probe_at(1030.0)
            self._probe_at(1061.0)
        lines = self._infos(caplog)
        assert len(lines) == 2
        assert "previous probe 31s ago" in lines[1]

    def test_changed_reading_logs_at_once(self, caplog):
        moved = dict(_UNIFIED_HEADERS)
        moved["anthropic-ratelimit-unified-5h-utilization"] = "0.41"
        with caplog.at_level(logging.INFO, logger="claude-swap"):
            self._probe_at(1000.0)
            self._probe_at(1010.0, headers=moved)
        lines = self._infos(caplog)
        assert len(lines) == 2
        assert "5h 41%" in lines[1]
        assert "previous probe 10s ago" in lines[1]

    def test_throttle_is_per_account(self, caplog):
        with caplog.at_level(logging.INFO, logger="claude-swap"):
            self._probe_at(1000.0, account="3")
            self._probe_at(1001.0, account="4")
        lines = self._infos(caplog)
        assert len(lines) == 2
        assert "account 4" in lines[1]
