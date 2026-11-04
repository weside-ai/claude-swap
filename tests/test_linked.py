"""Linked session profiles (CSWAP_LINKED_PROFILES=1, claude_swap.linked).

Red arms: each class names the 0.29.0 behaviour it replaces. Every test runs
in a temp HOME with made-up credentials; live instances are real ``sleep``
processes with matching session records.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from claude_swap import linked
from claude_swap import oauth
from claude_swap.claude_locks import credentials_lock_dir, oauth_refresh_lock_dir
from claude_swap.exceptions import SwitchError
from claude_swap.models import Platform
from claude_swap.process_detection import process_start_ticks
from claude_swap.session import scan_live_sessions, session_dir_for
from claude_swap.switcher import USAGE_TOKEN_EXPIRED, ClaudeAccountSwitcher

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="linked mode needs /proc"
)

NUM = "3"
EMAIL = "main@example.com"
ORG = "org-main"
OTHER_NUM = "1"
OTHER_EMAIL = "other@example.com"
HOUR_MS = 3600 * 1000


def now_ms() -> int:
    return int(time.time() * 1000)


def creds(tag: str, expires_at: int, extra: dict | None = None) -> str:
    doc = {
        "claudeAiOauth": {
            "accessToken": f"at-{tag}",
            "refreshToken": f"rt-{tag}",
            "expiresAt": expires_at,
            "scopes": ["user:inference"],
        }
    }
    doc.update(extra or {})
    return json.dumps(doc)


def wiped(expires_at: int) -> str:
    return json.dumps(
        {"claudeAiOauth": {"accessToken": "", "refreshToken": "", "expiresAt": expires_at}}
    )


def config_for(email: str, org: str) -> str:
    return json.dumps(
        {"oauthAccount": {"emailAddress": email, "accountUuid": f"u-{email}", "organizationUuid": org}}
    )


def gen(raw: str | None) -> str:
    return linked.generation_id(oauth.extract_oauth_data(raw) if raw else None)


# ---------------------------------------------------------------------------
# processes and registry
# ---------------------------------------------------------------------------


@pytest.fixture
def procs():
    started: list[subprocess.Popen] = []

    def spawn(cmd: list[str]) -> subprocess.Popen:
        p = subprocess.Popen(cmd)
        started.append(p)
        return p

    yield spawn
    for p in started:
        p.kill()
        p.wait()


def child_pid(parent: int, timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            kids = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
        except OSError:
            kids = []
        if kids:
            return int(kids[0])
        time.sleep(0.02)
    raise AssertionError("child did not start")


def write_record(registry: Path, pid: int) -> None:
    registry.mkdir(parents=True, exist_ok=True)
    (registry / f"{pid}.json").write_text(
        json.dumps({"pid": pid, "procStart": process_start_ticks(pid), "sessionId": f"s{pid}"})
    )


@pytest.fixture
def linked_on(monkeypatch):
    monkeypatch.setenv(linked.ENV_FLAG, "1")


@pytest.fixture
def shared_profile(tmp_path):
    """A profile whose sessions/ is a symlink to a shared registry."""
    registry = tmp_path / "global-sessions"
    registry.mkdir()
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "sessions").symlink_to(registry)
    return profile, registry


class TestSharedRegistryFilter:
    """0.29.0: every live record in a shared registry counts for the profile
    (18 foreign PIDs in the switch refusal, Konto 4 never refreshed)."""

    def test_only_run_pid_and_its_descendants_count(self, shared_profile, procs, linked_on):
        profile, registry = shared_profile
        foreign = procs(["sleep", "60"])
        owner = procs(["sh", "-c", "sleep 60 & wait"])  # stands in for claude rc
        kid = child_pid(owner.pid)
        for pid in (foreign.pid, owner.pid, kid):
            write_record(registry, pid)
        linked.record_run_pid(profile, owner.pid)

        sessions, unreadable = scan_live_sessions(profile)

        assert unreadable == 0
        assert sorted(s.pid for s in sessions) == sorted([owner.pid, kid])

    def test_red_arm_without_flag_counts_foreign(self, shared_profile, procs, monkeypatch):
        monkeypatch.delenv(linked.ENV_FLAG, raising=False)
        profile, registry = shared_profile
        foreign = procs(["sleep", "60"])
        owner = procs(["sleep", "60"])
        for pid in (foreign.pid, owner.pid):
            write_record(registry, pid)
        linked.record_run_pid(profile, owner.pid)

        sessions, _ = scan_live_sessions(profile)

        assert foreign.pid in {s.pid for s in sessions}  # 0.29.0 behaviour

    def test_no_marker_keeps_everything(self, shared_profile, procs, linked_on):
        """A live instance that predates the marker must never read as gone."""
        profile, registry = shared_profile
        foreign = procs(["sleep", "60"])
        write_record(registry, foreign.pid)

        sessions, _ = scan_live_sessions(profile)

        assert [s.pid for s in sessions] == [foreign.pid]

    def test_recycled_marker_pid_is_dropped(self, shared_profile, procs, linked_on):
        profile, registry = shared_profile
        foreign = procs(["sleep", "60"])
        write_record(registry, foreign.pid)
        # The pid is alive but started at other ticks than recorded: reused.
        (profile / linked.RUN_PIDS_FILE).write_text(f"{foreign.pid} 1\n")

        assert linked.read_run_pids(profile) == {}
        sessions, _ = scan_live_sessions(profile)
        assert [s.pid for s in sessions] == [foreign.pid]  # no live run: unfiltered

    def test_orphaned_child_without_live_root_still_counts(self, shared_profile, procs, linked_on):
        """claude rc exits, its session survives: never read as quiescent."""
        profile, registry = shared_profile
        owner = procs(["sh", "-c", "sleep 60 & wait"])
        kid = child_pid(owner.pid)
        write_record(registry, kid)
        linked.record_run_pid(profile, owner.pid)
        owner.kill()
        owner.wait()

        sessions, _ = scan_live_sessions(profile)

        assert kid in {s.pid for s in sessions}

    def test_own_registry_is_untouched(self, tmp_path, procs, linked_on):
        profile = tmp_path / "profile"
        p = procs(["sleep", "60"])
        write_record(profile / "sessions", p.pid)
        (profile / linked.RUN_PIDS_FILE).write_text("")

        sessions, _ = scan_live_sessions(profile)

        assert [s.pid for s in sessions] == [p.pid]

    def test_record_run_pid_prunes_dead_entries(self, tmp_path, procs):
        profile = tmp_path / "profile"
        profile.mkdir()
        dead = procs(["true"])
        dead.wait()
        (profile / linked.RUN_PIDS_FILE).write_text(f"{dead.pid} 5\n")
        live = procs(["sleep", "60"])

        linked.record_run_pid(profile, live.pid)

        assert linked.read_run_pids(profile) == {live.pid: process_start_ticks(live.pid)}
        assert (profile / linked.RUN_PIDS_FILE).stat().st_mode & 0o777 == 0o600


# ---------------------------------------------------------------------------
# switcher harness
# ---------------------------------------------------------------------------


@pytest.fixture
def switcher(temp_home: Path, monkeypatch) -> ClaudeAccountSwitcher:
    monkeypatch.setattr(Platform, "detect", classmethod(lambda cls: Platform.LINUX))
    sw = ClaudeAccountSwitcher(debug=True)
    sw._setup_directories()
    sw._write_json(
        sw.sequence_file,
        {
            "activeAccountNumber": int(OTHER_NUM),
            "lastUpdated": "2026-11-04T00:00:00Z",
            "sequence": [int(OTHER_NUM), int(NUM)],
            "accounts": {
                OTHER_NUM: {"email": OTHER_EMAIL, "uuid": "u-o", "organizationUuid": "org-o",
                            "organizationName": "O", "added": "2026-01-01T00:00:00Z"},
                NUM: {"email": EMAIL, "uuid": "u-m", "organizationUuid": ORG,
                      "organizationName": "M", "added": "2026-01-01T00:00:00Z"},
            },
        },
    )
    sw._write_account_credentials(OTHER_NUM, OTHER_EMAIL, creds("other", now_ms() + 5 * HOUR_MS))
    sw._write_account_config(OTHER_NUM, OTHER_EMAIL, config_for(OTHER_EMAIL, "org-o"))
    sw._write_account_config(NUM, EMAIL, config_for(EMAIL, ORG))
    return sw


def global_creds_path(home: Path) -> Path:
    return home / ".claude" / ".credentials.json"


def make_profile(sw, home: Path, procs, profile_creds: str, *, shared: bool = True) -> Path:
    profile = session_dir_for(sw.backup_dir, NUM, EMAIL)
    profile.mkdir(parents=True)
    (profile / ".credentials.json").write_text(profile_creds)
    (profile / ".claude.json").write_text(config_for(EMAIL, ORG))
    registry = home / ".claude" / "sessions"
    if shared:
        registry.mkdir(parents=True, exist_ok=True)
        (profile / "sessions").symlink_to(registry)
    else:
        registry = profile / "sessions"
    owner = procs(["sleep", "60"])
    write_record(registry, owner.pid)
    linked.record_run_pid(profile, owner.pid)
    return profile


def make_global(home: Path, email: str, org: str, raw: str) -> None:
    (home / ".claude.json").write_text(config_for(email, org))
    global_creds_path(home).write_text(raw)


class TestKonto4InactiveRefresh:
    """An abandoned pin profile with a shared registry and no live run keeps
    counting every Claude on the machine, flag or not (the conservative
    filter): its expired backup is never refreshed. The fix for that slot is
    removing the profile, not the filter."""

    @pytest.mark.parametrize("flag", ["", "1"])
    def test_abandoned_shared_profile_defers(self, switcher, temp_home, procs, monkeypatch, flag):
        monkeypatch.setenv(linked.ENV_FLAG, flag)
        foreign = procs(["sleep", "60"])
        profile = session_dir_for(switcher.backup_dir, NUM, EMAIL)
        profile.mkdir(parents=True)
        registry = temp_home / ".claude" / "sessions"
        registry.mkdir(parents=True)
        (profile / "sessions").symlink_to(registry)
        write_record(registry, foreign.pid)
        (profile / linked.RUN_PIDS_FILE).write_text("")
        expired = creds("old", now_ms() - HOUR_MS)
        switcher._write_account_credentials(NUM, EMAIL, expired)
        monkeypatch.setattr(
            oauth, "try_fetch_usage_for_account",
            lambda *a, **k: pytest.fail("0.29.0 never fetches here"),
        )

        rec = switcher._fetch_account_usage((int(NUM), EMAIL, "M", ORG, False, expired, ""))

        assert rec.sentinel == USAGE_TOKEN_EXPIRED

    def test_removed_profile_takes_refresh_path(self, switcher, temp_home, procs, linked_on, monkeypatch):
        expired = creds("old", now_ms() - HOUR_MS)
        switcher._write_account_credentials(NUM, EMAIL, expired)
        calls = []

        def fake_fetch(num, email, c, is_active, refresh_via=None, **kw):
            calls.append(refresh_via)
            return oauth.UsageOutcome(usage={"ok": True})

        monkeypatch.setattr(oauth, "try_fetch_usage_for_account", fake_fetch)

        rec = switcher._fetch_account_usage((int(NUM), EMAIL, "M", ORG, False, expired, ""))

        assert rec.sentinel != USAGE_TOKEN_EXPIRED
        assert calls and calls[0] is not None


class TestSwitchToLinked:
    """0.29.0: a live profile ahead of the backup refuses the switch with
    "rotated past the stored backup"."""

    def test_switch_adopts_live_profile_generation(self, switcher, temp_home, procs, linked_on, monkeypatch):
        switcher._write_account_credentials(NUM, EMAIL, creds("consumed", now_ms() - HOUR_MS))
        profile_raw = creds("live", now_ms() + 3 * HOUR_MS, {"mcpOAuth": {"x": {"t": "profile-only"}}})
        profile = make_profile(switcher, temp_home, procs, profile_raw)
        make_global(temp_home, OTHER_EMAIL, "org-o", creds("other", now_ms() + 5 * HOUR_MS))
        monkeypatch.setattr(switcher, "list_accounts", lambda **kw: None)
        monkeypatch.setattr(switcher, "_prefetch_live_identity", lambda: {"live": None, "resolved": None})

        switcher._perform_switch(NUM)

        g = global_creds_path(temp_home).read_text()
        assert gen(g) == gen(profile_raw)
        assert gen(switcher.read_account_credentials(NUM, EMAIL)) == gen(profile_raw)
        assert (profile / ".credentials.json").read_text() == profile_raw
        assert "profile-only" not in g  # store-local fields stay local

    def test_profile_locks_held_through_activation(self, switcher, temp_home, procs, linked_on, monkeypatch):
        switcher._write_account_credentials(NUM, EMAIL, creds("consumed", now_ms() - HOUR_MS))
        profile = make_profile(switcher, temp_home, procs, creds("live", now_ms() + 3 * HOUR_MS))
        make_global(temp_home, OTHER_EMAIL, "org-o", creds("other", now_ms() + 5 * HOUR_MS))
        monkeypatch.setattr(switcher, "list_accounts", lambda **kw: None)
        monkeypatch.setattr(switcher, "_prefetch_live_identity", lambda: {"live": None, "resolved": None})
        seen = []
        real = switcher._write_credentials

        def spy(raw):
            seen.append(oauth_refresh_lock_dir(profile).is_dir() and credentials_lock_dir(profile).is_dir())
            real(raw)

        monkeypatch.setattr(switcher, "_write_credentials", spy)

        switcher._perform_switch(NUM)

        assert seen and all(seen)

    def test_unadoptable_profile_refuses(self, switcher, temp_home, procs, linked_on, monkeypatch):
        switcher._write_account_credentials(NUM, EMAIL, creds("consumed", now_ms() - HOUR_MS))
        profile = make_profile(switcher, temp_home, procs, creds("live", now_ms() + 3 * HOUR_MS))
        (profile / ".claude.json").write_text("{broken")
        make_global(temp_home, OTHER_EMAIL, "org-o", creds("other", now_ms() + 5 * HOUR_MS))
        monkeypatch.setattr(switcher, "_prefetch_live_identity", lambda: {"live": None, "resolved": None})

        with pytest.raises(SwitchError, match="could not be adopted"):
            switcher._perform_switch(NUM)
        assert "at-other" in global_creds_path(temp_home).read_text()

    def test_red_arm_without_flag_refuses(self, switcher, temp_home, procs, monkeypatch):
        monkeypatch.delenv(linked.ENV_FLAG, raising=False)
        switcher._write_account_credentials(NUM, EMAIL, creds("consumed", now_ms() - HOUR_MS))
        make_profile(switcher, temp_home, procs, creds("live", now_ms() + 3 * HOUR_MS), shared=False)
        make_global(temp_home, OTHER_EMAIL, "org-o", creds("other", now_ms() + 5 * HOUR_MS))

        with pytest.raises(SwitchError, match="rotated past the stored backup"):
            switcher._perform_switch(NUM)

    def test_switch_away_leaves_profile_alone(self, switcher, temp_home, procs, linked_on, monkeypatch):
        live = creds("live", now_ms() + 3 * HOUR_MS)
        switcher._write_account_credentials(NUM, EMAIL, live)
        profile = make_profile(switcher, temp_home, procs, live)
        make_global(temp_home, EMAIL, ORG, live)
        switcher._write_json(
            switcher.sequence_file,
            {**switcher._get_sequence_data(), "activeAccountNumber": int(NUM)},
        )
        monkeypatch.setattr(switcher, "list_accounts", lambda **kw: None)
        monkeypatch.setattr(switcher, "_prefetch_live_identity", lambda: {"live": None, "resolved": None})
        before = (profile / ".credentials.json").read_bytes()

        switcher._perform_switch(OTHER_NUM)

        assert (profile / ".credentials.json").read_bytes() == before
        assert not global_creds_path(temp_home).is_symlink()
        assert json.loads(global_creds_path(temp_home).read_text())["claudeAiOauth"]["accessToken"] == "at-other"


class TestSyncLinked:
    """0.29.0 has no sync: profile and default login rotate one family
    independently, and the second refresh gets invalid_grant."""

    @pytest.fixture
    def linked_pair(self, switcher, temp_home, procs, linked_on):
        def build(g_raw: str, p_raw: str, backup: str | None = None):
            switcher._write_account_credentials(NUM, EMAIL, backup or g_raw)
            profile = make_profile(switcher, temp_home, procs, p_raw)
            make_global(temp_home, EMAIL, ORG, g_raw)
            return profile
        return build

    def sync(self, switcher):
        return linked.sync_linked_profile(switcher, NUM, EMAIL, ORG)

    def test_in_sync_does_nothing(self, switcher, temp_home, linked_pair, monkeypatch):
        raw = creds("a", now_ms() + 3 * HOUR_MS)
        linked_pair(raw, raw)
        monkeypatch.setattr(oauth, "try_refresh_oauth_credentials",
                            lambda c: pytest.fail("no refresh while fresh"))
        assert self.sync(switcher) == linked.IN_SYNC

    def test_freshens_once_under_both_lock_pairs(self, switcher, temp_home, linked_pair, monkeypatch):
        raw = creds("a", now_ms() + 5 * 60 * 1000)  # inside the 10-min window
        profile = linked_pair(raw, raw)
        fresh = creds("b", now_ms() + 8 * HOUR_MS)
        posts = []

        def fake_refresh(c):
            posts.append(c)
            for d in (oauth_refresh_lock_dir(), credentials_lock_dir(),
                      oauth_refresh_lock_dir(profile), credentials_lock_dir(profile)):
                assert d.is_dir(), f"{d} not held during refresh"
            return oauth.RefreshOutcome(fresh, None)

        monkeypatch.setattr(oauth, "try_refresh_oauth_credentials", fake_refresh)

        assert self.sync(switcher) == linked.FRESHENED
        assert len(posts) == 1
        assert gen(global_creds_path(temp_home).read_text()) == gen(fresh)
        assert gen((profile / ".credentials.json").read_text()) == gen(fresh)
        assert gen(switcher.read_account_credentials(NUM, EMAIL)) == gen(fresh)
        # Second pass: nothing left to do, no second POST.
        assert self.sync(switcher) == linked.IN_SYNC
        assert len(posts) == 1

    def test_profile_refresh_is_adopted_by_global(self, switcher, temp_home, linked_pair):
        old = creds("a", now_ms() + HOUR_MS)
        new = creds("b", now_ms() + 8 * HOUR_MS, {"mcpOAuth": {"p": 1}})
        linked_pair(old, new)
        assert self.sync(switcher) == linked.TO_GLOBAL
        g = global_creds_path(temp_home).read_text()
        assert gen(g) == gen(new)
        assert "mcpOAuth" not in json.loads(g)

    def test_global_refresh_is_adopted_by_profile(self, switcher, temp_home, linked_pair):
        old = creds("a", now_ms() + HOUR_MS)
        new = creds("b", now_ms() + 8 * HOUR_MS)
        profile = linked_pair(new, old)
        assert self.sync(switcher) == linked.TO_PROFILE
        assert gen((profile / ".credentials.json").read_text()) == gen(new)
        assert not (profile / ".credentials.json").is_symlink()

    def test_wiped_copy_is_restored(self, switcher, temp_home, linked_pair):
        good = creds("a", now_ms() + HOUR_MS)
        profile = linked_pair(good, wiped(now_ms() + 2 * HOUR_MS))
        assert self.sync(switcher) == linked.TO_PROFILE
        assert gen((profile / ".credentials.json").read_text()) == gen(good)

    def test_undecidable_divergence_alerts_and_writes_nothing(self, switcher, temp_home, linked_pair, capsys):
        exp = now_ms() + HOUR_MS
        g, p = creds("a", exp), creds("b", exp)
        profile = linked_pair(g, p)
        backup_before = switcher.read_account_credentials(NUM, EMAIL)

        assert self.sync(switcher) == linked.DIVERGED

        assert global_creds_path(temp_home).read_text() == g
        assert (profile / ".credentials.json").read_text() == p
        assert switcher.read_account_credentials(NUM, EMAIL) == backup_before
        alert = json.loads((switcher.backup_dir / f"{linked.ALERT_PREFIX}{NUM}.json").read_text())
        assert alert["global"] == gen(g) and alert["profile"] == gen(p)
        assert "neither is a later generation" in capsys.readouterr().err
        assert "at-" not in json.dumps(alert)  # hashes only

    def test_both_wiped_alerts(self, switcher, temp_home, linked_pair):
        linked_pair(wiped(1), wiped(2))
        assert self.sync(switcher) == linked.DIVERGED

    def test_alert_clears_when_back_in_sync(self, switcher, temp_home, linked_pair):
        raw = creds("a", now_ms() + 3 * HOUR_MS)
        linked_pair(raw, raw)
        alert = switcher.backup_dir / f"{linked.ALERT_PREFIX}{NUM}.json"
        alert.write_text("{}")
        assert self.sync(switcher) == linked.IN_SYNC
        assert not alert.exists()

    def test_not_linked_when_global_is_another_account(self, switcher, temp_home, procs, linked_on):
        raw = creds("a", now_ms() + 3 * HOUR_MS)
        make_profile(switcher, temp_home, procs, raw)
        make_global(temp_home, OTHER_EMAIL, "org-o", creds("other", now_ms() + HOUR_MS))
        assert linked.sync_linked_profile(switcher, NUM, EMAIL, ORG) == linked.NOT_LINKED

    def test_off_without_flag(self, switcher, monkeypatch):
        monkeypatch.delenv(linked.ENV_FLAG, raising=False)
        assert linked.sync_linked_profile(switcher, NUM, EMAIL, ORG) == linked.OFF

    def test_transient_refresh_failure_defers(self, switcher, temp_home, linked_pair, monkeypatch):
        raw = creds("a", now_ms() + 60 * 1000)
        linked_pair(raw, raw)
        monkeypatch.setattr(oauth, "try_refresh_oauth_credentials",
                            lambda c: oauth.RefreshOutcome(None, "transient"))
        assert self.sync(switcher) == linked.DEFERRED
        assert global_creds_path(temp_home).read_text() == raw

    def test_invalid_grant_alerts(self, switcher, temp_home, linked_pair, monkeypatch):
        raw = creds("a", now_ms() + 60 * 1000)
        linked_pair(raw, raw)
        monkeypatch.setattr(oauth, "try_refresh_oauth_credentials",
                            lambda c: oauth.RefreshOutcome(None, "invalid_grant"))
        assert self.sync(switcher) == linked.DIVERGED

    def test_unparseable_store_defers_and_keeps_fields(self, switcher, temp_home, linked_pair):
        good = creds("a", now_ms() + HOUR_MS)
        linked_pair(good, good)
        global_creds_path(temp_home).write_text('{"claudeAiOauth": {"acc')  # mid-write
        assert self.sync(switcher) == linked.DEFERRED
        assert global_creds_path(temp_home).read_text() == '{"claudeAiOauth": {"acc'

    def test_unreadable_profile_identity_alerts(self, switcher, temp_home, linked_pair):
        old, new = creds("a", now_ms() + HOUR_MS), creds("b", now_ms() + 8 * HOUR_MS)
        profile = linked_pair(old, new)
        (profile / ".claude.json").write_text("{broken")
        assert self.sync(switcher) == linked.DIVERGED
        assert global_creds_path(temp_home).read_text() == old

    def test_backup_witness_wins_after_failed_store_writes(self, switcher, temp_home, linked_pair):
        consumed = creds("a", now_ms() + 60 * 1000)
        fresh = creds("b", now_ms() + 8 * HOUR_MS)
        profile = linked_pair(consumed, consumed, backup=fresh)
        assert self.sync(switcher) == linked.FROM_BACKUP
        assert gen(global_creds_path(temp_home).read_text()) == gen(fresh)
        assert gen((profile / ".credentials.json").read_text()) == gen(fresh)

    def test_freshen_write_failure_alerts_and_next_pass_converges(
        self, switcher, temp_home, linked_pair, monkeypatch, capsys
    ):
        raw = creds("a", now_ms() + 60 * 1000)
        profile = linked_pair(raw, raw)
        fresh = creds("b", now_ms() + 8 * HOUR_MS)
        posts = []

        def fake_refresh(c):
            posts.append(c)
            return oauth.RefreshOutcome(fresh, None)

        monkeypatch.setattr(oauth, "try_refresh_oauth_credentials", fake_refresh)

        def boom(raw):
            raise OSError("disk full")

        monkeypatch.setattr(switcher, "_write_credentials", boom)
        monkeypatch.setattr(linked, "_write_profile", lambda d, r: boom(r))

        assert self.sync(switcher) == linked.FRESHENED
        assert "write failed" in capsys.readouterr().err
        monkeypatch.undo()
        monkeypatch.setenv(linked.ENV_FLAG, "1")

        assert self.sync(switcher) == linked.FROM_BACKUP
        assert len(posts) == 1  # the consumed generation is never POSTed again
        assert gen(global_creds_path(temp_home).read_text()) == gen(fresh)
        assert gen((profile / ".credentials.json").read_text()) == gen(fresh)
