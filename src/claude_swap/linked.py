"""Linked session profiles (opt-in: ``CSWAP_LINKED_PROFILES=1``).

Two problems, one per half of this module.

Shared session registry. A wrapper may symlink a profile's ``sessions/`` to
``~/.claude/sessions`` so that sessions started on the default login and
sessions inside the profile see each other (agent listings and cross-session
messaging read ``<config-dir>/sessions``; Claude Code has no other knob for
it). ``scan_live_sessions(profile)`` then sees every Claude on the machine
and the profile is never quiescent: adoption never runs, every backup write
marks it stale, and an inactive slot with such a profile is never refreshed.
``cswap run`` therefore records the pid it ``exec``s into (the pid stays the
same across ``execvpe``) in ``<profile>/.cswap-run-pids`` together with its
``/proc`` start ticks. With the registry shared, only records whose pid is a
recorded run pid -- or a descendant of one, since ``claude rc`` spawns its
sessions as children -- count for the profile. Without a readable marker the
filter does nothing: a profile whose live instance predates the marker must
never read as quiescent.

Linked active slot. When the account that is the default login also has a
live profile, the two stores hold the same generation and cswap is the one
that keeps them that way (see ``ClaudeAccountSwitcher.sync_linked_profile``).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from claude_swap.fsutil import atomic_write_text
from claude_swap.process_detection import is_pid_alive, process_start_ticks

if TYPE_CHECKING:  # pragma: no cover
    from claude_swap.process_detection import ClaudeSession

ENV_FLAG = "CSWAP_LINKED_PROFILES"
RUN_PIDS_FILE = ".cswap-run-pids"
# Ancestry walk bound: a session tree is a handful of levels deep.
_MAX_DEPTH = 64

_logger = logging.getLogger("claude-swap")


def linked_profiles_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


def registry_is_shared(session_dir: Path) -> bool:
    return (session_dir / "sessions").is_symlink()


def _marker(session_dir: Path) -> Path:
    return session_dir / RUN_PIDS_FILE


def read_run_pids(session_dir: Path) -> dict[int, str] | None:
    """Live recorded run pids -> start ticks; None when there is no marker
    or it cannot be read. Entries whose pid is gone or was recycled (start
    ticks differ) are dropped."""
    try:
        text = _marker(session_dir).read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    except OSError as exc:
        _logger.debug("Cannot read %s: %s", _marker(session_dir), exc)
        return None
    live: dict[int, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        pid, ticks = int(parts[0]), parts[1]
        if is_pid_alive(pid) and process_start_ticks(pid) == ticks:
            live[pid] = ticks
    return live


def record_run_pid(session_dir: Path, pid: int | None = None) -> None:
    """Append the pid ``cswap run`` is about to exec into, pruning dead
    entries. Linux only (needs ``/proc`` start ticks); best effort -- a
    failure here leaves the filter off for this profile, the safe side."""
    if not sys.platform.startswith("linux"):
        return
    pid = os.getpid() if pid is None else pid
    ticks = process_start_ticks(pid)
    if ticks is None:
        return
    entries = read_run_pids(session_dir) or {}
    entries[pid] = ticks
    try:
        atomic_write_text(
            _marker(session_dir),
            "".join(f"{p} {t}\n" for p, t in sorted(entries.items())),
            follow_symlinks=False,
        )
    except OSError as exc:
        _logger.warning("Cannot record run pid in %s: %s", _marker(session_dir), exc)


def _parent_pid(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii", errors="replace")
    except OSError:
        return None
    fields = text.rpartition(")")[2].split()
    if len(fields) < 2 or not fields[1].isdigit():
        return None
    return int(fields[1])


def descends_from(pid: int, roots: set[int]) -> bool:
    current: int | None = pid
    for _ in range(_MAX_DEPTH):
        if current is None or current <= 1:
            return False
        if current in roots:
            return True
        current = _parent_pid(current)
    return False


def owned_sessions(
    session_dir: Path, sessions: list[ClaudeSession]
) -> list[ClaudeSession]:
    """The subset of ``sessions`` that runs against ``session_dir``.

    Identity unless linked mode is on, the registry is shared and a marker
    is readable."""
    if not linked_profiles_enabled() or not registry_is_shared(session_dir):
        return sessions
    if not Path("/proc/self/stat").exists():
        return sessions
    roots = read_run_pids(session_dir)
    if roots is None:
        return sessions
    return [s for s in sessions if descends_from(s.pid, set(roots))]


# -- linked active slot ------------------------------------------------------

# cswap freshens a linked slot this long before expiry: ahead of Claude
# Code's own 5-min window (oauth.OAUTH_EXPIRY_BUFFER_MS), so neither group of
# Claude processes ever reaches the point where it would refresh itself.
LINKED_FRESHEN_MS = 10 * 60 * 1000
ALERT_PREFIX = "linked-alert-"

# Outcomes of sync_linked_profile.
OFF = "off"
NOT_LINKED = "not-linked"
IN_SYNC = "in-sync"
FRESHENED = "freshened"
TO_PROFILE = "copied-to-profile"
TO_GLOBAL = "copied-to-global"
DEFERRED = "deferred"
DIVERGED = "diverged"


def _oauth(credentials: str | None) -> dict | None:
    from claude_swap import oauth

    return oauth.extract_oauth_data(credentials) if credentials else None


def _usable(data: dict | None) -> bool:
    """A full token pair with a numeric expiry. Claude Code's invalid_grant
    path empties the token fields in place and keeps the rest, so a missing
    pair is a wiped copy, never a generation."""
    return bool(
        data
        and isinstance(data.get("accessToken"), str)
        and data.get("accessToken")
        and isinstance(data.get("refreshToken"), str)
        and data.get("refreshToken")
        and isinstance(data.get("expiresAt"), (int, float))
        and not isinstance(data.get("expiresAt"), bool)
    )


def generation_id(data: dict | None) -> str:
    """12-hex hash of the token pair: comparable in logs, never a secret."""
    import hashlib

    if not _usable(data):
        return "none"
    raw = f"{data['accessToken']}\0{data['refreshToken']}".encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def with_oauth(credentials: str | None, data: dict) -> str:
    """``credentials`` with its ``claudeAiOauth`` replaced by ``data``.

    Only the account token moves between the two stores: everything else in
    the file (``mcpOAuth`` and friends) belongs to the store it sits in."""
    import json

    try:
        doc = json.loads(credentials) if credentials else {}
    except json.JSONDecodeError:
        doc = {}
    if not isinstance(doc, dict):
        doc = {}
    doc["claudeAiOauth"] = data
    return json.dumps(doc, separators=(",", ":"))


def _alert_path(switcher, account_num: str) -> Path:
    return Path(switcher.backup_dir) / f"{ALERT_PREFIX}{account_num}.json"


def _raise_alert(switcher, account_num: str, email: str, reason: str, **ids: str) -> None:
    """Loud on every pass while it lasts: error log, stderr, alert file."""
    import json
    import time

    from claude_swap.printer import warning

    detail = " ".join(f"{k}={v}" for k, v in sorted(ids.items()))
    msg = (
        f"Linked profile of Account-{account_num} ({email}): {reason} "
        f"[{detail}]. Nothing was written; the two copies keep their "
        "generations until this is resolved."
    )
    switcher._logger.error(msg)
    try:
        warning(msg, file=sys.stderr)
    except Exception:  # pragma: no cover - output must never break a pass
        pass
    try:
        atomic_write_text(
            _alert_path(switcher, account_num),
            json.dumps({"account": account_num, "reason": reason, "at": int(time.time()), **ids}),
        )
    except OSError:
        pass


def _clear_alert(switcher, account_num: str) -> None:
    try:
        _alert_path(switcher, account_num).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _write_profile(session_dir: Path, credentials: str) -> None:
    # Never through a symlink: the profile's file is its own store.
    atomic_write_text(session_dir / ".credentials.json", credentials, follow_symlinks=False)


def sync_linked_profile(switcher, account_num: str, email: str, org_uuid: str) -> str:
    """Keep the default login and a live profile of the same account on one
    generation, with cswap as the one refresher.

    Runs under cswap's lock and BOTH Claude Code credential lock pairs (the
    default login's and the profile's), in that order everywhere, so neither
    group of Claude processes can refresh while the two copies are compared
    and written. Per pass:

    - same generation, more than ``LINKED_FRESHEN_MS`` left: nothing;
    - same generation, about to expire: refresh once, write both stores and
      the slot backup;
    - different generations, one later by ``expiresAt`` (every refresh and
      every login issues a later expiry): copy the later one over the other;
    - one copy wiped (no token pair): restore it from the other;
    - otherwise (equal expiry, both wiped, identity mismatch): alert loudly
      and write nothing -- activating an older generation is the failure
      this whole mode exists to prevent.
    """
    from claude_swap import oauth
    from claude_swap.claude_locks import claude_credentials_lock
    from claude_swap.exceptions import LockError
    from claude_swap.locking import FileLock
    from claude_swap.session import (
        read_session_credentials,
        scan_live_sessions,
        session_identity_drifted,
    )

    if not linked_profiles_enabled() or not sys.platform.startswith("linux"):
        return OFF
    session_dir = switcher._session_dir(account_num, email)
    if not session_dir.exists():
        return NOT_LINKED
    sessions, unreadable = scan_live_sessions(session_dir)
    if not sessions and not unreadable:
        return NOT_LINKED

    try:
        with (
            FileLock(switcher.lock_file, timeout=3.0),
            claude_credentials_lock(),
            claude_credentials_lock(config_home=session_dir),
        ):
            if not switcher._live_identity_matches(email, org_uuid):
                return NOT_LINKED
            if session_identity_drifted(session_dir, email, org_uuid):
                _raise_alert(switcher, account_num, email, "profile identity is another account")
                return DIVERGED
            g_raw = switcher._read_credentials()
            p_raw = read_session_credentials(session_dir)
            if g_raw is None:
                return DEFERRED
            g, p = _oauth(g_raw), _oauth(p_raw)
            g_id, p_id = generation_id(g), generation_id(p)
            ids = {"global": g_id, "profile": p_id}

            if _usable(g) and _usable(p) and g_id == p_id:
                now_ms = int(__import__("time").time() * 1000)
                if g["expiresAt"] - now_ms > LINKED_FRESHEN_MS:
                    _clear_alert(switcher, account_num)
                    return IN_SYNC
                outcome = oauth.try_refresh_oauth_credentials(g_raw)
                fresh = _oauth(outcome.credentials)
                if outcome.error or not _usable(fresh):
                    if outcome.error == "transient":
                        return DEFERRED
                    _raise_alert(
                        switcher, account_num, email,
                        f"refresh failed ({outcome.error or 'no token pair'})", **ids,
                    )
                    return DIVERGED
                _write_both(switcher, account_num, email, session_dir, g_raw, p_raw, fresh)
                switcher._logger.info(
                    "Linked Account-%s freshened: %s -> %s", account_num, g_id,
                    generation_id(fresh),
                )
                _clear_alert(switcher, account_num)
                return FRESHENED

            winner: str | None = None
            if _usable(g) and _usable(p):
                if g["expiresAt"] > p["expiresAt"]:
                    winner = "global"
                elif p["expiresAt"] > g["expiresAt"]:
                    winner = "profile"
            elif _usable(g):
                winner = "global"
            elif _usable(p):
                winner = "profile"
            if winner is None:
                _raise_alert(
                    switcher, account_num, email,
                    "copies differ and neither is a later generation", **ids,
                )
                return DIVERGED
            data = g if winner == "global" else p
            _write_both(switcher, account_num, email, session_dir, g_raw, p_raw, data)
            switcher._logger.warning(
                "Linked Account-%s: %s generation adopted (global=%s profile=%s)",
                account_num, winner, g_id, p_id,
            )
            _clear_alert(switcher, account_num)
            return TO_PROFILE if winner == "global" else TO_GLOBAL
    except LockError:
        return DEFERRED


def _write_both(switcher, account_num, email, session_dir, g_raw, p_raw, data) -> None:
    g_new = with_oauth(g_raw, data)
    p_new = with_oauth(p_raw, data)
    if g_new != g_raw:
        switcher._write_credentials(g_new)
    if p_new != p_raw:
        _write_profile(session_dir, p_new)
    backup, unreadable = switcher._read_account_credentials_ex(account_num, email)
    if not unreadable:
        b_new = with_oauth(backup, data)
        if b_new != backup:
            # The pure store write: _post_backup_write would mark the very
            # profile just synced stale.
            switcher._store._write_account_credentials(account_num, email, b_new)


def adopt_linked_profile(switcher, account_num: str, email: str, org_uuid: str) -> bool:
    """Switch-time adoption under a live profile. The caller holds cswap's
    lock and the default login's credential locks; this takes the profile's
    pair, so the profile's Claude cannot rotate between the adoption and the
    activation that follows. True when the backup now holds the profile's
    generation."""
    from claude_swap.claude_locks import claude_credentials_lock

    session_dir = switcher._session_dir(account_num, email)
    with claude_credentials_lock(config_home=session_dir):
        profile = switcher._session_profile_ahead(account_num, email, org_uuid)
        if profile is None:
            return False
        data = _oauth(profile)
        if not _usable(data):
            return False
        backup, unreadable = switcher._read_account_credentials_ex(account_num, email)
        if unreadable:
            return False
        switcher._store._write_account_credentials(account_num, email, with_oauth(backup, data))
    switcher._logger.info(
        "Linked Account-%s: adopted live profile generation %s into the backup",
        account_num, generation_id(data),
    )
    return True
