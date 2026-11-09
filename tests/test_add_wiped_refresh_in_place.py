"""`cswap add` on an EXISTING slot must refuse a wiped live credential too.

Field report (2026-11-09, fork 1.1.0b1): a re-add of an account that already
held a slot ran the refresh-in-place arm and stored the shape Claude Code
leaves after ``invalid_grant`` -- the full credential file with
``mcpOAuth``/``trustedDeviceToken`` intact and ``claudeAiOauth`` keeping
scopes and subscription fields, but ``accessToken``/``refreshToken`` emptied
and ``expiresAt`` zeroed. The slot then showed "no credentials" in rotation.

These tests read the credential through the real capture path (a file in a
temp HOME), not a patched ``_read_capture_credentials``, so they cover what
``cswap add`` actually reads on Linux.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap.exceptions import ConfigError
from claude_swap.switcher import ClaudeAccountSwitcher

EMAIL = "ax@example.com"

GOOD = json.dumps({
    "claudeAiOauth": {
        "accessToken": "sk-ant-oat01-GOOD", "refreshToken": "rt-good",
        "expiresAt": 99999999999000, "refreshTokenExpiresAt": 99999999999000,
        "scopes": ["user:inference"], "subscriptionType": "max",
        "subscriptionTypeSource": "x", "rateLimitTier": "t",
    },
    "mcpOAuth": {"srv": {"accessToken": "mcp-at"}},
    "mcpOAuthClientConfig": {}, "trustedDeviceToken": "tdt",
})


def _wiped(**overrides) -> str:
    data = json.loads(GOOD)
    data["claudeAiOauth"].update(
        {"accessToken": "", "refreshToken": "", "expiresAt": 0}, **overrides)
    return json.dumps(data)


def _live_file(home: Path) -> Path:
    return home / ".claude" / ".credentials.json"


def _setup(temp_home: Path, mock_claude_config: Path) -> ClaudeAccountSwitcher:
    s = ClaudeAccountSwitcher()
    s._setup_directories()
    s._init_sequence_file()
    s._get_claude_config_path().write_text(json.dumps({"oauthAccount": {
        "emailAddress": EMAIL, "organizationUuid": ""}}), encoding="utf-8")
    live = _live_file(temp_home)
    live.parent.mkdir(parents=True, exist_ok=True)
    live.write_text(GOOD, encoding="utf-8")
    with patch("claude_swap.oauth.fetch_oauth_profile",
               return_value={"uuid": "", "email": EMAIL, "organizationUuid": None}):
        s.add_account(slot=4, assume_yes=True)
    assert s._read_account_credentials("4", EMAIL) == GOOD
    return s


WIPES = {
    "empty-strings": _wiped(),
    "keys-removed": json.dumps({**json.loads(GOOD), "claudeAiOauth": {
        k: v for k, v in json.loads(GOOD)["claudeAiOauth"].items()
        if k not in ("accessToken", "refreshToken")} | {"expiresAt": 0}}),
    "null-tokens": _wiped(accessToken=None, refreshToken=None),
}


@pytest.mark.parametrize("slot", [None, 4], ids=["refresh-in-place", "explicit-slot"])
@pytest.mark.parametrize("wiped", list(WIPES.values()), ids=list(WIPES))
def test_readd_of_an_existing_slot_refuses_a_wiped_live_credential(
    temp_home: Path, mock_claude_config: Path, slot, wiped,
):
    s = _setup(temp_home, mock_claude_config)
    _live_file(temp_home).write_text(wiped, encoding="utf-8")

    with patch("claude_swap.oauth.fetch_oauth_profile") as fetch, \
         pytest.raises(ConfigError, match="Nothing was changed"):
        s.add_account(slot=slot, assume_yes=True)

    fetch.assert_not_called()
    assert s._read_account_credentials("4", EMAIL) == GOOD, (
        "DEFECT: the wiped live credential overwrote slot 4's working backup")
