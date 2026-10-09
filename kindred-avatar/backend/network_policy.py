"""Pure helpers for Jazzy's WebSocket network boundary."""

from __future__ import annotations

import hmac
from urllib.parse import parse_qs, urlsplit


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def validate_binding(host: str, token: str) -> None:
    if host not in LOOPBACK_HOSTS and not token:
        raise RuntimeError("JAZZY_WS_TOKEN is required when JAZZY_HOST is non-loopback")


def authenticated_path(path: str, expected_token: str) -> bool:
    if not expected_token:
        return True
    supplied = parse_qs(urlsplit(path).query).get("token", [""])[0]
    return hmac.compare_digest(supplied, expected_token)
