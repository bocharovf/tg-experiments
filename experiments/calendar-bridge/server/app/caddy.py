"""Управление маршрутом в работающем Caddy через admin API (без root)."""
from __future__ import annotations

import json
import logging
import urllib.request

log = logging.getLogger("caddy")

ADMIN = "http://127.0.0.1:2019"


def _req(method: str, path: str, body: dict | None = None) -> tuple[int, bytes]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        ADMIN + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def route(domain: str, upstream: str) -> dict:
    return {
        "match": [{"host": [domain]}],
        "handle": [
            {
                "handler": "headers",
                "response": {
                    "set": {
                        "Strict-Transport-Security": ["max-age=31536000; includeSubDomains"],
                        "X-Content-Type-Options": ["nosniff"],
                        "Referrer-Policy": ["no-referrer"],
                    }
                },
            },
            {
                "handler": "reverse_proxy",
                "upstreams": [{"dial": upstream}],
                "headers": {"request": {"set": {"X-Forwarded-Proto": ["https"]}}},
            },
        ],
        "terminal": True,
    }


def ensure(domain: str, upstream: str) -> bool:
    """Добавляет маршрут в srv0, если его ещё нет. True — если меняли конфиг."""
    status, raw = _req("GET", "/config/apps/http/servers/srv0/routes")
    if status != 200:
        log.warning("caddy admin недоступен (status=%s)", status)
        return False
    try:
        routes = json.loads(raw or b"[]") or []
    except Exception:  # noqa: BLE001
        routes = []

    for r in routes:
        for m in r.get("match", []) or []:
            if domain in (m.get("host") or []):
                return False

    status, raw = _req("POST", "/config/apps/http/servers/srv0/routes/0", route(domain, upstream))
    if status not in (200, 201):
        log.error("не удалось добавить маршрут: %s %s", status, raw[:200])
        return False
    log.info("маршрут Caddy добавлен: %s -> %s", domain, upstream)
    return True
