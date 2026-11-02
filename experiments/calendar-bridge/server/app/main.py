"""HTTP API синхронизатора: рабочий календарь -> Google Calendar."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from . import caddy, sync
from .config import Config
from .db import Store, now_ms
from .google_cal import GoogleCalendar, MemoryCalendar

log = logging.getLogger("api")


def _setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    fh = RotatingFileHandler(Config.db_path.parent / ".." / "logs" / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


_setup_logging()


class State:
    store: Store
    last_incoming: list = []
    cal = None
    started_ms: int = 0
    hits: dict[str, list[float]] = {}


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    problems = Config.validate()
    if problems:
        log.error("конфигурация неполная: %s", "; ".join(problems))

    state.store = Store(Config.db_path)
    state.started_ms = now_ms()

    if Config.dry_run:
        state.cal = MemoryCalendar(Config.calendar_id, Config.tz_name)
        log.warning("режим dry-run: события пишутся в память, Google не затрагивается")
    elif Config.sa_json.exists():
        try:
            state.cal = GoogleCalendar(str(Config.sa_json), Config.calendar_id, Config.tz_name)
            log.info("календарь Google доступен: %s", state.cal.check())
        except Exception:  # noqa: BLE001
            log.exception("не удалось открыть календарь Google — проверьте доступ сервисного аккаунта")
            state.cal = None
    else:
        log.error("нет ключа сервисного аккаунта: %s", Config.sa_json)

    try:
        caddy.ensure(Config.domain, f"127.0.0.1:{Config.port}")
    except Exception:  # noqa: BLE001
        log.exception("Caddy: не удалось проверить маршрут")

    keeper = asyncio.create_task(_caddy_keeper())
    yield
    keeper.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await keeper
    state.store.close()


async def _caddy_keeper() -> None:
    """Раз в 5 минут проверяем, что маршрут в Caddy на месте (после рестарта он теряется)."""
    while True:
        await asyncio.sleep(300)
        try:
            caddy.ensure(Config.domain, f"127.0.0.1:{Config.port}")
        except Exception:  # noqa: BLE001
            log.warning("Caddy: проверка маршрута не удалась")


app = FastAPI(title="Calendar Bridge", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "?"


def _authorized(request: Request) -> bool:
    if not Config.token:
        return False
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not token:
        token = request.headers.get("x-sync-token", "").strip()
    return bool(token) and hmac.compare_digest(token, Config.token)


def _rate_limited(ip: str, limit: int = 60, window: float = 600.0) -> bool:
    now = time.time()
    hits = [t for t in state.hits.get(ip, []) if now - t < window]
    hits.append(now)
    state.hits[ip] = hits
    if len(state.hits) > 5000:  # защита от роста памяти
        state.hits = {k: v for k, v in state.hits.items() if v and now - v[-1] < window}
    return len(hits) > limit


@app.get("/healthz")
async def healthz():
    return {"ok": True, "ts": now_ms()}


@app.post("/sync")
async def sync_endpoint(request: Request):
    ip = _client_ip(request)
    if not _authorized(request):
        log.warning("отказ в авторизации с %s", _ip_hash(ip))
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)

    if _rate_limited(ip):
        log.warning("превышен лимит запросов с %s", _ip_hash(ip))
        return JSONResponse({"ok": False, "error": "rate_limited"}, status_code=429)

    body = await request.body()
    if len(body) > Config.max_body_bytes:
        return JSONResponse({"ok": False, "error": "payload_too_large"}, status_code=413)

    if state.cal is None:
        return JSONResponse({"ok": False, "error": "google_not_configured"}, status_code=503)

    now = now_ms()
    try:
        incoming, sent_ms, skipped = sync.parse_payload(body, now, Config.max_events, Config.window_days)
    except sync.BadPayload as exc:
        log.warning("некорректный запрос с %s: %s", _ip_hash(ip), exc)
        return JSONResponse({"ok": False, "error": "bad_payload", "detail": str(exc)}, status_code=400)
    except Exception:  # noqa: BLE001
        log.exception("не удалось разобрать запрос")
        return JSONResponse({"ok": False, "error": "bad_payload"}, status_code=400)
    if skipped:
        log.warning("пропущено событий при разборе: %s", skipped)

    try:
        result = sync.run_sync(state.store, state.cal, incoming, now, Config.delete_after_misses)
    except Exception:  # noqa: BLE001
        log.exception("ошибка синхронизации")
        return JSONResponse({"ok": False, "error": "sync_failed"}, status_code=500)

    if Config.dry_run:
        state.last_incoming = incoming
    log.info(
        "sync от %s: получено=%s создано=%s перенесено=%s переименовано=%s длит=%s без_изменений=%s удалено=%s подавлено=%s",
        _ip_hash(ip), result["received"], result["created"], result["moved"], result["retitled"],
        result["extended"], result["unchanged"], result["deleted"],
        "да" if result["suppressed_deletes"] else "нет",
    )
    return result


@app.get("/macro")
async def macro(request: Request):
    """Отдаёт .macro файл, если в query передан верный токен (?t=...)."""
    token = request.query_params.get("t", "")
    if not Config.token or not hmac.compare_digest(token, Config.token):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    path = Config.db_path.parent.parent / "macrodroid" / "calendar_bridge.macro"
    if not path.exists():
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    return Response(
        content=path.read_bytes(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="calendar_bridge.macro"'},
    )


@app.get("/debug/last")
async def debug_last(request: Request):
    """Только для dry-run: что именно пришло с телефона (проверка разбора)."""
    if not _authorized(request) or not Config.dry_run:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    from datetime import datetime, timedelta, timezone

    tz = timezone(timedelta(hours=3))
    rows = [
        {
            "title": i.title,
            "start": datetime.fromtimestamp(i.start_ms / 1000, tz).strftime("%Y-%m-%d %H:%M"),
            "minutes": int((i.end_ms - i.start_ms) / 60000),
            "all_day": i.all_day,
        }
        for i in sorted(state.last_incoming, key=lambda x: x.start_ms)
    ]
    return {"ok": True, "count": len(rows), "events": rows}


@app.get("/status")
async def status(request: Request):
    if not _authorized(request):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    stats = state.store.stats()
    runs = state.store.last_runs(10)
    last = runs[0] if runs else None
    return {
        "ok": True,
        "uptime_s": int((now_ms() - state.started_ms) / 1000),
        "calendar_id": Config.calendar_id,
        "dry_run": Config.dry_run,
        "google_ready": state.cal is not None,
        "stats": stats,
        "last_run": last,
        "recent_runs": runs,
    }


def _ip_hash(ip: str) -> str:
    salt = Config.token.encode() or b"no-token"
    return hashlib.sha256(salt + ip.encode()).hexdigest()[:12]
