"""Движок синхронизации: снимок с телефона -> операции в Google Calendar."""
from __future__ import annotations

import json
import logging
import re
import uuid

from . import matcher
from .db import Store, now_ms
from .matcher import Incoming

log = logging.getLogger("sync")

MAX_TITLE = 200
# Отменённые/отклонённые встречи Exchange остаются в календаре с префиксом в теме
CANCELLED_PREFIXES = (
    "canceled:", "cancelled:", "declined:", "отменено:", "отменён:", "отклонено:",
    "[canceled]", "[cancelled]", "[отменено]", "(canceled)", "(cancelled)",
)
MIN_DUR_MIN = 1
MAX_DUR_MIN = 7 * 24 * 60


class BadPayload(ValueError):
    pass


def parse_payload(raw: bytes, now: int, max_events: int, window_days: int) -> tuple[list[Incoming], int | None]:
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise BadPayload("тело запроса не JSON") from exc

    if isinstance(data, list):
        data = {"events": data}
    if not isinstance(data, dict):
        raise BadPayload("ожидался объект")

    sent = data.get("sent_at") or data.get("sentAt")
    sent_ms = int(sent) if isinstance(sent, (int, float)) else None

    raw_events = data.get("events")
    if raw_events is None:
        # MacroDroid с выводом в словарь: {название: {поля события}, ...}
        # В словаре поле Title отсутствует — название лежит в ключе, а к дублям
        # MacroDroid дописывает номер ("Планёрка 2"). Восстанавливаем.
        values = list(data.values())
        if values and all(isinstance(v, dict) for v in values) and any(
            "Duration Minutes" in v or "Start Timestamp" in v for v in values
        ):
            keys = set(data)
            raw_events = []
            for key, value in data.items():
                item = dict(value)
                if not item.get("Title"):
                    base = _strip_suffix(key)
                    item["Title"] = base if (base != key and base in keys) else key
                raw_events.append(item)
    if not isinstance(raw_events, list):
        raise BadPayload("нет массива events")
    if len(raw_events) > max_events:
        raise BadPayload(f"слишком много событий ({len(raw_events)} > {max_events})")

    low = now - 3 * 24 * 3600 * 1000
    high = now + (window_days + 7) * 24 * 3600 * 1000

    out: list[Incoming] = []
    seen: set[tuple[str, int, int, bool]] = set()
    skipped = 0
    cancelled = 0
    for item in raw_events:
        parsed = _extract(item)
        if parsed is None:
            skipped += 1
            continue
        start_ms, dur_min, title, all_day = parsed

        if all_day and dur_min < 60:
            dur_min = 24 * 60
        if not (low <= start_ms <= high) or not (MIN_DUR_MIN <= dur_min <= MAX_DUR_MIN):
            skipped += 1
            continue

        title = " ".join(str(title or "").split())[:MAX_TITLE] or "Без названия"
        if title.casefold().startswith(CANCELLED_PREFIXES):
            cancelled += 1
            continue
        end_ms = start_ms + dur_min * 60_000

        key = (matcher.norm_title(title), start_ms, end_ms, all_day)
        if key in seen:
            continue
        seen.add(key)
        out.append(Incoming(title=title, start_ms=start_ms, end_ms=end_ms, all_day=all_day))

    out.sort(key=lambda i: i.start_ms)
    return out, sent_ms, skipped + cancelled

def _strip_suffix(key: str) -> str:
    """Убирает хвостовой номер, который MacroDroid добавляет к повторяющимся ключам."""
    return re.sub(r" \d+$", "", key)


def _extract(item) -> tuple[int, int, str, bool] | None:
    """Достаёт (start_ms, duration_min, title, all_day) из элемента любого формата."""
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        try:
            return int(item[0]), int(item[1]), item[2], False
        except (TypeError, ValueError):
            return None

    if not isinstance(item, dict):
        return None

    # компактный формат
    start = item.get("s", item.get("start"))

    # формат действия Get Calendar Events (MacroDroid)
    if "Start Timestamp" in item or "Duration Minutes" in item:
        start = item.get("Start Timestamp", start)
        dur = item.get("Duration Minutes")
        title = item.get("Title", "")
        all_day = bool(item.get("All day event") or item.get("All Day Event"))
        if dur is None and item.get("Start") and item.get("End"):
            try:
                from datetime import datetime

                s_dt = datetime.fromisoformat(str(item["Start"]))
                e_dt = datetime.fromisoformat(str(item["End"]))
                dur = int((e_dt - s_dt).total_seconds() // 60)
            except Exception:  # noqa: BLE001
                return None
    else:
        dur = item.get("d", item.get("dur"))
        title = item.get("t", item.get("title", ""))
        all_day = bool(item.get("all_day"))

    try:
        return int(start), int(dur or 0), title, all_day
    except (TypeError, ValueError):
        return None


def run_sync(
    store: Store,
    cal,
    incoming: list[Incoming],
    now: int,
    delete_after_misses: int = 2,
) -> dict:
    existing = store.active()
    p = matcher.plan(existing, incoming, now, delete_after_misses=delete_after_misses)

    created = moved = retitled = extended = unchanged = deleted = 0

    for inc in p.create:
        uid = uuid.uuid4().hex
        try:
            gid = cal.insert(uid, inc.title, inc.start_ms, inc.end_ms, inc.all_day)
        except Exception:  # noqa: BLE001
            log.exception("не удалось создать событие")
            continue
        store.add(uid, gid, inc.title, inc.start_ms, inc.end_ms, inc.all_day)
        created += 1

    for ex, inc in p.extend:
        try:
            cal.patch(ex.gid, None, None, inc.end_ms, inc.all_day)
        except Exception:  # noqa: BLE001
            log.exception("не удалось изменить длительность")
            continue
        store.touch(ex.uid, inc.title, inc.start_ms, inc.end_ms, inc.all_day)
        extended += 1

    for ex, inc in p.move:
        try:
            cal.patch(ex.gid, None, inc.start_ms, inc.end_ms, inc.all_day)
        except Exception:  # noqa: BLE001
            log.exception("не удалось перенести событие")
            continue
        store.touch(ex.uid, inc.title, inc.start_ms, inc.end_ms, inc.all_day)
        moved += 1

    for ex, inc in p.retitle:
        try:
            cal.patch(ex.gid, inc.title, None, None, inc.all_day)
        except Exception:  # noqa: BLE001
            log.exception("не удалось переименовать событие")
            continue
        store.touch(ex.uid, inc.title, inc.start_ms, inc.end_ms, inc.all_day)
        retitled += 1

    for ex in p.unchanged:
        if ex.miss:
            store.touch(ex.uid, ex.title, ex.start_ms, ex.end_ms, ex.all_day)
        unchanged += 1

    for ex in p.miss:
        store.mark_miss(ex.uid)

    for ex in p.delete:
        try:
            cal.delete(ex.gid)
        except Exception:  # noqa: BLE001
            log.exception("не удалось удалить событие")
            continue
        store.mark_deleted(ex.uid)
        deleted += 1

    result = {
        "ok": True,
        "received": len(incoming),
        "created": created,
        "moved": moved,
        "retitled": retitled,
        "extended": extended,
        "unchanged": unchanged,
        "deleted": deleted,
        "suppressed_deletes": p.suppressed_deletes,
    }
    store.record_run(
        ts_ms=now_ms(),
        ip_hash=None,
        sent_ms=None,
        received=len(incoming),
        created=created,
        moved=moved,
        retitled=retitled,
        extended=extended,
        unchanged=unchanged,
        deleted=deleted,
        suppressed=int(p.suppressed_deletes),
    )
    log.info(
        "sync received=%s created=%s moved=%s retitled=%s extended=%s unchanged=%s deleted=%s suppressed=%s",
        len(incoming), created, moved, retitled, extended, unchanged, deleted, p.suppressed_deletes,
    )
    return result
