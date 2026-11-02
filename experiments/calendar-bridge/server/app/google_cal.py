"""Клиент Google Calendar (сервисный аккаунт) + in-memory заглушка."""
from __future__ import annotations

import logging
import time
from datetime import datetime

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

log = logging.getLogger("google")

SCOPES = ["https://www.googleapis.com/auth/calendar"]
SYNC_TAG = "bridgeSync"


class GoogleError(RuntimeError):
    pass


def _with_retry(fn, *args, **kwargs):
    delay = 1.0
    last = None
    for attempt in range(4):
        try:
            return fn(*args, **kwargs).execute()
        except HttpError as exc:
            last = exc
            status = getattr(exc.resp, "status", 0)
            if status in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except Exception as exc:  # сетевые сбои
            last = exc
            if attempt < 3:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise GoogleError(str(last))


class GoogleCalendar:
    """Запись в конкретный календарь Google."""

    def __init__(self, sa_json: str, calendar_id: str, tz_name: str):
        creds = service_account.Credentials.from_service_account_file(sa_json, scopes=SCOPES)
        self.svc = build("calendar", "v3", credentials=creds, cache_discovery=False)
        self.calendar_id = calendar_id
        self.tz_name = tz_name
        self._tz = None

    def _date(self, ms: int) -> str:
        if self._tz is None:
            from zoneinfo import ZoneInfo

            self._tz = ZoneInfo(self.tz_name)
        return datetime.fromtimestamp(ms / 1000, tz=self._tz).date().isoformat()

    def _when(self, ms: int, all_day: bool) -> dict:
        if all_day:
            return {"date": self._date(ms)}
        return {"dateTime": self._iso(ms), "timeZone": self.tz_name}

    def _iso(self, ms: int) -> str:
        if self._tz is None:
            from zoneinfo import ZoneInfo

            self._tz = ZoneInfo(self.tz_name)
        return datetime.fromtimestamp(ms / 1000, tz=self._tz).isoformat()

    def check(self) -> str:
        meta = _with_retry(self.svc.calendars().get, calendarId=self.calendar_id)
        return meta.get("summary", "")

    def list_managed(self, time_min_ms: int, time_max_ms: int) -> list[dict]:
        """Все события, которые создали мы (по private extended property)."""
        out: list[dict] = []
        page = None
        while True:
            resp = _with_retry(
                self.svc.events().list,
                calendarId=self.calendar_id,
                timeMin=self._iso(time_min_ms),
                timeMax=self._iso(time_max_ms),
                singleEvents=True,
                showDeleted=False,
                maxResults=2500,
                privateExtendedProperty=f"{SYNC_TAG}=1",
                pageToken=page,
            )
            out.extend(resp.get("items", []))
            page = resp.get("nextPageToken")
            if not page:
                break
        return out

    def insert(self, uid: str, title: str, start_ms: int, end_ms: int, all_day: bool = False) -> str:
        body = {
            "summary": title,
            "start": self._when(start_ms, all_day),
            "end": self._when(end_ms, all_day),
            "extendedProperties": {"private": {SYNC_TAG: "1", "uid": uid}},
            "reminders": {"useDefault": False, "overrides": []},
            "transparency": "opaque",
        }
        res = _with_retry(
            self.svc.events().insert, calendarId=self.calendar_id, body=body, sendUpdates="none"
        )
        return res["id"]

    def patch(self, gid: str, title: str | None, start_ms: int | None, end_ms: int | None,
              all_day: bool = False) -> None:
        body: dict = {}
        if title is not None:
            body["summary"] = title
        if start_ms is not None:
            body["start"] = self._when(start_ms, all_day)
        if end_ms is not None:
            body["end"] = self._when(end_ms, all_day)
        if not body:
            return
        _with_retry(
            self.svc.events().patch,
            calendarId=self.calendar_id,
            eventId=gid,
            body=body,
            sendUpdates="none",
        )

    def delete(self, gid: str) -> None:
        try:
            _with_retry(self.svc.events().delete, calendarId=self.calendar_id, eventId=gid, sendUpdates="none")
        except HttpError as exc:
            if getattr(exc.resp, "status", 0) in (404, 410):
                return
            raise


class MemoryCalendar:
    """Заглушка для локальных тестов: тот же интерфейс, данные в памяти."""

    def __init__(self, calendar_id: str = "test", tz_name: str = "Europe/Moscow"):
        self.calendar_id = calendar_id
        self.tz_name = tz_name
        self._seq = 0
        self.events: dict[str, dict] = {}

    def check(self) -> str:
        return "Dry-run calendar"

    def list_managed(self, time_min_ms: int, time_max_ms: int) -> list[dict]:
        return [dict(id=gid, **ev) for gid, ev in self.events.items()
                if time_min_ms <= ev["start_ms"] <= time_max_ms]

    def insert(self, uid: str, title: str, start_ms: int, end_ms: int, all_day: bool = False) -> str:
        self._seq += 1
        gid = f"mem{self._seq}"
        self.events[gid] = {
            "summary": title,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "all_day": all_day,
            "extendedProperties": {"private": {SYNC_TAG: "1", "uid": uid}},
        }
        return gid

    def patch(self, gid: str, title, start_ms, end_ms, all_day: bool = False) -> None:
        ev = self.events[gid]
        if title is not None:
            ev["summary"] = title
        if start_ms is not None:
            ev["start_ms"] = start_ms
            ev["end_ms"] = end_ms
            ev["all_day"] = all_day

    def delete(self, gid: str) -> None:
        self.events.pop(gid, None)


def _moscow():
    from zoneinfo import ZoneInfo

    return ZoneInfo("Europe/Moscow")
