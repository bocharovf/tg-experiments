"""Проверка доступа сервисного аккаунта к календарю.

Использование: ./.venv/bin/python tools/check_google.py <calendar_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config  # noqa: E402
from app.google_cal import GoogleCalendar  # noqa: E402


def main() -> int:
    cal_id = sys.argv[1] if len(sys.argv) > 1 else Config.calendar_id
    if not cal_id:
        print("не задан calendar_id")
        return 2
    if not Config.sa_json.exists():
        print(f"нет ключа: {Config.sa_json}")
        return 2
    cal = GoogleCalendar(str(Config.sa_json), cal_id, Config.tz_name)
    print("календарь:", cal.check())
    print("событий синхронизатора в окне:", len(cal.list_managed(0, 4_000_000_000_000)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
