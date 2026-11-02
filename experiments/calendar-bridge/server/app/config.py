"""Конфигурация сервиса: читается из .env рядом с пакетом."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
SECRETS_DIR = BASE_DIR / "secrets"


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # переменные окружения имеют приоритет над файлом
        os.environ.setdefault(key, value)


_load_env(BASE_DIR / ".env")


def _bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    token: str = os.environ.get("SYNC_TOKEN", "")
    calendar_id: str = os.environ.get("CALENDAR_ID", "")
    sa_json: Path = Path(os.environ.get("GOOGLE_SA_JSON", str(SECRETS_DIR / "service-account.json")))
    tz_name: str = os.environ.get("TZ_NAME", "Europe/Moscow")
    host: str = os.environ.get("BIND_HOST", "127.0.0.1")
    port: int = int(os.environ.get("PORT", "8787"))
    domain: str = os.environ.get("DOMAIN", "sync.example.com")
    db_path: Path = Path(os.environ.get("DB_PATH", str(DATA_DIR / "state.sqlite3")))
    dry_run: bool = _bool("GOOGLE_DRY_RUN", False)
    # окно, за которое телефон присылает события (дни) — с запасом
    window_days: int = int(os.environ.get("WINDOW_DAYS", "14"))
    max_body_bytes: int = int(os.environ.get("MAX_BODY_BYTES", "262144"))
    max_events: int = int(os.environ.get("MAX_EVENTS", "500"))
    # сколько подряд пропусков нужно, чтобы удалить встречу
    delete_after_misses: int = int(os.environ.get("DELETE_AFTER_MISSES", "2"))

    @classmethod
    def validate(cls) -> list[str]:
        problems = []
        if len(cls.token) < 24:
            problems.append("SYNC_TOKEN пустой или короче 24 символов")
        if not cls.calendar_id:
            problems.append("CALENDAR_ID не задан")
        if not cls.dry_run and not cls.sa_json.exists():
            problems.append(f"нет ключа сервисного аккаунта: {cls.sa_json}")
        return problems


for _d in (DATA_DIR, LOG_DIR, SECRETS_DIR):
    _d.mkdir(mode=0o700, parents=True, exist_ok=True)
