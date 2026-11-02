import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from app import sync  # noqa: E402
from app.db import Store  # noqa: E402
from app.google_cal import MemoryCalendar  # noqa: E402

H = 3600 * 1000
NOW = 1_800_000_000_000


def make(tmp_path):
    return Store(tmp_path / "s.sqlite3"), MemoryCalendar()


def payload(events):
    return json.dumps({"v": 1, "sent_at": NOW, "events": [list(e) for e in events]}).encode()


def run(store, cal, events):
    inc, _, _ = sync.parse_payload(payload(events), NOW, 500, 14)
    return sync.run_sync(store, cal, inc, NOW, delete_after_misses=2)


def test_full_cycle(tmp_path):
    store, cal = make(tmp_path)
    evs = [(NOW + H, 60, "Созвон"), (NOW + 3 * H, 30, "1:1"), (NOW + 24 * H, 60, "Планёрка")]

    r1 = run(store, cal, evs)
    assert r1["created"] == 3 and len(cal.events) == 3

    r2 = run(store, cal, evs)
    assert r2["created"] == 0 and r2["unchanged"] == 3 and len(cal.events) == 3

    moved = [(NOW + 2 * H, 60, "Созвон"), evs[1], evs[2]]
    r3 = run(store, cal, moved)
    assert r3["moved"] == 1 and r3["created"] == 0 and len(cal.events) == 3
    db_starts = sorted(e.start_ms for e in store.active())
    assert db_starts == sorted(s for s, _, _ in moved)

    # одиночная отмена удаляется сразу, чтобы слот освободился без задержки
    r4 = run(store, cal, moved[:2])
    assert r4["deleted"] == 1 and len(cal.events) == 2

    r5 = run(store, cal, moved[:2])
    assert r5["deleted"] == 0 and r5["unchanged"] == 2 and len(cal.events) == 2


def test_mass_wipe_guard(tmp_path):
    store, cal = make(tmp_path)
    evs = [(NOW + i * H, 30, f"Встреча {i}") for i in range(10)]
    run(store, cal, evs)
    assert len(cal.events) == 10

    r = run(store, cal, [])
    assert r["suppressed_deletes"] is True and r["deleted"] == 0
    assert len(cal.events) == 10


def test_duplicate_snapshot_no_doubles(tmp_path):
    store, cal = make(tmp_path)
    evs = [(NOW + H, 60, "Созвон"), (NOW + H, 60, "Созвон")]
    r = run(store, cal, evs)
    assert r["created"] == 1 and len(cal.events) == 1


def test_parse_skips_bad_window():
    inc, _, skipped = sync.parse_payload(payload([(NOW + 60 * 24 * H, 60, "Далеко")]), NOW, 500, 14)
    assert inc == [] and skipped == 1


def test_parse_rejects_too_many(tmp_path):
    with pytest.raises(sync.BadPayload):
        sync.parse_payload(payload([(NOW + H, 60, f"E{i}") for i in range(5)]), NOW, 3, 14)


def test_parse_accepts_objects():
    raw = json.dumps({"events": [{"s": NOW + H, "d": 45, "t": "Звонок"}]}).encode()
    inc, _, _ = sync.parse_payload(raw, NOW, 500, 14)
    assert inc[0].title == "Звонок" and inc[0].end_ms - inc[0].start_ms == 45 * 60_000


def test_title_sanitized():
    raw = json.dumps({"events": [[NOW + H, 30, "  Много   пробелов  "]]}).encode()
    inc, _, _ = sync.parse_payload(raw, NOW, 500, 14)
    assert inc[0].title == "Много пробелов"


def test_macrodroid_format():
    raw = json.dumps([
        {"Title": "Планёрка", "Start Timestamp": NOW + H, "Duration Minutes": 30,
         "Start": "2027-01-15T10:00:00+0300", "All day event": False},
        {"Title": "Отпуск", "Start Timestamp": NOW + 2 * 24 * H, "Duration Minutes": 0,
         "All day event": True},
    ]).encode()
    inc, _, skipped = sync.parse_payload(raw, NOW, 500, 14)
    assert skipped == 0 and len(inc) == 2
    assert inc[0].title == "Планёрка" and inc[0].end_ms - inc[0].start_ms == 30 * 60_000
    assert inc[1].all_day and inc[1].end_ms - inc[1].start_ms == 24 * 60 * 60_000


def test_all_day_roundtrip(tmp_path):
    store, cal = make(tmp_path)
    inc, _, _ = sync.parse_payload(json.dumps([{"Title": "Отпуск", "Start Timestamp": NOW + 24 * H,
                                               "Duration Minutes": 0, "All day event": True}]).encode(),
                                   NOW, 500, 14)
    r = sync.run_sync(store, cal, inc, NOW)
    assert r["created"] == 1
    ev = list(cal.events.values())[0]
    assert ev["all_day"] is True
    assert store.active()[0].all_day is True
