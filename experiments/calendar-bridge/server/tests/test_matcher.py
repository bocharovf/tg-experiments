import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import matcher  # noqa: E402
from app.matcher import Existing, Incoming  # noqa: E402

H = 3600 * 1000
T0 = 1_800_000_000_000


def E(uid, title, start, dur_h=1, miss=0):
    return Existing(uid=uid, gid="g" + uid, title=title, start_ms=start, end_ms=start + dur_h * H, miss=miss)


def I(title, start, dur_h=1):
    return Incoming(title=title, start_ms=start, end_ms=start + dur_h * H)


def test_first_import():
    p = matcher.plan([], [I("A", T0), I("B", T0 + 2 * H)], T0)
    assert len(p.create) == 2
    assert not p.move and not p.delete


def test_idempotent():
    ex = [E("1", "Созвон", T0)]
    p = matcher.plan(ex, [I("Созвон", T0)], T0)
    assert p.unchanged and not p.create and not p.move and not p.delete


def test_move_single():
    ex = [E("1", "Созвон", T0)]
    p = matcher.plan(ex, [I("Созвон", T0 + 3 * H)], T0)
    assert len(p.move) == 1 and p.move[0][0].uid == "1"
    assert not p.create and not p.delete


def test_duration_change():
    ex = [E("1", "Созвон", T0, dur_h=1)]
    p = matcher.plan(ex, [I("Созвон", T0, dur_h=2)], T0)
    assert len(p.extend) == 1 and not p.create and not p.move


def test_rename_same_slot():
    ex = [E("1", "Созвон", T0)]
    p = matcher.plan(ex, [I("Созвон с командой", T0)], T0)
    assert len(p.retitle) == 1 and not p.create and not p.delete


def test_single_cancel_deletes_immediately():
    ex = [E("1", "Созвон", T0)]
    p = matcher.plan(ex, [], T0)
    assert len(p.delete) == 1 and not p.miss


def test_bulk_disappearance_needs_two_misses():
    ex = [E(str(i), f"Встреча {i}", T0 + i * H) for i in range(4)]
    p1 = matcher.plan(ex, [], T0)
    assert not p1.delete and len(p1.miss) == 4
    ex2 = [E(str(i), f"Встреча {i}", T0 + i * H, miss=1) for i in range(4)]
    p2 = matcher.plan(ex2, [], T0)
    assert len(p2.delete) == 4


def test_mass_wipe_guard():
    ex = [E(str(i), f"Встреча {i}", T0 + i * H) for i in range(10)]
    p = matcher.plan(ex, [], T0)
    assert p.suppressed_deletes
    assert not p.delete and len(p.miss) == 10


def test_partial_shrink_guard():
    ex = [E(str(i), f"Встреча {i}", T0 + i * H) for i in range(10)]
    inc = [I(f"Встреча {i}", T0 + i * H) for i in range(3)]
    p = matcher.plan(ex, inc, T0)
    assert p.suppressed_deletes and not p.delete


def test_recurring_series_shift():
    ex = [E(str(i), "Дейли", T0 + i * 24 * H) for i in range(5)]
    inc = [I("Дейли", T0 + i * 24 * H + 2 * H) for i in range(5)]
    p = matcher.plan(ex, inc, T0)
    assert len(p.move) == 5 and not p.create and not p.delete
    assert sorted(x.start_ms for _, x in p.move) == sorted(i.start_ms for i in inc)


def test_same_title_twice_same_day():
    ex = [E("1", "1:1", T0), E("2", "1:1", T0 + 4 * H)]
    p = matcher.plan(ex, [I("1:1", T0), I("1:1", T0 + 4 * H)], T0)
    assert len(p.unchanged) == 2 and not p.create and not p.move


def test_past_events_untouched():
    ex = [E("1", "Старая", T0 - 100 * H)]
    p = matcher.plan(ex, [], T0)
    assert not p.delete and not p.miss


def test_duplicate_replacement():
    # встреча перенесена и одновременно появилась вторая с тем же названием
    ex = [E("1", "Созвон", T0)]
    inc = [I("Созвон", T0 + 2 * H), I("Созвон", T0 + 10 * H)]
    p = matcher.plan(ex, inc, T0)
    assert len(p.move) == 1 and len(p.create) == 1 and not p.delete
