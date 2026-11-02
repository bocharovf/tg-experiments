"""Сверка снимка рабочего календаря с тем, что уже лежит в Google.

Чистые функции без сети и БД — вся логика подбора проверяется тестами.
Смысл: телефон присылает полный снимок окна (2 недели). Каждый цикл нужно
понять, какие встречи создать, какие перенести/переименовать, какие удалить.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

MOVE_MAX_DELTA_MS = 21 * 24 * 3600 * 1000
PAST_GRACE_MS = 60 * 60 * 1000

_WS = re.compile(r"\s+")


def norm_title(title: str) -> str:
    t = unicodedata.normalize("NFC", title or "").strip()
    t = _WS.sub(" ", t)
    return t.casefold()


@dataclass
class Existing:
    uid: str
    gid: str
    title: str
    start_ms: int
    end_ms: int
    miss: int = 0
    all_day: bool = False

    @property
    def key(self) -> str:
        return norm_title(self.title)


@dataclass
class Incoming:
    title: str
    start_ms: int
    end_ms: int
    all_day: bool = False

    @property
    def key(self) -> str:
        return norm_title(self.title)


@dataclass
class Plan:
    create: list[Incoming] = field(default_factory=list)
    move: list[tuple[Existing, Incoming]] = field(default_factory=list)
    retitle: list[tuple[Existing, Incoming]] = field(default_factory=list)
    extend: list[tuple[Existing, Incoming]] = field(default_factory=list)
    unchanged: list[Existing] = field(default_factory=list)
    miss: list[Existing] = field(default_factory=list)
    delete: list[Existing] = field(default_factory=list)
    suppressed_deletes: bool = False


def _pair_by_time(pool: list[Incoming], cands: list[Existing]) -> list[tuple[Existing, Incoming]]:
    """Жадное сопоставление «ближайший по времени», 1:1.

    Порядок пар не важен: важно, что после применения набора пар множество
    (название, начало) станет ровно таким, как в снимке.
    """
    pairs: list[tuple[Existing, Incoming]] = []
    free = list(cands)
    for inc in sorted(pool, key=lambda i: i.start_ms):
        best = None
        best_delta = None
        for ex in free:
            delta = abs(ex.start_ms - inc.start_ms)
            if delta <= MOVE_MAX_DELTA_MS and (best_delta is None or delta < best_delta):
                best, best_delta = ex, delta
        if best is not None:
            pairs.append((best, inc))
            free.remove(best)
    return pairs


def plan(
    existing: list[Existing],
    incoming: list[Incoming],
    now_ms: int,
    delete_after_misses: int = 2,
    suppress_ratio: float = 0.5,
    suppress_min_existing: int = 5,
    fast_delete_max: int = 3,
) -> Plan:
    p = Plan()
    # Прошлые встречи не трогаем: телефон присылает только будущее окно.
    relevant = [e for e in existing if e.start_ms >= now_ms - PAST_GRACE_MS]

    exact: dict[tuple[str, int], list[Existing]] = {}
    for ex in relevant:
        exact.setdefault((ex.key, ex.start_ms), []).append(ex)

    matched: set[str] = set()
    left_in: list[Incoming] = []
    done_in: set[int] = set()

    # 1. Точное совпадение по названию и началу.
    for idx, inc in enumerate(incoming):
        bucket = exact.get((inc.key, inc.start_ms))
        while bucket:
            ex = bucket.pop(0)
            if ex.uid in matched:
                continue
            matched.add(ex.uid)
            done_in.add(idx)
            if ex.end_ms != inc.end_ms:
                p.extend.append((ex, inc))
            else:
                p.unchanged.append(ex)
            break
        else:
            left_in.append(inc)

    left_ex = [e for e in relevant if e.uid not in matched]

    # 2. Переименование на том же слоте — только если пара однозначна.
    by_start_in: dict[int, list[Incoming]] = {}
    for inc in left_in:
        by_start_in.setdefault(inc.start_ms, []).append(inc)
    by_start_ex: dict[int, list[Existing]] = {}
    for ex in left_ex:
        by_start_ex.setdefault(ex.start_ms, []).append(ex)
    renamed: set[int] = set()
    for start, incs in by_start_in.items():
        exs = by_start_ex.get(start) or []
        if len(incs) == 1 and len(exs) == 1 and incs[0].key != exs[0].key:
            p.retitle.append((exs[0], incs[0]))
            matched.add(exs[0].uid)
            renamed.add(id(incs[0]))

    left_in = [i for i in left_in if id(i) not in renamed]
    left_ex = [e for e in left_ex if e.uid not in matched]

    # 3. Перенос: остатки группируем по названию и подбираем ближайшее время.
    in_by_key: dict[str, list[Incoming]] = {}
    for inc in left_in:
        in_by_key.setdefault(inc.key, []).append(inc)
    ex_by_key: dict[str, list[Existing]] = {}
    for ex in left_ex:
        ex_by_key.setdefault(ex.key, []).append(ex)

    for key, incs in in_by_key.items():
        exs = ex_by_key.get(key) or []
        if not exs:
            p.create.extend(incs)
            continue
        pairs = _pair_by_time(incs, exs)
        paired_in = {id(i) for _, i in pairs}
        for ex, inc in pairs:
            matched.add(ex.uid)
            if ex.end_ms != inc.end_ms:
                p.move.append((ex, inc))
            else:
                p.move.append((ex, inc))
        p.create.extend([i for i in incs if id(i) not in paired_in])

    # 4. Остатки: либо ждём подтверждения удаления, либо удаляем.
    leftovers = [e for e in relevant if e.uid not in matched]

    total_existing = len(relevant)
    total_incoming = len(incoming)
    suspicious = False
    if leftovers:
        # Пустой/сильно урезанный снимок при заметном числе событий — почти наверняка
        # сбой на телефоне, а не отмена всех встреч. Удаления в таком цикле не делаем.
        if total_existing >= suppress_min_existing and total_incoming < total_existing * suppress_ratio:
            suspicious = True

    if suspicious:
        p.suppressed_deletes = True
        p.miss.extend(leftovers)
    else:
        # Одиночные отмены (самый частый случай) удаляем сразу, чтобы слот
        # освобождался за один цикл. Для массовых исчезновений оставляем
        # двухшаговое подтверждение — на случай сбоя синхронизации на телефоне.
        threshold = 1 if len(leftovers) <= fast_delete_max else delete_after_misses
        for ex in leftovers:
            if ex.miss + 1 >= threshold:
                p.delete.append(ex)
            else:
                p.miss.append(ex)
    return p
