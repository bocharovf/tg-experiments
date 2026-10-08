"""Needle-in-a-haystack — does "lost in the middle" survive a careful prompt?

A classic long-context experiment: numeric "needles" are inserted into a
~115K-token haystack at evenly spaced positions (0%..100%), then the model is
asked about them. Two tasks:

  * `single` — "what is the value of marker m5?"  (retrieval control)
  * `sum`    — "what is the sum of m5 and m6?"     (must have read BOTH facts)

Two prompt styles are compared:
  * `neutral` — document + question, no extraction hint.
  * `primed`  — "answer using only the document; every value is stated in it."

The gap between the two styles, and the shape of accuracy vs. position, is the
"lost in the middle" effect.

Usage:
    python needle.py run --model openai/gpt-4o-mini
    python needle.py run --model deepseek/deepseek-v3.2 --task both --runs 3
    python needle.py run --dry-run                # build document only, no API
    python needle.py report results/<file>.csv    # accuracy table
    python scripts/plot.py                        # accuracy-vs-position chart

Any OpenAI-compatible provider works: pass --base-url and --api-key-env.
Defaults point at VseLLM (https://api.vsellm.ru/v1, VSELLM_API_KEY).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

DEFAULT_BASE_URL = "https://api.vsellm.ru/v1"
DEFAULT_API_KEY_ENV = "VSELLM_API_KEY"

NEUTRAL_INSTR = "Answer with just the number, without explanation."
PRIMED_INSTR = (
    "Answer using only the document above; every value you need is stated in it. "
    "Reply with just the number, without explanation."
)

FIELDS = [
    "timestamp", "model", "prompt_style", "task", "run", "qid",
    "pos_a", "pos_b", "id_a", "id_b", "value_a", "value_b", "expected",
    "answer", "correct", "prompt_tokens", "completion_tokens", "latency_s",
]


# --------------------------------------------------------------------------- utils

def load_env(path: Path = ROOT / ".env") -> None:
    """Minimal .env support: KEY=VALUE lines, existing env variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"\''))


def normalize(s: str) -> str:
    """Lowercase, strip accents/punctuation, collapse spaces; keep digits intact."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"(?<=\d)[, ](?=\d{3}\b)", "", s)  # 11,400 / 11 400 -> 11400
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- data

def build_haystack(target_chars: int) -> str:
    """One ~target-chars haystack from the bundled public-domain books."""
    parts = []
    for name in ("moby_dick.txt", "origin_of_species.txt", "voyage_of_beagle.txt"):
        parts.append((DATA / "texts" / name).read_text(encoding="utf-8"))
    body = "\n\n".join(parts)
    if target_chars <= len(body):
        return body[:target_chars]
    return (body * ((target_chars // len(body)) + 1))[:target_chars]


def make_needles(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return [
        {"id": f"m{i + 1}", "value": rng.randint(11, 499)}
        for i in range(n)
    ]


def insert_at_boundary(text: str, pct: float, sentence: str) -> tuple[str, int]:
    """Insert `sentence` as its own paragraph nearest to pct% of the text."""
    target = int(len(text) * pct / 100)
    k = text.rfind("\n\n", 0, max(target, 1))
    if k < 0:
        k = target
    else:
        k += 2  # after the blank-line separator
    return text[:k] + sentence + "\n\n" + text[k:], k


def build_document(target_chars: int, needles: list[dict]) -> tuple[str, list[dict]]:
    doc = build_haystack(target_chars)
    n = len(needles)
    placements = []
    # insert back-to-front so earlier offsets stay valid
    for i in range(n - 1, -1, -1):
        pct = i * 100 / (n - 1)
        sentence = f"Marker '{needles[i]['id']}' has the value {needles[i]['value']}."
        doc, offset = insert_at_boundary(doc, pct, sentence)
        placements.append({"offset": offset, **needles[i]})
    placements.sort(key=lambda p: p["offset"])
    for p in placements:
        p["pos_pct"] = round(100 * p["offset"] / len(doc))
    return doc, placements


def make_questions(placements: list[dict], task: str) -> list[dict]:
    """Build questions. `single` asks one marker's value; `sum` sums an adjacent pair."""
    qs: list[dict] = []
    if task in ("single", "both"):
        for p in placements:
            qs.append({
                "task": "single", "qid": f"s-{p['id']}",
                "id_a": p["id"], "value_a": p["value"], "pos_a": p["pos_pct"],
                "id_b": "", "value_b": None, "pos_b": None,
                "expected": p["value"],
                "question": f"What is the value of marker '{p['id']}'?",
            })
    if task in ("sum", "both"):
        for i in range(len(placements) - 1):
            qs.append({
                "task": "sum", "qid": f"q{i + 1}",
                "id_a": placements[i]["id"], "value_a": placements[i]["value"],
                "pos_a": placements[i]["pos_pct"],
                "id_b": placements[i + 1]["id"], "value_b": placements[i + 1]["value"],
                "pos_b": placements[i + 1]["pos_pct"],
                "expected": placements[i]["value"] + placements[i + 1]["value"],
                "question": (
                    f"What is the sum of the values of markers "
                    f"'{placements[i]['id']}' and '{placements[i + 1]['id']}'?"
                ),
            })
    return qs


# --------------------------------------------------------------------------- grading

def extract_numbers(text: str) -> list[int]:
    norm = normalize(text)
    return [int(x) for x in re.findall(r"\d+", norm)]


def is_correct(answer: str, expected: int) -> bool:
    """Strict: the exact sum must appear as a whole number; case/extra words OK."""
    return expected in extract_numbers(answer)


# --------------------------------------------------------------------------- commands

def cmd_run(args) -> None:
    load_env()
    RESULTS.mkdir(exist_ok=True)

    styles = ["neutral", "primed"] if args.prompt_style == "both" else [args.prompt_style]

    if args.dry_run:
        needles = make_needles(args.needles, args.seed)
        doc, placements = build_document(args.target_chars, needles)
        for p in placements:
            print(f"  {p['pos_pct']:>3}%  {p['id']} = {p['value']}")
        print(f"doc chars={len(doc)}")
        return

    from openai import OpenAI

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        sys.exit(f"set {args.api_key_env} (environment or .env)")
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout, max_retries=args.retries)

    out = Path(args.output) if args.output else RESULTS / f"v2_{args.model.replace('/', '_')}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    done = set()
    if out.exists():
        with out.open(encoding="utf-8", newline="") as f:
            done = {(r["model"], r["prompt_style"], r["run"], r["qid"]) for r in csv.DictReader(f)}
        print(f"resuming {out}: {len(done)} answers already there")
    new_file = not out.exists()
    fh = out.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    # base document/placements (seed args.seed == run 1), used only for the total
    needles = make_needles(args.needles, args.seed)
    doc, placements = build_document(args.target_chars, needles)
    questions = make_questions(placements, args.task)
    total = len(questions) * len(styles) * args.runs
    print(f"{len(doc)} chars, {len(placements)} markers, {len(questions)} questions "
          f"(task={args.task}), styles={styles}, runs={args.runs}")

    counter = {"n": len(done), "ok": 0, "failed": 0}
    lock = threading.Lock()

    def process(document: str, style: str, run: int, q: dict) -> None:
        instr = PRIMED_INSTR if style == "primed" else NEUTRAL_INSTR
        prompt = f"{document}\n\n{q['question']}\n\n{instr}"
        try:
            started = time.monotonic()
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=args.max_tokens,
            )
            answer = (resp.choices[0].message.content or "").strip()
            correct = is_correct(answer, q["expected"])
            usage = resp.usage
            with lock:
                writer.writerow({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "model": args.model, "prompt_style": style, "task": q["task"],
                    "run": run, "qid": q["qid"],
                    "pos_a": q["pos_a"], "pos_b": q["pos_b"],
                    "id_a": q["id_a"], "id_b": q["id_b"],
                    "value_a": q["value_a"], "value_b": q["value_b"], "expected": q["expected"],
                    "answer": answer, "correct": int(correct),
                    "prompt_tokens": usage.prompt_tokens if usage else None,
                    "completion_tokens": usage.completion_tokens if usage else None,
                    "latency_s": round(time.monotonic() - started, 2),
                })
                fh.flush()
                counter["n"] += 1
                counter["ok"] += correct
            mark = "+" if correct else "-"
            print(f"  [{counter['n']}/{total}] {style:7} {q['task']:6} {mark} pos={q['pos_a']:>3}% "
                  f"{q['id_a']}+{q['id_b']}={q['expected']}: {answer[:40]!r}")
        except Exception as e:  # noqa: BLE001 - keep going, row retried on resume
            with lock:
                counter["failed"] += 1
            print(f"  ERROR {style} {run} {q['qid']}: {e}", file=sys.stderr)

    for run in range(1, args.runs + 1):
        # run 1 reuses the base seed; later runs get fresh values so positions are
        # sampled independently (values rotate across positions between runs).
        run_needles = make_needles(args.needles, args.seed + run - 1)
        run_doc, run_placements = build_document(args.target_chars, run_needles)
        run_questions = make_questions(run_placements, args.task)
        for style in styles:
            todo = [q for q in run_questions if (args.model, style, str(run), q["qid"]) not in done]
            if not todo:
                continue
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                list(pool.map(lambda q: process(run_doc, style, run, q), todo))

    fh.close()
    print(f"\nsaved {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    if counter["failed"]:
        print(f"{counter['failed']} requests failed; rerun with --output {out} to retry them")
    report(out)


def report(path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("no results")
        return
    styles = sorted({r["prompt_style"] for r in rows})
    tasks = sorted({r["task"] for r in rows})
    models = sorted({r["model"] for r in rows})
    print(f"\nmodel(s): {', '.join(models)}")

    for task in tasks:
        subset = [r for r in rows if r["task"] == task]
        positions = sorted({int(r["pos_a"]) for r in subset})
        stats: dict[tuple[str, int], list[int]] = {}
        for r in subset:
            stats.setdefault((r["prompt_style"], int(r["pos_a"])), []).append(int(r["correct"]))

        def cell(style, pos):
            v = stats.get((style, pos))
            return f"{100 * sum(v) / len(v):.0f}% ({sum(v)}/{len(v)})" if v else "-"

        print(f"\n== task={task} — accuracy by position")
        header = "  " + "  ".join(f"{s:>14}" for s in styles)
        print(f"{'pos':>5}{header}")
        for pos in positions:
            print(f"{pos:>4}% " + "  ".join(f"{cell(s, pos):>14}" for s in styles))
        for s in styles:
            v = [int(r["correct"]) for r in subset if r["prompt_style"] == s]
            print(f"{s}: {sum(v)}/{len(v)} = {100 * sum(v) / len(v):.1f}%")

    wrong = [r for r in rows if r["correct"] == "0"]
    if wrong:
        print("\nwrong answers:")
        for r in wrong:
            print(f"  {r['prompt_style']:7} {r['task']:6} pos={r['pos_a']:>3}% {r['id_a']}+{r['id_b']} "
                  f"expected {r['expected']}, got {r['answer'][:60]!r}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--model", required=True, help="e.g. openai/gpt-4o-mini, deepseek/deepseek-v3.2")
    run.add_argument("--base-url", default=DEFAULT_BASE_URL)
    run.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    run.add_argument("--target-chars", type=int, default=500000, help="haystack size in chars (~120K tokens)")
    run.add_argument("--needles", type=int, default=12, help="number of markers inserted")
    run.add_argument("--prompt-style", choices=["neutral", "primed", "both"], default="both")
    run.add_argument("--task", choices=["single", "sum", "both"], default="both",
                     help="single = one marker's value (retrieval control); sum = sum of two")
    run.add_argument("--runs", type=int, default=1)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--max-tokens", type=int, default=32)
    run.add_argument("--concurrency", type=int, default=2)
    run.add_argument("--timeout", type=float, default=600)
    run.add_argument("--retries", type=int, default=3)
    run.add_argument("--output", help="CSV path; resume by pointing at an existing file")
    run.add_argument("--dry-run", action="store_true", help="only build the document, print markers")
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report")
    rep.add_argument("path", type=Path)
    rep.set_defaults(func=lambda a: report(a.path))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
