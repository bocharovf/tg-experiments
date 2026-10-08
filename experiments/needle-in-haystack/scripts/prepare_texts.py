"""Download public-domain books from Project Gutenberg and trim them to a target
token length at a paragraph boundary, producing data/texts/*.txt.

Optional: the trimmed books are already committed to data/texts/, so you only
need this script to regenerate them (e.g. at a different target size).

Requirements (extra): `pip install tokenizers huggingface_hub requests`

Usage:
    python scripts/prepare_texts.py [--target-tokens 100000]
"""

import argparse
import json
import re
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TEXTS_DIR = DATA / "texts"

TOKENIZER_REPO = "deepseek-ai/DeepSeek-V3.2"

BOOKS = [
    {"id": "moby_dick", "title": "Moby-Dick; or, The Whale", "author": "Herman Melville", "gutenberg_id": 2701,
     "start": ("CHAPTER 1. Loomings.", 2)},
    {"id": "origin_of_species", "title": "On the Origin of Species", "author": "Charles Darwin", "gutenberg_id": 1228,
     "start": ("INTRODUCTION.", 2)},
    {"id": "voyage_of_beagle", "title": "The Voyage of the Beagle", "author": "Charles Darwin", "gutenberg_id": 944,
     "start": ("CHAPTER I", 1)},
]


def load_tokenizer() -> Tokenizer:
    path = hf_hub_download(TOKENIZER_REPO, "tokenizer.json")
    return Tokenizer.from_file(path)


def download(gutenberg_id: int) -> str:
    url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content.decode("utf-8-sig")


def strip_boilerplate(raw: str) -> str:
    raw = raw.replace("\r\n", "\n")
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", raw)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", raw)
    return raw[start.end() if start else 0 : end.start() if end else len(raw)]


def to_paragraphs(body: str) -> list[str]:
    """Unwrap hard-wrapped lines: one paragraph per list item."""
    paragraphs = []
    for block in re.split(r"\n\s*\n", body):
        text = " ".join(line.strip() for line in block.splitlines() if line.strip())
        text = re.sub(r"\s+", " ", text).replace("_", "")
        if text:
            paragraphs.append(text)
    return paragraphs


def skip_front_matter(paragraphs: list[str], marker: str, occurrence: int) -> list[str]:
    """Drop table of contents etc.: start at the n-th paragraph equal to `marker`."""
    hits = [i for i, p in enumerate(paragraphs) if p == marker]
    return paragraphs[hits[occurrence - 1]:]


def trim(paragraphs: list[str], tokenizer: Tokenizer, target: int) -> tuple[str, int]:
    kept, total = [], 0
    for p in paragraphs:
        n = len(tokenizer.encode("\n\n" + p, add_special_tokens=False).ids)
        if total + n > target:
            break
        kept.append(p)
        total += n
    text = "\n\n".join(kept)
    return text, len(tokenizer.encode(text, add_special_tokens=False).ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-tokens", type=int, default=100_000)
    args = parser.parse_args()

    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer()
    meta = []
    for book in BOOKS:
        paragraphs = to_paragraphs(strip_boilerplate(download(book["gutenberg_id"])))
        paragraphs = skip_front_matter(paragraphs, *book["start"])
        text, n_tokens = trim(paragraphs, tokenizer, args.target_tokens)
        (TEXTS_DIR / f"{book['id']}.txt").write_text(text, encoding="utf-8")
        meta.append({
            **{k: v for k, v in book.items() if k != "start"},
            "source": f"https://www.gutenberg.org/ebooks/{book['gutenberg_id']}",
            "language": "en",
            "file": f"texts/{book['id']}.txt",
            "tokens": n_tokens,
            "chars": len(text),
            "paragraphs": text.count("\n\n") + 1,
            "tokenizer": TOKENIZER_REPO,
        })
        print(f"{book['id']}: {n_tokens} tokens, {len(text)} chars")
    (DATA / "texts.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
