# -*- coding: utf-8 -*-
"""Обезличивание реквизитов в PDF: два варианта — вырезание и сохранение формата.

Гибридный подход (из эксперимента "llm-anonym", пост https://t.me/technosurvival/54):
  - регэкспы — для жёстких форматов (ИНН, КПП, ОГРН, БИК, счета, индексы);
  - NER (spaCy ru + en) — для мягких понятий (ФИО, организации, города, email).

На выходе два PDF:
  - <имя>_redacted.pdf             реквизиты закрыты чёрными полосами («вырезание»);
  - <имя>_format_preserving.pdf    синтетика с сохранённой структурой
                                   («сохранение формата»).

Запуск:
    python anonymize.py --input договор.pdf --output out/

Требования: presidio-analyzer, pymupdf, spacy + модели en_core_web_lg и
ru_core_news_lg. Проще всего запускать через Docker (см. README).
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider


# --------------------------------------------------------------------------- движок

def build_engine() -> AnalyzerEngine:
    nlp_conf = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_lg"},
            {"lang_code": "ru", "model_name": "ru_core_news_lg"},
        ],
    }
    provider = NlpEngineProvider(nlp_configuration=nlp_conf)
    engine = AnalyzerEngine(
        nlp_engine=provider.create_engine(), supported_languages=["en", "ru"]
    )

    def pr(entity: str, regex: str, score: float, ctx: list[str]) -> PatternRecognizer:
        return PatternRecognizer(
            supported_entity=entity,
            patterns=[Pattern(name=entity.lower(), regex=regex, score=score)],
            context=ctx,
            supported_language="ru",
        )

    for r in [
        pr("INN", r"\b\d{10}\b", 0.9, ["ИНН", "инн"]),
        pr("KPP", r"\b(?!04)\d{9}\b", 0.85, ["КПП", "кпп"]),
        pr("OGRN", r"\b\d{13}\b", 0.9, ["ОГРН", "огрн"]),
        pr("BIK", r"\b04\d{7}\b", 0.95, ["БИК", "бик"]),
        pr("ACCOUNT_NUMBER", r"\b\d{20}\b", 0.9, ["р/с", "к/с", "Р/с"]),
        pr("POSTAL_INDEX", r"\b\d{6}\b", 0.7, ["индекс", "Индекс"]),
        pr("PHONE_NUMBER", r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}", 0.95, ["тел", "Тел", "телефон"]),
    ]:
        engine.registry.add_recognizer(r)
    return engine


def analyze(engine: AnalyzerEngine, text: str, score_threshold: float):
    results = engine.analyze(text=text, language="ru", score_threshold=score_threshold)
    # дедупликация пересекающихся сущностей (выше скор)
    results = sorted(results, key=lambda r: (r.start, -float(r.score)))
    kept, last = [], -1
    for r in results:
        if r.start >= last:
            kept.append(r)
            last = r.end
    return kept


# --------------------------------------------------------------------------- вариант 1: вырезание

def entity_tokens(text: str, results) -> list[str]:
    toks = []
    for r in results:
        sub = text[r.start : r.end]
        for tok in re.split(r"\s+", sub):
            tok = tok.strip("()«»\"'")
            # не обрубаем инициалы «А.», отсекаем одиночные символы и короткие числа
            if len(tok) >= 2 and not (tok.isdigit() and len(tok) < 6):
                toks.append(tok)
    seen = set()
    return [t for t in toks if not (t in seen or seen.add(t))]


def redact_pdf(src: Path, dst: Path, text: str, results) -> int:
    doc = pymupdf.open(src)
    hits = 0
    for tok in entity_tokens(text, results):
        for page in doc:
            for rect in page.search_for(tok):
                x0, y0, x1, y1 = rect
                page.add_redact_annot(
                    pymupdf.Rect(x0, y0 - 1.5, x1, y1 + 1.5), fill=(0, 0, 0)
                )
                hits += 1
    for page in doc:
        page.apply_redactions()
    doc.save(dst, garbage=3, deflate=True)
    return hits


# --------------------------------------------------------------------------- вариант 2: сохранение формата

def fp_replace(sub: str, etype: str, rng: random.Random) -> str:
    def digits(n: int) -> str:
        return "".join(rng.choice("0123456789") for _ in range(n))

    if etype == "PHONE_NUMBER":
        out = list(sub)
        ds = list(re.finditer(r"\d", sub))
        for m in ds[4:]:
            out[m.start()] = rng.choice("0123456789")
        return "".join(out)
    if etype in ("INN", "KPP", "BIK", "POSTAL_INDEX"):
        return sub[:2] + digits(len(sub) - 2)
    if etype == "OGRN":
        return sub[:1] + digits(len(sub) - 1)
    if etype == "ACCOUNT_NUMBER":
        return sub[:5] + digits(len(sub) - 5)
    # не числовой формат -> читаемый плейсхолдер
    return f"[{etype}]"


def format_preserving(text: str, results, seed: int = 42) -> str:
    rng = random.Random(seed)
    out = text
    for r in sorted(results, key=lambda r: -r.start):
        sub = text[r.start : r.end]
        out = out[: r.start] + fp_replace(sub, r.entity_type, rng) + out[r.end :]
    return out


# --------------------------------------------------------------------------- текст из PDF

def pdf_text(path: Path) -> str:
    doc = pymupdf.open(path)
    return "\n".join(page.get_text() for page in doc)


FONTS = Path(__file__).resolve().parent / "fonts"


def render_pdf(text: str, dst: Path, font_regular: Path, font_bold: Path) -> None:
    """Перерендер обезличенного текста в PDF (кириллица через DejaVu)."""
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    pdfmetrics.registerFont(TTFont("DejaVu", str(font_regular)))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(font_bold)))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")

    title = ParagraphStyle("t", fontName="DejaVu-Bold", fontSize=15, leading=19,
                           alignment=TA_CENTER, spaceAfter=8)
    body = ParagraphStyle("b", fontName="DejaVu", fontSize=11, leading=15,
                          alignment=TA_JUSTIFY, spaceAfter=5)

    story = []
    prev_blank = False
    for i, ln in enumerate(text.split("\n")):
        s = ln.strip()
        if not s:
            if not prev_blank:
                story.append(Spacer(1, 6))
            prev_blank = True
            continue
        prev_blank = False
        story.append(Paragraph(s, title if i == 0 else body))

    SimpleDocTemplate(str(dst), pagesize=A4, leftMargin=56, rightMargin=56,
                      topMargin=50, bottomMargin=50).build(story)


# --------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="путь к PDF")
    parser.add_argument("--output", default="out", help="каталог для результатов (по умолчанию out/)")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="порог уверенности NER (по умолчанию 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="seed для синтетики (воспроизводимость)")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"нет файла: {src}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    text = pdf_text(src)
    print(f"{src.name}: {len(text)} символов текста")

    engine = build_engine()
    results = analyze(engine, text, args.score_threshold)
    print(f"найдено сущностей: {len(results)}")

    # отчёт в консоль
    c = Counter(r.entity_type for r in results)
    print("=== найденные сущности ===")
    for etype, n in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"  {etype}: {n}")
    for r in sorted(results, key=lambda r: r.start):
        print(f"  [{r.entity_type}] {text[r.start:r.end]!r} ({round(float(r.score), 2)})")

    # вариант 1: вырезание
    red_path = out_dir / f"{stem}_redacted.pdf"
    hits = redact_pdf(src, red_path, text, results)
    print(f"вырезание: {hits} полос -> {red_path.name}")

    # вариант 2: сохранение формата
    fp = format_preserving(text, results, args.seed)
    fp_pdf = out_dir / f"{stem}_format_preserving.pdf"
    render_pdf(fp, fp_pdf, FONTS / "DejaVuSans.ttf", FONTS / "DejaVuSans-Bold.ttf")
    print(f"сохранение формата -> {fp_pdf.name}")


if __name__ == "__main__":
    main()
