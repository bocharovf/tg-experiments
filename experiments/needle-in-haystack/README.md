# Needle-in-a-haystack (lost in the middle)

Эксперимент из поста: **https://t.me/technosurvival/54**

Проверяем, правда ли LLM «теряет» факты в середине длинного контекста
(эффект *lost in the middle*). В текст ~115K токенов вставляются 12 числовых
«иголок» на позициях 0%–100%, а модель спрашивают про них двумя способами:

- `single` — «какое значение у маркера m5?» (извлечение одного факта);
- `sum` — «какова сумма m5 и m6?» (нужно прочитать **оба** факта и свести их).

Кривая точности по позиции — и есть тот самый график «слепого пятна».

## Что нужно на компьютере

- Python 3.10+ (проверено на 3.14);
- `pip` (идёт вместе с Python).

Больше ничего: стога сена уже лежат в репозитории (`data/texts/`), отдельно
скачивать их не нужно.

## Доступ к LLM

По умолчанию всё работает через **VseLLM** (`https://api.vsellm.ru/v1`,
OpenAI-совместимый роутер). Нужен ключ:

1. Зарегистрируйся на [vsellm.ru](https://vsellm.ru) и пополни баланс.
2. Скопируй API-ключ.
3. `cp .env.example .env` и впиши: `VSELLM_API_KEY=...`

**Другой провайдер?** Подойдёт любой OpenAI-совместимый API (OpenRouter,
ProxyAPI, DeepSeek, Groq, …) — просто передай `--base-url` и `--api-key-env`.
Код менять не нужно.

## Установка

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# Linux/macOS: .venv/bin/pip install -r requirements.txt
cp .env.example .env                             # вставь VSELLM_API_KEY
```

## Запуск

Дешёвая проверка, что всё собралось (без вызовов API):

```bash
.venv/Scripts/python needle.py run --model openai/gpt-4o-mini --dry-run
```

Реальный прогон (одна модель, обе задачи, 3 прогона):

```bash
.venv/Scripts/python needle.py run --model openai/gpt-4o-mini --task both --runs 3
```

Результаты пишутся в `results/v2_<model>_<timestamp>.csv`, сразу печатается
таблица точности по позициям.

## График

```bash
.venv/Scripts/python scripts/plot.py                 # все results/*.csv -> results/plot.png
.venv/Scripts/python scripts/plot.py --paths a.csv b.csv --out results/plot.png
```

## Сколько это стоит

Оценка на **один прогон** (`--task both --runs 1`, ~46 запросов по ~115K токенов
входного контекста), без учёта кэша:

| модель | цена входа | ~1 прогон | ~3 прогона |
|---|---|---|---|
| `openai/gpt-4o-mini` | 17 ₽/1M | ~90 ₽ | ~270 ₽ |
| `deepseek/deepseek-v3.2` | 44 ₽/1M | ~230 ₽ | ~700 ₽ |
| `qwen/qwen3-235b-a22b` | 44 ₽/1M | ~190 ₽* | ~570 ₽ |

\* у qwen реальное окно ~95K токенов — запускай с `--target-chars 400000`.

Реально выйдет дешевле: общий префикс документа кэшируется, повторы идут из кэша.

**Копеечный smoke-тест** (маленький контекст, одна задача, один прогон):

```bash
.venv/Scripts/python needle.py run --model openai/gpt-4o-mini --task single --target-chars 50000 --runs 1
```

## Флаги

- `--model` — id модели у провайдера (`openai/gpt-4o-mini`, `deepseek/deepseek-v3.2`, …).
- `--task` — `single` / `sum` / `both`.
- `--prompt-style` — `neutral` / `primed` / `both` (есть ли подсказка «ответь по документу»).
- `--runs` — число прогонов (в каждом прогоне новые значения иголок).
- `--target-chars` — размер стога в символах (~500000 ≈ 115K токенов).
- `--needles` — число иголок (по умолчанию 12, распределены по 0%–100%).
- `--concurrency` — параллельные запросы.
- `--output` — путь к CSV; повторный запуск с тем же файлом досчитывает пропущенное.
- `--base-url` / `--api-key-env` — смена провайдера без правки кода.

## Структура

```
needle.py                основной скрипт (run / report)
scripts/plot.py          график точности по позиции
scripts/prepare_texts.py (опционально) пересобрать data/texts/
data/texts/              стога сена: 3 книги Project Gutenberg, ~100K токенов каждая
results/                 сюда пишутся CSV и plot.png (в git не попадают)
```

## Результат

Итоги прогона (3 модели × 3 прогона × 2 задачи × 2 промпта) — в посте:
**https://t.me/technosurvival/54**

Коротко: классический single-needle «lost in the middle» на 128K у современных
моделей исчез (97–100% на любой позиции). Эффект остаётся только в задаче
сведения двух фактов и только у слабой модели (`gpt-4o-mini` — U-форма ~45%),
а прайминг («отвечай по документу») не помогает.
