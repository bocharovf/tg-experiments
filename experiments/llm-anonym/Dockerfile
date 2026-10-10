FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# spaCy модели для Presidio (en + ru, large)
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download ru_core_news_lg

COPY . .

ENTRYPOINT ["python", "anonymize.py"]
CMD ["--help"]
