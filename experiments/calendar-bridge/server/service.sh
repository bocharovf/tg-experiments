#!/usr/bin/env bash
# Запуск/остановка синхронизатора без root (обычный фоновый процесс).
set -euo pipefail
cd "$(dirname "$0")"
umask 077

case "${1:-start}" in
  start)
    if [ -f data/app.pid ] && kill -0 "$(cat data/app.pid)" 2>/dev/null; then
      echo "уже запущен (pid $(cat data/app.pid))"
      exit 0
    fi
    setsid .venv/bin/uvicorn app.main:app \
      --host 127.0.0.1 --port 8787 --workers 1 --no-access-log \
      >> logs/stdout.log 2>&1 &
    echo $! > data/app.pid
    sleep 1.5
    curl -sf http://127.0.0.1:8787/healthz >/dev/null && echo "запущен (pid $(cat data/app.pid))"
    ;;
  stop)
    if [ -f data/app.pid ]; then kill "$(cat data/app.pid)" 2>/dev/null || true; rm -f data/app.pid; fi
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
    echo "остановлен"
    ;;
  restart)
    "$0" stop; sleep 0.5; "$0" start
    ;;
  status)
    if [ -f data/app.pid ] && kill -0 "$(cat data/app.pid)" 2>/dev/null; then
      echo "работает (pid $(cat data/app.pid))"
    else
      echo "не запущен"
    fi
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status}" >&2; exit 1;;
esac
