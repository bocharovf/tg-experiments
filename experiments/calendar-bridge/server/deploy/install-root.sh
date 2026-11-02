#!/usr/bin/env bash
# Установка синхронизатора как системного сервиса. Запускать с sudo.
# Без этого скрипта сервис работает как обычный фоновый процесс (./service.sh start),
# но не поднимется сам после перезагрузки сервера, а маршрут в Caddy
# придётся добавлять заново после каждого рестарта Caddy.
set -euo pipefail

APP=/opt/calendar-sync

if [ "$(id -u)" -ne 0 ]; then
  echo "нужен root: sudo $0" >&2
  exit 1
fi

install -m 644 "$APP/deploy/calendar-sync.service" /etc/systemd/system/calendar-sync.service
systemctl daemon-reload
systemctl enable --now calendar-sync.service
systemctl restart calendar-sync.service

install -m 644 "$APP/deploy/Caddyfile" /etc/caddy/Caddyfile
systemctl reload caddy || systemctl restart caddy

sleep 2
systemctl --no-pager --lines=5 status calendar-sync.service || true
curl -sf https://sync.example.com/healthz && echo " — API отвечает"
