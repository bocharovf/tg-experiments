/**
 * Calendly -> рабочий календарь (через приглашение по почте).
 *
 * Что делает:
 *  1. Читает брони из календаря, куда пишет Calendly.
 *  2. Создаёт их копии в отдельном календаре с нейтральной темой.
 *  3. Приглашает рабочий адрес: Exchange/Outlook получает обычное приглашение.
 *  4. Перенос брони  -> patch копии  -> Outlook получает обновление.
 *     Отмена брони  -> удаление копии -> Outlook получает отмену.
 *
 * Почему копия, а не приглашение на саму бронь: если Calendly удалит своё
 * событие, Google не разошлёт отмену гостям. Копией владеет скрипт, поэтому
 * он сам удаляет её с sendUpdates='all' — и Exchange корректно снимает встречу.
 *
 * Отмену Calendly оформляет тремя способами: удаляет событие, ставит
 * status=cancelled или переименовывает в «Canceled: ...». Обрабатываются все три.
 *
 * Требуется включённый сервис Calendar API (Services -> + -> Calendar API).
 */

const SOURCE_CALENDAR_ID = 'YOUR_CALENDLY_CALENDAR_ID@group.calendar.google.com';
const TARGET_CALENDAR_NAME = 'Личные брони (авто)';
const WORK_EMAIL = 'you@company.com';
const PRIVATE_TITLE = 'Частная встреча';
const WINDOW_DAYS = 60;
const CANCEL_MARKERS = ['canceled:', 'cancelled:', 'declined:', 'отменено:', 'отменён:', 'отклонено:', '[canceled]', '[cancelled]', '(canceled)', '(cancelled)'];

function syncBookings() {
  const target = getTargetCalendar_();
  const targetId = target.getId();
  const tz = target.getTimeZone();
  const now = Date.now();

  const from = new Date(now - 24 * 3600 * 1000).toISOString();
  const to = new Date(now + WINDOW_DAYS * 24 * 3600 * 1000).toISOString();

  // showDeleted=true — чтобы видеть события, которые Calendly пометил отменёнными
  const source = Calendar.Events.list(SOURCE_CALENDAR_ID, {
    timeMin: from,
    timeMax: to,
    singleEvents: true,
    showDeleted: true,
    maxResults: 2500,
  }).items || [];

  const copies = Calendar.Events.list(targetId, {
    timeMin: from,
    timeMax: to,
    singleEvents: true,
    showDeleted: false,
    maxResults: 2500,
  }).items || [];

  const bySource = {};
  copies.forEach(function (c) {
    const priv = (c.extendedProperties && c.extendedProperties.private) || {};
    if (priv.mirrorOf) bySource[priv.mirrorOf] = c;
  });

  const seen = {};
  let created = 0, updated = 0, deleted = 0, failed = 0;

  source.forEach(function (ev) {
    if (isCancelled_(ev)) {
      Logger.log('источник отменён (%s): %s', ev.summary, ev.id);
      return; // seen не ставим -> копия будет удалена ниже
    }
    if (!ev.start || !ev.start.dateTime) return; // события на весь день пропускаем

    seen[ev.id] = true;
    const body = buildEvent_(ev, tz);
    const copy = bySource[ev.id];

    try {
      if (!copy) {
        Calendar.Events.insert(body, targetId, { sendUpdates: 'all' });
        created++;
        Logger.log('создана копия: %s', body.start.dateTime);
      } else if (changed_(copy, body)) {
        Calendar.Events.patch(body, targetId, copy.id, { sendUpdates: 'all' });
        updated++;
        Logger.log('изменена копия: %s', body.start.dateTime);
      }
    } catch (e) {
      failed++;
      Logger.log('ошибка по событию %s: %s | %s',
                 ev.id, e.message, JSON.stringify(e.details || {}));
    }
  });

  // Отмены: копия есть, а живой брони в источнике больше нет
  Object.keys(bySource).forEach(function (srcId) {
    if (seen[srcId]) return;

    const copy = bySource[srcId];
    const startMs = copy.start && copy.start.dateTime
      ? new Date(copy.start.dateTime).getTime() : 0;
    if (startMs < now - 3600 * 1000) return; // прошедшие не трогаем

    try {
      Logger.log('отменяю копию: %s | %s', copy.summary,
                 (copy.start && copy.start.dateTime) || '');
      Calendar.Events.remove(targetId, copy.id, { sendUpdates: 'all' });
      deleted++;
    } catch (e) {
      failed++;
      Logger.log('ошибка отмены %s: %s', copy.id, e.message);
    }
  });

  Logger.log('создано=%s изменено=%s отменено=%s ошибок=%s',
             created, updated, deleted, failed);
}

function buildEvent_(ev, tz) {
  return {
    summary: PRIVATE_TITLE,
    location: '',
    description: '',
    start: { dateTime: ev.start.dateTime, timeZone: (ev.start.timeZone || tz) },
    end: { dateTime: ev.end.dateTime, timeZone: (ev.end.timeZone || tz) },
    attendees: [{ email: WORK_EMAIL }],
    reminders: { useDefault: false, overrides: [] },
    guestsCanModify: false,
    guestsCanInviteOthers: false,
    extendedProperties: { private: { mirrorOf: ev.id, source: 'calendly' } },
  };
}

function changed_(copy, body) {
  return (copy.summary || '') !== body.summary ||
         (copy.location || '') !== (body.location || '') ||
         new Date(copy.start.dateTime).getTime() !== new Date(body.start.dateTime).getTime() ||
         new Date(copy.end.dateTime).getTime() !== new Date(body.end.dateTime).getTime();
}

function isCancelled_(ev) {
  if (ev.status === 'cancelled') return true;
  const s = (ev.summary || '').trim().toLowerCase();
  return CANCEL_MARKERS.some(function (m) { return s.indexOf(m) === 0; });
}

function getTargetCalendar_() {
  const found = CalendarApp.getCalendarsByName(TARGET_CALENDAR_NAME);
  if (found && found.length > 0) return found[0];
  return CalendarApp.createCalendar(TARGET_CALENDAR_NAME, { timeZone: 'Europe/Moscow' });
}

function installTriggers() {
  ScriptApp.getProjectTriggers().forEach(function (t) {
    if (t.getHandlerFunction() === 'syncBookings') ScriptApp.deleteTrigger(t);
  });
  ScriptApp.newTrigger('syncBookings').timeBased().everyMinutes(5).create();
  Logger.log('триггер установлен: каждые 5 минут');
}

function debugState() {
  const target = getTargetCalendar_();
  const now = Date.now();
  const from = new Date(now - 7 * 24 * 3600 * 1000).toISOString();
  const to = new Date(now + WINDOW_DAYS * 24 * 3600 * 1000).toISOString();

  const src = Calendar.Events.list(SOURCE_CALENDAR_ID, {
    timeMin: from, timeMax: to, singleEvents: true, showDeleted: true, maxResults: 2500
  }).items || [];
  Logger.log('ИСТОЧНИК: %s', src.length);
  src.forEach(function (e) {
    Logger.log('  src %s | status=%s | %s | %s',
      e.id, e.status, e.summary,
      (e.start && (e.start.dateTime || e.start.date)) || '');
  });

  const cp = Calendar.Events.list(target.getId(), {
    timeMin: from, timeMax: to, singleEvents: true, showDeleted: true, maxResults: 2500
  }).items || [];
  Logger.log('КОПИИ: %s', cp.length);
  cp.forEach(function (e) {
    const p = (e.extendedProperties && e.extendedProperties.private) || {};
    Logger.log('  copy %s | mirrorOf=%s | %s | %s | guests=%s',
      e.id, p.mirrorOf, e.summary,
      (e.start && e.start.dateTime) || '',
      (e.attendees || []).map(function (a) { return a.email; }).join(','));
  });
}
