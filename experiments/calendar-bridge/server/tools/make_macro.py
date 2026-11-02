"""Генератор .macro для MacroDroid.

Собирает файл импорта: таймер -> Get Calendar Events -> JSON Output -> HTTP POST.
Формат сверен со схемой, которая лежит внутри APK MacroDroid 6.x
(assets/ai/macrodroid-full-llm-schema.yaml).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent


def load_env() -> dict[str, str]:
    env = {}
    for line in (HERE / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def var(name: str, vtype: int, is_array: bool = False) -> dict:
    return {
        "dictionary": {
            "entries": [],
            "isArray": is_array,
            "variableType": vtype,
            "type": "Array" if is_array else "Dictionary",
        },
        "isActionBlockWorkingVar": False,
        "isLocalVar": True,
        "isSecure": False,
        "m_booleanValue": False,
        "m_decimalValue": 0.0,
        "m_intValue": 0,
        "m_name": name,
        "m_stringValue": "",
        "m_type": vtype,
        "supportsInput": False,
        "supportsOutput": True,
    }


COMMON = {
    "disableLogging": False,
    "m_SIGUID": 0,
    "m_comment": "",
    "m_constraintList": [],
    "m_isDisabled": False,
    "m_isOrCondition": False,
}


def build(token: str, domain: str, calendar_id: str = "", calendar_name: str = "",
          interval_seconds: int = 1800, window_days: int = 14) -> dict:
    actions = [
        {
            **COMMON,
            "m_classType": "GetCalendarEventsAction",
            "calendarId": calendar_id,
            "calendarName": calendar_name,
            "startOffsetValue": "0",
            "startOffsetUnit": 2,          # дни
            "durationValue": str(window_days),
            "durationUnit": 2,             # дни
            "outputDictionaryVarName": "work",
        },
        {
            **COMMON,
            "m_classType": "JsonOutputAction",
            "dictionaryVarName": "work",
            "stringVarName": "body",
        },
        {
            **COMMON,
            "m_classType": "HttpRequestAction",
            "requestConfig": {
                "requestType": 1,          # POST
                "urlToOpen": f"https://{domain}/sync",
                "blockNextAction": True,
                "contentBodySource": 0,
                "contentBodyText": "{lv=body}",
                "contentType": "application/json",
                "headerParams": [
                    {"paramName": "Authorization", "paramValue": f"Bearer {token}"},
                ],
                "queryParams": [],
                "followRedirects": True,
                "allowAnyCertificate": False,
                "requestTimeOutSeconds": 30,
                "maxTotalDurationSeconds": 120,
                "saveResponseType": 1,
                "responseVariableName": "resp",
                "saveReturnCodeToVariable": False,
                "saveReturnHeadersToVariable": False,
                "basicAuthEnabled": False,
                "proxyEnabled": False,
                "prettifyJson": False,
                "clientCertEnabled": False,
            },
        },
    ]

    trigger = {
        **COMMON,
        "m_classType": "RegularIntervalTrigger",
        "m_seconds": interval_seconds,
        "m_startHour": 0,
        "m_startMinute": 0,
        "m_ignoreReferenceStartTime": True,
        "m_useAlarm": False,
    }

    macro = {
        "aiGenerated": 1,
        "breakpoints": [],
        "disabledTimestamp": 0,
        "exportedActionBlocks": [],
        "forceEvenIfNotEnabledTimestamp": 0,
        "isActionBlock": False,
        "isExtra": False,
        "isFavourite": False,
        "lastEditedTimestamp": 0,
        "localVariables": [
            var("work", 4, is_array=False),
            var("body", 2),
            var("resp", 2),
        ],
        "localVarsAlphabetical": True,
        "loggingLevel": 1,
        "m_GUID": 0,
        "m_actionList": actions,
        "m_category": "Работа",
        "m_completed": True,
        "m_constraintList": [],
        "m_description": (
            "Каждые 30 минут выгружает рабочие встречи на 2 недели вперёд. "
            "После импорта откройте действие Get Calendar Events и выберите свой рабочий календарь."
        ),
        "m_descriptionOpen": False,
        "m_enabled": True,
        "m_excludeLog": False,
        "m_headingColor": 0,
        "m_isOrCondition": False,
        "m_name": "Рабочий календарь → Google",
        "m_triggerList": [trigger],
    }

    return {"globalVariables": [], "macro": macro, "macroExportVersion": 1}


def main(argv: list[str]) -> int:
    env = load_env()
    token = env.get("SYNC_TOKEN", "")
    domain = env.get("DOMAIN", "")
    if len(token) < 24 or not domain:
        print("нет SYNC_TOKEN/DOMAIN в .env", file=sys.stderr)
        return 1

    out = HERE / "macrodroid" / "calendar_bridge.macro"
    out.parent.mkdir(exist_ok=True)
    data = build(
        token=token,
        domain=domain,
        calendar_id=env.get("WORK_CALENDAR_ID", ""),
        calendar_name=env.get("WORK_CALENDAR_NAME", ""),
        interval_seconds=int(env.get("PUSH_INTERVAL_SECONDS", "1800")),
        window_days=int(env.get("WINDOW_DAYS", "14")),
    )
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.chmod(out, 0o600)
    print(f"готово: {out} ({out.stat().st_size} байт)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
