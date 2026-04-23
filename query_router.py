from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta


DATE_PATTERNS = (
    re.compile("(?P<year>\\d{4})[\\u5e74\\-\\.](?P<month>\\d{1,2})[\\u6708\\-\\.](?P<day>\\d{1,2})\\u65e5?"),
    re.compile("(?P<month>\\d{1,2})\\u6708(?P<day>\\d{1,2})\\u65e5?"),
)

PREFERENCE_HINTS = (
    "\u559c\u6b22",
    "\u504f\u597d",
    "\u7231\u597d",
    "\u8ba8\u538c",
    "\u4e0d\u559c\u6b22",
    "\u6211\u662f\u8c01",
    "\u6211\u53eb\u4ec0\u4e48",
    "\u6211\u7684\u4e13\u4e1a",
    "\u6211\u7684\u753b\u50cf",
    "\u6700\u559c\u6b22",
)

EVENT_HINTS = (
    "\u51e0\u6708\u51e0\u53f7",
    "\u54ea\u5929",
    "\u4ec0\u4e48\u65f6\u5019",
    "\u5f53\u5929",
    "\u90a3\u5929",
    "\u505a\u4e86\u4ec0\u4e48",
    "\u53d1\u751f\u4e86\u4ec0\u4e48",
    "\u8bb0\u5f55",
    "\u5386\u53f2",
)

HISTORICAL_RECALL_HINTS = (
    "\u66fe\u7ecf",
    "\u4e4b\u524d",
    "\u4ee5\u524d",
    "\u63d0\u5230\u8fc7",
    "\u5206\u4eab\u8fc7",
    "\u63a8\u8350\u8fc7",
    "\u544a\u8bc9\u8fc7",
    "\u8bf4\u8fc7\u4ec0\u4e48",
    "\u63a8\u8350\u4e86\u4ec0\u4e48",
    "\u804a\u8fc7",
    "\u7ed9\u6211\u63a8\u8350\u8fc7",
    "\u548c\u4f60\u63a8\u8350\u8fc7",
    "\u548c\u4f60\u5206\u4eab\u8fc7",
    "\u548c\u4f60\u63d0\u5230\u8fc7",
)

EXPERIENCE_HINTS = (
    "\u53bb\u8fc7",
    "\u53bb\u4e86",
    "\u53bb\u770b\u4e86",
    "\u770b\u8fc7",
    "\u770b\u4e86",
    "\u8bfb\u8fc7",
    "\u8bfb\u4e86",
    "\u63d0\u5230",
    "\u5206\u4eab",
    "\u63a8\u8350",
    "\u53c2\u89c2\u4e86",
    "\u5b66\u8fc7",
)

RECENT_HINTS = (
    "\u521a\u624d",
    "\u521a\u521a",
    "\u4e0a\u4e00\u8f6e",
    "\u4e0a\u4e00\u6761",
    "\u4e0a\u4e00\u6b21",
    "\u521a\u8bf4",
    "\u7b2c\u4e8c\u672c",
    "\u7b2c\u4e09\u672c",
    "\u521a\u624d\u4f60\u8bf4",
)


@dataclass
class QueryRoute:
    intent: str
    date_filters: list[str] = field(default_factory=list)
    retrieve_semantic: bool = False
    retrieve_events: bool = False
    retrieve_messages: bool = False
    recent_turn_window: int = 0
    rationale: str = ""


def _safe_date(year: int, month: int, day: int) -> str | None:
    try:
        return datetime(year, month, day).date().isoformat()
    except ValueError:
        return None


def extract_date_filters(query: str, now: datetime | None = None) -> list[str]:
    now = now or datetime.now()
    date_filters: list[str] = []

    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(query):
            year = int(match.groupdict().get("year") or now.year)
            month = int(match.group("month"))
            day = int(match.group("day"))
            date_value = _safe_date(year, month, day)
            if date_value and date_value not in date_filters:
                date_filters.append(date_value)

    relative_dates = {
        "\u4eca\u5929": now.date(),
        "\u6628\u65e5": (now - timedelta(days=1)).date(),
        "\u6628\u5929": (now - timedelta(days=1)).date(),
        "\u524d\u5929": (now - timedelta(days=2)).date(),
    }
    for token, value in relative_dates.items():
        if token in query:
            iso = value.isoformat()
            if iso not in date_filters:
                date_filters.append(iso)

    return date_filters


def infer_query_route(query: str, now: datetime | None = None) -> QueryRoute:
    now = now or datetime.now()
    date_filters = extract_date_filters(query, now=now)

    has_preference = any(token in query for token in PREFERENCE_HINTS)
    has_event = bool(date_filters) or any(token in query for token in EVENT_HINTS)
    has_historical_recall = any(token in query for token in HISTORICAL_RECALL_HINTS)
    has_experience = any(token in query for token in EXPERIENCE_HINTS)
    has_recent = any(token in query for token in RECENT_HINTS)

    if has_recent:
        return QueryRoute(
            intent="recent_context",
            date_filters=date_filters,
            retrieve_messages=True,
            recent_turn_window=6,
            rationale="\u95ee\u9898\u6307\u5411\u6700\u8fd1\u51e0\u8f6e\u5bf9\u8bdd\u6216\u987a\u5e8f\u76f8\u5173\u7684\u4e0a\u4e0b\u6587\u3002",
        )

    if has_event and has_preference:
        return QueryRoute(
            intent="hybrid",
            date_filters=date_filters,
            retrieve_semantic=True,
            retrieve_events=True,
            retrieve_messages=True,
            rationale="\u95ee\u9898\u540c\u65f6\u6d89\u53ca\u5e26\u65e5\u671f\u7684\u5386\u53f2\u4fe1\u606f\u548c\u7a33\u5b9a\u504f\u597d\uff0c\u9700\u8981\u6df7\u5408\u68c0\u7d22\u3002",
        )

    if has_event:
        return QueryRoute(
            intent="event_query",
            date_filters=date_filters,
            retrieve_events=True,
            retrieve_messages=True,
            rationale="\u95ee\u9898\u5e26\u6709\u660e\u786e\u65f6\u95f4\u7ea6\u675f\uff0c\u5e94\u4f18\u5148\u67e5\u4e8b\u4ef6\u548c\u539f\u59cb\u6d88\u606f\u3002",
        )

    if has_historical_recall and has_preference:
        return QueryRoute(
            intent="hybrid",
            date_filters=date_filters,
            retrieve_semantic=True,
            retrieve_events=True,
            retrieve_messages=True,
            rationale="\u95ee\u9898\u65e2\u5728\u56de\u5fc6\u8fc7\u5f80\u5bf9\u8bdd\uff0c\u53c8\u5305\u542b\u504f\u597d\u5224\u65ad\uff0c\u9002\u5408\u6df7\u5408\u68c0\u7d22\u3002",
        )

    if has_experience and has_preference:
        return QueryRoute(
            intent="hybrid",
            date_filters=date_filters,
            retrieve_semantic=True,
            retrieve_events=True,
            retrieve_messages=True,
            rationale="\u95ee\u9898\u540c\u65f6\u5305\u542b\u8fc7\u5f80\u7ecf\u5386\u548c\u504f\u597d\u5224\u65ad\uff0c\u9002\u5408\u6df7\u5408\u68c0\u7d22\u3002",
        )

    if has_historical_recall or has_experience:
        return QueryRoute(
            intent="historical_recall",
            date_filters=date_filters,
            retrieve_semantic=False,
            retrieve_events=True,
            retrieve_messages=True,
            rationale="\u95ee\u9898\u672c\u8d28\u662f\u5728\u56de\u5fc6\u4ee5\u524d\u67d0\u6b21\u5bf9\u8bdd\u6216\u7ecf\u5386\uff0c\u5e94\u4f18\u5148\u67e5 events \u548c messages\u3002",
        )

    if has_preference:
        return QueryRoute(
            intent="semantic_profile",
            date_filters=date_filters,
            retrieve_semantic=True,
            retrieve_events=False,
            retrieve_messages=False,
            rationale="\u95ee\u9898\u5728\u8be2\u95ee\u957f\u671f\u504f\u597d\u3001\u7a33\u5b9a\u753b\u50cf\u6216\u8eab\u4efd\u4fe1\u606f\u3002",
        )

    return QueryRoute(
        intent="hybrid",
        date_filters=date_filters,
        retrieve_semantic=True,
        retrieve_events=True,
        retrieve_messages=False,
        rationale="\u9ed8\u8ba4\u8d70\u6df7\u5408\u68c0\u7d22\uff0c\u517c\u987e\u957f\u671f\u8bed\u4e49\u8bb0\u5fc6\u548c\u5386\u53f2\u4e8b\u4ef6\u3002",
    )
