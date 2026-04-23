from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


BOOK_PATTERN = re.compile("\u300a([^\u300b]+)\u300b")
LIST_SPLIT_PATTERN = re.compile("[\u3001,\uff0c/]|\u548c|\u53ca|\u4ee5\u53ca|\u4e0e")


@dataclass
class SemanticMemoryOperation:
    action: str
    category: str
    memory_text: str = ""
    slot: str = ""
    fact_value: str = ""
    target_value: str = ""
    confidence: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TurnAnalysis:
    event_type: str
    event_summary: str
    event_metadata: dict[str, Any] = field(default_factory=dict)
    memory_operations: list[SemanticMemoryOperation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "event_summary": self.event_summary,
            "event_metadata": dict(self.event_metadata),
            "memory_operations": [item.to_dict() for item in self.memory_operations],
        }


class RuleBasedMemoryExtractor:
    """Fallback extractor for event memory and semantic memory updates."""

    def analyze_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        timestamp: str,
    ) -> TurnAnalysis:
        event_type = self._infer_event_type(user_msg, assistant_msg)
        event_metadata = self._build_event_metadata(user_msg, assistant_msg)
        event_summary = self._build_event_summary(
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            event_type=event_type,
            timestamp=timestamp,
            event_metadata=event_metadata,
        )
        memory_operations = self._extract_semantic_memories(user_msg)

        return TurnAnalysis(
            event_type=event_type,
            event_summary=event_summary,
            event_metadata=event_metadata,
            memory_operations=memory_operations,
        )

    def _infer_event_type(self, user_msg: str, assistant_msg: str) -> str:
        merged = f"{user_msg}\n{assistant_msg}"
        if "\u63a8\u8350" in merged or BOOK_PATTERN.search(assistant_msg):
            return "recommendation"
        if any(
            token in merged
            for token in (
                "\u8ba1\u5212",
                "\u5b89\u6392",
                "\u63d0\u9192",
                "\u5f85\u529e",
            )
        ):
            return "plan"
        if any(
            token in merged
            for token in (
                "\u7b54\u5e94",
                "\u4f1a\u5e2e\u4f60",
                "\u7a0d\u540e",
                "\u4e4b\u540e",
            )
        ):
            return "commitment"
        return "dialogue"

    def _build_event_metadata(
        self,
        user_msg: str,
        assistant_msg: str,
    ) -> dict[str, Any]:
        titles = [title.strip() for title in BOOK_PATTERN.findall(assistant_msg)]
        metadata: dict[str, Any] = {}
        if titles:
            metadata["recommended_titles"] = titles
        if "\u63a8\u8350" in user_msg or "\u63a8\u8350" in assistant_msg:
            metadata["recommendation"] = True
        if assistant_msg:
            metadata["assistant_preview"] = assistant_msg[:160]
        return metadata

    def _build_event_summary(
        self,
        user_msg: str,
        assistant_msg: str,
        event_type: str,
        timestamp: str,
        event_metadata: dict[str, Any],
    ) -> str:
        date_value = self._extract_date(timestamp)
        if event_type == "recommendation" and event_metadata.get("recommended_titles"):
            titles = "\u3001".join(
                f"\u300a{name}\u300b" for name in event_metadata["recommended_titles"]
            )
            return (
                f"{date_value}\uff0c"
                f"\u52a9\u624b\u9488\u5bf9\u201c{user_msg}\u201d"
                f"\u63a8\u8350\u4e86 {titles}"
            )

        preview = assistant_msg.replace("\n", " ").strip()
        if len(preview) > 120:
            preview = preview[:117] + "..."
        return (
            f"{date_value}\uff0c"
            f"\u7528\u6237\u63d0\u5230\u201c{user_msg}\u201d\uff0c"
            f"\u52a9\u624b\u56de\u590d\u201c{preview}\u201d"
        )

    def _extract_date(self, timestamp: str) -> str:
        if not timestamp:
            return datetime.now().date().isoformat()
        return timestamp.split("T", 1)[0]

    def _extract_semantic_memories(
        self,
        user_msg: str,
    ) -> list[SemanticMemoryOperation]:
        text = user_msg.strip()
        if not text or self._is_task_request(text):
            return []

        operations: list[SemanticMemoryOperation] = []
        operations.extend(self._extract_identity_memories(text))
        operations.extend(self._extract_positive_preferences(text))
        operations.extend(self._extract_negative_preferences(text))
        operations.extend(self._extract_favorite_song(text))
        operations.extend(self._extract_delete_operations(text))
        return self._dedupe_operations(operations)

    def _extract_identity_memories(self, text: str) -> list[SemanticMemoryOperation]:
        operations: list[SemanticMemoryOperation] = []
        identity_patterns = (
            (
                "\u6211\u53eb([^\uff0c\u3002\uff01\uff1f\\s]{1,20})",
                "identity_name",
                "identity",
                "\u7528\u6237\u59d3\u540d\u662f{value}",
            ),
            (
                "(?:\u6211\u7684\u4e13\u4e1a\u662f|\u6211\u5b66\u7684\u662f|\u6211\u5b66)([^\uff0c\u3002\uff01\uff1f]{1,30})",
                "identity_major",
                "identity",
                "\u7528\u6237\u4e13\u4e1a\u662f{value}",
            ),
            (
                "(?:\u6211\u5728|\u6211\u662f)([^\uff0c\u3002\uff01\uff1f]{2,30}(?:\u5927\u5b66|\u5b66\u9662|\u5b66\u6821))",
                "identity_school",
                "identity",
                "\u7528\u6237\u5b66\u6821\u662f{value}",
            ),
            (
                "\u6211\u662f([^\uff0c\u3002\uff01\uff1f]{1,20}(?:\u5b66\u751f|\u8001\u5e08|\u5de5\u7a0b\u5e08|\u7a0b\u5e8f\u5458|\u8bbe\u8ba1\u5e08|\u7814\u7a76\u751f|\u672c\u79d1\u751f))",
                "identity_role",
                "identity",
                "\u7528\u6237\u8eab\u4efd\u662f{value}",
            ),
        )
        for pattern, slot, category, template in identity_patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            value = match.group(1).strip()
            operations.append(
                SemanticMemoryOperation(
                    action="UPDATE",
                    category=category,
                    memory_text=template.format(value=value),
                    slot=slot,
                    fact_value=value,
                    confidence=0.95,
                )
            )
        return operations

    def _extract_positive_preferences(self, text: str) -> list[SemanticMemoryOperation]:
        operations: list[SemanticMemoryOperation] = []
        positive_patterns = (
            "\u6211(?:\u5f88|\u4e5f|\u6700)?\u559c\u6b22(?:\u770b|\u542c|\u5403|\u8bfb|\u505a)?(?P<value>[^\u3002\uff01\uff1f\uff1b?]{1,40})",
            "\u6211\u66f4\u559c\u6b22(?P<value>[^\u3002\uff01\uff1f\uff1b?]{1,40})",
            "\u6211\u504f\u597d(?P<value>[^\u3002\uff01\uff1f\uff1b?]{1,40})",
            "\u6211\u7231(?P<value>[^\u3002\uff01\uff1f\uff1b?]{1,40})",
        )

        for pattern in positive_patterns:
            for match in re.finditer(pattern, text):
                raw_value = self._clean_fact_value(match.group("value"))
                for value in self._split_preference_values(raw_value):
                    slot, category, template = self._infer_preference_schema(value)
                    operations.append(
                        SemanticMemoryOperation(
                            action="ADD",
                            category=category,
                            memory_text=template.format(value=value),
                            slot=slot,
                            fact_value=value,
                            confidence=0.86,
                        )
                    )
        return operations

    def _extract_negative_preferences(self, text: str) -> list[SemanticMemoryOperation]:
        operations: list[SemanticMemoryOperation] = []
        pattern = "\u6211(?:\u4e0d\u559c\u6b22|\u8ba8\u538c|\u4e0d\u7231)(?P<value>[^\u3002\uff01\uff1f\uff1b?]{1,40})"
        for match in re.finditer(pattern, text):
            raw_value = self._clean_fact_value(match.group("value"))
            for value in self._split_preference_values(raw_value):
                operations.append(
                    SemanticMemoryOperation(
                        action="ADD",
                        category="preference_dislike",
                        memory_text=f"\u7528\u6237\u4e0d\u559c\u6b22{value}",
                        slot="preference_dislike",
                        fact_value=value,
                        confidence=0.86,
                    )
                )
        return operations

    def _extract_favorite_song(self, text: str) -> list[SemanticMemoryOperation]:
        operations: list[SemanticMemoryOperation] = []
        song_patterns = (
            "\u6700\u559c\u6b22(?:\u7684)?(?:\u6b4c|\u6b4c\u66f2)(?:\u662f|\u53eb)?(?P<song>\u300a[^\u300b]+\u300b)",
            "\u6700\u559c\u6b22(?P<song>\u300a[^\u300b]+\u300b)",
        )
        for pattern in song_patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            song = match.group("song").strip()
            operations.append(
                SemanticMemoryOperation(
                    action="UPDATE",
                    category="preference_music_song",
                    memory_text=f"\u7528\u6237\u6700\u559c\u6b22\u7684\u6b4c\u66f2\u662f{song}",
                    slot="favorite_song",
                    fact_value=song,
                    confidence=0.92,
                )
            )
            break
        return operations

    def _extract_delete_operations(self, text: str) -> list[SemanticMemoryOperation]:
        operations: list[SemanticMemoryOperation] = []

        delete_preference = re.search(
            "\u6211\u4e0d\u518d\u559c\u6b22(?P<value>[^\u3002\uff01\uff1f\uff1b?]{1,40})",
            text,
        )
        if delete_preference:
            target_value = self._clean_fact_value(delete_preference.group("value"))
            operations.append(
                SemanticMemoryOperation(
                    action="DELETE",
                    category="preference_like",
                    slot="preference_like",
                    target_value=target_value,
                    confidence=0.9,
                )
            )

        delete_generic = re.search(
            "(?:\u5fd8\u6389|\u5220\u9664|\u5220\u6389|\u4e0d\u8981\u518d\u8bb0|\u522b\u518d\u8bb0)(?P<value>[^\u3002\uff01\uff1f\uff1b?]{1,40})",
            text,
        )
        if delete_generic:
            target_value = self._clean_fact_value(delete_generic.group("value"))
            operations.append(
                SemanticMemoryOperation(
                    action="DELETE",
                    category="generic",
                    target_value=target_value,
                    confidence=0.9,
                )
            )

        return operations

    def _infer_preference_schema(self, value: str) -> tuple[str, str, str]:
        if any(
            token in value
            for token in (
                "\u83dc",
                "\u83dc\u7cfb",
                "\u706b\u9505",
                "\u70e7\u70e4",
            )
        ):
            return (
                "preference_cuisine",
                "preference_cuisine",
                "\u7528\u6237\u559c\u6b22\u7684\u83dc\u7cfb\u662f{value}",
            )
        if any(
            token in value
            for token in (
                "\u559c\u5267",
                "\u7231\u60c5\u7247",
                "\u79d1\u5e7b",
                "\u60ac\u7591",
                "\u52a8\u4f5c\u7247",
                "\u7535\u5f71",
                "\u5f71\u7247",
            )
        ):
            return (
                "preference_movie",
                "preference_movie",
                "\u7528\u6237\u559c\u6b22\u7684\u7535\u5f71\u7c7b\u578b\u662f{value}",
            )
        if any(
            token in value
            for token in (
                "\u7235\u58eb",
                "\u6447\u6eda",
                "\u6c11\u8c23",
                "\u53e4\u5178",
                "\u6d41\u884c",
                "\u97f3\u4e50",
            )
        ):
            return (
                "preference_music",
                "preference_music",
                "\u7528\u6237\u559c\u6b22\u7684\u97f3\u4e50\u7c7b\u578b\u662f{value}",
            )
        if any(
            token in value
            for token in (
                "\u5c0f\u8bf4",
                "\u6587\u5b66",
                "\u6563\u6587",
                "\u8bd7\u6b4c",
                "\u4e66",
            )
        ):
            return (
                "preference_reading",
                "preference_reading",
                "\u7528\u6237\u559c\u6b22\u9605\u8bfb{value}",
            )
        return "preference_like", "preference_like", "\u7528\u6237\u559c\u6b22{value}"

    def _split_preference_values(self, value: str) -> list[str]:
        if not value:
            return []

        normalized = (
            value.replace("\u6bd4\u5982", "")
            .replace("\u4f8b\u5982", "")
            .replace("\u5c24\u5176\u662f", "")
            .replace("\u5c24\u5176\u559c\u6b22", "")
            .strip("\uff0c\u3002\uff01\uff1f\uff1b ")
        )

        parts = [
            part.strip("\uff0c\u3002\uff01\uff1f\uff1b ")
            for part in LIST_SPLIT_PATTERN.split(normalized)
        ]
        cleaned = [part for part in parts if 1 <= len(part) <= 20]

        if cleaned:
            return cleaned
        return [normalized] if normalized else []

    def _clean_fact_value(self, value: str) -> str:
        cleaned = value.strip()
        clause_breakers = (
            "\uff0c\u4e0d\u8fc7",
            "\uff0c\u4f46\u662f",
            "\uff0c\u7136\u540e",
            "\uff0c\u6240\u4ee5",
            "\uff0c\u89c9\u5f97",
            "\uff0c\u8ba9\u6211",
            "\u3002",
            "\uff01",
            "\uff1f",
            ";",
            "\uff1b",
        )
        for marker in clause_breakers:
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[0].strip()
        return cleaned.strip("\uff0c\u3002\uff01\uff1f\uff1b ")

    def _dedupe_operations(
        self,
        operations: list[SemanticMemoryOperation],
    ) -> list[SemanticMemoryOperation]:
        deduped: list[SemanticMemoryOperation] = []
        seen: set[tuple[str, str, str, str]] = set()
        for op in operations:
            signature = (
                op.action,
                op.category,
                op.slot,
                op.fact_value or op.target_value,
            )
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(op)
        return deduped

    def _is_task_request(self, text: str) -> bool:
        if "\uff1f" in text or "?" in text:
            return True
        request_prefixes = (
            "\u8bf7\u95ee",
            "\u5e2e\u6211",
            "\u8bf7\u4f60",
            "\u63a8\u8350\u6211",
            "\u7ed9\u6211\u63a8\u8350",
            "\u4f60\u53ef\u4ee5",
            "\u80fd\u4e0d\u80fd",
            "\u53ef\u4e0d\u53ef\u4ee5",
        )
        return text.startswith(request_prefixes)
