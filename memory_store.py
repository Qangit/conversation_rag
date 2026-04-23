from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from memory_extractor import RuleBasedMemoryExtractor, SemanticMemoryOperation, TurnAnalysis
from query_router import QueryRoute, infer_query_route


def _now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _normalize_timestamp(value: str | datetime | None) -> str:
    if value is None:
        return _now_iso()
    if isinstance(value, datetime):
        return value.replace(microsecond=0).isoformat()
    text = str(value).strip()
    if not text:
        return _now_iso()
    if len(text) == 10 and text.count("-") == 2:
        return f"{text}T00:00:00"
    return text


class MemoryStore:
    """
    Hybrid memory store for personalized dialogue.

    Persistent collections:
      - messages: raw audit log for every user/assistant message
      - events: structured event memories with explicit date/timestamp
      - semantic_memories: long-term stable facts and preferences
      - memory_history: evolution log for semantic memories
    """

    DEFAULT_EVENT_BOOST = 1.05
    DEFAULT_SEMANTIC_BOOST = 1.10
    DEFAULT_MESSAGE_BOOST = 0.95
    SEMANTIC_FALLBACK_SCORE_THRESHOLD = 0.52
    SEMANTIC_FALLBACK_PENALTY = 0.88

    def __init__(
        self,
        embed_model_path: str = "/data/2024/wenqiang000/bge-m3",
        index_dir: str = "./memory_data",
        user_id: str = "default_user",
        session_id: str = "default_session",
        agent_id: str | None = None,
        embed_model=None,
        extractor: RuleBasedMemoryExtractor | None = None,
        decay_rate: float | None = None,
        summary_interval: int | None = None,
    ):
        self.embed_model_path = embed_model_path
        self.index_dir = index_dir
        self.user_id = user_id
        self.session_id = session_id
        self.agent_id = agent_id
        self.user_dir = os.path.join(index_dir, user_id)

        # Legacy options are kept for CLI/backward compatibility.
        self.decay_rate = decay_rate
        self.summary_interval = summary_interval

        if embed_model is not None:
            self.embed_model = embed_model
        else:
            print(f"[MemoryStore] Loading embedding model from: {embed_model_path}")
            self.embed_model = SentenceTransformer(embed_model_path)
        self.embed_dim = self.embed_model.get_sentence_embedding_dimension()

        self.extractor = extractor or RuleBasedMemoryExtractor()
        self._turn_analyzer: Optional[
            Callable[[str, str, str], TurnAnalysis | dict[str, Any]]
        ] = None
        self._classify_callback: Optional[Callable[[str], str]] = None
        self._summary_callback: Optional[Callable[[list[str]], str]] = None

        self.messages: list[dict[str, Any]] = []
        self.events_meta: list[dict[str, Any]] = []
        self.semantic_meta: list[dict[str, Any]] = []
        self.memory_history: list[dict[str, Any]] = []

        self.event_index = faiss.IndexFlatIP(self.embed_dim)
        self.semantic_index = faiss.IndexFlatIP(self.embed_dim)

        self._turn_counter = 0
        self._load_from_disk()

        print(
            f"[MemoryStore] Ready. user={self.user_id}, session={self.session_id}, "
            f"messages={len(self.messages)}, events={len(self.events_meta)}, "
            f"semantic={len(self.semantic_meta)}"
        )

    def _messages_path(self) -> str:
        return os.path.join(self.user_dir, "messages.json")

    def _events_meta_path(self) -> str:
        return os.path.join(self.user_dir, "events_meta.json")

    def _events_index_path(self) -> str:
        return os.path.join(self.user_dir, "events.faiss")

    def _semantic_meta_path(self) -> str:
        return os.path.join(self.user_dir, "semantic_memories_meta.json")

    def _semantic_index_path(self) -> str:
        return os.path.join(self.user_dir, "semantic_memories.faiss")

    def _history_path(self) -> str:
        return os.path.join(self.user_dir, "memory_history.json")

    def set_turn_analyzer(
        self,
        callback: Callable[[str, str, str], TurnAnalysis | dict[str, Any]],
    ):
        self._turn_analyzer = callback

    def set_classify_callback(self, callback: Callable[[str], str]):
        self._classify_callback = callback

    def set_summary_callback(self, callback: Callable[[list[str]], str]):
        self._summary_callback = callback

    def _load_json_list(self, path: str) -> list[dict[str, Any]]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    def _load_or_rebuild_index(
        self,
        path: str,
        texts: list[str],
    ) -> faiss.IndexFlatIP:
        if os.path.exists(path):
            index = faiss.read_index(path)
            if index.ntotal == len(texts):
                return index

        index = faiss.IndexFlatIP(self.embed_dim)
        if texts:
            index.add(self._encode(texts))
        return index

    def _load_from_disk(self):
        os.makedirs(self.user_dir, exist_ok=True)

        self.messages = self._load_json_list(self._messages_path())
        self.events_meta = self._load_json_list(self._events_meta_path())
        self.semantic_meta = self._load_json_list(self._semantic_meta_path())
        self.memory_history = self._load_json_list(self._history_path())

        event_texts = [item["summary"] for item in self.events_meta]
        semantic_texts = [item["memory_text"] for item in self.semantic_meta]

        self.event_index = self._load_or_rebuild_index(self._events_index_path(), event_texts)
        self.semantic_index = self._load_or_rebuild_index(
            self._semantic_index_path(), semantic_texts
        )

        if self.messages:
            self._turn_counter = max(int(item.get("turn_id", 0)) for item in self.messages)

    def save(self):
        os.makedirs(self.user_dir, exist_ok=True)

        with open(self._messages_path(), "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)
        with open(self._events_meta_path(), "w", encoding="utf-8") as f:
            json.dump(self.events_meta, f, ensure_ascii=False, indent=2)
        with open(self._semantic_meta_path(), "w", encoding="utf-8") as f:
            json.dump(self.semantic_meta, f, ensure_ascii=False, indent=2)
        with open(self._history_path(), "w", encoding="utf-8") as f:
            json.dump(self.memory_history, f, ensure_ascii=False, indent=2)

        faiss.write_index(self.event_index, self._events_index_path())
        faiss.write_index(self.semantic_index, self._semantic_index_path())

        print(
            f"[MemoryStore] Saved messages={len(self.messages)}, events={len(self.events_meta)}, "
            f"semantic={len(self.semantic_meta)} to {self.user_dir}"
        )

    def clear(self):
        self.messages = []
        self.events_meta = []
        self.semantic_meta = []
        self.memory_history = []
        self.event_index = faiss.IndexFlatIP(self.embed_dim)
        self.semantic_index = faiss.IndexFlatIP(self.embed_dim)
        self._turn_counter = 0
        self.save()
        print("[MemoryStore] Cleared all hybrid memories.")

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embed_dim), dtype="float32")
        embeddings = self.embed_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype="float32")

    def _score_candidates(self, query_embedding: np.ndarray, texts: list[str]) -> list[float]:
        if not texts:
            return []
        candidate_embeddings = self._encode(texts)
        return (candidate_embeddings @ query_embedding[0]).tolist()

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def _make_scope(
        self,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> tuple[str, str | None]:
        return session_id or self.session_id, agent_id if agent_id is not None else self.agent_id

    def _agent_scope_matches(self, item: dict[str, Any], agent_id: str | None) -> bool:
        if agent_id is None:
            return True
        item_agent = item.get("agent_id")
        return item_agent in (None, agent_id)

    def _append_message(
        self,
        role: str,
        content: str,
        turn_id: int,
        timestamp: str,
        session_id: str,
        agent_id: str | None,
    ) -> dict[str, Any]:
        record = {
            "id": self._new_id("msg"),
            "user_id": self.user_id,
            "session_id": session_id,
            "agent_id": agent_id,
            "turn_id": turn_id,
            "role": role,
            "content": content,
            "created_at": timestamp,
            "date": timestamp.split("T", 1)[0],
        }
        self.messages.append(record)
        return record

    def _append_event(
        self,
        event_type: str,
        summary: str,
        timestamp: str,
        raw_message_ids: list[str],
        metadata: dict[str, Any],
        session_id: str,
        agent_id: str | None,
    ) -> dict[str, Any]:
        record = {
            "id": self._new_id("evt"),
            "user_id": self.user_id,
            "session_id": session_id,
            "agent_id": agent_id,
            "event_type": event_type,
            "date": timestamp.split("T", 1)[0],
            "timestamp": timestamp,
            "summary": summary,
            "raw_message_ids": list(raw_message_ids),
            "metadata": dict(metadata),
        }
        self.events_meta.append(record)
        self.event_index.add(self._encode([summary]))
        return record

    def _append_semantic_memory(
        self,
        memory_text: str,
        category: str,
        slot: str,
        fact_value: str,
        confidence: float,
        source_event_id: str | None,
        session_id: str,
        agent_id: str | None,
    ) -> dict[str, Any]:
        now_iso = _now_iso()
        record = {
            "id": self._new_id("mem"),
            "user_id": self.user_id,
            "session_id": session_id,
            "agent_id": agent_id,
            "memory_text": memory_text,
            "category": category,
            "slot": slot,
            "fact_value": fact_value,
            "confidence": confidence,
            "source_event_id": source_event_id,
            "created_at": now_iso,
            "updated_at": now_iso,
            "is_active": True,
        }
        self.semantic_meta.append(record)
        self.semantic_index.add(self._encode([memory_text]))
        return record

    def _append_memory_history(
        self,
        memory_id: str | None,
        old_value: str,
        new_value: str,
        action: str,
        source_event_id: str | None,
        slot: str,
        category: str,
    ):
        record = {
            "id": self._new_id("hist"),
            "memory_id": memory_id,
            "old_value": old_value,
            "new_value": new_value,
            "action": action,
            "slot": slot,
            "category": category,
            "source_event_id": source_event_id,
            "created_at": _now_iso(),
        }
        self.memory_history.append(record)

    def _normalize_analysis(self, result: TurnAnalysis | dict[str, Any]) -> TurnAnalysis:
        if isinstance(result, TurnAnalysis):
            return result
        ops = [
            SemanticMemoryOperation(**item)
            for item in result.get("memory_operations", [])
            if isinstance(item, dict)
        ]
        return TurnAnalysis(
            event_type=result.get("event_type", "dialogue"),
            event_summary=result.get("event_summary", ""),
            event_metadata=result.get("event_metadata", {}) or {},
            memory_operations=ops,
        )

    def _analyze_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        timestamp: str,
    ) -> TurnAnalysis:
        if self._turn_analyzer is not None:
            try:
                return self._normalize_analysis(
                    self._turn_analyzer(user_msg, assistant_msg, timestamp)
                )
            except Exception as exc:
                print(f"[MemoryStore] Turn analyzer failed, fallback to rules: {exc}")
        return self.extractor.analyze_turn(user_msg, assistant_msg, timestamp)

    def import_profile_meta(
        self,
        meta_information: dict[str, Any] | None,
        session_id: str | None = None,
        agent_id: str | None = None,
    ):
        if not meta_information:
            return

        current_session_id, current_agent_id = self._make_scope(session_id, agent_id)
        known_slots = {
            "name": ("profile_name", "identity", "用户姓名"),
            "personality": ("profile_personality", "profile", "用户性格"),
            "hobbies": ("profile_hobbies", "preference", "用户爱好"),
            "speaking_style": ("profile_speaking_style", "profile", "用户说话风格"),
        }
        for key, value in meta_information.items():
            if key not in known_slots or not value:
                continue
            slot, category, label = known_slots[key]
            self._upsert_semantic_from_operation(
                SemanticMemoryOperation(
                    action="UPDATE",
                    category=category,
                    memory_text=f"{label}是 {value}",
                    slot=slot,
                    fact_value=str(value),
                    confidence=0.9,
                ),
                source_event_id=None,
                session_id=current_session_id,
                agent_id=current_agent_id,
            )

    def add_dialogue_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        turn_id: int | None = None,
        timestamp: str | datetime | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        timestamp_iso = _normalize_timestamp(timestamp)
        current_session_id, current_agent_id = self._make_scope(session_id, agent_id)

        self._turn_counter = turn_id if turn_id is not None else self._turn_counter + 1
        current_turn_id = self._turn_counter

        user_record = self._append_message(
            role="user",
            content=user_msg,
            turn_id=current_turn_id,
            timestamp=timestamp_iso,
            session_id=current_session_id,
            agent_id=current_agent_id,
        )
        assistant_record = self._append_message(
            role="assistant",
            content=assistant_msg,
            turn_id=current_turn_id,
            timestamp=timestamp_iso,
            session_id=current_session_id,
            agent_id=current_agent_id,
        )

        analysis = self._analyze_turn(user_msg, assistant_msg, timestamp_iso)
        event_record = self._append_event(
            event_type=analysis.event_type,
            summary=analysis.event_summary,
            timestamp=timestamp_iso,
            raw_message_ids=[user_record["id"], assistant_record["id"]],
            metadata=analysis.event_metadata,
            session_id=current_session_id,
            agent_id=current_agent_id,
        )

        for operation in analysis.memory_operations:
            self._upsert_semantic_from_operation(
                operation=operation,
                source_event_id=event_record["id"],
                session_id=current_session_id,
                agent_id=current_agent_id,
            )

        return {
            "turn_id": current_turn_id,
            "message_ids": [user_record["id"], assistant_record["id"]],
            "event_id": event_record["id"],
            "analysis": analysis.to_dict(),
        }

    def _find_active_semantic_memories(
        self,
        *,
        slot: str | None = None,
        category: str | None = None,
        target_value: str | None = None,
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        target = (target_value or "").strip().lower()
        for item in self.semantic_meta:
            if not item.get("is_active", True):
                continue
            if slot and item.get("slot") != slot:
                continue
            if category and item.get("category") != category:
                continue
            if target:
                haystacks = (
                    str(item.get("fact_value", "")),
                    str(item.get("memory_text", "")),
                )
                if not any(target in text.lower() for text in haystacks):
                    continue
            matches.append(item)
        return matches

    def _deactivate_semantic_memory(self, item: dict[str, Any]):
        item["is_active"] = False
        item["updated_at"] = _now_iso()

    def _upsert_semantic_from_operation(
        self,
        operation: SemanticMemoryOperation,
        source_event_id: str | None,
        session_id: str,
        agent_id: str | None,
    ):
        action = operation.action.upper().strip()
        if action in {"", "NONE"}:
            return

        if action == "DELETE":
            matches = self._find_active_semantic_memories(
                slot=operation.slot or None,
                category=None if operation.category == "generic" else operation.category,
                target_value=operation.target_value or operation.fact_value,
            )
            for item in matches:
                self._deactivate_semantic_memory(item)
                self._append_memory_history(
                    memory_id=item["id"],
                    old_value=item["memory_text"],
                    new_value="",
                    action="DELETE",
                    source_event_id=source_event_id,
                    slot=item.get("slot", operation.slot),
                    category=item.get("category", operation.category),
                )
            return

        if not operation.memory_text:
            return

        if action == "UPDATE":
            existing = self._find_active_semantic_memories(slot=operation.slot or None)
            same_value = next(
                (
                    item
                    for item in existing
                    if str(item.get("fact_value", "")).strip() == operation.fact_value.strip()
                ),
                None,
            )
            if same_value is not None:
                same_value["updated_at"] = _now_iso()
                return

            for item in existing:
                self._deactivate_semantic_memory(item)
                self._append_memory_history(
                    memory_id=item["id"],
                    old_value=item["memory_text"],
                    new_value=operation.memory_text,
                    action="UPDATE",
                    source_event_id=source_event_id,
                    slot=item.get("slot", operation.slot),
                    category=item.get("category", operation.category),
                )

            new_item = self._append_semantic_memory(
                memory_text=operation.memory_text,
                category=operation.category,
                slot=operation.slot,
                fact_value=operation.fact_value,
                confidence=operation.confidence,
                source_event_id=source_event_id,
                session_id=session_id,
                agent_id=agent_id,
            )
            self._append_memory_history(
                memory_id=new_item["id"],
                old_value="",
                new_value=operation.memory_text,
                action="ADD",
                source_event_id=source_event_id,
                slot=operation.slot,
                category=operation.category,
            )
            return

        existing_duplicates = self._find_active_semantic_memories(
            slot=operation.slot or None,
            category=operation.category,
            target_value=operation.fact_value or operation.memory_text,
        )
        if existing_duplicates:
            return

        new_item = self._append_semantic_memory(
            memory_text=operation.memory_text,
            category=operation.category,
            slot=operation.slot,
            fact_value=operation.fact_value,
            confidence=operation.confidence,
            source_event_id=source_event_id,
            session_id=session_id,
            agent_id=agent_id,
        )
        self._append_memory_history(
            memory_id=new_item["id"],
            old_value="",
            new_value=operation.memory_text,
            action="ADD",
            source_event_id=source_event_id,
            slot=operation.slot,
            category=operation.category,
        )

    def _event_result(self, item: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "source_type": "event",
            "source_id": item["id"],
            "session_id": item.get("session_id"),
            "agent_id": item.get("agent_id"),
            "timestamp": item.get("timestamp"),
            "date": item.get("date"),
            "score": float(score),
            "text": item["summary"],
            "metadata": dict(item.get("metadata", {})),
            "provenance": f"event:{item['id']}",
        }

    def _get_event_by_id(self, event_id: str) -> dict[str, Any] | None:
        return next((item for item in self.events_meta if item.get("id") == event_id), None)

    def _build_event_replay_text(
        self,
        event_item: dict[str, Any],
        max_chars: int = 320,
    ) -> str:
        raw_message_ids = event_item.get("raw_message_ids", [])
        if not raw_message_ids:
            return ""

        messages = self.get_messages_by_ids(raw_message_ids)
        if not messages:
            return ""

        replay_lines: list[str] = []
        for message in messages:
            role = message.get("role", "message")
            content = str(message.get("content", "")).strip().replace("\n", " ")
            if not content:
                continue
            replay_lines.append(f"[{role}] {content}")

        if not replay_lines:
            return ""

        replay_text = "\n".join(replay_lines)
        if len(replay_text) > max_chars:
            replay_text = replay_text[: max_chars - 3].rstrip() + "..."
        return replay_text

    def _augment_event_query_results(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        augmented: list[dict[str, Any]] = []
        for item in results:
            if item.get("source_type") != "event":
                augmented.append(item)
                continue

            event_item = self._get_event_by_id(str(item.get("source_id", "")))
            if not event_item:
                augmented.append(item)
                continue

            replay_text = self._build_event_replay_text(event_item)
            if not replay_text:
                augmented.append(item)
                continue

            clone = dict(item)
            clone["text"] = f"{item['text']}\n事件回放:\n{replay_text}"
            metadata = dict(clone.get("metadata", {}))
            metadata["replayed_message_ids"] = list(event_item.get("raw_message_ids", []))
            clone["metadata"] = metadata
            augmented.append(clone)

        return augmented

    def _semantic_result(self, item: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "source_type": "semantic",
            "source_id": item["id"],
            "session_id": item.get("session_id"),
            "agent_id": item.get("agent_id"),
            "timestamp": item.get("updated_at") or item.get("created_at"),
            "date": (item.get("updated_at") or item.get("created_at") or "").split("T", 1)[0],
            "score": float(score),
            "text": item["memory_text"],
            "metadata": {
                "category": item.get("category"),
                "slot": item.get("slot"),
                "confidence": item.get("confidence"),
                "source_event_id": item.get("source_event_id"),
            },
            "provenance": f"semantic:{item['id']}",
        }

    def _message_result(self, item: dict[str, Any], score: float) -> dict[str, Any]:
        role = item.get("role", "message")
        return {
            "source_type": "message",
            "source_id": item["id"],
            "session_id": item.get("session_id"),
            "agent_id": item.get("agent_id"),
            "timestamp": item.get("created_at"),
            "date": item.get("date"),
            "score": float(score),
            "text": f"[{role}] {item['content']}",
            "metadata": {
                "role": role,
                "turn_id": item.get("turn_id"),
            },
            "provenance": f"message:{item['id']}",
        }

    def _search_semantic(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        agent_id: str | None,
    ) -> list[dict[str, Any]]:
        if not any(item.get("is_active", True) for item in self.semantic_meta):
            return []

        raw_k = min(max(top_k * 3, top_k), len(self.semantic_meta))
        scores, indices = self.semantic_index.search(query_embedding, raw_k)

        results: list[dict[str, Any]] = []
        for raw_score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.semantic_meta):
                continue
            item = self.semantic_meta[idx]
            if not item.get("is_active", True):
                continue
            if not self._agent_scope_matches(item, agent_id):
                continue
            boosted_score = float(raw_score) * self.DEFAULT_SEMANTIC_BOOST
            results.append(self._semantic_result(item, boosted_score))
            if len(results) >= top_k:
                break
        return results

    def _search_events(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        date_filters: list[str],
        agent_id: str | None,
    ) -> list[dict[str, Any]]:
        if not self.events_meta:
            return []

        if date_filters:
            candidates = [
                item
                for item in self.events_meta
                if item.get("date") in date_filters and self._agent_scope_matches(item, agent_id)
            ]
            if not candidates:
                return []
            scores = self._score_candidates(query_embedding, [item["summary"] for item in candidates])
            ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
            return [
                self._event_result(item, score * (self.DEFAULT_EVENT_BOOST + 0.10))
                for item, score in ranked[:top_k]
            ]

        raw_k = min(max(top_k * 3, top_k), len(self.events_meta))
        scores, indices = self.event_index.search(query_embedding, raw_k)

        results: list[dict[str, Any]] = []
        for raw_score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.events_meta):
                continue
            item = self.events_meta[idx]
            if not self._agent_scope_matches(item, agent_id):
                continue
            results.append(self._event_result(item, float(raw_score) * self.DEFAULT_EVENT_BOOST))
            if len(results) >= top_k:
                break
        return results

    def _recent_messages_for_session(
        self,
        session_id: str,
        turn_window: int,
        agent_id: str | None,
    ) -> list[dict[str, Any]]:
        session_messages = [
            item
            for item in self.messages
            if item.get("session_id") == session_id and self._agent_scope_matches(item, agent_id)
        ]
        if not session_messages:
            return []

        turn_ids: list[int] = []
        for item in session_messages:
            turn_id = int(item.get("turn_id", 0))
            if turn_id and turn_id not in turn_ids:
                turn_ids.append(turn_id)
        turn_ids = turn_ids[-turn_window:]
        if not turn_ids:
            return session_messages[-(turn_window * 2):]
        return [item for item in session_messages if int(item.get("turn_id", 0)) in turn_ids]

    def _search_messages(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        route: QueryRoute,
        session_id: str,
        agent_id: str | None,
    ) -> list[dict[str, Any]]:
        if not self.messages:
            return []

        if route.recent_turn_window > 0:
            candidates = self._recent_messages_for_session(
                session_id,
                route.recent_turn_window,
                agent_id,
            )
        elif route.date_filters:
            candidates = [
                item
                for item in self.messages
                if item.get("date") in route.date_filters and self._agent_scope_matches(item, agent_id)
            ]
        else:
            filtered_messages = [
                item for item in self.messages if self._agent_scope_matches(item, agent_id)
            ]
            if route.intent in {"historical_recall", "event_query"}:
                candidates = filtered_messages
            elif route.intent == "semantic_profile":
                candidates = filtered_messages[- min(len(filtered_messages), max(top_k * 20, 80)) :]
            else:
                candidates = filtered_messages[- min(len(filtered_messages), max(top_k * 10, 50)) :]

        if not candidates:
            return []

        scores = self._score_candidates(query_embedding, [item["content"] for item in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        results: list[dict[str, Any]] = []
        for item, score in ranked[:top_k]:
            adjusted_score = score * self.DEFAULT_MESSAGE_BOOST
            if route.intent == "historical_recall" and item.get("role") == "assistant":
                adjusted_score *= 1.05
            results.append(self._message_result(item, adjusted_score))
        return results

    def _apply_score_penalty(
        self,
        results: list[dict[str, Any]],
        penalty: float,
    ) -> list[dict[str, Any]]:
        adjusted: list[dict[str, Any]] = []
        for item in results:
            clone = dict(item)
            clone["score"] = float(clone["score"]) * penalty
            adjusted.append(clone)
        return adjusted

    def _needs_semantic_profile_fallback(
        self,
        route: QueryRoute,
        results: list[dict[str, Any]],
    ) -> bool:
        if route.intent != "semantic_profile":
            return False

        semantic_results = [item for item in results if item["source_type"] == "semantic"]
        if not semantic_results:
            return True

        best_score = max(item["score"] for item in semantic_results)
        return best_score < self.SEMANTIC_FALLBACK_SCORE_THRESHOLD

    def _dedupe_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in sorted(results, key=lambda record: record["score"], reverse=True):
            signature = (item["source_type"], item["source_id"])
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(item)
        return deduped

    def search_with_diagnostics(
        self,
        query: str,
        top_k: int = 5,
        now: datetime | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        current_session_id, current_agent_id = self._make_scope(session_id, agent_id)
        route = infer_query_route(query, now=now)
        query_embedding = self._encode([query])

        results: list[dict[str, Any]] = []
        if route.retrieve_semantic:
            results.extend(self._search_semantic(query_embedding, top_k, current_agent_id))
        if route.retrieve_events:
            results.extend(
                self._search_events(
                    query_embedding,
                    top_k,
                    route.date_filters,
                    current_agent_id,
                )
            )
        if route.retrieve_messages or route.recent_turn_window > 0:
            results.extend(
                self._search_messages(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    route=route,
                    session_id=current_session_id,
                    agent_id=current_agent_id,
                )
            )

        if self._needs_semantic_profile_fallback(route, results):
            fallback_route = QueryRoute(
                intent="historical_recall",
                retrieve_events=True,
                retrieve_messages=True,
            )
            fallback_results: list[dict[str, Any]] = []
            fallback_results.extend(
                self._search_events(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    date_filters=[],
                    agent_id=current_agent_id,
                )
            )
            fallback_results.extend(
                self._search_messages(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    route=fallback_route,
                    session_id=current_session_id,
                    agent_id=current_agent_id,
                )
            )
            results.extend(
                self._apply_score_penalty(
                    fallback_results,
                    self.SEMANTIC_FALLBACK_PENALTY,
                )
            )

        final_results = self._dedupe_results(results)[:top_k]
        if route.intent == "event_query":
            final_results = self._augment_event_query_results(final_results)
        return {
            "route": route,
            "results": final_results,
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        now: datetime | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        diagnostics = self.search_with_diagnostics(
            query=query,
            top_k=top_k,
            now=now,
            session_id=session_id,
            agent_id=agent_id,
        )
        return diagnostics["results"]

    def get_recent_messages(
        self,
        n: int = 10,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        current_session_id, _ = self._make_scope(session_id, None)
        candidates = [item for item in self.messages if item.get("session_id") == current_session_id]
        return candidates[-n:]

    def get_messages_by_ids(self, message_ids: list[str]) -> list[dict[str, Any]]:
        target_ids = set(message_ids)
        return [item for item in self.messages if item.get("id") in target_ids]

    def replay_event(self, event_id: str) -> dict[str, Any] | None:
        event = next((item for item in self.events_meta if item.get("id") == event_id), None)
        if event is None:
            return None
        return {
            "event": event,
            "messages": self.get_messages_by_ids(event.get("raw_message_ids", [])),
        }

    def stats(self) -> dict[str, Any]:
        active_semantic = sum(1 for item in self.semantic_meta if item.get("is_active", True))
        sessions = {item.get("session_id") for item in self.messages if item.get("session_id")}
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "message_count": len(self.messages),
            "event_count": len(self.events_meta),
            "semantic_total_count": len(self.semantic_meta),
            "semantic_active_count": active_semantic,
            "memory_history_count": len(self.memory_history),
            "session_count": len(sessions),
            "embed_model": self.embed_model_path,
            "embed_dim": self.embed_dim,
            "index_dir": self.user_dir,
        }
