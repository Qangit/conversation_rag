from __future__ import annotations

import re
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_store import MemoryStore


PROMPT_PREFIX = """\
你是一个带混合记忆能力的个性化助手。

你在回答时必须区分不同来源的记忆：
1. `semantic` 表示长期稳定事实与偏好。
2. `event` 表示带日期的历史事件。
3. `message` 表示原始对话记录或最近上下文。

统一回答规则：
1. 回答先给结论，再补最多1句必要说明。
2. 不要追加反问，不要寒暄，不要说“如果你还需要……”。
3. 除非用户要求分析或建议，否则不要扩展背景介绍。
4. 如果证据不足，就直接说记录不足，不要编造。
5. 不要暴露“记忆库”“检索”等系统措辞。
"""


ROUTE_PROMPT_RULES = {
    "semantic_profile": """\
你正在回答长期稳定信息问题。
请优先依据 `semantic` 记忆回答身份、偏好、兴趣和长期画像。
不要混入一次性事件，不要扩展成泛泛而谈。
答案尽量控制在1到2句。""",
    "historical_recall": """\
你正在回答历史回忆问题。
请优先依据 `event` 和 `message` 回忆过去某次对话中出现过的内容。
如果没有明确日期，不要自行补日期；如果证据不足，就直接说记录不足。
只回答回忆到的内容本身，不要追加泛化总结或追问。
答案尽量控制在1到2句。""",
    "event_query": """\
你正在回答带日期的历史事件问题。
请优先依据同日期的 `event` 和 `message` 回答。
只回答该日期下发生的事情，不要扩展到其他日期，不要追加追问。
答案尽量控制在1到2句。""",
    "recent_context": """\
你正在回答最近上下文问题。
请优先依据最近几轮消息及其顺序信息回答，尤其注意“刚才”“上一轮”“第二个”等表述。
直接给出结果，不要引入长期记忆，不要补充无关背景。
答案尽量控制在1句，必要时最多2句。""",
    "hybrid": """\
你正在回答需要综合多类记忆的问题。
可以同时参考长期语义记忆、事件记忆和原始消息，但必须优先回答用户问题本身。
先给结论，再补必要说明，不要无关扩展。
答案尽量控制在2到3句。""",
}


NO_MEMORY_SYSTEM_PROMPT = """\
你是一个个性化助手。当前没有可靠的历史记忆可用，请直接根据用户问题回答；如果问题明显依赖历史信息但你没有把握，请明确说明。回答应简洁直接，不要追问用户，也不要添加额外扩展。"""


class PersonalizedChatSystem:
    """Hybrid-memory chat system with routing, retrieval fusion, and provenance trace."""

    def __init__(
        self,
        llm_model_path: str = "/data/2024/wenqiang000/Qwen2.5-3B-Instruct",
        memory_store: MemoryStore | None = None,
        top_k: int = 5,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 1.0,
        device: str | None = None,
        session_id: str = "default_session",
        agent_id: str | None = None,
    ):
        self.memory_store = memory_store
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.turn_count = 0
        self.session_id = session_id
        self.agent_id = agent_id

        self.conversation_history: list[dict[str, str]] = []
        self.max_history_turns = 5
        self.last_trace: dict | None = None

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ChatSystem] Using device: {self.device}")

        print(f"[ChatSystem] Loading LLM model from: {llm_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[ChatSystem] Model loaded successfully.")

    def _format_memory_context(self, diagnostics: dict) -> str:
        route = diagnostics["route"]
        results = diagnostics["results"]
        if not results:
            return ""

        route_line = f"查询路由: {route.intent}"
        if route.date_filters:
            route_line += f" | 日期过滤: {', '.join(route.date_filters)}"

        lines = [route_line, ""]
        for idx, item in enumerate(results, 1):
            lines.append(
                f"[{idx}] source={item['source_type']} id={item['source_id']} "
                f"date={item.get('date', '')} session={item.get('session_id', '')} "
                f"score={item['score']:.3f}"
            )
            lines.append(item["text"])
            lines.append("")
        return "\n".join(lines).strip()

    def _retrieve_memories(self, query: str, now: datetime | None = None) -> dict | None:
        if self.memory_store is None:
            self.last_trace = None
            return None

        diagnostics = self.memory_store.search_with_diagnostics(
            query=query,
            top_k=self.top_k,
            now=now,
            session_id=self.session_id,
            agent_id=self.agent_id,
        )
        self.last_trace = diagnostics
        if diagnostics["results"]:
            print(
                f"[RAG] Route={diagnostics['route'].intent}, "
                f"hits={len(diagnostics['results'])}"
            )
        else:
            print(f"[RAG] Route={diagnostics['route'].intent}, no hits")
        return diagnostics

    def _build_system_prompt(self, memory_context: str, route_intent: str | None) -> str:
        if not memory_context:
            return NO_MEMORY_SYSTEM_PROMPT

        route_rules = ROUTE_PROMPT_RULES.get(route_intent or "hybrid", ROUTE_PROMPT_RULES["hybrid"])
        return (
            f"{PROMPT_PREFIX}\n\n"
            f"{route_rules}\n\n"
            f"下面是本轮可用的记忆上下文：\n{memory_context}"
        )

    def _build_messages(
        self,
        query: str,
        memory_context: str,
        route_intent: str | None,
    ) -> list[dict[str, str]]:
        system_prompt = self._build_system_prompt(memory_context, route_intent)
        messages = [{"role": "system", "content": system_prompt}]

        for turn in self.conversation_history[-self.max_history_turns :]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        messages.append({"role": "user", "content": query})
        return messages

    def _normalize_response(self, response: str) -> str:
        response = response.replace("\r", "\n").strip()
        response = re.sub(r"\n{2,}", "\n", response)
        response = re.sub(r"[ \t]+", " ", response)
        response = re.sub(
            r"^(根据(?:我的记忆|你的记录|记录|描述)|从记录来看|根据记录|根据你的描述)[，,:：]\s*",
            "",
            response,
        )
        return response.strip()

    def _split_sentences(self, response: str) -> list[str]:
        parts = re.split(r"(?<=[。！？!?])", response)
        sentences = [part.strip() for part in parts if part.strip()]
        return sentences if sentences else [response.strip()]

    def _is_followup_sentence(self, sentence: str) -> bool:
        followup_markers = (
            "如果你",
            "如果您",
            "如需",
            "需要更多",
            "请随时",
            "你喜欢",
            "您喜欢",
            "还有其他",
            "有没有其他",
            "如果愿意",
            "欢迎继续",
            "告诉我",
            "随时告诉我",
        )
        if any(marker in sentence for marker in followup_markers):
            return True
        if sentence.endswith(("？", "?")):
            return True
        return False

    def _trim_response_by_route(self, response: str, route_intent: str | None) -> str:
        response = self._normalize_response(response)
        sentences = self._split_sentences(response)

        if route_intent in {"event_query", "historical_recall", "semantic_profile", "recent_context"}:
            kept: list[str] = []
            max_sentences = 1 if route_intent == "recent_context" else 2

            for sentence in sentences:
                if self._is_followup_sentence(sentence):
                    continue
                kept.append(sentence)
                if len(kept) >= max_sentences:
                    break

            if kept:
                return "".join(kept).strip()

        if route_intent == "hybrid":
            kept = []
            for sentence in sentences:
                if self._is_followup_sentence(sentence):
                    continue
                kept.append(sentence)
                if len(kept) >= 3:
                    break
            if kept:
                return "".join(kept).strip()

        return response

    @torch.no_grad()
    def generate(
        self,
        query: str,
        now: datetime | None = None,
        persist: bool = True,
    ) -> str:
        diagnostics = self._retrieve_memories(query, now=now)
        memory_context = self._format_memory_context(diagnostics) if diagnostics else ""
        route_intent = diagnostics["route"].intent if diagnostics else None
        messages = self._build_messages(query, memory_context, route_intent)

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        response = self._trim_response_by_route(response, route_intent)

        if persist and self.memory_store is not None:
            timestamp = now or datetime.now()
            write_result = self.memory_store.add_dialogue_turn(
                user_msg=query,
                assistant_msg=response,
                timestamp=timestamp,
                session_id=self.session_id,
                agent_id=self.agent_id,
            )
            self.turn_count = int(write_result["turn_id"])

        self.conversation_history.append({"user": query, "assistant": response})
        return response

    def _print_last_trace(self):
        if not self.last_trace:
            print("[Trace] No retrieval trace available.\n")
            return

        route = self.last_trace["route"]
        print(f"[Trace] intent: {route.intent}")
        print(f"[Trace] date_filters: {route.date_filters}")
        print(f"[Trace] rationale: {route.rationale}")
        for idx, item in enumerate(self.last_trace["results"], 1):
            print(
                f"[Trace {idx}] {item['provenance']} "
                f"(date={item.get('date')}, score={item['score']:.3f})"
            )
            print(f"          {item['text']}")
        print()

    def _replay_event(self, event_id: str):
        if self.memory_store is None:
            print("[Replay] Memory store is disabled.\n")
            return

        payload = self.memory_store.replay_event(event_id)
        if payload is None:
            print(f"[Replay] Event not found: {event_id}\n")
            return

        event = payload["event"]
        print(f"[Replay] event_id: {event['id']}")
        print(f"[Replay] type: {event['event_type']}")
        print(f"[Replay] date: {event['date']}")
        print(f"[Replay] summary: {event['summary']}")
        for message in payload["messages"]:
            print(f"  [{message['role']}] {message['content']}")
        print()

    def chat_loop(self):
        print("\n" + "=" * 68)
        print("  Hybrid Memory Dialogue System")
        print("  Qwen + BGE-M3 + FAISS + Structured Hybrid Memory")
        print(f"  user_id={self.memory_store.user_id if self.memory_store else 'N/A'}")
        print(f"  session_id={self.session_id}")
        print("=" * 68)
        print("  commands: quit | clear | clear_memory | stats | save | trace | replay <event_id>")
        print("=" * 68 + "\n")

        try:
            while True:
                user_input = input("You: ").strip()
                if not user_input:
                    continue

                lowered = user_input.lower()
                if lowered in ("quit", "exit"):
                    if self.memory_store:
                        self.memory_store.save()
                    print("[System] Bye.")
                    break

                if lowered == "clear":
                    self.conversation_history.clear()
                    print("[System] Cleared in-memory short context.\n")
                    continue

                if lowered == "clear_memory":
                    if self.memory_store:
                        self.memory_store.clear()
                    self.conversation_history.clear()
                    print("[System] Cleared all hybrid memories.\n")
                    continue

                if lowered == "stats":
                    if self.memory_store:
                        stats = self.memory_store.stats()
                        for key, value in stats.items():
                            print(f"[Stats] {key}: {value}")
                    else:
                        print("[Stats] Memory store is disabled.")
                    print()
                    continue

                if lowered == "save":
                    if self.memory_store:
                        self.memory_store.save()
                    print("[System] Saved.\n")
                    continue

                if lowered == "trace":
                    self._print_last_trace()
                    continue

                if lowered.startswith("replay "):
                    _, _, event_id = user_input.partition(" ")
                    self._replay_event(event_id.strip())
                    continue

                print("\nAssistant: ", end="", flush=True)
                response = self.generate(user_input)
                print(response)
                print()

        except KeyboardInterrupt:
            if self.memory_store:
                self.memory_store.save()
            print("\n[System] Interrupted. Memory saved.")
