from __future__ import annotations

import argparse
import sys

from chat_system import PersonalizedChatSystem
from memory_store import MemoryStore


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid Memory Dialogue System",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="/data/2024/wenqiang000/Qwen2.5-3B-Instruct",
        help="Path to the LLM model.",
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default="/data/2024/wenqiang000/bge-m3",
        help="Path to the embedding model.",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="./memory_data",
        help="Directory used to persist hybrid memories.",
    )
    parser.add_argument(
        "--user_id",
        type=str,
        default="default_user",
        help="User scope for all memories.",
    )
    parser.add_argument(
        "--session_id",
        type=str,
        default="default_session",
        help="Session scope for recent context and raw messages.",
    )
    parser.add_argument(
        "--agent_id",
        type=str,
        default=None,
        help="Optional agent scope.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of retrieved memories.")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Inference device. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--max_history_turns",
        type=int,
        default=5,
        help="Recent in-memory turns used for local context.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 68)
    print("  Initializing Hybrid Memory Dialogue System")
    print("=" * 68)
    print(f"  LLM Model:   {args.llm_model}")
    print(f"  Embed Model: {args.embed_model}")
    print(f"  User ID:     {args.user_id}")
    print(f"  Session ID:  {args.session_id}")
    print(f"  Agent ID:    {args.agent_id or 'N/A'}")
    print(f"  Memory Dir:  {args.index_dir}")
    print("=" * 68)
    print()

    print("[Init] Step 1/2: Loading hybrid memory store...")
    try:
        memory_store = MemoryStore(
            embed_model_path=args.embed_model,
            index_dir=args.index_dir,
            user_id=args.user_id,
            session_id=args.session_id,
            agent_id=args.agent_id,
        )
    except Exception as exc:
        print(f"[Error] Failed to initialize memory store: {exc}")
        print("        Please check your embedding model path and memory directory.")
        sys.exit(1)

    print("[Init] Step 2/2: Loading chat system...")
    try:
        chat_system = PersonalizedChatSystem(
            llm_model_path=args.llm_model,
            memory_store=memory_store,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
            session_id=args.session_id,
            agent_id=args.agent_id,
        )
        chat_system.max_history_turns = args.max_history_turns
    except Exception as exc:
        print(f"[Error] Failed to initialize chat system: {exc}")
        print("        Please check your LLM model path and runtime environment.")
        sys.exit(1)

    chat_system.chat_loop()


if __name__ == "__main__":
    main()
