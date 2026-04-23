from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

from chat_system import PersonalizedChatSystem
from memory_store import MemoryStore


def load_memory_bank(data_file: str) -> dict:
    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_probing_questions(questions_file: str) -> dict:
    questions: dict[str, list[str]] = {}
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.update(json.loads(line))
    return questions


def load_ground_truths(gt_file: str | None) -> dict:
    if not gt_file or not os.path.exists(gt_file):
        return {}
    ground_truths: dict[str, list[str]] = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ground_truths.update(json.loads(line))
    return ground_truths


def inject_history(memory_store: MemoryStore, user_data: dict) -> int:
    meta_information = user_data.get("meta_information", {})
    memory_store.import_profile_meta(meta_information, session_id="profile_bootstrap")

    history = user_data.get("history", {})
    sorted_dates = sorted(history.keys())

    turn_id = 0
    total_turns = sum(len(turns) for turns in history.values())
    print(f"  [Inject] {total_turns} turns across {len(sorted_dates)} days...")

    for date_str in sorted_dates:
        for turn in history[date_str]:
            turn_id += 1
            memory_store.add_dialogue_turn(
                user_msg=turn["query"],
                assistant_msg=turn["response"],
                turn_id=turn_id,
                timestamp=date_str,
                session_id="history_import",
            )

    print(f"  [Inject] Done. {turn_id} turns injected.")
    return turn_id


def evaluate_user_ragas(
    chat_system: PersonalizedChatSystem,
    memory_store: MemoryStore,
    questions: list[str],
    user_name: str,
    ground_truths: list[str] | None = None,
    eval_now: datetime | None = None,
) -> list[dict]:
    results: list[dict] = []

    for idx, question in enumerate(questions, 1):
        print(f"  [Q{idx}/{len(questions)}] {question}")

        diagnostics = memory_store.search_with_diagnostics(
            query=question,
            top_k=chat_system.top_k,
            now=eval_now,
            session_id=chat_system.session_id,
            agent_id=chat_system.agent_id,
        )
        contexts = [item["text"] for item in diagnostics["results"]]

        chat_system.conversation_history = []
        answer = chat_system.generate(question, now=eval_now, persist=False)
        print(f"  [A{idx}] {answer[:120]}...")

        record = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "user": user_name,
            "route": diagnostics["route"].intent,
            "date_filters": diagnostics["route"].date_filters,
        }
        if ground_truths and idx <= len(ground_truths):
            record["ground_truth"] = ground_truths[idx - 1]
        results.append(record)

    return results


def export_ragas_dataset(results: list[dict], output_path: str):
    ragas_records = []
    for item in results:
        record = {
            "question": item["question"],
            "answer": item["answer"],
            "contexts": item["contexts"],
        }
        if "ground_truth" in item:
            record["ground_truth"] = item["ground_truth"]
        ragas_records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ragas_records, f, ensure_ascii=False, indent=2)
    print(f"  [RAGAS] Exported {len(ragas_records)} records -> {output_path}")


def export_ragas_csv(results: list[dict], output_path: str):
    import csv

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user", "route", "question", "answer", "contexts", "ground_truth"])
        for item in results:
            writer.writerow(
                [
                    item.get("user", ""),
                    item.get("route", ""),
                    item["question"],
                    item["answer"],
                    " || ".join(item["contexts"]),
                    item.get("ground_truth", ""),
                ]
            )
    print(f"  [CSV] Exported -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for hybrid memory dialogue")
    parser.add_argument("--llm_model", type=str, default="/root/Qwen2.5-7B-Instruct")
    parser.add_argument("--embed_model", type=str, default="/root/bge-m3")
    parser.add_argument("--data_file", type=str, default="eval_data/memory_bank_cn.json")
    parser.add_argument("--questions_file", type=str, default="eval_data/probing_questions_cn.jsonl")
    parser.add_argument("--ground_truths_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--eval_date",
        type=str,
        default="2023-05-07",
        help="Evaluation date used by the router when parsing relative dates.",
    )
    parser.add_argument("--users", nargs="*", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 68)
    print("  Hybrid Memory Batch Evaluation")
    print("=" * 68)

    eval_now = datetime.strptime(args.eval_date, "%Y-%m-%d") if args.eval_date else None

    memory_bank = load_memory_bank(args.data_file)
    probing_questions = load_probing_questions(args.questions_file)
    ground_truths = load_ground_truths(args.ground_truths_file)

    available_users = list(memory_bank.keys())
    eval_users = [user for user in (args.users or available_users) if user in available_users]

    print(f"[Info] Users in data:  {len(available_users)}")
    print(f"[Info] Users to eval: {len(eval_users)} -> {eval_users}")

    print(f"\n[Init] Loading shared embedding model from {args.embed_model}...")
    shared_store = MemoryStore(
        embed_model_path=args.embed_model,
        index_dir=os.path.join(args.output_dir, "_tmp"),
        user_id="_init",
        session_id="_init_session",
    )
    chat_system = PersonalizedChatSystem(
        llm_model_path=args.llm_model,
        memory_store=shared_store,
        top_k=args.top_k,
        session_id="eval_session",
    )
    print("[Init] Shared models loaded.\n")

    all_results: list[dict] = []

    for idx, user_name in enumerate(eval_users, 1):
        print(f"\n{'-' * 68}")
        print(f"  [{idx}/{len(eval_users)}] User: {user_name}")
        print(f"{'-' * 68}")

        user_store = MemoryStore(
            embed_model_path=args.embed_model,
            index_dir=os.path.join(args.output_dir, "memory_data"),
            user_id=user_name,
            session_id="eval_session",
            embed_model=shared_store.embed_model,
        )
        user_store.clear()

        t0 = time.time()
        injected_turns = inject_history(user_store, memory_bank[user_name])
        print(f"  [Time] inject: {time.time() - t0:.1f}s")
        user_store.save()

        chat_system.memory_store = user_store
        chat_system.session_id = "eval_session"
        chat_system.conversation_history = []
        chat_system.turn_count = injected_turns

        questions = probing_questions.get(user_name, [])
        if not questions:
            print(f"  [Skip] No probing questions for {user_name}")
            continue

        t1 = time.time()
        user_results = evaluate_user_ragas(
            chat_system=chat_system,
            memory_store=user_store,
            questions=questions,
            user_name=user_name,
            ground_truths=ground_truths.get(user_name),
            eval_now=eval_now,
        )
        print(f"  [Time] qa: {time.time() - t1:.1f}s")

        all_results.extend(user_results)
        user_file = os.path.join(args.output_dir, f"results_{user_name}.json")
        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(user_results, f, ensure_ascii=False, indent=2)

    ragas_json = os.path.join(args.output_dir, "ragas_dataset.json")
    export_ragas_dataset(all_results, ragas_json)

    ragas_csv = os.path.join(args.output_dir, "ragas_dataset.csv")
    export_ragas_csv(all_results, ragas_csv)

    all_file = os.path.join(args.output_dir, "all_results.json")
    with open(all_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 68}")
    print("  Evaluation complete")
    print(f"  Users:     {len(eval_users)}")
    print(f"  Questions: {len(all_results)}")
    print(f"  Output:    {args.output_dir}/")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
