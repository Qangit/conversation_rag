"""
generate_ground_truths.py — 使用 LLM API 生成 probing questions 的标准答案

读取 memory_bank_cn.json（历史对话）和 probing_questions_cn.jsonl（测试题），
将历史对话作为上下文，让 LLM 根据历史对话生成每个问题的标准答案。

Usage:
    python generate_ground_truths.py \
        --api_key YOUR_API_KEY \
        --data_file eval_data/memory_bank_cn.json \
        --questions_file eval_data/probing_questions_cn.jsonl \
        --output eval_data/ground_truths.jsonl \
        --users 张曼婷 王峰      # 可选，指定用户
"""

import argparse
import json
import os
import time
from openai import OpenAI


PROMPT_TEMPLATE = """\
以下是用户"{user_name}"与AI助手之间的完整历史对话记录。请你仔细阅读，然后根据对话内容回答问题。

要求：
1. 只根据对话记录中的事实来回答，不要编造
2. 回答要简洁准确，直接给出答案
3. 如果对话中没有相关信息，回答"对话中未提及"

═══ 对话记录 ═══
{dialogue_history}

═══ 问题 ═══
{question}

═══ 标准答案 ═══"""


def load_memory_bank(data_file: str) -> dict:
    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_probing_questions(questions_file: str) -> dict:
    questions = {}
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            questions.update(data)
    return questions


def format_dialogue_history(user_data: dict) -> str:
    """将用户的多天历史对话格式化为纯文本。"""
    history = user_data.get("history", {})
    sorted_dates = sorted(history.keys())

    lines = []
    for date_str in sorted_dates:
        lines.append(f"\n── {date_str} ──")
        turns = history[date_str]
        for turn in turns:
            lines.append(f"用户: {turn['query']}")
            lines.append(f"助手: {turn['response']}")

    return "\n".join(lines)


def generate_answer(client: OpenAI, model: str, user_name: str,
                    dialogue_history: str, question: str,
                    max_retries: int = 3) -> str:
    """调用 LLM API 生成标准答案。"""
    prompt = PROMPT_TEMPLATE.format(
        user_name=user_name,
        dialogue_history=dialogue_history,
        question=question,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个严谨的阅读理解助手，只根据给定的对话记录回答问题。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # 低温度确保答案确定性
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [Retry {attempt + 1}/{max_retries}] API error: {e}")
            time.sleep(2 * (attempt + 1))

    return "[生成失败]"


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth answers using LLM API"
    )
    parser.add_argument("--api_key", type=str, default="sk-yswgblrprxhgrxmzniintjxjssodnvajgalszzzddnlsvolt",required=True,
                        help="SiliconFlow API key")
    parser.add_argument("--base_url", type=str,
                        default="https://api.siliconflow.cn/v1")
    parser.add_argument("--model", type=str,
                        default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--data_file", type=str,
                        default="eval_data/memory_bank_cn.json")
    parser.add_argument("--questions_file", type=str,
                        default="eval_data/probing_questions_cn.jsonl")
    parser.add_argument("--output", type=str,
                        default="eval_data/ground_truths.jsonl")
    parser.add_argument("--users", nargs="*", default=None,
                        help="Specify user names. Default: all.")
    args = parser.parse_args()

    # ── Init API client ──────────────────────────────────────
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    print(f"[Init] API: {args.base_url}")
    print(f"[Init] Model: {args.model}")

    # ── Load data ────────────────────────────────────────────
    memory_bank = load_memory_bank(args.data_file)
    probing_questions = load_probing_questions(args.questions_file)

    available_users = list(memory_bank.keys())
    if args.users:
        eval_users = [u for u in args.users if u in available_users]
    else:
        eval_users = [u for u in available_users if u in probing_questions]

    print(f"[Info] Users to process: {len(eval_users)}")

    # ── Generate ground truths ───────────────────────────────
    all_ground_truths = {}
    total_questions = 0

    for idx, user_name in enumerate(eval_users, 1):
        questions = probing_questions.get(user_name, [])
        if not questions:
            continue

        print(f"\n{'─' * 60}")
        print(f"  [{idx}/{len(eval_users)}] {user_name} — {len(questions)} questions")
        print(f"{'─' * 60}")

        user_data = memory_bank[user_name]
        dialogue_history = format_dialogue_history(user_data)

        # 显示对话长度
        n_chars = len(dialogue_history)
        print(f"  Dialogue history: {n_chars} chars")

        answers = []
        for i, question in enumerate(questions, 1):
            print(f"  [Q{i}] {question}")
            answer = generate_answer(
                client, args.model, user_name,
                dialogue_history, question
            )
            print(f"  [A{i}] {answer[:120]}")
            answers.append(answer)
            total_questions += 1

            # 避免 API 限流
            time.sleep(0.5)

        all_ground_truths[user_name] = answers

    # ── Save results ─────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for user_name, answers in all_ground_truths.items():
            record = {user_name: answers}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"  Ground truths generated!")
    print(f"  Users:     {len(all_ground_truths)}")
    print(f"  Questions: {total_questions}")
    print(f"  Output:    {args.output}")
    print(f"{'=' * 60}")

    # 同时存一份完整 JSON 方便查看
    json_output = args.output.replace(".jsonl", ".json")
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(all_ground_truths, f, ensure_ascii=False, indent=2)
    print(f"  Also saved: {json_output}")


if __name__ == "__main__":
    main()
