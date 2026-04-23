"""
ragas_eval.py 鈥?Run RAGAS evaluation on the output of eval_batch.py

Evaluates using RAGAS metrics:
  - Faithfulness:       Is the answer faithful to the retrieved contexts?
  - Answer Relevancy:   Is the answer relevant to the question?
  - Context Precision:  Are the top-ranked contexts actually relevant?
  - Context Recall:     Do the contexts cover all info needed? (requires ground_truth)
  - Answer Correctness: How correct is the answer? (requires ground_truth)

Usage:
    pip install ragas datasets
    python ragas_eval.py --input eval_results/ragas_dataset.json
    python ragas_eval.py --input eval_results/ragas_dataset.json --limit 50 --api_key <key>

Note:
    RAGAS by default uses OpenAI models as the LLM judge.
    To use a local model or other provider, see RAGAS documentation.
    Set OPENAI_API_KEY environment variable before running.
"""

import argparse
import json
import warnings
from numbers import Number

# Suppress some langchain warnings
warnings.filterwarnings("ignore")


def extract_summary_scores(result, selected_metrics):
    """Return a flat metric->score dict from RAGAS EvaluationResult across versions."""
    metric_names = [m.name for m in selected_metrics]

    if isinstance(result, dict):
        return {k: float(v) for k, v in result.items() if isinstance(v, Number)}

    if hasattr(result, "to_dict"):
        try:
            raw = result.to_dict()
            if isinstance(raw, dict):
                return {k: float(v) for k, v in raw.items() if isinstance(v, Number)}
        except Exception as e:
            print(f"[Warn] Failed to read summary from result.to_dict(): {e}")

    if hasattr(result, "to_pandas"):
        try:
            df = result.to_pandas()
            summary = {}
            for name in metric_names:
                if name in df.columns:
                    series = df[name].dropna()
                    if len(series) > 0:
                        summary[name] = float(series.mean())
            return summary
        except Exception as e:
            print(f"[Warn] Failed to read summary from result.to_pandas(): {e}")

    return {}


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation with SiliconFlow")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to ragas_dataset.json from eval_batch.py")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--metrics", nargs="*", default=None,
                        help="Specific metrics to evaluate (default: all applicable)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N records after filtering.")
    parser.add_argument("--api_key", type=str, required=True,
                        help="SiliconFlow API key for the LLM judge")
    parser.add_argument("--base_url", type=str, default="https://api.siliconflow.cn/v1",
                        help="SiliconFlow API Base URL")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V3.2",
                        help="Judge LLM model name")
    parser.add_argument("--embed_model", type=str, default="/root/bge-m3",
                        help="Local embedding model path for metrics assessing semantic similarity")
    args = parser.parse_args()

    # 鈹€鈹€ Load data 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print(f"[Load] Reading {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"[Load] {len(records)} records loaded")
    if args.limit is not None:
        if args.limit <= 0:
            print("[Error] --limit must be a positive integer.")
            raise SystemExit(1)
        before_limit_count = len(records)
        records = records[:args.limit]
        print(f"[Filter] Limit applied: {len(records)}/{before_limit_count}")
        if not records:
            print("[Error] No records left after applying --limit.")
            raise SystemExit(1)

    gt_present_count = sum(
        1 for r in records if "ground_truth" in r and str(r.get("ground_truth", "")).strip()
    )
    has_ground_truth = gt_present_count == len(records) and len(records) > 0
    print(f"[Info] Ground truth coverage: {gt_present_count}/{len(records)}")
    print(f"[Info] Ground truth fully available: {has_ground_truth}")

    # 鈹€鈹€ Initialize LLM & Embeddings 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        )
        from langchain_openai import ChatOpenAI
        try:
            # New package path (preferred)
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            # Backward-compatible fallback
            from langchain_community.embeddings import HuggingFaceEmbeddings
            print("[Warn] Using deprecated langchain_community.HuggingFaceEmbeddings."
                  " Install `langchain-huggingface` to remove this warning.")
    except ImportError as e:
        print(f"\n[Error] Missing dependency: {e}")
        print("  Please run: pip install ragas datasets langchain-openai sentence-transformers")
        print("  Optional (recommended): pip install -U langchain-huggingface")
        return

    print(f"\n[Init] Setting up Judge LLM: {args.model}")
    judge_llm = ChatOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=0.1,
    )
    
    # 鈹€鈹€ Monkey Patch for DeepSeek/SiliconFlow: Force n=1 鈹€鈹€
    # RAGAS requests n=3 for some metrics, but SiliconFlow DeepSeek strict-rejects n>1.
    # We patch the default `bind` method to ignore the `n` runtime arg.
    original_bind = judge_llm.bind

    def custom_bind(**kwargs):
        if "n" in kwargs:
            kwargs.pop("n")
        return original_bind(**kwargs)

    try:
        # ChatOpenAI is a pydantic model in some versions; normal setattr may fail.
        object.__setattr__(judge_llm, "bind", custom_bind)
    except Exception as e:
        print(f"[Warn] Failed to patch judge_llm.bind (n-forcing workaround disabled): {e}")
    # 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    print(f"[Init] Setting up Embeddings: {args.embed_model}")
    judge_embeddings = HuggingFaceEmbeddings(model_name=args.embed_model)

    # 鈹€鈹€ Build HuggingFace Dataset 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    ds_dict = {
        "question": [r["question"] for r in records],
        "answer": [r["answer"] for r in records],
        "contexts": [r["contexts"] for r in records],
    }
    if has_ground_truth:
        ds_dict["ground_truth"] = [r.get("ground_truth", "") for r in records]

    dataset = Dataset.from_dict(ds_dict)
    print(f"\n[Dataset] Configured testing dataset with {len(dataset)} items.")

    # 鈹€鈹€ Select metrics 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
    }
    gt_required_metrics = {"context_recall", "answer_correctness"}

    if args.metrics:
        unknown_metrics = [m for m in args.metrics if m not in metric_map]
        if unknown_metrics:
            print(f"[Error] Unknown metrics: {unknown_metrics}")
            print(f"        Supported metrics: {list(metric_map.keys())}")
            raise SystemExit(1)

        invalid_gt_metrics = [m for m in args.metrics if m in gt_required_metrics and not has_ground_truth]
        if invalid_gt_metrics:
            print(f"[Error] Metrics require complete ground_truth but dataset is incomplete: {invalid_gt_metrics}")
            print(f"        Ground truth coverage is {gt_present_count}/{len(records)}.")
            raise SystemExit(1)

        selected = [metric_map[m] for m in args.metrics if m in metric_map]
    else:
        # Default: use metrics that don't require ground_truth
        selected = [faithfulness, answer_relevancy]
        if has_ground_truth:
            selected.extend([context_precision, context_recall, answer_correctness])
        else:
            print("[Warn] Ground truth not complete; skipping context_recall/answer_correctness.")

    if not selected:
        print("[Error] No valid metrics selected.")
        raise SystemExit(1)

    # Some RAGAS metrics sample multiple generations by default (e.g. strictness=3),
    # which is expensive and may be unsupported by certain providers.
    for m in selected:
        if hasattr(m, "strictness"):
            try:
                cur = getattr(m, "strictness")
                if isinstance(cur, int) and cur > 1:
                    setattr(m, "strictness", 1)
                    print(f"[Config] Set metric strictness to 1 for: {m.name}")
            except Exception as e:
                print(f"[Warn] Could not set strictness for {getattr(m, 'name', m)}: {e}")

    print(f"[Metrics] Metrics to calculate: {[m.name for m in selected]}")

    # 鈹€鈹€ Run evaluation 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print(f"\n[Eval] Running RAGAS evaluation...")
    print(f"       (Please wait, this issues API calls to the LLM judge)")

    result = evaluate(
        dataset,
        metrics=selected,
        llm=judge_llm,
        embeddings=judge_embeddings
    )

    # 鈹€鈹€ Display results 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print(f"\n{'=' * 60}")
    print(f"  RAGAS Evaluation Results")
    print(f"{'=' * 60}")

    scores_dict = extract_summary_scores(result, selected)
    if not scores_dict:
        print("[Warn] No aggregate scores were extracted from RAGAS result.")
        print("       Check ragas version compatibility or inspect per-sample CSV output.")

    for metric_name, score in scores_dict.items():
        if isinstance(score, (float, int)):
            print(f"  {metric_name:.<40} {score:.4f}")

    # 鈹€鈹€ Save results 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    if args.output is None:
        args.output = args.input.replace("ragas_dataset", "ragas_scores").replace(".json", ".json")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(scores_dict, f, ensure_ascii=False, indent=2)
    print(f"\n[Save] Scores saved to {args.output}")

    if hasattr(result, "to_pandas"):
        try:
            df = result.to_pandas()
            per_sample_file = args.output.replace(".json", "_per_sample.csv")
            df.to_csv(per_sample_file, index=False, encoding="utf-8-sig")
            print(f"[Save] Per-sample scores -> {per_sample_file}")
        except Exception as e:
            print(f"[Warn] Failed to save per-sample CSV: {e}")

    print(f"\n{'=' * 60}")
    print(f"  Evaluation complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
