"""
Golden-set evaluation CLI.

Retrieval (no LLM, compares vector / vector+rerank / hybrid / hybrid+rerank):
    py scripts/eval.py retrieval --out evals/fairy_tales/baselines/retrieval.json

End-to-end answers through AdvancedRAG.query_detailed. By default the AI
model running this command answers RAG's prompts (AgentLLM): each run writes
the prompts it still needs to <eval-dir>/runs/agent_requests.json and exits
with code 3; answer them in <eval-dir>/agent_answers.json and run again.
    py scripts/eval.py answer --judge

Or call an OpenAI-compatible API instead (spends credit):
    py scripts/eval.py answer --llm api --base-url https://openrouter.ai/api/v1 \
        --llm-model openai/gpt-4o-mini --limit 20

Compare a new run with a stored baseline:
    py scripts/eval.py compare evals/fairy_tales/baselines/retrieval.json new.json

Local models are downloaded on first use (bge-m3 + reranker, about 4.5 GB);
``--cache-dir`` points the HuggingFace cache somewhere with free space.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_EVAL_DIR = ROOT / "evals" / "fairy_tales"

RETRIEVAL_CONFIGS = {
    "vector": {"use_hybrid": False, "use_reranking": False},
    "vector_rerank": {"use_hybrid": False, "use_reranking": True},
    "hybrid": {"use_hybrid": True, "use_reranking": False},
    "hybrid_rerank": {"use_hybrid": True, "use_reranking": True},
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_pipeline_options(sub):
        sub.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR,
                         help="Folder with corpus/ and golden.jsonl")
        sub.add_argument("--k", type=int, default=5, help="Documents retrieved per question")
        sub.add_argument("--chunk-size", type=int, default=500)
        sub.add_argument("--chunk-overlap", type=int, default=50)
        sub.add_argument("--embedding-model", default=None, help="Default: BAAI/bge-m3")
        sub.add_argument("--device", default=None, help="Embedding device: cpu or cuda")
        sub.add_argument("--vector-store", choices=["faiss", "qdrant"], default="faiss",
                         help="qdrant runs in memory, no server needed")
        sub.add_argument("--parent-context", action=argparse.BooleanOptionalAction, default=True,
                         help="Search child chunks, return their parent chunks (pipeline default)")
        sub.add_argument("--parent-fanout", type=int, default=3,
                         help="Child chunks searched per parent returned")
        sub.add_argument("--cache-dir", type=Path, default=None, help="HuggingFace cache (HF_HOME)")
        sub.add_argument("--out", type=Path, default=None, help="Report path (default: <eval-dir>/runs/)")
        sub.add_argument("--baseline", type=Path, default=None, help="Report to compare against")

    retrieval = commands.add_parser("retrieval", help="Retrieval-only benchmark (no LLM)")
    add_pipeline_options(retrieval)
    retrieval.add_argument("--configs", default=",".join(RETRIEVAL_CONFIGS),
                           help=f"Comma-separated subset of: {', '.join(RETRIEVAL_CONFIGS)}")

    answer = commands.add_parser("answer", help="End-to-end answer benchmark (uses an LLM)")
    add_pipeline_options(answer)
    answer.add_argument("--config", choices=list(RETRIEVAL_CONFIGS), default="hybrid_rerank")
    answer.add_argument("--llm", choices=["agent", "api"], default="agent",
                        help="agent (default): the AI model running this command answers RAG's "
                             "prompts via AgentLLM, over several runs; api: --llm-provider/--llm-model")
    answer.add_argument("--answers", type=Path, default=None,
                        help="Agent answers file (default: <eval-dir>/agent_answers.json)")
    answer.add_argument("--requests", type=Path, default=None,
                        help="Agent requests file (default: <eval-dir>/runs/agent_requests.json)")
    answer.add_argument("--llm-provider", default="openai")
    answer.add_argument("--llm-model", default=None)
    answer.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, e.g. OpenRouter")
    answer.add_argument("--temperature", type=float, default=0.0)
    answer.add_argument("--judge", action="store_true",
                        help="Also score faithfulness, judged by the same model as the pipeline")
    answer.add_argument("--no-transform", action="store_true", help="Skip LLM query rewriting")
    answer.add_argument("--no-grade", action="store_true", help="Skip LLM document grading")
    answer.add_argument("--limit", type=int, default=None, help="Only the first N questions")
    answer.add_argument("--ids", default=None, help="Comma-separated question ids")

    compare = commands.add_parser("compare", help="Diff two reports")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("current", type=Path)

    return parser.parse_args(argv)


def git_revision() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_rag(args, use_hybrid: bool, use_reranking: bool, llm_provider="openai", llm_model=None):
    """Index the eval corpus with AdvancedRAG; returns (rag, seconds spent indexing)."""
    from src.rag.advanced_rag import AdvancedRAG

    rag = AdvancedRAG(
        llm_provider=llm_provider,
        llm_model=llm_model,
        embedding_model=args.embedding_model,
        embedding_device=args.device,
        vector_store_provider=args.vector_store,
        vector_store_url=":memory:" if args.vector_store == "qdrant" else None,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        retrieval_k=args.k,
        use_hybrid=use_hybrid,
        use_reranking=use_reranking,
        use_parent_context=args.parent_context,
        parent_fanout=args.parent_fanout,
    )

    start = time.perf_counter()
    rag.add_documents(args.eval_dir / "corpus")
    return rag, time.perf_counter() - start


def display_path(path: Path) -> str:
    """Repo-relative POSIX path for reports, so they read the same on every machine."""
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def pipeline_settings(args, rag, index_seconds: float) -> dict:
    retriever = rag._retriever
    return {
        "eval_dir": display_path(args.eval_dir),
        "k": args.k,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "embedding_model": rag.embeddings.config.model_name,
        "embedding_device": args.device or "auto",
        "vector_store": args.vector_store,
        "parent_context": args.parent_context,
        "parent_fanout": args.parent_fanout if args.parent_context else None,
        "reranker_model": retriever.active_reranker_model if retriever else None,
        "documents": rag.num_documents,
        "retrieval_chunks": rag.num_chunks,
        "index_seconds": round(index_seconds, 2),
    }


def run_metadata(kind: str) -> dict:
    return {
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "python": platform.python_version(),
    }


def write_report(report: dict, out, eval_dir: Path) -> Path:
    if out is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = eval_dir / "runs" / f"{report['kind']}-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 10 else f"{value:.0f}"
    return str(value)


def print_table(title: str, configs: dict, metrics) -> None:
    print(f"\n{title}")
    width = max(len(name) for name in configs) + 2
    print("config".ljust(width) + "".join(m[:14].rjust(16) for m in metrics))
    for name, run in configs.items():
        print(name.ljust(width) + "".join(fmt(run["summary"].get(m)).rjust(16) for m in metrics))
    for name, run in configs.items():
        errors = run["summary"].get("errors")
        if errors:
            # Failed questions score zero and drag the averages down: never compare such a run.
            first = next(case["error"] for case in run["cases"] if case.get("error"))
            print(f"WARNING: {name}: {errors} questions failed (e.g. {first[:120]})")


def print_comparison(baseline: dict, current: dict) -> None:
    from src.evaluation.benchmark import compare_reports

    rows = compare_reports(baseline, current)
    if not rows:
        print("\nNo config in common with the baseline.")
        return
    print(f"\nCompared with baseline {baseline.get('git_revision', '?')} ({baseline.get('created_at', '?')}):")
    for row in rows:
        marker = {"better": "+", "worse": "!", "same": " "}[row["status"]]
        print(
            f" {marker} {row['config']:<14} {row['metric']:<24} "
            f"{fmt(row['baseline']):>10} -> {fmt(row['current']):>10} ({row['delta']:+.3f})"
        )


def cmd_retrieval(args) -> int:
    from src.evaluation.benchmark import load_golden_set, run_retrieval_benchmark

    names = [name.strip() for name in args.configs.split(",") if name.strip()]
    unknown = [name for name in names if name not in RETRIEVAL_CONFIGS]
    if unknown:
        print(f"Unknown config(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    cases = load_golden_set(args.eval_dir / "golden.jsonl")
    needs_rerank = any(RETRIEVAL_CONFIGS[name]["use_reranking"] for name in names)
    rag, index_seconds = build_rag(args, use_hybrid=True, use_reranking=needs_rerank)

    # Warm up lazy model loading and CUDA kernels so latencies are comparable.
    rag.retrieve(cases[0].question, k=args.k, use_hybrid=True, use_reranking=needs_rerank)

    configs = {}
    for name in names:
        options = RETRIEVAL_CONFIGS[name]
        configs[name] = run_retrieval_benchmark(
            lambda question, k, options=options: rag.retrieve(question, k=k, **options),
            cases,
            k=args.k,
        )

    report = {**run_metadata("retrieval"), "settings": pipeline_settings(args, rag, index_seconds), "configs": configs}
    path = write_report(report, args.out, args.eval_dir)

    print_table(
        f"Retrieval @k={args.k} on {configs[names[0]]['summary']['cases']} answerable questions",
        configs,
        ["recall_at_k", "mrr", "ndcg", "evidence_recall", "context_chars", "latency_p50_ms"],
    )
    if args.baseline:
        print_comparison(json.loads(args.baseline.read_text(encoding="utf-8")), report)
    print(f"\nReport: {path}")
    return 0


AGENT_SYSTEM_PROMPT = (
    "Bạn là LLM bên trong một pipeline RAG đang được đo đánh giá. Làm đúng yêu cầu của "
    "từng prompt (định dạng, ngôn ngữ, độ dài). Khi prompt yêu cầu chỉ dựa vào ngữ cảnh "
    "hoặc tài liệu đi kèm, tuyệt đối không bổ sung hiểu biết riêng về các truyện."
)


def cmd_answer(args) -> int:
    from src.core.agent_llm import AgentChatModel, AgentLLM
    from src.core.llm import LLMManager
    from src.evaluation.benchmark import (
        LLMCallCounter,
        load_golden_set,
        make_faithfulness_judge,
        run_answer_benchmark,
    )

    cases = load_golden_set(args.eval_dir / "golden.jsonl")
    if args.ids:
        wanted = {item.strip() for item in args.ids.split(",")}
        cases = [case for case in cases if case.id in wanted]
    if args.limit:
        cases = cases[:args.limit]

    options = RETRIEVAL_CONFIGS[args.config]
    rag, index_seconds = build_rag(
        args, llm_provider=args.llm_provider, llm_model=args.llm_model, **options,
    )

    agent = None
    if args.llm == "agent":
        # Whichever model runs this command answers every prompt; no API is called.
        answers_path = args.answers or args.eval_dir / "agent_answers.json"
        requests_path = args.requests or args.eval_dir / "runs" / "agent_requests.json"
        agent = AgentLLM(answers_path, requests_path, system_prompt=AGENT_SYSTEM_PROMPT)
        rag.llm._llm = AgentChatModel(agent)
        rag.llm.config.provider = "agent"
        rag.llm.config.model = agent.answered_by or "agent"
        judge_llm = agent

        # The grading prompts of one question do not depend on each other: ask them in one run.
        grade_documents = rag._grade_documents

        def grade_independently(question, docs):
            with agent.independent():
                return grade_documents(question, docs)

        rag._grade_documents = grade_independently
    else:
        # The chat client is created lazily, so these still apply.
        rag.llm.config.temperature = args.temperature
        if args.base_url:
            rag.llm.config.base_url = args.base_url
        judge_llm = LLMManager(
            provider=args.llm_provider, model=rag.llm.config.model,
            base_url=args.base_url, temperature=0.0,
        )

    counter = LLMCallCounter()
    counter.attach(rag.llm)
    judge = make_faithfulness_judge(judge_llm) if args.judge else None

    def answer_fn(question):
        if agent:
            # Later prompts of a question depend on earlier answers (rewrite -> grade -> answer).
            agent.start_chain()
        result = rag.query_detailed(
            question,
            transform_query=not args.no_transform,
            grade_documents=not args.no_grade,
        )
        return {
            "answer": result["answer"],
            "contexts": [source["content"] for source in result["relevant_docs"]],
        }

    run = run_answer_benchmark(answer_fn, cases, counter=counter, judge=judge)

    if agent and not agent.complete:
        path = agent.write_requests()
        print(
            f"\n{len(agent.pending)} prompts wait for the calling model "
            f"({agent.deferred} more depend on them).\n"
            f"Requests: {path}\nAnswer them in {agent.answers_path}, then run the same command again."
        )
        return 3

    settings = {
        **pipeline_settings(args, rag, index_seconds),
        "llm_provider": rag.llm.config.provider,
        "llm_model": rag.llm.config.model,
        "base_url": args.base_url if not agent else None,
        "temperature": args.temperature if not agent else None,
        "agent_answers": display_path(agent.answers_path) if agent else None,
        "judge": "same model as the pipeline" if args.judge else None,
        "transform_query": not args.no_transform,
        "grade_documents": not args.no_grade,
    }
    if agent:
        settings["note"] = "Agent answers are replayed: latency excludes LLM generation."
    report = {**run_metadata("answer"), "settings": settings, "configs": {args.config: run}}
    path = write_report(report, args.out, args.eval_dir)

    print_table(
        f"Answers on {run['summary']['cases']} questions ({rag.llm.config.model})",
        report["configs"],
        ["answer_recall", "faithfulness", "abstention_accuracy", "false_abstention_rate",
         "citation_rate", "llm_calls_per_query"],
    )
    if args.baseline:
        print_comparison(json.loads(args.baseline.read_text(encoding="utf-8")), report)
    print(f"\nReport: {path}")
    return 0


def cmd_compare(args) -> int:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))
    print_comparison(baseline, current)
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    if getattr(args, "cache_dir", None):
        os.environ["HF_HOME"] = str(args.cache_dir.resolve())
    handlers = {"retrieval": cmd_retrieval, "answer": cmd_answer, "compare": cmd_compare}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
