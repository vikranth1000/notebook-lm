#!/usr/bin/env python3
"""
Notebook LM CLI utilities.

Usage:
    python scripts/notebooklm_cli.py notebooks
    python scripts/notebooklm_cli.py jobs
    python scripts/notebooklm_cli.py diagnostics
"""

from __future__ import annotations

import argparse
import json
import asyncio
from datetime import datetime
from pathlib import Path

from notebooklm_backend.config import get_settings, reset_settings_cache
from notebooklm_backend.services.notebook_store import NotebookStore
from notebooklm_backend.services.metrics_store import MetricsStore
from notebooklm_backend.services.agent import AgentService


def human_ts(value: datetime | None) -> str:
    if not value:
        return "-"
    return value.strftime("%Y-%m-%d %H:%M")


def cmd_notebooks(_: argparse.Namespace) -> None:
    reset_settings_cache()
    settings = get_settings()
    settings.ensure_directories()
    store = NotebookStore(settings)
    notebooks = store.list_notebooks()
    if not notebooks:
        print("No notebooks indexed yet.")
        return
    print(f"{'Notebook':<12} {'Title':<30} {'Docs':>4} {'Chunks':>6} {'Updated'}")
    print("-" * 70)
    for notebook in notebooks:
        print(
            f"{notebook.notebook_id[:8]:<12} "
            f"{notebook.title[:29]:<30} "
            f"{notebook.source_count:>4} "
            f"{notebook.chunk_count:>6} "
            f"{human_ts(notebook.updated_at)}"
        )


def cmd_jobs(_: argparse.Namespace) -> None:
    reset_settings_cache()
    settings = get_settings()
    settings.ensure_directories()
    store = NotebookStore(settings)
    jobs = store.list_jobs()
    if not jobs:
        print("No ingestion jobs recorded yet.")
        return
    print(f"{'Job':<12} {'Notebook':<12} {'Status':<10} {'Docs':>4} {'Chunks':>6} {'Started':<17} {'Completed'}")
    print("-" * 90)
    for job in jobs:
        print(
            f"{job.job_id[:8]:<12} "
            f"{job.notebook_id[:8]:<12} "
            f"{job.status:<10} "
            f"{(job.documents_processed or 0):>4} "
            f"{(job.chunks_indexed or 0):>6} "
            f"{human_ts(job.started_at):<17} "
            f"{human_ts(job.completed_at)}"
        )


def cmd_diagnostics(_: argparse.Namespace) -> None:
    reset_settings_cache()
    settings = get_settings()
    settings.ensure_directories()
    print("Configuration")
    print("-" * 40)
    print(f"Workspace: {settings.workspace_root}")
    print(f"LLM Provider: {settings.llm_provider}")
    print(f"Ollama Model: {settings.resolved_ollama_model or settings.ollama_model}")
    if settings.model_selection_reason:
        print(f"Model Reason: {settings.model_selection_reason}")
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"LangChain Splitter: {settings.use_langchain_splitter}")
    print(f"LlamaIndex RAG: {settings.use_llamaindex_rag}")
    print()

    baseline_path = Path(__file__).parent.parent / "docs" / "baseline_metrics.json"
    if baseline_path.exists():
        print("Baseline Metrics (docs/baseline_metrics.json)")
        print("-" * 40)
        data = json.loads(baseline_path.read_text())
        embedding = data.get("embedding", {})
        if embedding:
            print("Embedding throughput:")
            for label, metrics in embedding.items():
                tps = metrics.get("texts_per_second")
                print(f"  {label}: {tps:.1f} texts/sec")
        queries = data.get("queries", {})
        if queries:
            print(f"Average query time: {queries.get('average_query_time', 0):.3f}s")
        print()
    else:
        print("No baseline metrics found. Run scripts/measure_baseline_auto.py first.\n")


def cmd_metrics(_: argparse.Namespace) -> None:
    reset_settings_cache()
    settings = get_settings()
    store = MetricsStore(settings)
    summary = store.summary()
    print("Chat Metrics Summary")
    print("-" * 40)
    print(f"Conversations: {summary.conversations}")
    print(f"Avg total latency: {summary.avg_total_ms or 0:.1f} ms")
    print(f"Avg LLM latency: {summary.avg_llm_ms or 0:.1f} ms")
    print(f"Avg retrieval latency: {summary.avg_retrieval_ms or 0:.1f} ms")
    print("Providers:")
    for provider, count in summary.provider_breakdown.items():
        print(f"  {provider}: {count}")


def cmd_agent(args: argparse.Namespace) -> None:
    if not args.goal:
        raise SystemExit("Provide a goal with --goal")
    reset_settings_cache()
    settings = get_settings()
    agent = AgentService(settings)
    plan = asyncio.run(agent.plan(args.goal, args.notebook))
    print(plan)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Notebook LM utility CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("notebooks", help="List indexed notebooks")
    sub.add_parser("jobs", help="List recent ingestion jobs")
    sub.add_parser("diagnostics", help="Show configuration and baseline metrics")
    sub.add_parser("metrics", help="Show chat latency summary")
    agent_parser = sub.add_parser("agent-plan", help="Generate a plan for a goal")
    agent_parser.add_argument("--goal", required=True, help="Goal description")
    agent_parser.add_argument("--notebook", help="Notebook ID to ground the plan")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "notebooks":
        cmd_notebooks(args)
    elif args.command == "jobs":
        cmd_jobs(args)
    elif args.command == "diagnostics":
        cmd_diagnostics(args)
    elif args.command == "metrics":
        cmd_metrics(args)
    elif args.command == "agent-plan":
        cmd_agent(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
