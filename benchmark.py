"""Benchmark: compare three context modes across multiple prospects.

Measures:
- Context tokens (input size)
- Output tokens (compressed/pruned size)
- Context assembly time
- LLM response time (TTFT proxy)
- Total time

Outputs a comparison table and saves results to JSON.
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.sales_agent import run_agent

DB_PATH = Path(__file__).parent / "data" / "crm.db"
RESULTS_PATH = Path(__file__).parent / "data" / "benchmark_results.json"

# Longer delay between runs to avoid 429 rate limits
INTER_RUN_DELAY = 3.0  # seconds between each mode run
INTER_PROSPECT_DELAY = 5.0  # seconds between prospects


def get_prospects(limit: int = 5) -> list[dict[str, str]]:
    """Get prospects with multi-touch call histories."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT c.name as company, ct.name as contact, d.deal_stage
        FROM call_summaries cs
        JOIN contacts ct ON ct.id = cs.contact_id
        JOIN companies c ON c.id = cs.company_id
        JOIN deals d ON d.company_id = cs.company_id
        GROUP BY cs.company_id, ct.id
        ORDER BY RANDOM()
        LIMIT ?
    """,
        (limit,),
    ).fetchall()

    conn.close()
    return [dict(r) for r in rows]


def run_benchmark(
    prospects: list[dict[str, str]], modes: list[str] | None = None
) -> dict[str, list[dict[str, object]]]:
    """Run benchmark across all prospects and modes."""
    if modes is None:
        modes = ["naive", "compress", "prune"]

    results: dict[str, list[dict[str, object]]] = {mode: [] for mode in modes}
    total_start = time.time()

    for i, p in enumerate(prospects, 1):
        company = p["company"]
        contact = p["contact"]
        deal_stage = p.get("deal_stage", "unknown")
        prospect_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(prospects)}] {contact} at {company} (Stage: {deal_stage})")
        print(f"{'=' * 60}")

        for mode in modes:
            mode_start = time.time()
            print(f"\n  → Running {mode} mode...")
            try:
                result = run_agent(company, contact, mode)  # type: ignore[arg-type]
                results[mode].append(result)
                mode_elapsed = time.time() - mode_start
                print(
                    f"  ✓ {mode} complete: {mode_elapsed:.1f}s "
                    f"(raw: {result['input_context_tokens']}t → effective: {result['effective_context_tokens']}t → response: {result['response_tokens']}t)"
                )
            except Exception as e:
                mode_elapsed = time.time() - mode_start
                print(f"  ✗ {mode} failed after {mode_elapsed:.1f}s: {e}")
                results[mode].append({"error": str(e)})

            # Rate limit between modes
            if mode != modes[-1]:
                print(f"  ⏳ Waiting {INTER_RUN_DELAY}s to avoid rate limits...")
                time.sleep(INTER_RUN_DELAY)

        prospect_elapsed = time.time() - prospect_start
        print(f"\n  Prospect {i} completed in {prospect_elapsed:.1f}s")

        # Rate limit between prospects
        if i < len(prospects):
            print(f"  ⏳ Waiting {INTER_PROSPECT_DELAY}s before next prospect...")
            time.sleep(INTER_PROSPECT_DELAY)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Benchmark completed in {total_elapsed:.1f}s")
    print(f"{'=' * 60}")

    return results


def print_summary(results: dict[str, list[dict[str, object]]]) -> None:
    """Print benchmark summary table."""
    print(f"\n{'=' * 110}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 110}")
    print(
        f"{'Mode':<12} {'Raw Context Tokens':<20} {'Effective Context Tokens':<24} {'Avg Response Tokens':<20} {'Avg LLM Time':<14} {'Avg Total':<12}"
    )
    print(f"{'-' * 110}")

    for mode, mode_results in results.items():
        valid = [r for r in mode_results if "error" not in r]
        errors = [r for r in mode_results if "error" in r]
        if not valid:
            print(f"{mode:<12} ERROR ({len(errors)} failures)")
            continue

        avg_raw = sum(int(r["input_context_tokens"]) for r in valid) / len(valid)  # type: ignore[arg-type]
        avg_eff = sum(int(r["effective_context_tokens"]) for r in valid) / len(valid)  # type: ignore[arg-type]
        avg_resp = sum(int(r["response_tokens"]) for r in valid) / len(valid)  # type: ignore[arg-type]
        avg_llm = sum(int(r["llm_time_ms"]) for r in valid) / len(valid)  # type: ignore[arg-type]
        avg_total = sum(int(r["total_time_ms"]) for r in valid) / len(valid)  # type: ignore[arg-type]

        error_note = f" ({len(errors)} failed)" if errors else ""
        print(
            f"{mode:<12} {avg_raw:<20.0f} {avg_eff:<24.0f} {avg_resp:<20.0f} {avg_llm:<14.0f}ms {avg_total:<12.0f}ms{error_note}"
        )

    # Calculate savings
    naive = [r for r in results.get("naive", []) if "error" not in r]
    prune = [r for r in results.get("prune", []) if "error" not in r]

    if naive and prune:
        naive_tokens = sum(int(r["input_context_tokens"]) for r in naive) / len(naive)  # type: ignore[arg-type]
        prune_tokens = sum(int(r["effective_context_tokens"]) for r in prune) / len(prune)  # type: ignore[arg-type]
        naive_time = sum(int(r["total_time_ms"]) for r in naive) / len(naive)  # type: ignore[arg-type]
        prune_time = sum(int(r["total_time_ms"]) for r in prune) / len(prune)  # type: ignore[arg-type]

        token_reduction = ((naive_tokens - prune_tokens) / max(naive_tokens, 1)) * 100
        time_reduction = ((naive_time - prune_time) / max(naive_time, 1)) * 100

        print(f"\n{'=' * 110}")
        print("KEY METRICS")
        print(f"{'=' * 110}")
        print(f"Token reduction (naive → prune): {token_reduction:.1f}%")
        print(f"Time reduction (naive → prune):  {time_reduction:.1f}%")
        print(
            "\nInspired by: 'Is a Large Context Window All You Need?' (Roy et al., Alchemyst AI)"
        )
        print("Paper results: 38.5% latency reduction, 99.73% token savings")


def main() -> None:
    """Benchmark entry point."""
    print("\n" + "=" * 90)
    print("CONTEXT-CUT: Benchmark — Naive vs Compressed vs Pruned Context")
    print(
        "Inspired by: 'Is a Large Context Window All You Need?' (Roy et al., Alchemyst AI)"
    )
    print("=" * 90)

    limit_str = input("\nNumber of prospects to benchmark? (default 3): ").strip()
    limit = int(limit_str) if limit_str.isdigit() else 3

    prospects = get_prospects(limit)
    if not prospects:
        print("No prospects found. Run data ingestion first.")
        return

    print("\nProspects to benchmark:")
    for i, p in enumerate(prospects, 1):
        print(f"  {i}. {p['contact']} at {p['company']} ({p.get('deal_stage', '?')})")

    confirm = (
        input(f"\nBenchmark {len(prospects)} prospects × 3 modes? (y/n): ")
        .strip()
        .lower()
    )
    if confirm != "y":
        print("Cancelled.")
        return

    print("\nStarting benchmark...")
    print(
        f"Rate limits: {INTER_RUN_DELAY}s between modes, {INTER_PROSPECT_DELAY}s between prospects"
    )

    results = run_benchmark(prospects)

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    print_summary(results)


if __name__ == "__main__":
    main()
