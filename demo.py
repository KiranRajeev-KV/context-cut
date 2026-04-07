"""Interactive demo: compare three context modes side-by-side."""

import sqlite3
import sys
import time
from pathlib import Path
from typing import Literal

# Add parent to path for imports before importing project modules
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.sales_agent import run_agent  # noqa: E402

DB_PATH = Path(__file__).parent / "data" / "crm.db"


def list_prospects(limit: int = 10) -> list[sqlite3.Row]:
    """List prospects with multi-touch call histories."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    prospects = conn.execute(
        """
        SELECT c.name as company, c.sector, c.revenue,
               ct.name as contact, ct.job_title, ct.decision_maker,
               d.deal_stage, d.product, d.close_value,
               COUNT(DISTINCT cs.id) as call_count
        FROM call_summaries cs
        JOIN contacts ct ON ct.id = cs.contact_id
        JOIN companies c ON c.id = cs.company_id
        JOIN deals d ON d.company_id = cs.company_id
        GROUP BY cs.company_id, ct.id
        ORDER BY call_count DESC
        LIMIT ?
    """,
        (limit,),
    ).fetchall()

    conn.close()

    print(f"\n{'=' * 80}")
    print(
        f"{'#':<4} {'Company':<20} {'Contact':<20} {'Title':<22} {'Stage':<15} {'Calls':<6}"
    )
    print(f"{'=' * 80}")

    for i, p in enumerate(prospects, 1):
        print(
            f"{i:<4} {(p['company'] or '')[:19]:<20} {(p['contact'] or '')[:19]:<20} "
            f"{(p['job_title'] or '')[:21]:<22} {(p['deal_stage'] or '')[:14]:<15} "
            f"{p['call_count']:<6}"
        )

    return prospects  # type: ignore[return-value]


def run_comparison(company: str, contact: str) -> None:
    """Run all three modes and compare."""
    print(f"\n{'=' * 80}")
    print(f"CONTEXT-CUT: Comparing context modes for {contact} at {company}")
    print(f"{'=' * 80}")

    results: dict[str, dict[str, object]] = {}
    for mode in ["naive", "compress", "prune"]:
        print(f"\n⏳ Running {mode} mode...")
        start = time.time()
        try:
            result = run_agent(company, contact, mode)  # type: ignore[arg-type]
            results[mode] = result
            elapsed = time.time() - start
            print(f"   ✓ Complete in {elapsed:.1f}s")
            print(f"   Context tokens: {result['context_tokens']}")
            print(f"   Output tokens: {result['compressed_tokens']}")
            print(f"   LLM time: {result['llm_time_ms']}ms")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            results[mode] = {"error": str(e)}

    # Print comparison table
    print(f"\n{'=' * 110}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 110}")
    print(
        f"{'Mode':<12} {'Raw Context Tokens':<20} {'Effective Context Tokens':<24} {'Response Tokens':<18} {'LLM Time':<12} {'Total Time':<12}"
    )
    print(f"{'-' * 110}")

    for mode in ["naive", "compress", "prune"]:
        r = results.get(mode, {})
        if "error" in r:
            print(f"{mode:<12} ERROR: {r['error']}")
        else:
            print(
                f"{mode:<12} {r['input_context_tokens']:<20} {r['effective_context_tokens']:<24} {r['response_tokens']:<18} {r['llm_time_ms']}ms{'':<7} {r['total_time_ms']}ms"
            )

    # Print responses
    for mode in ["naive", "compress", "prune"]:
        r = results.get(mode, {})
        if "error" in r:
            continue

        print(f"\n{'=' * 80}")
        print(f"{mode.upper()} MODE RESPONSE")
        print(f"{'=' * 80}")
        print(r["response"])

    # Calculate savings
    if all(m in results and "error" not in results[m] for m in ["naive", "prune"]):
        naive_tokens = int(results["naive"]["input_context_tokens"])  # type: ignore[arg-type]
        prune_tokens = int(results["prune"]["effective_context_tokens"])  # type: ignore[arg-type]
        naive_time = int(results["naive"]["total_time_ms"])  # type: ignore[arg-type]
        prune_time = int(results["prune"]["total_time_ms"])  # type: ignore[arg-type]

        token_savings = ((naive_tokens - prune_tokens) / max(naive_tokens, 1)) * 100
        time_savings = ((naive_time - prune_time) / max(naive_time, 1)) * 100

        print(f"\n{'=' * 80}")
        print("KEY METRICS")
        print(f"{'=' * 80}")
        print(f"Token reduction (naive → prune): {token_savings:.1f}%")
        print(f"Time reduction (naive → prune):  {time_savings:.1f}%")
        print(
            "\nInspired by: 'Is a Large Context Window All You Need?' (Roy et al., Alchemyst AI)"
        )
        print("Paper results: 38.5% latency reduction, 99.73% token savings")


def main() -> None:
    """Interactive demo entry point."""
    print("\n" + "=" * 80)
    print("CONTEXT-CUT: Context-Aware Sales Agent Demo")
    print(
        "Inspired by: 'Is a Large Context Window All You Need?' (Roy et al., Alchemyst AI)"
    )
    print("=" * 80)

    while True:
        print("\nOptions:")
        print("  1. List prospects")
        print("  2. Run comparison for a prospect")
        print("  3. Run single mode for a prospect")
        print("  q. Quit")

        choice = input("\nChoice: ").strip().lower()

        if choice == "q":
            break
        elif choice == "1":
            limit_str = input("How many prospects? (default 10): ").strip()
            limit = int(limit_str) if limit_str.isdigit() else 10
            list_prospects(limit)
        elif choice == "2":
            company = input("Company name: ").strip()
            contact = input("Contact name: ").strip()
            if company and contact:
                run_comparison(company, contact)
        elif choice == "3":
            company = input("Company name: ").strip()
            contact = input("Contact name: ").strip()
            mode_str = input("Mode (naive/compress/prune): ").strip()
            mode: Literal["naive", "compress", "prune"] = (
                mode_str if mode_str in ("naive", "compress", "prune") else "prune"
            )
            if company and contact:
                result = run_agent(company, contact, mode)
                print(f"\n{'=' * 60}")
                print(f"Mode: {result['mode']}")
                print(f"Raw context tokens: {result['input_context_tokens']}")
                print(f"Effective context tokens: {result['effective_context_tokens']}")
                print(f"Response tokens: {result['response_tokens']}")
                print(f"{'=' * 60}")
                print(f"\n{result['response']}")
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
