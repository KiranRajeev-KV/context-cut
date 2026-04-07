"""Sales agent with three context modes: naive, compress, prune."""

import time
from typing import Literal

import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL
from src.context.naive import assemble_naive_context, count_tokens
from src.context.compress import assemble_compressed_context
from src.context.prune import assemble_pruned_context

SALES_AGENT_PROMPT = """You are an expert sales development representative preparing for an outbound call.

Given the CRM context below, generate a personalized call script for contacting {contact_name} at {company_name}.

Your call script should:
1. Reference their company and role appropriately
2. Acknowledge prior interactions and current deal stage
3. Address any known objections proactively
4. Follow up on previous action items
5. Have a clear next step

CRM Context:
{context}

Generate a concise, natural-sounding call script (3-5 paragraphs)."""

MAX_RETRIES = 5
BASE_BACKOFF = 2.0


def get_llm() -> ChatOpenAI:
    """Initialize the LLM with OpenRouter."""
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=SecretStr(OPENROUTER_API_KEY) if OPENROUTER_API_KEY else None,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.7,
    )


def invoke_with_retry(chain, inputs: dict[str, object]) -> object:
    """Invoke LangChain chain with exponential backoff retry."""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            backoff = min(BASE_BACKOFF * (2 ** (attempt - 1)), 60)
            print(f"    LLM retry {attempt}/{MAX_RETRIES} after {backoff:.1f}s backoff")
            time.sleep(backoff)
        try:
            return chain.invoke(inputs)
        except Exception as e:
            last_error = e
            if "429" in str(e) or "rate" in str(e).lower():
                print("    Rate limited — will retry")
            else:
                print(f"    Error: {e} — will retry")
    raise last_error  # type: ignore[misc]


def run_agent(
    company_name: str,
    contact_name: str,
    mode: Literal["naive", "compress", "prune"] = "prune",
) -> dict[str, object]:
    """Run the sales agent with the specified context mode.

    Returns:
        Dictionary with response, context info, and timing.
    """
    print(f"  [agent] Running in {mode} mode...")
    start_time = time.time()

    # Assemble context based on mode
    if mode == "naive":
        print("  [agent] Assembling naive context...")
        context = assemble_naive_context(company_name, contact_name)
        context_tokens = count_tokens(context)
        compressed_tokens = context_tokens
        print(f"  [agent] Naive context: {context_tokens} tokens")
    elif mode == "compress":
        context, context_tokens, compressed_tokens = assemble_compressed_context(
            company_name, contact_name
        )
    else:  # prune
        print("  [agent] Assembling pruned context (Qdrant vector search)...")
        context, context_tokens, compressed_tokens = assemble_pruned_context(
            company_name, contact_name
        )
        print(f"  [agent] Pruned context: {compressed_tokens}/{context_tokens} tokens")

    context_time = time.time() - start_time

    # Build prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert sales development representative."),
            ("human", SALES_AGENT_PROMPT),
        ]
    )

    # Run LLM with retry
    llm = get_llm()
    chain = prompt | llm

    print(f"  [agent] Calling LLM ({OPENROUTER_MODEL})...")
    llm_start = time.time()
    response = invoke_with_retry(
        chain,
        {
            "contact_name": contact_name,
            "company_name": company_name,
            "context": context,
        },
    )
    llm_time = time.time() - llm_start
    print(f"  [agent] LLM response received in {llm_time:.1f}s")

    total_time = time.time() - start_time

    return {
        "mode": mode,
        "response": response.content if hasattr(response, "content") else str(response),  # type: ignore[union-attr]
        "input_context_tokens": context_tokens,
        "effective_context_tokens": compressed_tokens,
        "response_tokens": _count_tokens(
            response.content if hasattr(response, "content") else str(response)  # type: ignore[union-attr]
        ),
        "context_time_ms": int(context_time * 1000),
        "llm_time_ms": int(llm_time * 1000),
        "total_time_ms": int(total_time * 1000),
    }


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base)."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: uv run python -m src.agents.sales_agent <company> <contact> [mode]"
        )
        print("Modes: naive, compress, prune (default: prune)")
        sys.exit(1)

    company = sys.argv[1]
    contact = sys.argv[2]
    mode_val = sys.argv[3] if len(sys.argv) > 3 else "prune"
    mode: Literal["naive", "compress", "prune"] = (
        mode_val if mode_val in ("naive", "compress", "prune") else "prune"
    )

    result = run_agent(company, contact, mode)

    print(f"\n{'=' * 60}")
    print(f"Mode: {result['mode']}")
    print(f"Raw context tokens: {result['input_context_tokens']}")
    print(f"Effective context tokens: {result['effective_context_tokens']}")
    print(f"Response tokens: {result['response_tokens']}")
    print(f"Context assembly time: {result['context_time_ms']}ms")
    print(f"LLM response time: {result['llm_time_ms']}ms")
    print(f"Total time: {result['total_time_ms']}ms")
    print(f"{'=' * 60}")
    print(f"\n{result['response']}")
