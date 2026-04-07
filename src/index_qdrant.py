"""Index CRM context pieces into Qdrant for vector search.

Run this once after data ingestion to populate the vector store.
Subsequent runs will skip already-indexed points.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.context.prune import index_context_pieces


def main() -> None:
    """Index all context pieces into Qdrant."""
    print("=== Qdrant Indexing ===")
    start = time.time()
    try:
        count = index_context_pieces()
        elapsed = time.time() - start
        print(f"\nDone. {count} new points indexed in {elapsed:.1f}s.")
        print("Subsequent runs will skip already-indexed points.")
    except Exception as e:
        print(f"\nFailed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
