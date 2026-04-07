"""Centralized configuration for context-cut."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# OpenRouter
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free"
)

# Embedding model (used for semantic similarity in context pruning)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")

# Qdrant vector store
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "context_pieces")

# Database
DB_PATH = Path(__file__).parent.parent / "data" / "crm.db"

# Context pruning weights (from Alchemyst's context arithmetic)
WEIGHT_RECENCY = 0.35
WEIGHT_RELEVANCE = 0.40
WEIGHT_INFO_DENSITY = 0.25
TOP_K = 5
