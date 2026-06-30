"""
config.py — Application Settings

HOW THIS WORKS:
- Pydantic's BaseSettings reads values from a .env file
- If a value is missing from .env, the app crashes at startup with a clear error
- This is GOOD — you find out immediately, not when a user hits the endpoint

USAGE IN OTHER FILES:
    from app.config import settings
    print(settings.groq_api_key)  # reads from .env
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Every field here maps to an environment variable.
    Field name (lowercase) → ENV VAR (UPPERCASE)

    Example: groq_api_key → GROQ_API_KEY in .env
    """

    # --- Required (no default = must be in .env) ---
    pinecone_api_key: str
    pinecone_index_name: str
    groq_api_key: str
    nextauth_secret: str

    # --- Optional (have defaults) ---
    frontend_url: str = "http://localhost:3000"
    embedding_model: str = "./models/all-MiniLM-L6-v2"  # Local path (avoids corporate firewall)
    llm_model: str = "llama-3.3-70b-versatile"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 5
    recency_weight: float = 0.2
    recency_half_life_days: float = 30.0

    # Tell Pydantic WHERE to find the .env file
    model_config = SettingsConfigDict(
        env_file=".env",         # Look for .env in the working directory
        env_file_encoding="utf-8",
        extra="ignore",          # Don't crash if .env has extra vars we don't use
    )


# Create ONE instance — import this everywhere
# This runs validation immediately when the app starts
settings = Settings()
