"""
embedder.py — Text Embedding Service

WHAT THIS DOES:
Turns text strings into 384-dimensional vectors (lists of numbers)
that capture the MEANING of the text.

THE MODEL: all-MiniLM-L6-v2
- Produces 384-dimensional embeddings
- Trained on over 1 billion sentence pairs
- Fast: ~14,000 sentences/second on CPU
- Size: ~80MB (runs locally, no API calls needed)

HOW EMBEDDINGS WORK (simplified):
1. Text goes in: "The cat sat on the mat"
2. Numbers come out: [0.12, -0.03, 0.87, ..., 0.45] (384 numbers)
3. Similar meanings → similar numbers

WHY SINGLETON PATTERN:
Loading the model takes ~2 seconds and ~500MB RAM.
We load it ONCE when the app starts, then reuse it for every request.
Without singleton: every request waits 2 seconds. With: instant.

IMPORTANT RULE:
The SAME model must embed both documents and queries.
If you embed docs with Model A and queries with Model B, nothing will match —
the numbers live in different "spaces."
"""

from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    """
    Singleton embedding service — loads model once, reuses forever.

    Usage:
        embedder = EmbeddingService()
        vector = embedder.embed_text("What is our PTO policy?")
        vectors = embedder.embed_batch(["chunk 1", "chunk 2", "chunk 3"])
    """

    # Class-level variable — shared across ALL instances
    _model = None

    def __init__(self):
        """
        Load the model if it hasn't been loaded yet.

        First time: takes ~2 seconds (downloads + loads model)
        Every subsequent time: instant (model already in memory)
        """
        if EmbeddingService._model is None:
            print(f"Loading embedding model: {settings.embedding_model}...")
            EmbeddingService._model = SentenceTransformer(
                settings.embedding_model
            )
            print("Embedding model loaded!")

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Parameters:
        -----------
        text : str
            Any text (a chunk, a query, a sentence)

        Returns:
        --------
        list[float]
            384-dimensional vector (list of 384 decimal numbers)

        Example:
        --------
        >>> embedder = EmbeddingService()
        >>> vec = embedder.embed_text("Hello world")
        >>> len(vec)
        384
        >>> vec[:3]
        [0.123, -0.456, 0.789]
        """
        # .encode() runs the model on the text
        # .tolist() converts numpy array to plain Python list
        # (Pinecone expects plain lists, not numpy arrays)
        return EmbeddingService._model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts at once (MUCH faster than one-by-one).

        Parameters:
        -----------
        texts : list[str]
            List of text strings to embed

        Returns:
        --------
        list[list[float]]
            List of 384-dim vectors, one per input text

        WHY BATCH:
        GPU/CPU processes multiple texts in parallel.
        - 100 texts one-by-one: ~7 seconds
        - 100 texts as batch: ~0.5 seconds (14x faster!)

        Example:
        --------
        >>> vecs = embedder.embed_batch(["text 1", "text 2", "text 3"])
        >>> len(vecs)
        3
        >>> len(vecs[0])
        384
        """
        # batch_size=32 means process 32 texts at a time on the GPU/CPU
        # show_progress_bar shows a progress indicator for large batches
        embeddings = EmbeddingService._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 50,  # Only show for large batches
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """
        Returns the embedding dimension (384 for MiniLM).
        Useful when creating the Pinecone index.
        """
        return EmbeddingService._model.get_sentence_embedding_dimension()
