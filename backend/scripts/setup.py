"""
scripts/setup.py — One-time setup script

Run this ONCE after adding your API keys to .env:
    cd backend
    source venv/bin/activate
    python scripts/setup.py

It will:
1. Verify your API keys work
2. Create the Pinecone index (if it doesn't exist)
3. Verify the embedding model loads
4. Print a success message
"""

import sys
sys.path.insert(0, ".")

from app.config import settings


def main():
    print("=" * 60)
    print("SELF-HEALING RAG — SETUP")
    print("=" * 60)

    # 1. Check API keys aren't placeholders
    print("\n1️⃣  Checking API keys...")
    if "placeholder" in settings.pinecone_api_key:
        print("   ❌ PINECONE_API_KEY is still a placeholder. Edit .env!")
        return
    if "placeholder" in settings.groq_api_key:
        print("   ❌ GROQ_API_KEY is still a placeholder. Edit .env!")
        return
    print("   ✅ API keys look real")

    # 2. Test Pinecone connection
    print("\n2️⃣  Testing Pinecone connection...")
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.pinecone_api_key)
        indexes = [idx.name for idx in pc.list_indexes()]
        print(f"   ✅ Connected. Existing indexes: {indexes}")

        if settings.pinecone_index_name not in indexes:
            print(f"   📦 Creating index '{settings.pinecone_index_name}'...")
            from pinecone import ServerlessSpec
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print("   ✅ Index created!")
        else:
            print(f"   ✅ Index '{settings.pinecone_index_name}' already exists")
    except Exception as e:
        print(f"   ❌ Pinecone failed: {e}")
        return

    # 3. Test Groq connection
    print("\n3️⃣  Testing Groq connection...")
    try:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=10,
        )
        print(f"   ✅ Groq works. Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"   ❌ Groq failed: {e}")
        return

    # 4. Test embedding model
    print("\n4️⃣  Loading embedding model...")
    try:
        from app.services.embedder import EmbeddingService
        embedder = EmbeddingService()
        vec = embedder.embed_text("test")
        print(f"   ✅ Model loaded. Dimension: {len(vec)}")
    except Exception as e:
        print(f"   ❌ Embedding model failed: {e}")
        return

    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE — Everything works!")
    print("=" * 60)
    print("\nNext step: python scripts/ingest.py <path-to-pdf>")


if __name__ == "__main__":
    main()
