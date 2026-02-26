"""One-time script: embed filtered data into ChromaDB for RAG retrieval."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    QUIZ_QUESTIONS_PATH,
    TUTORIAL_CHUNKS_PATH,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
)
from modules.knowledge_retrieval import KnowledgeBase


def main():
    print(f"Initializing KnowledgeBase with model '{EMBEDDING_MODEL}'...")
    start = time.time()
    kb = KnowledgeBase(str(CHROMA_DB_PATH), EMBEDDING_MODEL)

    # 1. Tutorial chunks
    print(f"\nLoading tutorial chunks from {TUTORIAL_CHUNKS_PATH}...")
    try:
        with open(TUTORIAL_CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"  {len(chunks)} chunks loaded")

        if chunks:
            docs = [c["content"] for c in chunks]
            metas = [{"topic": c["topic"], "section": c.get("section", "")} for c in chunks]
            ids = [c["chunk_id"] for c in chunks]
            print("  Embedding and storing tutorial chunks...")
            kb.add_documents("tutorial_chunks", docs, metas, ids)
            print(f"  Done. Collection count: {kb.tutorial_chunks.count()}")
    except FileNotFoundError:
        print("  File not found, skipping.")

    # 2. Quiz questions
    print(f"\nLoading quiz questions from {QUIZ_QUESTIONS_PATH}...")
    try:
        with open(QUIZ_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            questions = json.load(f)
        print(f"  {len(questions)} questions loaded")

        if questions:
            docs = [q["question"] for q in questions]
            metas = [{"topic": q["topic"], "difficulty": q["difficulty"]} for q in questions]
            ids = [q["quiz_id"] for q in questions]
            print("  Embedding and storing quiz questions...")
            kb.add_documents("quiz_questions", docs, metas, ids)
            print(f"  Done. Collection count: {kb.quiz_collection.count()}")
    except FileNotFoundError:
        print("  File not found, skipping.")

    elapsed = time.time() - start
    print(f"\nVector store built in {elapsed:.1f}s")
    print(f"Stored at: {CHROMA_DB_PATH}")


if __name__ == "__main__":
    main()
