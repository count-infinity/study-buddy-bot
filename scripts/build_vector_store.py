"""One-time script: embed filtered data into ChromaDB for RAG retrieval."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FILTERED_DATASET_PATH,
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

    # 1. Load and embed filtered exercises
    print(f"\nLoading filtered exercises from {FILTERED_DATASET_PATH}...")
    with open(FILTERED_DATASET_PATH, "r", encoding="utf-8") as f:
        exercises = json.load(f)
    print(f"  {len(exercises)} exercises loaded")

    if exercises:
        docs = [e["combined"] for e in exercises]
        metas = [{"topic": e["topic"], "id": str(e["id"])} for e in exercises]
        ids = [f"ex_{e['id']}" for e in exercises]
        print("  Embedding and storing exercises...")
        kb.add_documents("exercises", docs, metas, ids)
        print(f"  Done. Collection count: {kb.exercises.count()}")

    # 2. Load and embed tutorial chunks
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

    # 3. Load and embed quiz questions (for hint retrieval)
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
