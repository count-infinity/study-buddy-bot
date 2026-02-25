"""Sentence-transformer embedding + ChromaDB vector search for RAG retrieval."""

import json
import random

from sentence_transformers import SentenceTransformer
import chromadb

from config import EMBEDDING_MODEL, CHROMA_DB_PATH, QUIZ_QUESTIONS_PATH


class KnowledgeBase:
    def __init__(self, persist_directory: str = None, embedding_model_name: str = None):
        persist_dir = persist_directory or str(CHROMA_DB_PATH)
        model_name = embedding_model_name or EMBEDDING_MODEL

        self.embedder = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Get or create the three collections
        self.exercises = self.client.get_or_create_collection("exercises")
        self.tutorial_chunks = self.client.get_or_create_collection("tutorial_chunks")
        self.quiz_collection = self.client.get_or_create_collection("quiz_questions")

        # Load quiz questions from JSON for direct access
        self._quiz_questions = []
        try:
            with open(QUIZ_QUESTIONS_PATH, "r", encoding="utf-8") as f:
                self._quiz_questions = json.load(f)
        except FileNotFoundError:
            pass

    def add_documents(self, collection_name: str, documents: list[str],
                      metadatas: list[dict], ids: list[str]):
        """Embed and upsert documents into the named collection."""
        collection = self.client.get_or_create_collection(collection_name)
        embeddings = self.embedder.encode(documents).tolist()
        collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def query(self, query_text: str, collection_name: str,
              n_results: int = 3, where_filter: dict = None) -> list[dict]:
        """Embed the query and search the collection for similar documents.

        Returns list of {"content": str, "metadata": dict, "distance": float}
        """
        collection = self.client.get_or_create_collection(collection_name)
        if collection.count() == 0:
            return []

        query_embedding = self.embedder.encode([query_text]).tolist()
        kwargs = {
            "query_embeddings": query_embedding,
            "n_results": min(n_results, collection.count()),
        }
        if where_filter:
            kwargs["where"] = where_filter

        try:
            results = collection.query(**kwargs)
        except Exception:
            # If where_filter causes issues, retry without it
            kwargs.pop("where", None)
            results = collection.query(**kwargs)

        output = []
        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                })
        return output

    def get_quiz_question(self, topic: str, difficulty: str,
                          exclude_ids: list[str] = None) -> dict | None:
        """Retrieve a quiz question by topic and difficulty from the JSON data.

        Uses direct JSON access for reliable metadata matching.
        """
        exclude = set(exclude_ids or [])
        candidates = [
            q for q in self._quiz_questions
            if q["topic"] == topic
            and q["difficulty"] == difficulty
            and q["quiz_id"] not in exclude
        ]
        if not candidates:
            # Try any difficulty for this topic
            candidates = [
                q for q in self._quiz_questions
                if q["topic"] == topic and q["quiz_id"] not in exclude
            ]
        if not candidates:
            # Try any available question
            candidates = [
                q for q in self._quiz_questions
                if q["quiz_id"] not in exclude
            ]
        if not candidates:
            return None
        return random.choice(candidates)
