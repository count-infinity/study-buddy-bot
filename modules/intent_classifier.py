"""spaCy-based intent classification for the Study Buddy Bot."""

import spacy
from spacy.matcher import PhraseMatcher

from config import SPACY_MODEL, TOPICS

# Phrase patterns for each intent
_INTENT_PHRASES = {
    "quiz": [
        "quiz me", "test me", "give me a question", "ask me a question",
        "quiz", "test", "question please", "ask me about", "practice",
        "give me a quiz", "try a question", "challenge me",
    ],
    "hint": [
        "give me a hint", "hint", "hint please", "help me",
        "i'm stuck", "im stuck", "i need help", "can i get a hint",
        "i don't know", "i dont know", "clue", "give me a clue",
    ],
    "explain": [
        "explain", "what is", "what are", "how does", "how do",
        "tell me about", "teach me", "describe", "can you explain",
        "what does", "help me understand", "i want to learn",
    ],
    "progress": [
        "how am i doing", "my score", "my progress", "show stats",
        "progress", "show my progress", "how am i performing",
        "what's my score", "score", "stats", "performance",
    ],
    "greeting": [
        "hello", "hi", "hey", "good morning", "good afternoon",
        "good evening", "start", "hi there", "howdy", "greetings",
    ],
    "farewell": [
        "bye", "goodbye", "quit", "exit", "see you", "later",
        "done", "i'm done", "im done", "stop", "end session",
        "thanks bye", "thank you bye",
    ],
}

# Topic synonyms for detection
_TOPIC_SYNONYMS = {
    "variables": ["variable", "variables", "var", "assignment", "assign"],
    "data_types": [
        "data type", "data types", "types", "string", "integer", "float",
        "boolean", "int", "str", "bool", "type casting", "type conversion",
    ],
    "control_structures": [
        "if", "else", "elif", "loop", "loops", "for loop", "while loop",
        "for", "while", "conditional", "conditionals", "control structure",
        "control flow", "iteration", "branching",
    ],
    "functions": [
        "function", "functions", "def", "return", "parameter", "parameters",
        "argument", "arguments", "lambda", "decorator", "recursion",
    ],
    "lists": [
        "list", "lists", "append", "index", "indexing", "slicing",
        "slice", "list comprehension",
    ],
}


class IntentClassifier:
    def __init__(self):
        self.nlp = spacy.load(SPACY_MODEL)
        self._matchers: dict[str, PhraseMatcher] = {}
        self._setup_matchers()

    def _setup_matchers(self):
        """Create a PhraseMatcher for each intent."""
        for intent, phrases in _INTENT_PHRASES.items():
            matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            patterns = [self.nlp.make_doc(phrase) for phrase in phrases]
            matcher.add(intent, patterns)
            self._matchers[intent] = matcher

    def classify(self, text: str, quiz_pending: bool = False) -> dict:
        """Classify user intent from text.

        Returns: {"intent": str, "confidence": float, "topic_mentioned": str|None}
        """
        doc = self.nlp(text)
        text_lower = text.lower().strip()

        # Check each intent matcher, track best match
        best_intent = None
        best_score = 0

        for intent, matcher in self._matchers.items():
            matches = matcher(doc)
            if matches:
                # Score by how much of the input the match covers
                for match_id, start, end in matches:
                    span_len = end - start
                    score = span_len / len(doc) if len(doc) > 0 else 0
                    if score > best_score:
                        best_score = score
                        best_intent = intent

        # Extract topic regardless of intent
        topic = self.extract_topic(text_lower)

        # Default behavior when quiz is pending
        if best_intent is None:
            if quiz_pending:
                return {"intent": "answer", "confidence": 0.7, "topic_mentioned": topic}
            return {"intent": "off_topic", "confidence": 0.3, "topic_mentioned": topic}

        # "explain" intent requires a Python topic to be mentioned, otherwise it's off_topic
        # This prevents "What is the weather?" from triggering explain
        if best_intent == "explain" and topic is None:
            return {"intent": "off_topic", "confidence": 0.4, "topic_mentioned": None}

        confidence = min(0.5 + best_score, 1.0)
        return {"intent": best_intent, "confidence": confidence, "topic_mentioned": topic}

    def extract_topic(self, text: str) -> str | None:
        """Check if the user mentioned one of the 5 topics."""
        import re
        # Strip punctuation for matching, keep spaces
        text_clean = re.sub(r"[^\w\s]", "", text.lower())
        for topic, synonyms in _TOPIC_SYNONYMS.items():
            for synonym in synonyms:
                if f" {synonym} " in f" {text_clean} " or text_clean.strip() == synonym:
                    return topic
        return None
