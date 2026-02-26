from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
QUIZ_QUESTIONS_PATH = DATA_DIR / "quiz_questions.json"
TUTORIAL_CHUNKS_PATH = DATA_DIR / "python_tutorial_chunks.json"
CHROMA_DB_PATH = KNOWLEDGE_DIR / "chroma_db"

# Models
HF_MODEL_NAME = "google/flan-t5-base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

# RAG settings
TOP_K_RETRIEVAL = 3

# Topics
TOPICS = ["variables", "data_types", "control_structures", "functions", "lists"]
DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced"]

# Adaptive controller thresholds
ACCURACY_PROMOTE_THRESHOLD = 0.8
ACCURACY_DEMOTE_THRESHOLD = 0.4
MIN_ATTEMPTS_FOR_ADJUSTMENT = 3

# Intent labels
INTENTS = ["quiz", "hint", "explain", "answer", "progress", "greeting", "farewell", "off_topic"]
