from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
RAW_DATASET_PATH = DATA_DIR / "CodeExercise-Python-27k.json"
FILTERED_DATASET_PATH = DATA_DIR / "filtered_exercises.json"
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

# Topic keyword mappings for dataset filtering and topic detection
TOPIC_KEYWORDS = {
    "variables": [
        "variable", "variables", "assignment", "assign a value",
        "global variable", "local variable", "scope", "namespace",
    ],
    "data_types": [
        "data type", "integer", "float", "string", "boolean",
        "type conversion", "casting", "type()", "isinstance",
        "str(", "int(", "complex number", "numeric type",
    ],
    "control_structures": [
        "for loop", "while loop", "if statement", "if-else", "elif",
        "conditional", "iteration", "iterate", "break statement",
        "continue statement", "nested loop", "comprehension",
        "for i in range", "ternary",
    ],
    "functions": [
        "function", "def ", "return value", "parameter", "argument",
        "lambda", "decorator", "recursive", "recursion", "generator",
        "yield", "closure", "callback",
    ],
    "lists": [
        "list", "append", "extend", "insert", "remove from list",
        "pop", "slice", "sort a list", "list comprehension",
        "nested list", "index of", "enumerate",
    ],
}
