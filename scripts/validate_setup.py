"""Smoke test: verify all dependencies, models, and data files are ready."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name, fn):
    try:
        fn()
        results.append((name, PASS, ""))
        print(f"  [{PASS}] {name}")
    except Exception as e:
        results.append((name, FAIL, str(e)))
        print(f"  [{FAIL}] {name}: {e}")


def test_imports():
    import nltk, spacy, chromadb, gradio
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline


def test_spacy_model():
    import spacy
    spacy.load("en_core_web_sm")


def test_nltk_data():
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords, wordnet
    # Verify they actually load
    word_tokenize("hello world")
    stopwords.words("english")
    wordnet.synsets("test")


def test_hf_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from config import HF_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
    inputs = tokenizer("What is a Python variable?", return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    assert result, "Empty response from HF model"


def test_data_files():
    from config import FILTERED_DATASET_PATH, QUIZ_QUESTIONS_PATH, TUTORIAL_CHUNKS_PATH
    import json
    for path in [FILTERED_DATASET_PATH, QUIZ_QUESTIONS_PATH, TUTORIAL_CHUNKS_PATH]:
        assert path.exists(), f"{path} not found"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) > 0, f"{path} is empty"


def test_vector_store():
    from config import CHROMA_DB_PATH
    assert CHROMA_DB_PATH.exists(), f"{CHROMA_DB_PATH} not found"
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    for name in ["exercises", "tutorial_chunks", "quiz_questions"]:
        col = client.get_collection(name)
        assert col.count() > 0, f"Collection '{name}' is empty"


def main():
    print("\nStudy Buddy Bot - Setup Validation\n" + "=" * 40)

    print("\n1. Python packages:")
    check("Import all packages", test_imports)

    print("\n2. NLP models:")
    check("spaCy en_core_web_sm", test_spacy_model)
    check("NLTK data files", test_nltk_data)

    print("\n3. HuggingFace LLM:")
    check("HuggingFace flan-t5-base loads and responds", test_hf_model)

    print("\n4. Data files:")
    check("Data JSON files exist and are valid", test_data_files)

    print("\n5. Vector store:")
    check("ChromaDB collections populated", test_vector_store)

    print("\n" + "=" * 40)
    passed = sum(1 for _, s, _ in results if s == PASS)
    total = len(results)
    print(f"Results: {passed}/{total} passed")

    if passed < total:
        print("\nFailed checks:")
        for name, status, err in results:
            if status == FAIL:
                print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("\nAll checks passed! Ready to run: python app.py")


if __name__ == "__main__":
    main()
