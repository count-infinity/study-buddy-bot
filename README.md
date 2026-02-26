---
title: Study Buddy Bot
emoji: "\U0001F40D"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
---

# Study Buddy Bot

A RAG-based Python tutoring chatbot built for CSC525 (Principles of Machine Learning). The bot quizzes students on five Python topics, adapts difficulty based on performance, and provides explanations grounded in retrieved educational content.

## Features

- **Quiz mode** with 75 questions across 5 topics (variables, data types, control structures, functions, lists)
- **Adaptive difficulty** (beginner/intermediate/advanced) that adjusts based on student accuracy
- **Graduated hints** (3 levels per question)
- **RAG-powered explanations** using ChromaDB vector search over curated tutorial content
- **Progress tracking** with per-topic statistics
- **Intent classification** using spaCy rule-based matching

## Tech Stack

- **LLM**: HuggingFace `google/flan-t5-base` (downloads automatically, ~1 GB)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: ChromaDB
- **Intent Classification**: spaCy (`en_core_web_sm`)
- **Text Preprocessing**: NLTK
- **UI**: Gradio

## Setup (5 minutes)

Requires **Python 3.10+**.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the spaCy language model
python -m spacy download en_core_web_sm

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# 5. Build the vector store (embeds data into ChromaDB, takes ~2 minutes)
python scripts/build_vector_store.py

# 6. (Optional) Verify everything is set up correctly
python scripts/validate_setup.py
```

## Running

```bash
python app.py
```

This launches a Gradio web interface. Open the URL shown in the terminal (typically `http://127.0.0.1:7860`).

## Usage

Once the bot is running, try:

- **"Quiz me"** or **"Quiz me on lists"** to start a quiz
- **"Give me a hint"** when stuck on a question
- **"Explain how for loops work"** for a topic explanation
- **"How am I doing?"** to see your progress
- Use the **Quiz Me**, **Hint**, and **My Progress** buttons for quick access

## Project Structure

```
study-buddy-bot/
├── app.py                  # Main orchestrator + Gradio UI
├── config.py               # All constants and paths
├── requirements.txt
├── data/
│   ├── filtered_exercises.json      # 2,182 Python exercises (filtered from 27k dataset)
│   ├── quiz_questions.json          # 75 curated quiz questions
│   └── python_tutorial_chunks.json  # 68 educational reference chunks
├── modules/
│   ├── preprocessor.py         # NLTK tokenization + lemmatization
│   ├── intent_classifier.py    # spaCy rule-based intent detection
│   ├── knowledge_retrieval.py  # ChromaDB vector search
│   ├── response_generator.py   # HuggingFace model + RAG response assembly
│   ├── student_model.py        # Per-session performance tracking
│   └── adaptive_controller.py  # Difficulty adjustment logic
└── scripts/
    ├── build_vector_store.py   # One-time: embed data into ChromaDB
    ├── filter_dataset.py       # One-time: filter raw 27k dataset (already done)
    └── validate_setup.py       # Smoke test for all components
```
