"""HuggingFace Transformers response generation for the Study Buddy Bot.

Uses flan-t5-base for answer evaluation (yes/no classification) and presents
RAG-retrieved content directly for explanations, since small seq2seq models
excel at classification but struggle with long-form generation.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import HF_MODEL_NAME

_model = None
_tokenizer = None


def _get_model():
    """Lazy-load the HuggingFace model and tokenizer."""
    global _model, _tokenizer
    if _model is None:
        print(f"Loading HuggingFace model '{HF_MODEL_NAME}'...")
        _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
        print("Model loaded.")
    return _model, _tokenizer


def _generate(prompt: str, max_tokens: int = 64) -> str:
    """Generate text using the HuggingFace model directly."""
    try:
        model, tokenizer = _get_model()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        return f"An error occurred generating a response: {e}"


class ResponseGenerator:
    def __init__(self):
        _get_model()

    def generate_explanation(self, topic: str, user_question: str, context_chunks: list[str]) -> str:
        """Generate a RAG-grounded explanation.

        Presents retrieved tutorial content directly since the small model
        cannot produce coherent long-form explanations on its own.
        """
        if not context_chunks:
            return (
                f"I don't have specific reference material on that yet, but "
                f"try asking about one of these Python topics: "
                f"variables, data types, control structures, functions, or lists."
            )

        topic_label = topic.replace("_", " ").title() if topic else "Python"
        header = f"**{topic_label}**\n\n"
        body = "\n\n".join(context_chunks[:3])

        # Use the model to generate a brief summary sentence
        summary_prompt = f"Summarize in one sentence: {context_chunks[0][:300]}"
        summary = _generate(summary_prompt, max_tokens=48)

        # Only use the model summary if it's reasonable (>10 chars, not gibberish)
        if summary and len(summary) > 10 and not _looks_degenerate(summary):
            return f"{header}**In short:** {summary}\n\n{body}"
        return f"{header}{body}"

    def generate_hint(self, question: str, correct_answer: str, hint_level: int,
                      context_chunks: list[str], quiz_question: dict | None = None) -> str:
        """Generate a graduated hint without revealing the answer.

        Prefers pre-authored hints from the quiz question JSON when available,
        falling back to rule-based hints.
        """
        # Use pre-authored hints if available
        if quiz_question:
            hint_key = f"hint_{hint_level}"
            if hint_key in quiz_question and quiz_question[hint_key]:
                return f"**Hint {hint_level}:** {quiz_question[hint_key]}"

        # Rule-based fallback
        if hint_level == 1:
            return f"**Hint:** Think carefully about the concept this question is testing."
        elif hint_level == 2:
            # Give a partial reveal based on the answer
            answer_words = correct_answer.split()
            if len(correct_answer) <= 3:
                return f"**Hint:** The answer is {len(correct_answer)} character(s) long."
            elif len(answer_words) == 1:
                return f"**Hint:** The answer starts with **{correct_answer[0]}** and is {len(correct_answer)} characters long."
            else:
                return f"**Hint:** The answer has {len(answer_words)} word(s). The first word is **{answer_words[0]}**."
        else:
            # Level 3: nearly reveal
            if len(correct_answer) <= 4:
                masked = correct_answer[0] + "_" * (len(correct_answer) - 1)
            else:
                masked = correct_answer[:2] + "_" * (len(correct_answer) - 3) + correct_answer[-1]
            return f"**Strong hint:** The answer looks like: `{masked}`"

    def evaluate_answer(self, question: str, correct_answer: str,
                        student_answer: str, context_chunks: list[str]) -> dict:
        """Evaluate the student's answer.

        Uses string matching first, then the LLM for semantic yes/no comparison.
        Returns: {"is_correct": bool, "explanation": str, "score": float}
        """
        answer_lower = student_answer.lower().strip()
        correct_lower = correct_answer.lower().strip()

        if answer_lower == correct_lower:
            return {"is_correct": True, "explanation": "Correct!", "score": 1.0}

        if correct_lower in answer_lower or answer_lower in correct_lower:
            return {"is_correct": True, "explanation": "Correct!", "score": 0.9}

        # Use the model for semantic yes/no classification
        prompt = (
            f"Is the student's answer correct? "
            f"Question: {question} "
            f"Correct answer: {correct_answer} "
            f"Student answer: {student_answer} "
            f"Reply with only 'yes' or 'no'."
        )
        raw = _generate(prompt, max_tokens=16)
        raw_lower = raw.lower()

        is_correct = raw_lower.startswith("yes") or "correct" in raw_lower
        if is_correct:
            return {"is_correct": True, "explanation": "Correct!", "score": 0.8}
        return {
            "is_correct": False,
            "explanation": f"Not quite. The correct answer was: **{correct_answer}**",
            "score": 0.0,
        }

    def generate_greeting(self) -> str:
        """Generate a welcome message."""
        return (
            "Hello! I'm Study Buddy Bot, your Python tutoring assistant.\n\n"
            "I can help you learn Python in these areas:\n"
            "- **Variables** - naming, assignment, scope\n"
            "- **Data Types** - int, float, str, bool, type conversion\n"
            "- **Control Structures** - if/else, for loops, while loops\n"
            "- **Functions** - def, parameters, return values, lambda\n"
            "- **Lists** - indexing, slicing, methods, comprehensions\n\n"
            "Try saying:\n"
            '- "Quiz me on lists"\n'
            '- "Explain how for loops work"\n'
            '- "How am I doing?"'
        )

    def generate_farewell(self, session_feedback: str) -> str:
        """Generate a goodbye message with session summary."""
        return f"Thanks for studying with me! Here's your session summary:\n\n{session_feedback}\n\nGoodbye and keep coding!"

    def generate_off_topic_response(self) -> str:
        """Redirect off-topic queries."""
        return (
            "I'm focused on helping you learn Python! I can help with:\n"
            "- Variables, Data Types, Control Structures, Functions, and Lists\n\n"
            'Try asking "Quiz me" or "Explain [topic]".'
        )

    def format_quiz_question(self, question: dict) -> str:
        """Format a quiz question for display."""
        q_text = question["question"]
        parts = [f"**{question['difficulty'].title()} - {question['topic'].replace('_', ' ').title()}**\n\n{q_text}"]

        if question.get("question_type") == "multiple_choice" and question.get("options"):
            parts.append("")
            for letter, text in question["options"].items():
                parts.append(f"  **{letter}.** {text}")

        return "\n".join(parts)


def _looks_degenerate(text: str) -> bool:
    """Check if model output is repetitive/degenerate."""
    words = text.split()
    if len(words) < 3:
        return False
    # If more than half the words are the same token, it's degenerate
    from collections import Counter
    counts = Counter(words)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count > len(words) * 0.5
