"""Study Buddy Bot - Main entry point with Gradio UI."""

import subprocess
import sys
from pathlib import Path

import gradio as gr

from config import TOP_K_RETRIEVAL, CHROMA_DB_PATH, SPACY_MODEL


def _ensure_setup():
    """Auto-setup for HuggingFace Spaces: download spaCy model + build vector store."""
    # Download spaCy model if not installed
    try:
        import spacy
        spacy.load(SPACY_MODEL)
    except OSError:
        print(f"Downloading spaCy model '{SPACY_MODEL}'...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", SPACY_MODEL])

    # Build vector store if it doesn't exist
    if not CHROMA_DB_PATH.exists() or not any(CHROMA_DB_PATH.iterdir()):
        print("Building vector store (first run)...")
        subprocess.check_call([sys.executable, str(Path(__file__).parent / "scripts" / "build_vector_store.py")])


_ensure_setup()

from modules.preprocessor import initialize as init_nltk
from modules.intent_classifier import IntentClassifier
from modules.knowledge_retrieval import KnowledgeBase
from modules.response_generator import ResponseGenerator
from modules.student_model import StudentModel
from modules.adaptive_controller import AdaptiveController


class StudyBuddyBot:
    def __init__(self):
        print("Initializing Study Buddy Bot...")
        init_nltk()
        self.classifier = IntentClassifier()
        self.kb = KnowledgeBase()
        self.generator = ResponseGenerator()
        self.student = StudentModel()
        self.controller = AdaptiveController(self.student)
        self._asked_ids: list[str] = []
        print("Study Buddy Bot ready!")

    def chat(self, user_message: str, history: list[dict]) -> tuple[list[dict], str]:
        """Main conversation handler called by Gradio."""
        if not user_message.strip():
            return history, ""

        # Add user message to history
        history = history + [{"role": "user", "content": user_message}]

        # Classify intent
        quiz_pending = self.student.current_quiz is not None
        result = self.classifier.classify(user_message, quiz_pending=quiz_pending)
        intent = result["intent"]
        topic = result["topic_mentioned"]

        # Route to handler
        if intent == "greeting":
            response = self.generator.generate_greeting()
        elif intent == "quiz":
            response = self._handle_quiz(topic)
        elif intent == "answer":
            response = self._handle_answer(user_message)
        elif intent == "hint":
            response = self._handle_hint()
        elif intent == "explain":
            response = self._handle_explain(user_message, topic)
        elif intent == "progress":
            response = self._handle_progress()
        elif intent == "farewell":
            response = self._handle_farewell()
        elif intent == "off_topic":
            response = self.generator.generate_off_topic_response()
        else:
            response = self.generator.generate_off_topic_response()

        history = history + [{"role": "assistant", "content": response}]
        return history, ""

    def _handle_quiz(self, topic: str = None) -> str:
        """Select and deliver a quiz question."""
        if topic is None:
            topic = self.controller.get_recommended_topic()

        difficulty = self.controller.get_recommended_difficulty(topic)
        question = self.kb.get_quiz_question(topic, difficulty, exclude_ids=self._asked_ids)

        if question is None:
            return "I've run out of questions! Great job working through them all. Try asking me to explain a concept instead."

        self.student.set_current_quiz(question)
        self._asked_ids.append(question["quiz_id"])
        return self.generator.format_quiz_question(question)

    def _handle_answer(self, user_answer: str) -> str:
        """Evaluate the student's answer."""
        quiz = self.student.current_quiz
        if quiz is None:
            return 'It doesn\'t look like there\'s a pending question. Say "Quiz me" to get a question!'

        correct_answer = quiz["correct_answer"]
        question_text = quiz["question"]
        topic = quiz["topic"]
        difficulty = quiz["difficulty"]

        # For multiple choice, do simple letter matching first
        if quiz.get("question_type") == "multiple_choice":
            user_letter = user_answer.strip().upper()
            if len(user_letter) == 1 and user_letter in ("A", "B", "C", "D"):
                is_correct = user_letter == correct_answer.upper()
                explanation = quiz.get("explanation", "")
                result = {
                    "is_correct": is_correct,
                    "explanation": explanation,
                    "score": 1.0 if is_correct else 0.0,
                }
            else:
                # Use LLM evaluation for longer answers to MC questions
                context = self._get_context(question_text, topic)
                result = self.generator.evaluate_answer(question_text, correct_answer, user_answer, context)
        else:
            # Short answer / code output: use LLM evaluation
            context = self._get_context(question_text, topic)
            result = self.generator.evaluate_answer(question_text, correct_answer, user_answer, context)

        # Record result
        hints_used = self.student.hint_count
        self.student.record_answer(topic, result["is_correct"], difficulty, hints_used)
        self.student.clear_current_quiz()
        self.controller.adjust_difficulty(topic)

        # Build response
        if result["is_correct"]:
            response = f"**Correct!** {result['explanation']}"
        else:
            response = f"**Not quite.** {result['explanation']}"
            if quiz.get("explanation"):
                response += f"\n\n{quiz['explanation']}"

        # Check if difficulty changed
        new_diff = self.student.topics[topic].current_difficulty
        if new_diff != difficulty:
            response += f"\n\n*Difficulty adjusted to {new_diff}.*"

        # Offer next step
        if self.controller.should_offer_hint():
            response += '\n\nWould you like to try another question? Say "Quiz me".'
        else:
            response += '\n\nReady for another? Say "Quiz me" or ask me to explain something.'

        return response

    def _handle_hint(self) -> str:
        """Provide a hint for the current question."""
        quiz = self.student.current_quiz
        if quiz is None:
            return "There's no active question to give a hint for. Say \"Quiz me\" to get started!"

        hint_level = self.controller.get_hint_level()
        self.student.hint_count += 1

        # Try pre-authored hints first
        hint_key = f"hint_{hint_level}"
        if hint_key in quiz and quiz[hint_key]:
            return f"**Hint (level {hint_level}/3):** {quiz[hint_key]}"

        # Fallback to LLM-generated hint
        context = self._get_context(quiz["question"], quiz["topic"])
        hint = self.generator.generate_hint(
            quiz["question"], quiz["correct_answer"], hint_level, context
        )
        return f"**Hint (level {hint_level}/3):** {hint}"

    def _handle_explain(self, query: str, topic: str = None) -> str:
        """Retrieve context and generate an explanation."""
        # Search both tutorial chunks and exercises for context
        context_chunks = []

        where = {"topic": topic} if topic else None
        tutorial_results = self.kb.query(query, "tutorial_chunks", n_results=TOP_K_RETRIEVAL, where_filter=where)
        context_chunks.extend([r["content"] for r in tutorial_results])

        exercise_results = self.kb.query(query, "exercises", n_results=2, where_filter=where)
        context_chunks.extend([r["content"] for r in exercise_results])

        return self.generator.generate_explanation(topic or "general", query, context_chunks)

    def _handle_progress(self) -> str:
        """Show the student's progress."""
        summary = self.student.get_progress_summary()
        return f"**Your Progress:**\n\n{summary}"

    def _handle_farewell(self) -> str:
        """Say goodbye with session feedback."""
        feedback = self.controller.get_session_feedback()
        summary = self.student.get_progress_summary()
        return self.generator.generate_farewell(f"{summary}\n\n{feedback}")

    def _get_context(self, query: str, topic: str) -> list[str]:
        """Retrieve context chunks for a query."""
        where = {"topic": topic} if topic else None
        results = self.kb.query(query, "tutorial_chunks", n_results=TOP_K_RETRIEVAL, where_filter=where)
        return [r["content"] for r in results]

    # Button handlers for Gradio
    def quiz_button(self, history: list[dict]) -> tuple[list[dict], str]:
        """Handle the Quiz Me button click."""
        return self.chat("Quiz me", history)

    def progress_button(self, history: list[dict]) -> tuple[list[dict], str]:
        """Handle the My Progress button click."""
        return self.chat("How am I doing?", history)

    def hint_button(self, history: list[dict]) -> tuple[list[dict], str]:
        """Handle the Hint button click."""
        return self.chat("Give me a hint", history)


def create_interface() -> gr.Blocks:
    """Build the Gradio Blocks interface."""
    bot = StudyBuddyBot()

    with gr.Blocks(title="Study Buddy Bot") as demo:
        gr.Markdown("# Study Buddy Bot\n*Your Python tutoring assistant*")

        chatbot = gr.Chatbot(
            value=[{"role": "assistant", "content": bot.generator.generate_greeting()}],
            height=500,
        )
        msg = gr.Textbox(
            placeholder="Ask me about Python, or say 'Quiz me'...",
            show_label=False,
            container=False,
        )

        with gr.Row():
            quiz_btn = gr.Button("Quiz Me!", variant="primary")
            hint_btn = gr.Button("Hint")
            progress_btn = gr.Button("My Progress")

        # Wire up events
        msg.submit(bot.chat, [msg, chatbot], [chatbot, msg])
        quiz_btn.click(bot.quiz_button, [chatbot], [chatbot, msg])
        hint_btn.click(bot.hint_button, [chatbot], [chatbot, msg])
        progress_btn.click(bot.progress_button, [chatbot], [chatbot, msg])

    return demo


# Expose demo at module level for HuggingFace Spaces
demo = create_interface()

if __name__ == "__main__":
    demo.launch()
