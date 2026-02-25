"""Per-session student performance tracking for the Study Buddy Bot."""

from config import TOPICS, DIFFICULTY_LEVELS


class TopicStats:
    """Tracks stats for a single topic."""

    def __init__(self):
        self.attempted = 0
        self.correct = 0
        self.current_difficulty = "beginner"
        self.hints_used = 0
        self.history: list[bool] = []

    @property
    def accuracy(self) -> float:
        if self.attempted == 0:
            return 0.0
        return self.correct / self.attempted

    @property
    def last_n(self) -> list[bool]:
        """Return the last 5 results for trend analysis."""
        return self.history[-5:]


class StudentModel:
    def __init__(self):
        self.topics: dict[str, TopicStats] = {t: TopicStats() for t in TOPICS}
        self.session_history: list[dict] = []
        self.current_quiz: dict | None = None
        self.hint_count: int = 0

    def record_answer(self, topic: str, is_correct: bool, difficulty: str, used_hints: int):
        """Record a quiz answer attempt."""
        stats = self.topics[topic]
        stats.attempted += 1
        if is_correct:
            stats.correct += 1
        stats.history.append(is_correct)
        stats.hints_used += used_hints
        self.session_history.append({
            "topic": topic,
            "is_correct": is_correct,
            "difficulty": difficulty,
            "hints_used": used_hints,
        })

    def get_topic_accuracy(self, topic: str) -> float:
        return self.topics[topic].accuracy

    def get_topic_stats(self, topic: str) -> dict:
        stats = self.topics[topic]
        return {
            "attempted": stats.attempted,
            "correct": stats.correct,
            "accuracy": stats.accuracy,
            "current_difficulty": stats.current_difficulty,
            "hints_used": stats.hints_used,
            "last_n_correct": stats.last_n,
        }

    def get_weakest_topic(self) -> str | None:
        """Return the topic with the lowest accuracy (min 1 attempt)."""
        attempted = {t: s for t, s in self.topics.items() if s.attempted > 0}
        if not attempted:
            return None
        return min(attempted, key=lambda t: attempted[t].accuracy)

    def get_progress_summary(self) -> str:
        """Generate a human-readable progress report."""
        lines = []
        for topic in TOPICS:
            stats = self.topics[topic]
            name = topic.replace("_", " ").title()
            if stats.attempted == 0:
                lines.append(f"  {name}: Not attempted yet")
            else:
                pct = stats.accuracy * 100
                level = stats.current_difficulty.title()
                marker = " - Needs review!" if stats.accuracy <= 0.4 and stats.attempted >= 3 else ""
                lines.append(f"  {name}: {stats.correct}/{stats.attempted} correct ({pct:.0f}%) - {level}{marker}")
        total_attempted = sum(s.attempted for s in self.topics.values())
        total_correct = sum(s.correct for s in self.topics.values())
        lines.append(f"\n  Overall: {total_correct}/{total_attempted} correct" if total_attempted else "")
        return "\n".join(lines)

    def set_current_quiz(self, question: dict):
        """Store the pending quiz question."""
        self.current_quiz = question
        self.hint_count = 0

    def clear_current_quiz(self):
        """Clear the pending quiz after evaluation."""
        self.current_quiz = None
        self.hint_count = 0

    def update_difficulty(self, topic: str, new_difficulty: str):
        """Update the current difficulty for a topic."""
        self.topics[topic].current_difficulty = new_difficulty
