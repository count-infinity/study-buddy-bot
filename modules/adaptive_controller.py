"""Adaptive difficulty and topic recommendation for the Study Buddy Bot."""

from config import (
    ACCURACY_PROMOTE_THRESHOLD,
    ACCURACY_DEMOTE_THRESHOLD,
    MIN_ATTEMPTS_FOR_ADJUSTMENT,
    DIFFICULTY_LEVELS,
    TOPICS,
)
from modules.student_model import StudentModel


def _promote(level: str) -> str:
    idx = DIFFICULTY_LEVELS.index(level)
    return DIFFICULTY_LEVELS[min(idx + 1, len(DIFFICULTY_LEVELS) - 1)]


def _demote(level: str) -> str:
    idx = DIFFICULTY_LEVELS.index(level)
    return DIFFICULTY_LEVELS[max(idx - 1, 0)]


class AdaptiveController:
    def __init__(self, student: StudentModel):
        self.student = student

    def get_recommended_difficulty(self, topic: str) -> str:
        """Determine the difficulty for the next question on this topic."""
        stats = self.student.get_topic_stats(topic)
        current = stats["current_difficulty"]

        if stats["attempted"] < MIN_ATTEMPTS_FOR_ADJUSTMENT:
            return current

        # Emergency demote: 3 wrong in a row
        recent = stats["last_n_correct"]
        if len(recent) >= 3 and not any(recent[-3:]):
            return _demote(current)

        if stats["accuracy"] >= ACCURACY_PROMOTE_THRESHOLD:
            return _promote(current)
        elif stats["accuracy"] <= ACCURACY_DEMOTE_THRESHOLD:
            return _demote(current)

        return current

    def get_recommended_topic(self) -> str:
        """Choose which topic to quiz on next."""
        # Phase 1: Coverage — pick any untouched topic
        for topic in TOPICS:
            if self.student.get_topic_stats(topic)["attempted"] == 0:
                return topic

        # Phase 2: Remediation — pick weakest topic
        weakest = self.student.get_weakest_topic()
        if weakest and self.student.get_topic_accuracy(weakest) <= ACCURACY_DEMOTE_THRESHOLD:
            return weakest

        # Phase 3: Balance — pick least-attempted topic
        return min(TOPICS, key=lambda t: self.student.get_topic_stats(t)["attempted"])

    def should_offer_hint(self) -> bool:
        """Return True if the student should be proactively offered a hint."""
        if self.student.current_quiz is None:
            return False
        if self.student.hint_count > 0:
            return False
        # Offer hint if last answer was wrong
        if self.student.session_history:
            return not self.student.session_history[-1]["is_correct"]
        return False

    def get_hint_level(self) -> int:
        """Return the current hint escalation level (1-3)."""
        return min(self.student.hint_count + 1, 3)

    def adjust_difficulty(self, topic: str):
        """Compute and apply new difficulty after an answer."""
        new_difficulty = self.get_recommended_difficulty(topic)
        self.student.update_difficulty(topic, new_difficulty)

    def get_session_feedback(self) -> str:
        """Generate end-of-session feedback."""
        lines = []
        strong = []
        weak = []
        for topic in TOPICS:
            stats = self.student.get_topic_stats(topic)
            if stats["attempted"] == 0:
                continue
            name = topic.replace("_", " ").title()
            if stats["accuracy"] >= ACCURACY_PROMOTE_THRESHOLD:
                strong.append(name)
            elif stats["accuracy"] <= ACCURACY_DEMOTE_THRESHOLD:
                weak.append(name)

        if strong:
            lines.append(f"Great work on: {', '.join(strong)}!")
        if weak:
            lines.append(f"You might want to review: {', '.join(weak)}.")
        if not strong and not weak:
            lines.append("Keep practicing to build your Python skills!")

        return " ".join(lines)
