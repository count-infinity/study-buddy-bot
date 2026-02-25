"""Filter the CodeExercise-Python-27k dataset to entries matching the 5 target topics."""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DATASET_PATH, FILTERED_DATASET_PATH, TOPIC_KEYWORDS

MAX_PER_TOPIC = 500


def match_topic(text: str) -> str | None:
    """Return the first matching topic for a given text, or None."""
    text_lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return topic
    return None


def main():
    print(f"Reading {RAW_DATASET_PATH}...")
    by_topic: dict[str, list] = {topic: [] for topic in TOPIC_KEYWORDS}

    with open(RAW_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            rounds = entry.get("chat_rounds", [])
            if len(rounds) < 2:
                continue
            question = rounds[0].get("content", "")
            answer = rounds[1].get("content", "")
            topic = match_topic(question)
            if topic:
                by_topic[topic].append({
                    "id": entry["id"],
                    "topic": topic,
                    "question": question,
                    "answer": answer,
                    "combined": f"Question: {question}\nAnswer: {answer}",
                })

    # Sample and combine
    filtered = []
    for topic, entries in by_topic.items():
        print(f"  {topic}: {len(entries)} raw matches", end="")
        if len(entries) > MAX_PER_TOPIC:
            entries = random.sample(entries, MAX_PER_TOPIC)
            print(f" -> sampled to {MAX_PER_TOPIC}")
        else:
            print()
        filtered.extend(entries)

    random.shuffle(filtered)
    print(f"\nTotal filtered entries: {len(filtered)}")

    with open(FILTERED_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"Saved to {FILTERED_DATASET_PATH}")


if __name__ == "__main__":
    main()
