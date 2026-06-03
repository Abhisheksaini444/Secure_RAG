from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.models.schemas import PromptAssessment


@dataclass(frozen=True)
class PromptInjectionRule:
    name: str
    pattern: re.Pattern[str]
    severity: str = "high"


class PromptInjectionDetector:
    """Detects common prompt-injection and jailbreak patterns.

    The detector is intentionally conservative. Security is prioritized over
    convenience, so false positives are acceptable when they reduce the chance of
    leaking hidden instructions, embeddings, or retrieved context.
    """

    def __init__(self, rules: Iterable[PromptInjectionRule] | None = None) -> None:
        self._rules = list(rules) if rules is not None else self._default_rules()

    @staticmethod
    def _default_rules() -> list[PromptInjectionRule]:
        phrases = [
            ("ignore_previous", r"\bignore\s+(?:all\s+)?previous\s+instructions\b"),
            ("ignore_system_prompt", r"\bignore\s+system\s+prompt\b"),
            ("reveal_hidden_prompt", r"\breveal\s+(?:the\s+)?hidden\s+prompt\b"),
            ("reveal_embeddings", r"\breveal\s+embeddings?\b"),
            ("dump_context", r"\bdump\s+(?:the\s+)?context\b"),
            ("show_documents", r"\bshow\s+(?:all\s+)?documents\b"),
            ("developer_mode", r"\bdeveloper\s+mode\b"),
            ("act_as_developer", r"\bact\s+as\s+developer\b"),
            ("jailbreak", r"\bjailbreak\b"),
            ("override_instructions", r"\boverride\s+instructions\b"),
            ("print_vector_db", r"\bprint\s+vector\s+database\b"),
            ("show_vector_db", r"\bshow\s+vector\s+database\b"),
            ("internal_state", r"\bshow\s+internal\s+state\b"),
            ("system_prompt_request", r"\bshow\s+system\s+prompt\b"),
            ("prompt_extraction", r"\bexfiltrat(?:e|ion)\b"),
            ("role_override", r"\b(?:you\s+are\s+now|from\s+now\s+on)\b"),
            ("chain_of_thought_request", r"\bshow\s+(?:your\s+)?(?:chain\s+of\s+thought|reasoning)\b"),
            ("secret_request", r"\breveal\s+(?:api\s+)?keys?\b"),
        ]
        return [PromptInjectionRule(name=name, pattern=re.compile(pattern, re.IGNORECASE)) for name, pattern in phrases]

    def assess(self, text: str) -> PromptAssessment:
        if not text:
            return PromptAssessment(is_suspicious=False)

        matches: list[str] = []
        reasons: list[str] = []
        for rule in self._rules:
            if rule.pattern.search(text):
                matches.append(rule.name)
                reasons.append(f"matched {rule.name}")

        is_suspicious = bool(matches)
        return PromptAssessment(is_suspicious=is_suspicious, reasons=reasons, matched_rules=matches)

    def should_block(self, text: str) -> bool:
        return self.assess(text).is_suspicious


DEFAULT_PROMPT_INJECTION_DETECTOR = PromptInjectionDetector()
