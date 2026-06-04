from __future__ import annotations

import re
from typing import Tuple, Optional

from app.security.prompt_injection import DEFAULT_PROMPT_INJECTION_DETECTOR


SECRET_PATTERNS = [
    ("private_key", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |)PRIVATE KEY-----", re.IGNORECASE)),
    ("ssh_rsa", re.compile(r"ssh-rsa\s+[A-Za-z0-9+/=]{100,}", re.IGNORECASE)),
    ("api_key_like", re.compile(r"\b(?:api[_-]?key|secret|token)\b[:=]?\s*[A-Za-z0-9_\-]{16,}", re.IGNORECASE)),
    ("password_like", re.compile(r"\b(?:password|passwd|pwd)\b[:=]?\s*[^\s]{8,}", re.IGNORECASE)),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9_\-\.=]{20,}", re.IGNORECASE)),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("jwt_token", re.compile(r"eyJ[0-9A-Za-z_-]+\.[0-9A-Za-z_-]+\.[0-9A-Za-z_-]+")),
]


def assess_and_filter(text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Assess model output for sensitive content and either allow, sanitize, or block.

    Returns (allowed, reason, output). If blocked, output will be None.
    """
    if not text:
        return True, None, ""

    # First, run the prompt-injection detector on the model output as a guard
    # against the model following injected instructions embedded in documents.
    if DEFAULT_PROMPT_INJECTION_DETECTOR.should_block(text):
        return False, "prompt_injection_in_output", None

    # Check for high-confidence secret patterns that should be blocked.
    for name, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return False, f"contains_{name}", None

    # If nothing suspicious, return the original text (no redaction performed).
    return True, None, text
