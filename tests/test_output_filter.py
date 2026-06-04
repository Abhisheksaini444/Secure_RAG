from app.services.output_filter import assess_and_filter


def test_blocks_private_key():
    txt = "This contains a key: -----BEGIN PRIVATE KEY----- ABCDEF -----END PRIVATE KEY-----"
    allowed, reason, out = assess_and_filter(txt)
    assert not allowed
    assert reason and "private_key" in reason


def test_allows_safe_text():
    txt = "This is a concise factual answer without secrets."
    allowed, reason, out = assess_and_filter(txt)
    assert allowed
    assert out == txt
