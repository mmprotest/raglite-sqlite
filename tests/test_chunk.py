from raglite import chunk


def test_split_fixed_tokens_overlap():
    text = " ".join(str(i) for i in range(100))
    parts = chunk.split_fixed_tokens(text, max_tokens=20, overlap=5)
    assert parts
    assert all(len(chunk.TOKEN_PATTERN.findall(p)) <= 20 for p in parts)


def test_split_recursive_handles_headings():
    text = "# Title\ncontent\n# Heading\nmore content"
    parts = chunk.split_recursive(text, max_tokens=10)
    assert len(parts) == 2
