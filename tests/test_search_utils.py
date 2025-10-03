from raglite.search import RankedChunk, normalize_scores


def test_normalize_scores_handles_uniform_values():
    data = [RankedChunk(chunk_id=1, score=2.0), RankedChunk(chunk_id=2, score=2.0)]
    normalized = normalize_scores(data)
    assert all(value == 1.0 for value in normalized.values())


def test_normalize_scores_scales_range():
    data = [RankedChunk(chunk_id=1, score=0.5), RankedChunk(chunk_id=2, score=1.5)]
    normalized = normalize_scores(data)
    assert normalized[2] == 0.0
    assert normalized[1] == 1.0
