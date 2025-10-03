from raglite.embed import DebugEmbeddingStore, embedding_from_bytes, get_embedding_store


def test_debug_embedding_repeatable():
    store = DebugEmbeddingStore(dimension=32)
    vec1 = embedding_from_bytes(store.embed_many(["hello world"])[0])
    vec2 = embedding_from_bytes(store.embed_many(["hello world"])[0])
    assert len(vec1) == 32
    assert list(vec1) == list(vec2)


def test_get_embedding_store_debug():
    store = get_embedding_store("debug")
    assert isinstance(store, DebugEmbeddingStore)
