from pathlib import Path

from raglite.api import RagliteAPI, RagliteConfig


def test_demo_corpus_query(tmp_path: Path):
    db = tmp_path / "demo.db"
    api = RagliteAPI(RagliteConfig(db_path=db, embed_model="debug"))
    api.init_db()
    demo_dir = Path(__file__).resolve().parents[1] / "src" / "demo" / "mini_corpus"
    api.index(demo_dir, strategy="fixed")
    results = api.query("quick start guide", top_k=5)
    assert results
    assert any("quick start" in r.text.lower() for r in results)
