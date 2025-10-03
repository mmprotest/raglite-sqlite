from pathlib import Path

from raglite.api import RagliteAPI, RagliteConfig


def main() -> None:
    db = Path("example.db")
    api = RagliteAPI(RagliteConfig(db))
    api.init_db()
    demo_dir = Path(__file__).resolve().parents[1] / "demo" / "mini_corpus"
    api.index(demo_dir, strategy="fixed", ocr=False)
    for result in api.query("quick start guide", top_k=3):
        print(f"score={result.score:.3f} text={result.text[:60]}...")


if __name__ == "__main__":
    main()
