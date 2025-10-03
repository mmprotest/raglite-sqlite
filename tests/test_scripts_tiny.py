import io
from contextlib import redirect_stdout

from scripts import bench_basic, eval_small


def test_eval_small_tiny_runs():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = eval_small.main(["--tiny", "--embed-model", "debug", "--alpha", "0.6"])
    output = buffer.getvalue()
    assert code == 0
    assert "Hybrid" in output


def test_bench_basic_tiny_runs(tmp_path):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = bench_basic.main(["--tiny", "--embed-model", "debug", str(tmp_path / "bench.db")])
    output = buffer.getvalue()
    assert code == 0
    assert "Query latency" in output
