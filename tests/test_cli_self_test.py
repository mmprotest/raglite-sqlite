import pytest

typer = pytest.importorskip("typer")
from typer.testing import CliRunner  # noqa: E402

from raglite.cli import app  # noqa: E402


def test_cli_self_test_runs_successfully():
    runner = CliRunner()
    result = runner.invoke(app, ["self-test", "--alpha", "0.6"])
    assert result.exit_code == 0
    assert "raglite self-test" in result.stdout
    assert "Vector backend detected" in result.stdout
