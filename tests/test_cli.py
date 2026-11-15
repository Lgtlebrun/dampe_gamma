import importlib

import pytest

from dampe_gamma import cli
from dampe_gamma.config import Paths


@pytest.mark.parametrize("command", sorted(cli.COMMANDS))
def test_every_command_module_exposes_the_interface(command):
    module = importlib.import_module(f"dampe_gamma.{cli.COMMANDS[command][0]}")
    assert callable(module.add_arguments)
    assert callable(module.run)


@pytest.mark.parametrize("command", sorted(cli.COMMANDS))
def test_help_works_without_root(command, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([command, "--help"])
    assert exit_info.value.code == 0
    assert "usage" in capsys.readouterr().out


def test_paths_follow_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DAMPE_WORK_DIR", str(tmp_path))
    paths = Paths.from_env()
    assert paths.filtered_mc_dir(0) == tmp_path / "tt_filtered_cos_theta_bins" / "exp0.0"
