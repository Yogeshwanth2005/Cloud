import textwrap
from aco.config import load_config


def test_load_config_reads_required_fields(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(textwrap.dedent("""
        seed: 42
        data_root: "D:/Cloud Computing"
        output_dir: "runs/test"
    """))
    cfg = load_config(str(cfg_path))
    assert cfg["seed"] == 42
    assert cfg["data_root"] == "D:/Cloud Computing"


def test_load_config_raises_on_missing_field(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("seed: 42\n")
    try:
        load_config(str(cfg_path))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "data_root" in str(e)
