import yaml

REQUIRED_FIELDS = ["seed", "data_root", "output_dir"]


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    missing = [f for f in REQUIRED_FIELDS if f not in cfg]
    if missing:
        raise ValueError(f"config {path} missing required fields: {missing}")
    return cfg
