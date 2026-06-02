from pathlib import Path
import yaml

from .paths import get_project_root


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_path: str | Path = "config/main.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = get_project_root() / config_path
    cfg = load_yaml(config_path)
    required = ["project", "sample", "factors", "regressions",
                "table3", "table4", "validation", "outputs"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")
    # Auto-detect the project root from the package location so the project is portable
    # (works after cloning anywhere) and does not depend on a hard-coded path in the config.
    cfg.setdefault("project", {})["root"] = str(get_project_root())
    return cfg


def load_benchmarks(path: str | Path = "config/published_benchmarks.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = get_project_root() / path
    return load_yaml(path)
