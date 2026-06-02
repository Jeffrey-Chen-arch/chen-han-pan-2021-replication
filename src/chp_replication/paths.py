from pathlib import Path

# All project directories, relative to project root.
ALL_DIRS = [
    "config",
    "data/raw/public",
    "data/raw/professor",
    "data/raw/author_replication/original",
    "data/raw/author_replication/extracted",
    "data/raw/author_replication/documents",
    "data/raw/restricted",
    "data/raw/_seed_originals",
    "data/interim",
    "data/processed",
    "data/logs",
    "metadata",
    "output/tables",
    "output/figures",
    "output/diagnostics",
    "output/memo",
    "output/package_for_professor",
    "output/runs",
    "output/expert_review",
    "scripts",
    "src",
    "tests",
]


def get_project_root() -> Path:
    # src/chp_replication/paths.py -> parents[2] is the project root
    return Path(__file__).resolve().parents[2]


def ensure_project_dirs(root: Path | None = None) -> Path:
    root = Path(root) if root is not None else get_project_root()
    for d in ALL_DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)
    return root
