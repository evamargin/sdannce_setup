"""Load and lightly validate config.yaml."""
from __future__ import annotations

from pathlib import Path
import yaml

from .board import BoardSpec


class Config(dict):
    """A dict with attribute access and a couple of convenience accessors."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    @property
    def board_spec(self) -> BoardSpec:
        b = self["board"]
        return BoardSpec(cols=int(b["cols"]), rows=int(b["rows"]),
                         square_mm=float(b["square_mm"]))

    @property
    def camera_names(self):
        return [c["name"] for c in self["cameras"]]

    @property
    def root(self) -> Path:
        """Directory the config file lives in; all relative paths resolve from here."""
        return Path(self["_config_path"]).resolve().parent

    def path(self, relative: str) -> Path:
        """Resolve a config-relative path to an absolute Path."""
        p = Path(relative)
        return p if p.is_absolute() else (self.root / p)


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = Config(raw)
    cfg["_config_path"] = str(path)

    # --- minimal validation so failures are obvious, not cryptic ----------
    if not cfg.get("cameras"):
        raise ValueError("config: 'cameras' list is empty")
    names = cfg.camera_names
    if len(names) != len(set(names)):
        raise ValueError(f"config: camera names must be unique, got {names}")
    b = cfg["board"]
    if b["cols"] < 2 or b["rows"] < 2:
        raise ValueError("config: board cols/rows are INNER-CORNER counts and must be >= 2")
    if b["cols"] == b["rows"]:
        # Square boards have a 4-fold orientation ambiguity that breaks the
        # shared-world assumption (each camera may pick a different corner as
        # origin). Warn loudly rather than silently producing garbage extrinsics.
        print("WARNING: board has equal cols/rows -> rotational ambiguity. "
              "Use a non-square inner-corner count (e.g. 9x6).")
    return cfg
