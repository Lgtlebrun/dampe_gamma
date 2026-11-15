"""File discovery, ROOT access and small persistence helpers.

ROOT is imported lazily so that the rest of the package works without it.
"""

from __future__ import annotations

import glob
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def browse_files(patterns: str | Path | Iterable[str | Path]) -> list[str]:
    """Expand one or several glob patterns into a sorted list of files."""
    if isinstance(patterns, (str, Path)):
        patterns = [patterns]
    files: list[str] = []
    for pattern in patterns:
        files += sorted(glob.glob(str(pattern)))
    return files


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def import_root():
    """Import PyROOT, with a helpful message when it is missing."""
    try:
        import ROOT
    except ImportError as err:
        raise ImportError(
            "PyROOT is required for this step. On Baobab, source slurm/env.sh first."
        ) from err
    ROOT.gROOT.SetBatch(True)
    return ROOT


def load_rdf(files: str | Sequence[str], tree_name: str):
    """Open one or several ROOT files as an RDataFrame."""
    ROOT = import_root()
    if isinstance(files, str):
        return ROOT.RDataFrame(tree_name, files)
    return ROOT.RDataFrame(tree_name, list(files))


def save_rdf(rdf, output_file: str | Path, tree_name: str, columns: Sequence[str] | None = None):
    """Write an RDataFrame to disk, keeping only existing columns when a list is given."""
    if columns is None:
        rdf.Snapshot(tree_name, str(output_file))
    else:
        available = {str(c) for c in rdf.GetColumnNames()}
        missing = [c for c in columns if c not in available]
        if missing:
            log.warning("Columns not found and skipped: %s", missing)
        rdf.Snapshot(tree_name, str(output_file), [c for c in columns if c in available])
    log.info("Saved %s", output_file)


def read_root_columns(
    files: str | Sequence[str], tree_name: str, columns: Sequence[str]
) -> pd.DataFrame:
    """Read selected branches of a ROOT tree into a DataFrame."""
    return pd.DataFrame(load_rdf(files, tree_name).AsNumpy(list(columns)))


def summarize(df: pd.DataFrame, name: str = "DataFrame") -> str:
    """Short text summary of an event table."""
    lines = [f"{name}: {len(df)} entries"]
    for col in ("E_true", "cosTheta", "weight"):
        if col in df and len(df):
            s = df[col]
            lines.append(f"  {col}: min {s.min():.4g}, max {s.max():.4g}, mean {s.mean():.4g}")
    if "tt_isHET" in df and len(df):
        lines.append(f"  HET fraction: {100 * (df['tt_isHET'] == 1).mean():.2f}%")
    return "\n".join(lines)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return dict(data)
