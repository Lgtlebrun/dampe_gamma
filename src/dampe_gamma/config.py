"""Filesystem locations used by the pipeline.

Every location can be overridden with an environment variable. Defaults point to the
Baobab cluster layout used for the original analysis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


@dataclass(frozen=True)
class Paths:
    """Input and output directories of the pipeline."""

    mc_dir: Path
    hk_dir: Path
    work_dir: Path
    results_dir: Path
    figures_dir: Path
    data_dir: Path

    @classmethod
    def from_env(cls) -> Paths:
        return cls(
            mc_dir=_env_path(
                "DAMPE_MC_DIR", "/dpnc/beegfs/dampe/tryf/MC_Skimmed/MC_gamma_wo1GeVCut"
            ),
            hk_dir=_env_path("DAMPE_HK_DIR", "/dpnc/beegfs/dampe/prod/FM/skim/6.0.0/v2"),
            work_dir=_env_path("DAMPE_WORK_DIR", "/dpnc/beegfs/dampe/tryf/ForKimberly"),
            results_dir=_env_path("DAMPE_RESULTS_DIR", "~/results"),
            figures_dir=_env_path("DAMPE_FIGURES_DIR", "~/figures"),
            data_dir=_env_path("DAMPE_DATA_DIR", REPO_ROOT / "data"),
        )

    def scaled_mc_dir(self, exponent: float) -> Path:
        """Per cos(theta) bin MC files carrying spectral weights."""
        return self.work_dir / "tt_scaled_cos_theta_bins" / f"exp{exponent:.1f}"

    def filtered_mc_dir(self, exponent: float) -> Path:
        """Per cos(theta) bin MC files after the photon selection."""
        return self.work_dir / "tt_filtered_cos_theta_bins" / f"exp{exponent:.1f}"

    @property
    def effective_area_file(self) -> Path:
        return self.results_dir / "effArea" / "effectiveArea_weighted.npz"

    @property
    def giem_bins_file(self) -> Path:
        return self.results_dir / "all_bins_GIEM.npz"

    @property
    def flight_data_file(self) -> Path:
        return self.data_dir / "DmlNtup_Data_9years_FullPhotonData_AllSky.root"

    @property
    def available_dates_file(self) -> Path:
        return self.data_dir / "available_dates.txt"


MC_INPUT_PATTERNS = (
    "DmlNtup_gamma_part*_wo1GeVCut_AddVarCorr.root",
    "DmlNtup_gamma_100GeV_1TeV_part*_wo1GeVCut_AddVarCorr.root",
    "DmlNtup_gamma_1TeV_10TeV_part*_wo1GeVCut_AddVarCorr.root",
)
