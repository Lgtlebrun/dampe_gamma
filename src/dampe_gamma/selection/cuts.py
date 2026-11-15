"""Photon selection cuts, following Jennifer Frieden's HET selection algorithm.

Cuts are ROOT expressions so that they can be applied directly with RDataFrame.Filter.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

log = logging.getLogger(__name__)

PSD_X1 = -284.5
PSD_Y1 = -310.7
PSD_X2 = -298.5
PSD_Y2 = -324.7

E = "tt_bgoTotalE_GeV"


class Cut(NamedTuple):
    label: str
    expression: str


def _all(*conditions: str) -> str:
    return " && ".join(conditions)


def _sanity() -> str:
    psd = [f"tt_maxE{p}Psd/({E})<1.5" for p in ("X1", "X2", "Y1", "Y2")]
    clusters = [f"tt_QcloCluster{a}TrackCorr/({E})<0.5" for a in ("X", "Y")]
    projections = [
        f"(X_ProjectionBest_X1_E+X_ProjectionBest_X2_E)/({E})<1.",
        f"(Y_ProjectionBest_Y1_E+Y_ProjectionBest_Y2_E)/{E}<1.",
        f"(X_ProjectionBest_X1_E_1bar+X_ProjectionBest_X2_E_1bar)/{E}<1.",
        f"(Y_ProjectionBest_Y1_E_1bar+Y_ProjectionBest_Y2_E_1bar)/{E}<1.",
        f"(X_ProjectionBest_X1_E_3bar+X_ProjectionBest_X2_E_3bar)/{E}<1.",
        f"(Y_ProjectionBestY1_E_3bar+Y_ProjectionBestY2_E_3bar)/{E}<1.",
    ]
    track_slope = {"02mm": 2.8, "1mm": 2.8, "5mm": 3.8, "10mm": 4.0}
    tracks = [
        f"Etrack_STK_Lay{layer}_{width}_Best<exp({slope}*{E})"
        for width, slope in track_slope.items()
        for layer in (0, 1)
    ]
    return _all(*psd, *clusters, *projections, *tracks)


CUT_STAGES: tuple[Cut, ...] = (
    Cut("Ereco", f"{E} >= 1"),
    Cut("Ecore", "tt_Ecore3 <= 1 && tt_Ecore3 > 0.9"),
    Cut(
        "BGO trajectory",
        _all(
            f"fabs(tt_bgoRecSlopeX * {PSD_X2} + tt_bgoRecInterceptX) < 420",
            f"fabs(tt_bgoRecSlopeY * {PSD_Y2} + tt_bgoRecInterceptY) < 420",
        ),
    ),
    Cut("HET", "tt_isHET == 1"),
    Cut(
        "Geometry",
        _all(
            "tt_F0 < 0.35",
            "tt_F1 < 0.45",
            "CUT_TRACK_NO_HOLES_Best == 1",
            "theta_BestSTK < 60",
            "tt_Xtrl < 100",
            "STKPlaneCut_Best == 1",
            "fabs(X_ProjectionBest_X1) < 392",
            "fabs(X_ProjectionBest_X2) < 392",
            "fabs(Y_ProjectionBest_Y1) < 392",
            "fabs(Y_ProjectionBest_Y2) < 392",
        ),
    ),
    Cut("Tracks", "nnTracks > 0"),
    Cut(
        "Charge",
        "(ProjSTKPSD_countX_Best<=1 && ProjSTKPSD_countY_Best<=1)"
        f" || (ProjSTKPSD_count_Best/log({E})<=0.65)",
    ),
    Cut("CNN", "CNN_linear > 1.84"),
    Cut("Sanity", _sanity()),
    Cut(
        "BDT",
        f"({E}<=100 && BDT_cleaned>0.18-0.038*log({E}))"
        f" || (({E}>100 && {E}<=1000) && BDT_cleaned>0.05-0.035*log(0.01*{E}))"
        f" || ({E}>1000 && BDT_cleaned>0.05-0.02*log(0.001*{E}))",
    ),
)
"""Selection stages in the order they are applied."""


def apply_cuts(rdf, stages: tuple[Cut, ...] = CUT_STAGES):
    """Chain every cut stage onto an RDataFrame."""
    for i, cut in enumerate(stages, start=1):
        log.info("Cut stage %d: %s", i, cut.label)
        rdf = rdf.Filter(cut.expression, cut.label)
    return rdf
