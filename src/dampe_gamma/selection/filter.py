"""Stage 2: apply the photon selection to one weighted cos(theta) bin."""

from __future__ import annotations

import argparse
import logging

from dampe_gamma.config import Paths
from dampe_gamma.io import ensure_dir, load_rdf, save_rdf
from dampe_gamma.selection.cuts import apply_cuts

log = logging.getLogger(__name__)

TREE = "MC"
KEPT_COLUMNS = (
    "tt_ekin",
    "tt_bgoTotalE_GeV",
    "tt_isLET",
    "tt_isHET",
    "tt_theta",
    "tt_phi",
    "theta_BestSTK",
    "phi_BestSTK",
    "tt_bgoTotalEcorr_GeV",
    "E_true",
    "weight",
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("bin_index", type=int, help="index of the cos(theta) bin to process")
    parser.add_argument(
        "-e", "--exponent", type=float, default=0.0, help="spectral index of the input"
    )


def run(args: argparse.Namespace, paths: Paths) -> None:
    input_file = paths.scaled_mc_dir(args.exponent) / f"scaled_costheta_bin_{args.bin_index}.root"
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    output_dir = ensure_dir(paths.filtered_mc_dir(args.exponent))
    output_file = output_dir / f"filtered_costheta_bin_{args.bin_index}.root"

    rdf = load_rdf(str(input_file), TREE)
    n_before = rdf.Count().GetValue()
    rdf = apply_cuts(rdf)
    log.info("Selected %d of %d events", rdf.Count().GetValue(), n_before)
    save_rdf(rdf, output_file, TREE, KEPT_COLUMNS)
