"""Stage 1: extract one cos(theta) bin of the raw MC and attach spectral weights."""

from __future__ import annotations

import argparse
import logging

import numpy as np

from dampe_gamma.config import MC_INPUT_PATTERNS, Paths
from dampe_gamma.io import browse_files, ensure_dir, load_rdf, save_rdf
from dampe_gamma.selection.weighting import add_spectral_weights, plot_weighted_spectrum

log = logging.getLogger(__name__)

INPUT_TREE = "DmlNtup"
OUTPUT_TREE = "MC"
DROPPED_COLUMNS = {"BDT_new"}


def select_cos_theta_bin(rdf, bin_index: int, n_bins: int, theta_branch: str = "tt_theta"):
    """Keep events whose cos(theta) falls in bin 'bin_index' of a uniform [0, 1] binning."""
    cos_edges = np.linspace(0, 1, n_bins + 1)
    lo, hi = cos_edges[bin_index], cos_edges[bin_index + 1]
    rdf = rdf.Define("tt_cosTheta", f"cos({theta_branch})")
    return rdf.Filter(f"tt_cosTheta >= {lo} && tt_cosTheta < {hi}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("n_bins", type=int, help="number of cos(theta) bins in [0, 1]")
    parser.add_argument("bin_index", type=int, help="index of the bin to process")
    parser.add_argument("-e", "--exponent", type=float, default=0.0, help="target spectral index")


def run(args: argparse.Namespace, paths: Paths) -> None:
    input_files = browse_files(paths.mc_dir / pattern for pattern in MC_INPUT_PATTERNS)
    if not input_files:
        raise FileNotFoundError(f"No MC files found in {paths.mc_dir}")
    log.info("Reading %d MC files", len(input_files))

    output_dir = ensure_dir(paths.scaled_mc_dir(args.exponent))
    output_file = output_dir / f"scaled_costheta_bin_{args.bin_index}.root"
    figure_dir = paths.figures_dir / "scaling" / "script" / f"exp{args.exponent:.1f}"

    rdf = load_rdf(input_files, INPUT_TREE)
    rdf = select_cos_theta_bin(rdf, args.bin_index, args.n_bins)
    rdf = add_spectral_weights(rdf, exponent=args.exponent)
    plot_weighted_spectrum(rdf, figure_dir / f"Scaled_Spectrum_bin{args.bin_index}.pdf")

    columns = [str(c) for c in rdf.GetColumnNames() if str(c) not in DROPPED_COLUMNS]
    save_rdf(rdf, output_file, OUTPUT_TREE, columns)
