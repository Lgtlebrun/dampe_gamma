"""Diagnostic: energy spectrum of the weighted MC after each successive selection cut."""

from __future__ import annotations

import argparse
import logging

from dampe_gamma.binning import E_MAX_GEV, E_MIN_GEV, log_edges
from dampe_gamma.config import Paths
from dampe_gamma.io import browse_files, ensure_dir, import_root
from dampe_gamma.selection.cuts import CUT_STAGES

log = logging.getLogger(__name__)

N_ENERGY_BINS = 50


def _draw_cutflow(ROOT, rdf, edges_root, weighted: bool, output):
    colors = [
        ROOT.kBlack,
        ROOT.kRed,
        ROOT.kBlue,
        ROOT.kGreen + 2,
        ROOT.kMagenta,
        ROOT.kOrange + 7,
        ROOT.kCyan + 2,
        ROOT.kViolet,
        ROOT.kTeal + 3,
        ROOT.kGray + 2,
    ]
    tag = "weighted" if weighted else "unweighted"
    canvas = ROOT.TCanvas(f"c_{tag}", f"Spectra after each cut ({tag})", 1000, 800)
    canvas.SetLogx()
    canvas.SetLogy()
    legend = ROOT.TLegend(0.25, 0.1, 0.75, 0.45)

    stages = [("No cut", None), *CUT_STAGES]
    histograms = []
    for i, (label, cut) in enumerate(stages):
        if cut:
            rdf = rdf.Filter(cut)
        model = (
            f"hist_{tag}_{i}",
            f"Spectra of successive cuts ({tag}); Energy [GeV]; Counts",
            N_ENERGY_BINS,
            edges_root.data(),
        )
        hist = rdf.Histo1D(model, "E_true", "weight") if weighted else rdf.Histo1D(model, "E_true")
        hist.SetLineColor(colors[i % len(colors)])
        hist.SetLineWidth(2)
        legend.AddEntry(hist.GetPtr(), label, "l")
        histograms.append(hist)
        log.info("Stage %d (%s): %d entries", i, label, hist.GetEntries())

    histograms[0].SetMaximum(max(h.GetMaximum() for h in histograms) * 1.2)
    histograms[0].SetMinimum(max(min(h.GetMinimum(1) for h in histograms), 1e-3))
    for i, hist in enumerate(histograms):
        hist.Draw("HIST" if i == 0 else "HIST SAME")
    legend.Draw()
    canvas.SaveAs(str(output))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-e", "--exponent", type=float, default=0.0, help="spectral index of the input"
    )
    parser.add_argument("--bin", default="*", help="restrict to one cos(theta) bin index")


def run(args: argparse.Namespace, paths: Paths) -> None:
    ROOT = import_root()
    files = browse_files(
        paths.scaled_mc_dir(args.exponent) / f"scaled_costheta_bin_{args.bin}.root"
    )
    if not files:
        raise FileNotFoundError(paths.scaled_mc_dir(args.exponent))

    ROOT.EnableImplicitMT()
    ROOT.gStyle.SetOptStat(0)
    rdf = ROOT.RDataFrame("MC", files)
    edges_root = ROOT.std.vector("double")(log_edges(E_MIN_GEV, E_MAX_GEV, N_ENERGY_BINS).tolist())

    figure_dir = ensure_dir(paths.figures_dir)
    _draw_cutflow(ROOT, rdf, edges_root, False, figure_dir / "spectra_all_cuts.pdf")
    _draw_cutflow(ROOT, rdf, edges_root, True, figure_dir / "spectra_all_cuts_weighted.pdf")
