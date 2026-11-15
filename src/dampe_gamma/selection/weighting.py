"""Spectral reweighting of the MC sample.

The simulated spectrum is not uniform in log(E). Each event receives a weight so that
the weighted spectrum follows a power law in log(E) bins; exponent 0 gives a flat
spectrum, which is what the IRF computation needs.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from dampe_gamma.binning import E_MAX_GEV, E_MIN_GEV, log_edges
from dampe_gamma.io import import_root

_weight_function_ids = itertools.count(1)


def power_law_bin_weights(
    counts: np.ndarray, centers: np.ndarray, exponent: float, reference_energy: float = 1.0
) -> np.ndarray:
    """Per bin weights mapping observed counts onto a power law with the same total."""
    counts = np.asarray(counts, dtype=float)
    target = (np.asarray(centers, dtype=float) / reference_energy) ** exponent
    if target.sum() > 0:
        target *= counts.sum() / target.sum()
    return np.where(counts > 0, target / np.where(counts > 0, counts, 1.0), 1.0)


def _declare_weight_function(edges: np.ndarray, weights: np.ndarray) -> str:
    """JIT compile a C++ lookup from energy to bin weight and return its name."""
    ROOT = import_root()
    name = f"GetHistogramBasedWeight{next(_weight_function_ids)}"
    ROOT.gInterpreter.Declare(
        f"""
        #include <algorithm>
        #include <vector>
        double {name}(double energy) {{
            static std::vector<double> edges = {{{", ".join(map(str, edges))}}};
            static std::vector<double> weights = {{{", ".join(map(str, weights))}}};
            auto it = std::upper_bound(edges.begin(), edges.end(), energy);
            int i = std::distance(edges.begin(), it) - 1;
            i = std::max(0, std::min(i, (int)weights.size() - 1));
            return weights[i];
        }}
        """
    )
    return name


def add_spectral_weights(
    rdf,
    exponent: float = 0.0,
    e_min: float = E_MIN_GEV,
    e_max: float = E_MAX_GEV,
    n_bins: int = 50,
    reference_energy: float = 1.0,
    energy_column: str = "tt_ekin",
):
    """Define 'E_true' (GeV) and a 'weight' column flattening the spectrum to E^exponent."""
    ROOT = import_root()
    rdf = rdf.Define("E_true", f"{energy_column} / 1000.0")

    edges = log_edges(e_min, e_max, n_bins)
    edges_root = ROOT.std.vector("double")(edges)
    model = ROOT.RDF.TH1DModel("h_current", "Current Distribution", n_bins, edges_root.data())
    hist = rdf.Histo1D(model, "E_true").GetPtr()

    centers = np.array([hist.GetBinCenter(i) for i in range(1, n_bins + 1)])
    counts = np.array([hist.GetBinContent(i) for i in range(1, n_bins + 1)])
    weights = power_law_bin_weights(counts, centers, exponent, reference_energy)

    name = _declare_weight_function(edges, weights)
    return rdf.Define("weight", f"{name}(E_true)")


def plot_weighted_spectrum(
    rdf, output: str | Path, e_min: float = E_MIN_GEV, e_max: float = E_MAX_GEV, n_bins: int = 50
) -> None:
    """Draw the weighted spectrum with a power law fit, as a check of the weighting."""
    ROOT = import_root()
    ROOT.gInterpreter.Declare(
        "double dampePowerLaw(double *x, double *p) { return p[1] * pow(x[0], p[0]); }"
    )
    edges_root = ROOT.std.vector("double")(log_edges(e_min, e_max, n_bins))
    model = ROOT.RDF.TH1DModel(
        "h_weighted", "Power Law Weighted;Energy [GeV];Weighted Counts", n_bins, edges_root.data()
    )
    hist = rdf.Histo1D(model, "E_true", "weight").GetPtr()

    ROOT.gStyle.SetOptStat(0)
    canvas = ROOT.TCanvas()
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetGrid()
    hist.SetLineColor(ROOT.kRed)
    hist.SetLineWidth(2)
    hist.GetXaxis().SetLabelOffset(0.005)
    hist.GetXaxis().SetTitleOffset(1.3)
    hist.Draw()

    fit = ROOT.TF1("fitFunction", ROOT.dampePowerLaw, e_min, e_max, 2)
    fit.SetParameters(-2.5, 4)
    fit.SetLineColor(ROOT.kBlack)
    fit.SetLineWidth(2)
    hist.Fit(fit, "R")

    legend = ROOT.TLegend(0.55, 0.7, 0.82, 0.8)
    legend.AddEntry(fit, rf"log_{{10}}(y) \propto {fit.GetParameter(0):.2f} log_{{10}}(x)", "l")
    legend.SetTextSize(0.03)
    legend.Draw("same")
    canvas.Update()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(str(output))
