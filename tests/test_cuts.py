import numpy as np

from dampe_gamma.selection import cuts
from dampe_gamma.selection.weighting import power_law_bin_weights


def test_stage_order_matches_selection_algorithm():
    labels = [c.label for c in cuts.CUT_STAGES]
    assert labels == [
        "Ereco",
        "Ecore",
        "BGO trajectory",
        "HET",
        "Geometry",
        "Tracks",
        "Charge",
        "CNN",
        "Sanity",
        "BDT",
    ]


def test_expressions_are_balanced_and_use_psd_constants():
    for cut in cuts.CUT_STAGES:
        assert cut.expression.count("(") == cut.expression.count(")"), cut.label
    trajectory = dict(cuts.CUT_STAGES)["BGO trajectory"]
    assert str(cuts.PSD_X2) in trajectory and str(cuts.PSD_Y2) in trajectory


class FakeRDataFrame:
    def __init__(self, filters=()):
        self.filters = list(filters)

    def Filter(self, expression, name=""):  # noqa: N802
        return FakeRDataFrame([*self.filters, expression])


def test_apply_cuts_chains_every_stage():
    rdf = cuts.apply_cuts(FakeRDataFrame())
    assert rdf.filters == [c.expression for c in cuts.CUT_STAGES]


def test_power_law_weights_flatten_spectrum():
    centers = np.logspace(0, 4, 10)
    counts = 1000 * centers**-1.0
    weights = power_law_bin_weights(counts, centers, exponent=0.0)
    weighted = counts * weights
    assert np.allclose(weighted, weighted[0])
    assert np.isclose(weighted.sum(), counts.sum())


def test_power_law_weights_default_to_one_for_empty_bins():
    weights = power_law_bin_weights(np.array([0.0, 10.0]), np.array([1.0, 10.0]), exponent=0.0)
    assert weights[0] == 1.0
