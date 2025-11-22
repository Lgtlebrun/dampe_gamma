import ROOT
import numpy as np
import os
import glob

# ------------------------- Configuration -------------------------

# Paths
BASE_DIR = "/dpnc/beegfs/dampe/tryf/ForKimberly/"

## Change this to your home directory
HOME_DIR = "/home/users/l/lebrun/"

DATA_DIR = os.path.join(BASE_DIR, "tt_scaled_cos_theta_bins")
FIGURE_DIR = os.path.join(HOME_DIR, "figures")

# PSD constants
PSD_X2 = -298.5
PSD_Y2 = -324.7

# Parameters
EXPONENT = 0
N_BINS = 50
MIN_VAL = 1
MAX_VAL = 1e4

# Colors
COLORS = [
    ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen + 2, ROOT.kMagenta,
    ROOT.kOrange + 7, ROOT.kCyan + 2, ROOT.kViolet, ROOT.kTeal + 3, ROOT.kGray + 2
]

# ------------------------- Helpers -------------------------

def browse_files(filenames):
    """
    Return sorted list of files from a string pattern or list of patterns.
    """
    if isinstance(filenames, str):
        return sorted(glob.glob(filenames))
    elif isinstance(filenames, list):
        files = []
        for pattern in filenames:
            files += sorted(glob.glob(pattern))
        return files
    else:
        raise ValueError("Input must be a string or a list of strings.")

def create_log_bins(min_val, max_val, n_bins):
    """
    Return logarithmically spaced bin edges.
    """
    return np.logspace(np.log10(min_val), np.log10(max_val), n_bins + 1)

# ------------------------- Load Files -------------------------
# To speed up, you can focus on one specific bin by changin * to a valid index (e.g. 19 for closest to zenith)
data_path = os.path.join(DATA_DIR, f"exp{EXPONENT:.1f}", "scaled_costheta_bin_*.root")
files = browse_files(data_path)
print("Files found:", files)

# ------------------------- ROOT Initialization -------------------------

ROOT.EnableImplicitMT()
rdf = ROOT.RDataFrame("MC", files)
ROOT.gStyle.SetOptStat(0)
log_bins = create_log_bins(MIN_VAL, MAX_VAL, N_BINS)
log_bins_root = ROOT.std.vector('double')(log_bins.tolist())

# ------------------------- Cut Stages -------------------------

cut_stages = [
    ("No cut", ""),
    ("Ereco", "tt_bgoTotalE_GeV >= 1"),
    ("Ecore", "tt_Ecore3 <= 1 && tt_Ecore3 > 0.9"),
    ("BGO trajectory", f"fabs(tt_bgoRecSlopeX * {PSD_X2} + tt_bgoRecInterceptX) < 420 && fabs(tt_bgoRecSlopeY * {PSD_Y2} + tt_bgoRecInterceptY) < 420"),
    ("HET", "tt_isHET == 1"),
    ("Geometry", "tt_F0 < 0.35 && tt_F1 < 0.45 && CUT_TRACK_NO_HOLES_Best == 1 && theta_BestSTK < 60 && tt_Xtrl < 100 && STKPlaneCut_Best == 1 && fabs(X_ProjectionBest_X1) < 392 && fabs(X_ProjectionBest_X2) < 392 && fabs(Y_ProjectionBest_Y1) < 392 && fabs(Y_ProjectionBest_Y2) < 392"),
    ("Tracks", "nnTracks > 0"),
    ("Charge", "(ProjSTKPSD_countX_Best<=1 && ProjSTKPSD_countY_Best<=1) || (ProjSTKPSD_count_Best/log(tt_bgoTotalE_GeV)<=0.65)"),
    ("CNN", "CNN_linear > 1.84"),
    ("Sanity", "tt_maxEX1Psd/(tt_bgoTotalE_GeV)<1.5 && tt_maxEX2Psd/(tt_bgoTotalE_GeV)<1.5 && tt_maxEY1Psd/(tt_bgoTotalE_GeV)<1.5 && tt_maxEY2Psd/(tt_bgoTotalE_GeV)<1.5 && tt_QcloClusterXTrackCorr/(tt_bgoTotalE_GeV)<0.5 && tt_QcloClusterYTrackCorr/(tt_bgoTotalE_GeV)<0.5 && (X_ProjectionBest_X1_E+X_ProjectionBest_X2_E)/(tt_bgoTotalE_GeV)<1. && (Y_ProjectionBest_Y1_E+Y_ProjectionBest_Y2_E)/tt_bgoTotalE_GeV<1. && (X_ProjectionBest_X1_E_1bar+X_ProjectionBest_X2_E_1bar)/tt_bgoTotalE_GeV<1. && (Y_ProjectionBest_Y1_E_1bar+Y_ProjectionBest_Y2_E_1bar)/tt_bgoTotalE_GeV<1. && (X_ProjectionBest_X1_E_3bar+X_ProjectionBest_X2_E_3bar)/tt_bgoTotalE_GeV<1. && (Y_ProjectionBestY1_E_3bar+Y_ProjectionBestY2_E_3bar)/tt_bgoTotalE_GeV<1. && Etrack_STK_Lay0_02mm_Best<exp(2.8*tt_bgoTotalE_GeV) && Etrack_STK_Lay1_02mm_Best<exp(2.8*tt_bgoTotalE_GeV) && Etrack_STK_Lay0_1mm_Best<exp(2.8*tt_bgoTotalE_GeV) && Etrack_STK_Lay1_1mm_Best<exp(2.8*tt_bgoTotalE_GeV) && Etrack_STK_Lay0_5mm_Best<exp(3.8*tt_bgoTotalE_GeV) && Etrack_STK_Lay1_5mm_Best<exp(3.8*tt_bgoTotalE_GeV) && Etrack_STK_Lay0_10mm_Best<exp(4.0*tt_bgoTotalE_GeV) && Etrack_STK_Lay1_10mm_Best<exp(4.0*tt_bgoTotalE_GeV)"),
    ("BDT", "(tt_bgoTotalE_GeV<=100 && BDT_cleaned>0.18-0.038*log(tt_bgoTotalE_GeV)) || ((tt_bgoTotalE_GeV>100 && tt_bgoTotalE_GeV<=1000) && BDT_cleaned>0.05-0.035*log(0.01*tt_bgoTotalE_GeV)) || (tt_bgoTotalE_GeV>1000 && BDT_cleaned>0.05-0.02*log(0.001*tt_bgoTotalE_GeV))")
]

# ------------------------- Plotting -------------------------

# Unweighted
canvas = ROOT.TCanvas("c", "Spectra after each cut", 1000, 800)
canvas.SetLogx()
canvas.SetLogy()

legend = ROOT.TLegend(0.25, 0.1, 0.75, 0.45)
rdf_step = rdf
histograms = []

for i, (label, cut) in enumerate(cut_stages):
    if cut:
        rdf_step = rdf_step.Filter(cut)
    
    hist = rdf_step.Histo1D(
        (f"hist_{i}", "Spectra of successive cuts (unweighted); Energy [GeV]; Counts", N_BINS, log_bins_root.data()),
        "E_true"
    )
    hist.SetLineColor(COLORS[i % len(COLORS)])
    hist.SetLineWidth(2)
    histograms.append(hist)
    legend.AddEntry(hist.GetPtr(), label, "l")
    
    print(f"Stage {i}: {label} - Entries: {hist.GetEntries()}")

max_y = max(h.GetMaximum() for h in histograms) * 1.2
min_y = min(h.GetMinimum(1) for h in histograms)
min_y = max(min_y, 1e-3)

histograms[0].SetMaximum(max_y)
histograms[0].SetMinimum(min_y)

for i, hist in enumerate(histograms):
    hist.Draw("HIST" if i == 0 else "HIST SAME")

legend.Draw()
output_path = os.path.join(FIGURE_DIR, "spectra_all_cuts.pdf")
canvas.SaveAs(output_path)
print("Saved figure to:", output_path)

# Weighted
canvas_w = ROOT.TCanvas("c_w", "Spectra after each cut (weighted); Energy [GeV]; Counts", 1000, 800)
canvas_w.SetLogx()
canvas_w.SetLogy()

legend_w = ROOT.TLegend(0.25, 0.1, 0.75, 0.45)
rdf_step = rdf
histograms_w = []

for i, (label, cut) in enumerate(cut_stages):
    if cut:
        rdf_step = rdf_step.Filter(cut)
    
    hist_w = rdf_step.Histo1D(
        (f"hist_w_{i}", "Spectra of successive cuts (weighted); Energy [GeV]; Counts", N_BINS, log_bins_root.data()),
        "E_true", "weight"
    )
    hist_w.SetLineColor(COLORS[i % len(COLORS)])
    hist_w.SetLineWidth(2)
    histograms_w.append(hist_w)
    legend_w.AddEntry(hist_w.GetPtr(), label, "l")

max_y_w = max(h.GetMaximum() for h in histograms_w) * 1.2
min_y_w = min(h.GetMinimum(1) for h in histograms_w)
min_y_w = max(min_y_w, 1e-3)

histograms_w[0].SetMaximum(max_y_w)
histograms_w[0].SetMinimum(min_y_w)

for i, hist in enumerate(histograms_w):
    hist.Draw("HIST" if i == 0 else "HIST SAME")

legend_w.Draw()
output_path_w = os.path.join(FIGURE_DIR, "spectra_all_cuts_weighted.pdf")
canvas_w.SaveAs(output_path_w)
print("Saved figure to:", output_path_w)
