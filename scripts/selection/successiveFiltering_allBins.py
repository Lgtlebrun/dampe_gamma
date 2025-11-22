import ROOT
import numpy as np
import os, glob

def browse_files(filenames):
    """
    Browse through the list of filenames and return a sorted list of files.
    """
    if isinstance(filenames, str):
        files = sorted(glob.glob(filenames))
    elif isinstance(filenames, list):
        files = []
        for filename in filenames:
            files += sorted(glob.glob(filename))
    else:
        raise ValueError("Input should be a string or a list of strings.")
    
    return files

def create_log_bins(min_val, max_val, n_bins):
    return np.logspace(np.log10(min_val), np.log10(max_val), n_bins + 1)

PSD_X1=-284.5
PSD_Y1=-310.7
PSD_X2=-298.5
PSD_Y2=-324.7

exponent = 0
dirDir= "/home/users/l/lebrun/"
dataPath = dirDir + "data/" + "scaled" + "_cos_theta_bins/"
dataPath += f"exp{exponent:.1f}/"
files = browse_files(dataPath + "scaled_costheta_bin_*.root")
print("Files found:", files)
figPath = dirDir + "figures/"

# Initialize
# base_input_dir = f"/home/users/l/lebrun/data/scaled_cos_theta_bins/exp{exponent:.1f}"
# file = os.path.join(base_input_dir, f"scaled_costheta_bin_19.root")
ROOT.EnableImplicitMT()
rdf = ROOT.RDataFrame("MC", files)

# Canvas and styles
c = ROOT.TCanvas("c", "Spectra after each cut", 1000, 800)
c.SetLogx()
c.SetLogy()
colors = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen + 2, ROOT.kMagenta,
          ROOT.kOrange + 7, ROOT.kCyan + 2, ROOT.kViolet, ROOT.kTeal + 3, ROOT.kGray + 2]
ROOT.gStyle.SetOptStat(0)

# Binning
n_bins = 50
min_val = 1
max_val = 1e4
log_bins = create_log_bins(min_val, max_val, n_bins)
log_bins_root = ROOT.std.vector('double')(log_bins)

# Define cut stages
cut_stages = [
    ("No cut", ""),
    ("BGO trajectory", f"fabs(tt_bgoRecSlopeX * {PSD_X2} + tt_bgoRecInterceptX) < 420 && fabs(tt_bgoRecSlopeY * {PSD_Y2} + tt_bgoRecInterceptY) < 420"),
    ("Ecore", "tt_Ecore3 <= 1 && tt_Ecore3 > 0.9"),
    ("Tracks", "nnTracks > 0"),
    ("Geometry", "tt_F0 < 0.35 && tt_F1 < 0.45 && CUT_TRACK_NO_HOLES_Best == 1 && theta_BestSTK < 60 && tt_Xtrl < 100 && STKPlaneCut_Best == 1 && fabs(X_ProjectionBest_X1) < 392 && fabs(X_ProjectionBest_X2) < 392 && fabs(Y_ProjectionBest_Y1) < 392 && fabs(Y_ProjectionBest_Y2) < 392"),
    ("Charge", "(ProjSTKPSD_countX_Best<=1 && ProjSTKPSD_countY_Best<=1) || (ProjSTKPSD_count_Best/log(tt_bgoTotalE_GeV)<=0.65)"),
    ("CNN", "CNN_linear > 1.84"),
    ("Sanity", "tt_maxEX1Psd/tt_bgoTotalE_GeV<1.5 && tt_maxEX2Psd/tt_bgoTotalE_GeV<1.5 && tt_maxEY1Psd/tt_bgoTotalE_GeV<1.5 && tt_maxEY2Psd/tt_bgoTotalE_GeV<1.5 && tt_QcloClusterXTrackCorr/tt_bgoTotalE_GeV<0.5 && tt_QcloClusterYTrackCorr/tt_bgoTotalE_GeV<0.5 && (X_ProjectionBest_X1_E+X_ProjectionBest_X2_E)/tt_bgoTotalE_GeV<1. && (Y_ProjectionBest_Y1_E+Y_ProjectionBest_Y2_E)/tt_bgoTotalE_GeV<1. && (X_ProjectionBest_X1_E_1bar+X_ProjectionBest_X2_E_1bar)/tt_bgoTotalE_GeV<1. && (Y_ProjectionBest_Y1_E_1bar+Y_ProjectionBest_Y2_E_1bar)/tt_bgoTotalE_GeV<1. && (X_ProjectionBest_X1_E_3bar+X_ProjectionBest_X2_E_3bar)/tt_bgoTotalE_GeV<1. && (Y_ProjectionBestY1_E_3bar+Y_ProjectionBestY2_E_3bar)/tt_bgoTotalE_GeV<1. && Etrack_STK_Lay0_02mm_Best<exp(2.8*tt_bgoTotalE_GeV) && Etrack_STK_Lay1_02mm_Best<exp(2.8*tt_bgoTotalE_GeV) && Etrack_STK_Lay0_1mm_Best<exp(2.8*tt_bgoTotalE_GeV) && Etrack_STK_Lay1_1mm_Best<exp(2.8*tt_bgoTotalE_GeV) && Etrack_STK_Lay0_5mm_Best<exp(3.8*tt_bgoTotalE_GeV) && Etrack_STK_Lay1_5mm_Best<exp(3.8*tt_bgoTotalE_GeV) && Etrack_STK_Lay0_10mm_Best<exp(4.0*tt_bgoTotalE_GeV) && Etrack_STK_Lay1_10mm_Best<exp(4.0*tt_bgoTotalE_GeV)"),
    ("BDT", "(tt_bgoTotalE_GeV<=100 && BDT_cleaned>0.18-0.038*log(tt_bgoTotalE_GeV)) || ((tt_bgoTotalE_GeV>100 && tt_bgoTotalE_GeV<=1000) && BDT_cleaned>0.05-0.035*log(0.01*tt_bgoTotalE_GeV)) || (tt_bgoTotalE_GeV>1000 && BDT_cleaned>0.05-0.02*log(0.001*tt_bgoTotalE_GeV))")
]

# Loop through stages and plot
legend = ROOT.TLegend(0.25, 0.1, 0.75, 0.45)
legend_w = ROOT.TLegend(0.25, 0.1, 0.75, 0.45)
rdf_step = rdf
histograms = []
histograms_w = []

for i, (label, cut) in enumerate(cut_stages):
    if cut:
        rdf_step = rdf_step.Filter(cut)

    hist = rdf_step.Histo1D(
        ("hist_" + str(i), f"Spectra of successive cuts (unweighted)", n_bins, log_bins_root.data()),
        "E_true"
    )
    hist.SetLineColor(colors[i % len(colors)])
    hist.SetLineWidth(2)
    histograms.append(hist)
    legend.AddEntry(hist.GetPtr(), label, "l")

    hist_w = rdf_step.Histo1D(
        ("hist_w_" + str(i), f"Spectra of successive cuts (weighted)", n_bins, log_bins_root.data()),
        "E_true", "weight"
    )
    hist_w.SetLineColor(colors[i % len(colors)])
    hist_w.SetLineWidth(2)
    histograms_w.append(hist_w)
    legend_w.AddEntry(hist_w.GetPtr(), label, "l")
    print(f"Stage {i}: {label} - Entries: {hist.GetEntries()}")

# Compute the maximum y value among all histograms
max_y = max(hist.GetMaximum() for hist in histograms)
min_y = min(hist.GetMinimum(1) for hist in histograms)  # avoid log(0)

# Optionally pad the y-range a bit (for log scale)
max_y *= 1.2
min_y = min_y if min_y > 0 else 1e-3  # ROOT log axis needs > 0

# Set axis range on the first histogram before drawing
histograms[0].SetMaximum(max_y)
histograms[0].SetMinimum(min_y)

# Draw histograms
for i, hist in enumerate(histograms):
    draw_opt = "HIST" if i == 0 else "HIST SAME"
    hist.Draw(draw_opt)

legend.Draw()

c.SaveAs(figPath + "spectra_all_cuts.pdf")
print("Saved figure to:", figPath + "spectra_all_cuts.pdf")


c_w = ROOT.TCanvas("c_w", "Spectra after each cut", 1000, 800)
c_w.SetLogx()
c_w.SetLogy()
# Compute the maximum y value among all histograms
max_y = max(hist_w.GetMaximum() for hist_w in histograms_w)
min_y = min(hist.GetMinimum(1) for hist in histograms_w)  # avoid log(0)

# Optionally pad the y-range a bit (for log scale)
max_y *= 1.2
min_y = min_y if min_y > 0 else 1e-3  # ROOT log axis needs > 0

# Set axis range on the first histogram before drawing
histograms_w[0].SetMaximum(max_y)
histograms_w[0].SetMinimum(min_y)

# Draw histograms
for i, hist in enumerate(histograms_w):
    draw_opt = "HIST" if i == 0 else "HIST SAME"
    hist.Draw(draw_opt)

legend_w.Draw()

c_w.SaveAs(figPath + "spectra_all_cuts_weighted.pdf")
print("Saved figure to:", figPath + "spectra_all_cuts_weighted.pdf")