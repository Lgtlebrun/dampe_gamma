import ROOT
import sys
import os, argparse
import numpy as np

PSD_X1=-284.5
PSD_Y1=-310.7
PSD_X2=-298.5
PSD_Y2=-324.7

def load_rdf(input_file, tree_name):
    """
    Load an RDataFrame from a ROOT file.
    """
    try:
        rdf = ROOT.RDataFrame(tree_name, input_file)
        return rdf
    except Exception as e:
        print(f"Error loading RDataFrame: {e}")
        sys.exit(1)

def filter_rdf(rdf, bin_index=None):
    """
    Filter the RDataFrame for a specific cos(theta) bin.
    """

    # Filter on BGO trajectory
    rdf_filtered = rdf.Filter(f"fabs(tt_bgoRecSlopeX * {PSD_X2} + tt_bgoRecInterceptX) < 420").Filter(
        f"fabs(tt_bgoRecSlopeY * {PSD_Y2} + tt_bgoRecInterceptY) < 420")

    print("Number of entries remaining : ", rdf_filtered.Count().GetValue())

    print("filtered BGO trajectory")

    # Filter on Ecore
    rdf_filtered = rdf_filtered.Filter("tt_Ecore3 <= 1").Filter("tt_Ecore3 > 0.9")

    print("filtered Ecore")
    print("Number of entries remaining : ", rdf_filtered.Count().GetValue())

    # With tracks
    rdf_filtered = rdf_filtered.Filter("nnTracks > 0")
    print("Number of entries remaining : ", rdf_filtered.Count().GetValue())

    print("filtered tracks")

    # Geometry cuts
    rdf_filtered = rdf_filtered.Filter("tt_F0 <0.35").Filter("tt_F1<0.45")
    rdf_filtered = rdf_filtered.Filter("CUT_TRACK_NO_HOLES_Best == 1").Filter("theta_BestSTK < 60").Filter("tt_Xtrl < 100")
    rdf_filtered = rdf_filtered.Filter("STKPlaneCut_Best == 1").Filter("fabs(X_ProjectionBest_X1) < 392").Filter(
        "fabs(X_ProjectionBest_X2) < 392").Filter("fabs(Y_ProjectionBest_Y1) < 392").Filter(
        "fabs(Y_ProjectionBest_Y2) < 392")
    
    print("filtered geometry cuts")
    print("Number of entries remaining : ", rdf_filtered.Count().GetValue())
    
    # Cuts on charge
    rdf_filtered = rdf_filtered.Filter("(ProjSTKPSD_countX_Best<=1 && ProjSTKPSD_countY_Best<=1) || (ProjSTKPSD_count_Best/log(tt_bgoTotalE_GeV)<=0.65)")

    print("filtered charge cuts")

    # Cut on CNN
    rdf_filtered = rdf_filtered.Filter("CNN_linear>1.84")

    print("filtered CNN")

    # Sanity cuts 
    rdf_filtered = rdf_filtered.Filter("tt_maxEX1Psd/tt_bgoTotalE_GeV<1.5").Filter("tt_maxEX2Psd/tt_bgoTotalE_GeV<1.5").Filter("tt_maxEY1Psd/tt_bgoTotalE_GeV<1.5").Filter("tt_maxEY2Psd/tt_bgoTotalE_GeV<1.5")
    rdf_filtered = rdf_filtered.Filter("tt_QcloClusterXTrackCorr/tt_bgoTotalE_GeV<0.5").Filter("tt_QcloClusterYTrackCorr/tt_bgoTotalE_GeV<0.5").Filter("(X_ProjectionBest_X1_E+X_ProjectionBest_X2_E)/tt_bgoTotalE_GeV<1.").Filter(
        "(Y_ProjectionBest_Y1_E+Y_ProjectionBest_Y2_E)/tt_bgoTotalE_GeV<1.").Filter("(X_ProjectionBest_X1_E_1bar+X_ProjectionBest_X2_E_1bar)/tt_bgoTotalE_GeV<1.").Filter("(Y_ProjectionBest_Y1_E_1bar+Y_ProjectionBest_Y2_E_1bar)/tt_bgoTotalE_GeV<1.").Filter(
            "(X_ProjectionBest_X1_E_3bar+X_ProjectionBest_X2_E_3bar)/tt_bgoTotalE_GeV<1.").Filter("(Y_ProjectionBestY1_E_3bar+Y_ProjectionBestY2_E_3bar)/tt_bgoTotalE_GeV<1.").Filter("Etrack_STK_Lay0_02mm_Best<exp(2.8*tt_bgoTotalE_GeV)").Filter("Etrack_STK_Lay1_02mm_Best<exp(2.8*tt_bgoTotalE_GeV)").Filter(
                "Etrack_STK_Lay0_1mm_Best<exp(2.8*tt_bgoTotalE_GeV)").Filter("Etrack_STK_Lay1_1mm_Best<exp(2.8*tt_bgoTotalE_GeV)").Filter("Etrack_STK_Lay0_5mm_Best<exp(3.8*tt_bgoTotalE_GeV)").Filter("Etrack_STK_Lay1_5mm_Best<exp(3.8*tt_bgoTotalE_GeV)").Filter("Etrack_STK_Lay0_10mm_Best<exp(4.0*tt_bgoTotalE_GeV)").Filter(
                    "Etrack_STK_Lay1_10mm_Best<exp(4.0*tt_bgoTotalE_GeV)")
    print("filtered sanity cuts")
    # Cut on BDT 
    rdf_filtered = rdf_filtered.Filter("(tt_bgoTotalE_GeV<=100 && BDT_cleaned>0.18-0.038*log(tt_bgoTotalE_GeV) ) || ((tt_bgoTotalE_GeV>100) && (tt_bgoTotalE_GeV<=1000) && (BDT_cleaned>0.05-0.035*log(0.01*tt_bgoTotalE_GeV)) ) || (tt_bgoTotalE_GeV>1000 and BDT_cleaned>0.05-0.02*log(0.001*tt_bgoTotalE_GeV)) ")
    #rdf = rdf.Filter("BDT_cleaned>0")

    print("filtered BDT")

    return rdf_filtered

def save_rdf(rdf, output_file, tree_name, columns_to_keep=None):
    """
    Save an RDataFrame to a ROOT file.
    """
    try:
        if columns_to_keep is None:
            rdf.Snapshot(tree_name, output_file)
        else:
            # Ensure the columns to keep are valid
            for col in columns_to_keep:
                if col not in rdf.GetColumnNames():
                    print(f"Warning: Column '{col}' not found in RDataFrame. It will be skipped.")
                    columns_to_keep.remove(col)
            rdf.Snapshot(tree_name, output_file, columns_to_keep)
        print(f"Saved RDataFrame to {output_file}")
    except Exception as e:
        print(f"Error saving RDataFrame: {e}")
        sys.exit(1)



def main():
    parser = argparse.ArgumentParser()

    # Usage : python scale_thetaBin.py <bin_index> [-e <exponent>]
    parser.add_argument("bin_index", type=int, help="Index of the cos(theta) bin to process")
    parser.add_argument("-e", "--exponent", type=float, help="Exponent for rescaling", default=0.0)
    parser.add_argument("-f", "--figPath", type=str, help="Path to save figures", default="/home/users/l/lebrun/figures/scaling/script/")
    args = parser.parse_args()
    bin_index = args.bin_index
    exponent = args.exponent
    figPath = args.figPath
    figPath = os.path.join(figPath, f"exp{exponent:.1f}/")
    os.makedirs(figPath, exist_ok=True)
    
    BASE_DIR = "/dpnc/beegfs/dampe/tryf/ForKimberly/"

    # Parameters
    base_input_dir = BASE_DIR + f"tt_scaled_cos_theta_bins/exp{exponent:.1f}"
    base_output_dir = BASE_DIR + f"tt_filtered_cos_theta_bins/exp{exponent:.1f}"
    os.makedirs(base_output_dir, exist_ok=True)

    tree_name = "MC"
    columns_to_keep = ["tt_ekin", "tt_bgoTotalE_GeV", "tt_isLET", "tt_isHET", "tt_theta", "tt_phi",
    "theta_BestSTK", "phi_BestSTK", "tt_bgoTotalEcorr_GeV", "E_true", "weight"]

    # Input and output filenames
    input_file = os.path.join(base_input_dir, f"scaled_costheta_bin_{bin_index}.root")
    output_file = os.path.join(base_output_dir, f"filtered_costheta_bin_{bin_index}.root")

    if not os.path.isfile(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    rdf = load_rdf(input_file, tree_name)
    print(f"Loaded RDataFrame from {input_file}")

    # Print original number of entries
    original_entries = rdf.Count().GetValue()
    print(f"Original number of entries: {original_entries}")

    rdf = filter_rdf(rdf, bin_index=bin_index)
    print(f"Filtered RDataFrame for bin {bin_index}")

    # Print number of entries after filtering
    filtered_entries = rdf.Count().GetValue()
    print(f"Number of entries after filtering: {filtered_entries}")

    save_rdf(rdf, output_file, tree_name, columns_to_keep)
    print(f"Saved rescaled RDataFrame to {output_file}")

if __name__ == "__main__":
    main()