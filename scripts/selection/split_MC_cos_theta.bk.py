import ROOT, glob
import numpy as np
import os, sys

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

def process_one_costheta_bin(input_file, tree_name, bin_index, n_bins, theta_branch="theta_True", output_dir="cos_theta_bins"):
    """
    Process a single cos(theta) bin and write it to a ROOT file.
    """
    os.makedirs(output_dir, exist_ok=True)

    rdf = ROOT.RDataFrame(tree_name, input_file)
    cos_theta_branch = "cosTheta"
    rdf = rdf.Define(cos_theta_branch, f"cos({theta_branch} * 3.14159265359 / 180)")
    print("Generated cos(theta) branch.")

    cos_bins = np.linspace(0, 1, n_bins + 1)

    cos_min = cos_bins[bin_index]
    cos_max = cos_bins[bin_index + 1]

    rdf_bin = rdf.Filter(f"{cos_theta_branch} >= {cos_min} && {cos_theta_branch} < {cos_max}")

    all_columns = list(rdf.GetColumnNames())
    columns_to_save = [col for col in all_columns if col != "BDT_new"]

    out_file = os.path.join(output_dir, f"costheta_bin_{bin_index}.root")
    print(f"Writing bin {bin_index} with cos(theta) in [{cos_min:.2f}, {cos_max:.2f}) to {out_file}")
    
    rdf_bin.Snapshot("MC", out_file, columns_to_save)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_MC_cos_theta.py <n_bins> <bin_index>")
        sys.exit(1)

    n_bins = int(sys.argv[1])
    bin_index = int(sys.argv[2])

    if bin_index < 0 or bin_index >= n_bins:
        print(f"Error: bin_index must be in range [0, {n_bins - 1}]")
        sys.exit(1)

    dirDir= "/home/users/l/lebrun/"
    dataDir = "/dpnc/beegfs/dampe/tryf/MC_Skimmed/MC_gamma/"
    resultPath = dirDir + "results/"
    dataPath = dirDir + "data/"
    filename1 = dataDir + "DmlNtup_gamma_part*_TQ_FullSel.root"
    filename2 = dataDir + "DmlNtup_gamma_100GeV_1TeV_part*_TQ_FullSel.root"
    filename3 = dataDir + "DmlNtup_gamma_1TeV_10TeV_part*_MPhotonSelProj_TQ_FullSel.root"

    filenames = [filename1, filename2, filename3]

    files = browse_files(filenames)

    tree_name = "DmlNtup"
    output_dir = dataPath + "cos_theta_bins/"
    theta_branch = "theta_True"

    branches_to_keep = [ "tt_bgoTotalE_GeV", "tt_Ecore3", "tt_bgoRecSlopeX", "tt_bgoRecSlopeY", "tt_bgoRecInterceptX",
    "tt_bgoRecInterceptY", "tt_isHET", "tt_isLET", "tt_F0", "tt_F1", "CUT_TRACK_NO_HOLES_Best", "theta_BestSTK",
    "X_ProjectionBest_X1", "X_ProjectionBest_X2", "Y_ProjectionBest_Y1", "Y_ProjectionBest_Y2" , "BgoHit_barID",
    "BgoHit_layerID", "BgoHit_E", "tt_ekin", "DNN_score", "tt_bgoTotalEcorr_GeV", "tt_Xtrl","theta_True","TQ_bestT",
    "tt_maxEX1Psd","tt_maxEX2Psd","tt_maxEY1Psd","tt_maxEY2Psd","tt_QcloClusterXTrackCorr","tt_QcloClusterYTrackCorr",
    "X_ProjectionBest_X1_E","X_ProjectionBest_X2_E","Y_ProjectionBest_Y1_E","Y_ProjectionBest_Y2_E",
    "X_ProjectionBest_X1_E_1bar","X_ProjectionBest_X2_E_1bar","Y_ProjectionBest_Y1_E_1bar","Y_ProjectionBest_Y2_E_1bar",
    "X_ProjectionBest_X1_E_3bar","X_ProjectionBest_X2_E_3bar","Etrack_STK_Lay0_02mm_Best","Etrack_STK_Lay1_02mm_Best",
    "Etrack_STK_Lay0_1mm_Best","Etrack_STK_Lay1_1mm_Best","Etrack_STK_Lay0_5mm_Best","Etrack_STK_Lay1_5mm_Best",
    "Etrack_STK_Lay0_10mm_Best","Etrack_STK_Lay1_10mm_Best","X_ProjectionBest_X1","X_ProjectionBest_X2",
    "Y_ProjectionBest_Y1","Y_ProjectionBest_Y2", "BDT_cleaned", "tt_weights", "CNN_score_Max_Norm_sigmoid_AUC", "CNN_score_Max_Norm_linear_AUC", "TQ_Corr", "MLP"]

    process_one_costheta_bin(files, tree_name, bin_index, n_bins, theta_branch, output_dir)
