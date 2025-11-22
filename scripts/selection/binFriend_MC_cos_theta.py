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

def create_bin_friend_tree(input_files, tree_name, theta_branch="theta_True", output_file="bin_friend.root", n_bins=20):
    # Create cos(theta) bin edges
    cos_bins = np.linspace(0, 1, n_bins + 1)
    h_bins = ROOT.TH1D("h_bins", "Bins", n_bins, cos_bins)

    # Prepare TChain
    chain = ROOT.TChain(tree_name)
    for file in input_files:
        chain.Add(file)

    # Set up input branch
    theta_value = np.zeros(1, dtype=np.float64)
    chain.SetBranchAddress(theta_branch, theta_value)

    # Prepare output file and tree
    out_file = ROOT.TFile(output_file, "RECREATE")
    out_tree = ROOT.TTree("binFriend", "Cos(theta) bin indices")

    bin_index = np.zeros(1, dtype=np.int32)
    out_tree.Branch("bin_index", bin_index, "bin_index/I")

    # Loop over events
    n_entries = chain.GetEntries()
    for i in range(n_entries):
        chain.GetEntry(i)
        theta = theta_value[0]
        cos_theta = np.cos(theta * np.pi / 180.0)
        bin_index[0] = h_bins.FindBin(cos_theta) - 1  # ROOT bins start at 1
        out_tree.Fill()

    # Write and close
    out_tree.Write()
    out_file.Close()
    print(f"Friend tree saved to: {output_file}")

if __name__ == "__main__":
    # Adjust paths
    dirDir = "/home/users/l/lebrun/"
    dataDir = "/dpnc/beegfs/dampe/tryf/MC_Skimmed/MC_gamma/"
    resultPath = dirDir + "results/"
    dataPath = dirDir + "data/"
    output_friend = os.path.join(dataPath, "cosTheta_bin_friend.root")

    # Input patterns
    filenames = [
        dataDir + "DmlNtup_gamma_part*_TQ_FullSel.root",
        dataDir + "DmlNtup_gamma_100GeV_1TeV_part*_TQ_FullSel.root",
        dataDir + "DmlNtup_gamma_1TeV_10TeV_part*_MPhotonSelProj_TQ_FullSel.root"
    ]
    files = browse_files(filenames)
    print(f"Detected {len(files)} input files")

    create_bin_friend_tree(
        input_files=files,
        tree_name="DmlNtup",
        theta_branch="theta_True",
        output_file=output_friend,
        n_bins=20
    )