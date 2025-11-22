import ROOT
import sys
import os, argparse
import numpy as np
import glob

COUNTER = 0

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

def save_rdf(rdf, output_file, tree_name, columns):
    """
    Save an RDataFrame to a ROOT file.
    """
    try:
        rdf.Snapshot(tree_name, output_file, columns)
        print(f"Saved RDataFrame to {output_file}")
    except Exception as e:
        print(f"Error saving RDataFrame: {e}")
        sys.exit(1)


def create_log_bins(min_val, max_val, n_bins):
    """Create logarithmically spaced bin edges"""
    return np.logspace(np.log10(min_val), np.log10(max_val), n_bins + 1)

def process_one_costheta_bin(rdf, bin_index, n_bins, theta_branch="tt_theta"):
    """
    Process a single cos(theta) bin and filter on a cos theta bin.
    """

    cos_theta_branch = "tt_cosTheta"
    rdf = rdf.Define(cos_theta_branch, f"cos({theta_branch})")
    print("Generated cos(theta) branch.")

    cos_bins = np.linspace(0, 1, n_bins + 1)

    cos_min = cos_bins[bin_index]
    cos_max = cos_bins[bin_index + 1]

    rdf_bin = rdf.Filter(f"{cos_theta_branch} >= {cos_min} && {cos_theta_branch} < {cos_max}")

    return rdf_bin
    

def declare_fit_function():
    cpp_code = f"""
        #include <cmath>        
        double powerLaw(double *x, double a, double b){{
            return b*pow(x[0], a);
        }}
        double fitFunction(double *x, double *par){{
            return powerLaw(x, par[0], par[1]);
        }}
        """
        
    ROOT.gInterpreter.Declare(cpp_code)

def rescale_rdf(rdf, Emin=1, Emax=1e4, exponent=0, reference_energy=1, n_bins = 50, energy_column:str ="tt_ekin", figPath=None, bin_index=None):
    """ Rescale the energy of the events in the RDataFrame
    
    Args:
        rdf (ROOT.RDataFrame): The input RDataFrame.
        Emin (float): Minimum energy in GeV.
        Emax (float): Maximum energy in GeV.
        exponent (float): Exponent for the energy rescaling.
        reference_energy (float): Reference energy for filtering in GeV.
        n_bins (int): Number of bins for the logarithmic binning.
        energy_column (str): Name of the column containing the energy values in MeV.
        figPath (str): Path to save figures, if provided.
        bin_index (int): Index of the cos(theta) bin being processed, used for naming figures.
        
    Returns:
        ROOT.RDataFrame: The RDataFrame with rescaled energy."""
    
    # Rescale the energy
    rdf = rdf.Define("E_true", energy_column + " / 1000.0")


    def create_histogram_based_weights_root(rdf, Emin=1, Emax=1e4, exponent=0, reference_energy=1, n_bins = 50, energy_column:str ="E_true"):
        """
        Create histogram-based power law weights in ROOT to smooth out step-like behavior
        This mimics the pandas approach that worked well
        """
        global COUNTER

        COUNTER += 1

        # Create logarithmic bins
        log_bins = create_log_bins(Emin, Emax, n_bins)
        log_bins_root = ROOT.std.vector('double')(log_bins)
        
        # Get current histogram
        h_current = rdf.Histo1D(ROOT.RDF.TH1DModel("h_current", "Current Distribution", 
                                                n_bins, log_bins_root.data()), energy_column)
        
        # Force evaluation to get the histogram
        h_current_ptr = h_current.GetPtr()
        
        # Step 2: Calculate target distribution (power law)
        bin_centers = []
        current_counts = []
        target_counts = []
        
        for i in range(1, n_bins + 1):  # ROOT bins start at 1
            bin_center = h_current_ptr.GetBinCenter(i)
            current_count = h_current_ptr.GetBinContent(i)
            
            bin_centers.append(bin_center)
            current_counts.append(current_count)
            
            # Target count following power law
            target_count = (bin_center/reference_energy) ** exponent
            target_counts.append(target_count)
        
        # Normalize target to same total as current
        total_current = sum(current_counts)
        total_target = sum(target_counts)
        
        if total_target > 0:
            normalization = total_current / total_target
            target_counts = [t * normalization for t in target_counts]
        
        # Step 3: Calculate bin weights
        bin_weights = []
        for current, target in zip(current_counts, target_counts):
            if current > 0:
                weight = target / current
            else:
                weight = 1.0  # Default weight for empty bins
            bin_weights.append(weight)
        
        # Step 4: Create C++ function that maps energy to appropriate bin weight
        # First, create arrays for the C++ function
        bin_edges_str = "{" + ", ".join(map(str, log_bins)) + "}"
        bin_weights_str = "{" + ", ".join(map(str, bin_weights)) + "}"
        
        cpp_code = f"""
        #include <algorithm>
        #include <vector>
        
        double GetHistogramBasedWeight{COUNTER}(double energy) {{
            static std::vector<double> bin_edges = {bin_edges_str};
            static std::vector<double> bin_weights = {bin_weights_str};
            
            // Find bin index
            auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), energy);
            int bin_idx = std::distance(bin_edges.begin(), it) - 1;
            
            // Clamp to valid range
            bin_idx = std::max(0, std::min(bin_idx, (int)bin_weights.size() - 1));
            
            return bin_weights[bin_idx];
        }}
        """
        
        ROOT.gInterpreter.Declare(cpp_code)
        
        return log_bins_root

    def create_smooth_power_law_rdf(rdf, Emin=1, Emax=1e4, exponent=-2.5, reference_energy=1, n_bins = 50, energy_column:str ="E_true"):
        """
        Create RDF with histogram-based power law weights that smooth out steps
        """
        
        # Create the histogram-based weighting system
        log_bins_root = create_histogram_based_weights_root(rdf, Emin, Emax, exponent, reference_energy, n_bins, energy_column)
        
        # Add the sophisticated weight column
        rdf_weighted = rdf.Define("weight", f"GetHistogramBasedWeight{COUNTER}({energy_column})")
        
        return rdf_weighted, log_bins_root
    
    rdf_weighted, log_bins_root = create_smooth_power_law_rdf(rdf, Emin, Emax, exponent, reference_energy, n_bins, "E_true")

    if figPath:
        
        h_weighted = rdf_weighted.Histo1D(ROOT.RDF.TH1DModel("h_weighted", "Power Law Weighted;Energy [GeV];Weighted Counts", 
                                                                n_bins, log_bins_root.data()), "E_true", "weight")
        ROOT.gStyle.SetOptStat(0)
        # Plot weighted
        c2 = ROOT.TCanvas()
        ROOT.gPad.SetLogx()
        ROOT.gPad.SetLogy() 
        ROOT.gPad.SetGrid()

        h_weight = h_weighted.GetPtr()
        h_weight.SetLineColor(ROOT.kRed)
        h_weight.SetLineWidth(2)
        h_weight.GetXaxis().SetLabelOffset(0.005)
        h_weight.GetXaxis().SetTitleOffset(1.3)
        # Draw histogram
        h_weight.Draw()

        # perform the fit
        fitFuncE = ROOT.TF1(f"fitFunction", ROOT.fitFunction, xmin=Emin, xmax=Emax, npar=2)
        fitFuncE.SetParameter(0, -2.5) # a
        fitFuncE.SetParameter(1, 4)    # b 
        fitFuncE.SetLineColor(ROOT.kBlack)
        fitFuncE.SetLineWidth(2)

        # Fit with "R+" option to draw the fit function
        res = h_weight.Fit(fitFuncE, "R")

        # Add legend
        alpha = fitFuncE.GetParameter(0)
        beta = np.log10(fitFuncE.GetParameter(1))
        leg = ROOT.TLegend(0.55, 0.7, 0.82, 0.8)
        leg.AddEntry(fitFuncE, f"log_{{10}}(y) \propto {alpha:.2f} log_{{10}}(x)", "l")
        leg.SetTextSize(0.03)
        leg.Draw("same")

        # Final update before saving
        c2.Update()

        c2.SaveAs(figPath+f"Scaled_Spectrum_bin{bin_index}.pdf")

    return rdf_weighted

def main():

    declare_fit_function()

    parser = argparse.ArgumentParser()

    # Usage : python split_and_scale_thetaBin.py <n_bins> <bin_index> [-e <exponent>]
    parser.add_argument("n_bins", type=int, help="Number of cos(theta) bins")
    parser.add_argument("bin_index", type=int, help="Index of the cos(theta) bin to process")
    parser.add_argument("-e", "--exponent", type=float, help="Exponent for rescaling", default=0.0)
    parser.add_argument("-f", "--figPath", type=str, help="Path to save figures", default="/home/users/l/lebrun/figures/scaling/script/")
    args = parser.parse_args()
    n_bins = args.n_bins
    bin_index = args.bin_index
    exponent = args.exponent
    figPath = args.figPath
    figPath = os.path.join(figPath, f"exp{exponent:.1f}/")
    os.makedirs(figPath, exist_ok=True)
    
    # Parameters
    base_input_dir = "/dpnc/beegfs/dampe/tryf/MC_Skimmed/MC_gamma_wo1GeVCut/"
    base_output_dir = f"/dpnc/beegfs/dampe/tryf/ForKimberly/tt_scaled_cos_theta_bins/exp{exponent:.1f}"
    os.makedirs(base_output_dir, exist_ok=True)

    input_tree_name = "DmlNtup"
    output_tree_name = "MC"

    # Input and output filenames
    filename1 = base_input_dir + "DmlNtup_gamma_part*_wo1GeVCut_AddVarCorr.root"
    filename2 = base_input_dir + "DmlNtup_gamma_100GeV_1TeV_part*_wo1GeVCut_AddVarCorr.root"
    filename3 = base_input_dir + "DmlNtup_gamma_1TeV_10TeV_part*_wo1GeVCut_AddVarCorr.root"
    filenames = [filename1, filename2, filename3]

    input_files = browse_files(filenames)
    output_file = os.path.join(base_output_dir, f"scaled_costheta_bin_{bin_index}.root")

    rdf = load_rdf(input_files, input_tree_name)
    print(f"Loaded RDataFrame from {input_files}")

    rdf = process_one_costheta_bin(rdf, bin_index, n_bins, "tt_theta")

    rdf = rescale_rdf(rdf, exponent=exponent, figPath=figPath, bin_index=bin_index)
    print(f"Rescaled RDataFrame for bin {bin_index}")

    all_columns = list(rdf.GetColumnNames())
    columns_to_save = [col for col in all_columns if col != "BDT_new"]

    save_rdf(rdf, output_file, output_tree_name, columns_to_save)

if __name__ == "__main__":
    main()