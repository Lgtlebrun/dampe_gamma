# %%
import ROOT, argparse, glob
import pandas as pd
import numpy as np
import multiprocessing as mp
from multiprocessing import cpu_count

PSD_X1=-284.5
PSD_Y1=-310.7
PSD_X2=-298.5
PSD_Y2=-324.7

COUNTER = 0

def create_log_bins(min_val, max_val, n_bins):
    """Create logarithmically spaced bin edges"""
    return np.logspace(np.log10(min_val), np.log10(max_val), n_bins + 1)

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

def rescale(rdf, Emin=1, Emax=1e4, exponent=-2.5, reference_energy=1, n_bins = 50, energy_column:str ="tt_ekin", figPath=None):
    """ Rescale the energy of the events in the RDataFrame
    
    Args:
        rdf (ROOT.RDataFrame): The input RDataFrame.
        Emin (float): Minimum energy in GeV.
        Emax (float): Maximum energy in GeV.
        exponent (float): Exponent for the energy rescaling.
        reference_energy (float): Reference energy for filtering in GeV.
        n_bins (int): Number of bins for the logarithmic binning.
        energy_column (str): Name of the column containing the energy values in MeV.
        
    Returns:
        ROOT.RDataFrame: The RDataFrame with rescaled energy."""
    
    # Rescale the energy
    rdf = rdf.Define("E_true", energy_column + " / 1000.0")


    def create_histogram_based_weights_root(rdf, Emin=1, Emax=1e4, exponent=-2.5, reference_energy=1, n_bins = 50, energy_column:str ="E_true"):
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

        c2.SaveAs(figPath+f"Scaled_Spectrum_exp{exponent}.pdf")

    return rdf_weighted

def load_and_filter(filenames = None, LET=False, HET=False, columns=[], exponent = -2.5, figPath = None):
    """ Loads root files and applies the selection algorithm on gamma rays
    
    Args:
        files (str or list): Path to the root files or a list of root files.
        LET (bool): If True OR if both LET and HET are false, includes the LET events. 
        HET (bool): If True OR if both LET and HET are false, includes the HET events.  
        columns (list): List of columns to keep in the DataFrame.
        exponent (float): Exponent for the energy rescaling.
        figPath (str): Path to save the figures.
        
    Returns:
        pd.DataFrame: A DataFrame containing the filtered data."""

    # Default behaviour if no filenames are provided
    if not filenames :
        # If HET selected, include all files.
        # If only LET selected, include only LET files.
        # If both LET and HET are false, include all files by default.
        if HET :
            filenames = filenames_HET
        else :
            filenames = filenames_LET

    if isinstance(filenames, str):
        files = sorted(glob.glob(files))
    elif isinstance(filenames, list):
        files = []
        for filename in filenames:
            files += sorted(glob.glob(filename))
    print("Files detected : ")
    print(files)
    # Pass the list of files to RDataFrame
    rdf = ROOT.RDataFrame("DmlNtup", files)
    print("created RDataFrame")
    print(rdf.GetColumnNames())

    Emin = 1  # Minimum energy in MeV
    Emax = None

    if (not LET) and (not HET):
        # If both LET and HET are false, include all events
        LET = True
        HET = True
        Emax = 1e4 # Maximum energy in MeV

    if LET and not HET:
        # Apply the selection for LET events : includes some HET events
        Emax = 1e2
    elif HET and not LET:
        # Apply the selection for HET events
        Emax = 1e4

    print("filtered LET/HET")

    rdf = rescale(rdf, Emin=Emin, Emax=Emax, exponent=exponent, reference_energy=1, 
                  n_bins = 50, energy_column ="tt_ekin", figPath=figPath)

    print("Rescaled")


    if LET and not HET:
        # Apply the selection for LET events : includes some HET events
        rdf_temp = rdf.Filter("tt_isLET == 1")
    elif HET and not LET:
        # Apply the selection for HET events
        rdf_temp = rdf.Filter("tt_isHET == 1")
    else:
        # If both LET and HET are selected, include all events
        rdf_temp = rdf

    # Filter on BGO trajectory
    rdf_filtered = rdf_temp.Filter(f"fabs(tt_bgoRecSlopeX * {PSD_X2} + tt_bgoRecInterceptX) < 420").Filter(
        f"fabs(tt_bgoRecSlopeY * {PSD_Y2} + tt_bgoRecInterceptY) < 420")

    print("filtered BGO trajectory")

    # Filter on Ecore
    rdf_filtered = rdf_filtered.Filter("tt_Ecore3 <= 1").Filter("tt_Ecore3 > 0.9")

    print("filtered Ecore")

    # With tracks
    rdf_filtered = rdf_filtered.Filter("nnTracks > 0")

    print("filtered tracks")

    # Geometry cuts
    rdf_filtered = rdf_filtered.Filter("tt_F0 <0.35").Filter("tt_F1<0.45")
    rdf_filtered = rdf_filtered.Filter("CUT_TRACK_NO_HOLES_Best == 1").Filter("theta_BestSTK < 60").Filter("tt_Xtrl < 100")
    rdf_filtered = rdf_filtered.Filter("STKPlaneCut_Best == 1").Filter("fabs(X_ProjectionBest_X1) < 392").Filter(
        "fabs(X_ProjectionBest_X2) < 392").Filter("fabs(Y_ProjectionBest_Y1) < 392").Filter(
        "fabs(Y_ProjectionBest_Y2) < 392")
    
    print("filtered geometry cuts")
    
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

    # Convert to DataFrame
    df = rdf_filtered.AsNumpy(columns)
    print("converted to numpy")
    df = pd.DataFrame(df)
    print("converted to pandas")
    
    return df, rdf, rdf_filtered


def process_data(filenames, kind, columns, columns_energy, columns_angle, dataPath, exponent = -2.5, figPath = None):
    is_LET = kind == "LET"
    df, rdf, rdf_filtered = load_and_filter(filenames, LET=is_LET, HET=not is_LET, columns=columns, exponent=exponent, figPath=figPath)
    
    df.to_csv(f"{dataPath}MC_{kind.lower()}_exp{exponent}_filtered.csv", index=False)
    rdf_filtered.Snapshot(f"MC_{kind.lower()}_filtered", dataPath + f"MC_{kind.lower()}_exp{exponent}_filtered.root")
    rdf.Snapshot(f"MC", dataPath + f"MC_exp{exponent}.root")

    print(f'saved filtered data for {kind} with exponent {exponent} under path {dataPath}MC_{kind.lower()}_exp{exponent}_filtered.root')

    df[columns_energy].to_csv(f"{dataPath}MC_{kind.lower()}_exp{exponent}_filtered_energy.csv", index=False)
    df[columns_angle].to_csv(f"{dataPath}MC_{kind.lower()}_exp{exponent}_filtered_angle.csv", index=False)

    print(f'saved energy and angle data for {kind} with exponent {exponent} under path {dataPath}MC_{kind.lower()}_exp{exponent}_filtered_energy.csv and {dataPath}MC_{kind.lower()}_exp{exponent}_filtered_angle.csv')


if __name__ == "__main__":
    
    print(f"Available CPUs: {cpu_count()}")

    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--exponent", type=float, help="Exponent for rescaling", default=-2.5)
    parser.add_argument("-f", "--figPath", type=str, help="Path to save figures", default="/home/users/l/lebrun/figures/scaling/script/")
    
    args = parser.parse_args()
    print(f"Exponent: {args.exponent}")

    dirDir= "/home/users/l/lebrun/"
    dataDir = "/dpnc/beegfs/dampe/tryf/MC_Skimmed/MC_gamma/"
    resultPath = dirDir + "results/"
    dataPath = dirDir + "data/"
    figPath = args.figPath
    filename1 = dataDir + "DmlNtup_gamma_part*_TQ_FullSel.root"
    filename2 = dataDir + "DmlNtup_gamma_100GeV_1TeV_part*_TQ_FullSel.root"
    filename3 = dataDir + "DmlNtup_gamma_1TeV_10TeV_part*_MPhotonSelProj_TQ_FullSel.root"

    filenames_LET = [filename1, filename2]
    filenames_HET = [filename1, filename2, filename3]

    # List appropriate columns
    # Add the columns to keep in the DataFrame

    columns_energy = set(["weight", "tt_ekin", "tt_bgoTotalEcorr_GeV", "theta_True", "tt_isHET", "tt_isLET", "E_true"])
    columns_angle = set(["weight", "tt_ekin", "theta_True", "phi_True", "theta_BestSTK", "phi_BestSTK", "tt_isHET", "tt_isLET", "E_true"])


    in_angle_but_not_in_energy = columns_angle - columns_energy

    columns = list(columns_energy) + list(in_angle_but_not_in_energy)

    exponent = args.exponent  # Exponent for the energy rescaling

    args_processing = [
        #(filenames_LET, "LET", columns, columns_energy, columns_angle, dataPath, exponent, figPath),
        (filenames_HET, "HET", columns, columns_energy, columns_angle, dataPath, exponent, figPath)
    ]

    declare_fit_function()

    for arg in args_processing:
        process_data(*arg)

