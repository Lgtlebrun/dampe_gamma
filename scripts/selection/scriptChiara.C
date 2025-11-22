#include <ROOT/RDataFrame.hxx>
#include <TCanvas.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TH1D.h>
#include <TFile.h>
#include <TString.h>
#include <TGaxis.h>

#include <vector>
#include <cmath>
#include <iostream>

// === CONFIGURAZIONE ===
//const std::string file_path = "/dpnc/beegfs/dampe/tryf/MC_Skimmed/MC_gamma_wo1GeVCut/merged_24June.root";
const std::string file_path = "/dpnc/beegfs/dampe/tryf/MC_Skimmed/MC_gamma_wo1GeVCut/DmlNtup_gamma_AddVarCorr_merged_29giugno.root";
//const std::string file_path = "/home/users/p/perrinac/scratch/merged_10jun25.root";
const std::string tree_name = "DmlNtup";
//const std::string tree_name = "MC";
const int N_BINS = 50;
const double MIN_VAL = 1;
const double MAX_VAL = 1e4;
const double PSD_X2 = -298.5;
const double PSD_Y2 = -324.7;


// === LOG BINNING ===
std::vector<double> create_log_bins(double min_val, double max_val, int n_bins) {
    std::vector<double> bins(n_bins + 1);
    double log_min = std::log10(min_val);
    double log_max = std::log10(max_val);
    double step = (log_max - log_min) / n_bins;
    for (int i = 0; i <= n_bins; ++i)
        bins[i] = std::pow(10, log_min + i * step);
    return bins;
}

// === PRINCIPALE ===
void runCutsSingleFile() {
    ROOT::EnableImplicitMT();
    ROOT::RDataFrame rdf(tree_name, file_path);
    gStyle->SetOptStat(0);

    constexpr double MIN_VAL = 1000;    // energia minima in GeV
    constexpr double MAX_VAL = 10000000;  // energia massima in GeV
    constexpr int N_BINS = 50;         // numero di bin

    // Creo i bin logaritmici
    std::vector<double> log_bins = create_log_bins(MIN_VAL, MAX_VAL, N_BINS);

    // Definizione dei cut successivi
    std::vector<std::pair<std::string, std::string>> cut_stages = {
        {"No cut", ""},
	{"oneGeV", "tt_bgoTotalE_GeV >= 1"},
        {"HET", "tt_isHET == 1"},
	{"Geometry", "tt_F0 < 0.35 && tt_F1 < 0.45 && CUT_TRACK_NO_HOLES_Best == 1 && theta_BestSTK < 60 && tt_Xtrl < 100 && STKPlaneCut_Best == 1 && fabs(X_ProjectionBest_X1) < 392 && fabs(X_ProjectionBest_X2) < 392 && fabs(Y_ProjectionBest_Y1) < 392 && fabs(Y_ProjectionBest_Y2) < 392"},
        {"BGO trajectory", Form("fabs(tt_bgoRecSlopeX * %.1f + tt_bgoRecInterceptX) < 420 && fabs(tt_bgoRecSlopeY * %.1f + tt_bgoRecInterceptY) < 420", PSD_X2, PSD_Y2)},
	{"Tracks", "nnTracks > 0"},
        {"Ecore", "tt_Ecore3 <= 1 && tt_Ecore3 > 0.9"},
        {"Charge", "(ProjSTKPSD_countX_Best<=1 && ProjSTKPSD_countY_Best<=1) || (ProjSTKPSD_count_Best/log(tt_bgoTotalE_GeV)<=0.65)"},
        {"CNN", "CNN_linear > 1.84"},
        {"Sanity", "tt_maxEX1Psd/tt_bgoTotalE_GeV<1.5 && tt_maxEX2Psd/tt_bgoTotalE_GeV<1.5 && tt_maxEY1Psd/tt_bgoTotalE_GeV<1.5 && tt_maxEY2Psd/tt_bgoTotalE_GeV<1.5"},
        {"BDT", "(tt_bgoTotalE_GeV<=100 && BDT_cleaned>0.18-0.038*log(tt_bgoTotalE_GeV)) || ((tt_bgoTotalE_GeV>100 && tt_bgoTotalE_GeV<=1000) && BDT_cleaned>0.05-0.035*log(0.01*tt_bgoTotalE_GeV)) || (tt_bgoTotalE_GeV>1000 && BDT_cleaned>0.05-0.02*log(0.001*tt_bgoTotalE_GeV))"}
    };

    std::vector<ROOT::RDF::RNode> rdf_chain;
    rdf_chain.push_back(rdf);

    std::vector<ROOT::RDF::RResultPtr<TH1D>> histograms;
    std::vector<int> colors = {
        kBlack, kRed, kBlue, kGreen + 2, kMagenta,
        kOrange + 7, kCyan + 2, kViolet, kTeal + 3, kGray + 2
    };
    // Definiamo prima la nuova colonna come divisione
    //auto rdf_scaled = rdf_chain.back().Define(
    //					      "ekin_MeV",   // nuovo nome della colonna
    //					      "tt_ekin/1000.0"  // espressione aritmetica
    //					      );
    // Creo gli istogrammi per ogni cut, con bin logaritmici
    for (size_t i = 0; i < cut_stages.size(); ++i) {
      const auto& [label, cut] = cut_stages[i];
      
      if (!cut.empty()) {
	rdf_chain.push_back(rdf_chain.back().Filter(cut));
        } else {
            rdf_chain.push_back(rdf_chain.back());
      }
      
      std::string hname = "hist_" + std::to_string(i);
      //auto h = rdf_scaled.Histo1D(
      //			  { hname.c_str(),
      //			    "Spectra of successive cuts",
      //			    N_BINS,
      //			    log_bins },       // std::vector<double> per i bin logaritmici
      //			  "ekin_MeV"          // nome colonna appena definita
      //			  );
      
      auto h = rdf_chain.back().Histo1D<double>({hname.c_str(), "Spectra of successive cuts", N_BINS, log_bins.data()},"tt_ekin" );
      //	auto h = rdf_chain.back().Histo1D<double>({hname.c_str(), "Spectra of successive cuts", N_BINS, log_bins.data()},"E_true" );
      h->SetLineColor(colors[i % colors.size()]);
      h->SetLineWidth(2);
      histograms.push_back(h);
      
      std::cout << "Stage " << i << ": " << label << " - Entries: " << h->GetEntries() << std::endl;
    }

    // --- Calcolo delle efficienze globali ---
    std::vector<double> efficiencies;
    efficiencies.reserve(cut_stages.size());

    for (size_t i = 0; i < histograms.size(); ++i) {
        if (i == 0) {
            efficiencies.push_back(1.0);  // prima efficienza = 100%
        } else {
            double prev = histograms[i-1]->GetEntries();
            double curr = histograms[i]->GetEntries();
            double eff = (prev > 0) ? curr / prev : 0;
            efficiencies.push_back(eff);
        }
    }

    TCanvas *c_eff = new TCanvas("c_eff", "Cut Efficiencies", 800, 600);
    c_eff->SetGrid();

    int n = (int)cut_stages.size();
    TGraph *gr_eff = new TGraph(n);
    for (int i = 0; i < n; ++i) {
        gr_eff->SetPoint(i, i, efficiencies[i]);
    }
    gr_eff->SetTitle("Relative Efficiency per Cut;Cut Index;Efficiency");
    gr_eff->SetMarkerStyle(20);
    gr_eff->SetMarkerColor(kBlue);
    gr_eff->SetLineColor(kBlue);
    gr_eff->SetMinimum(0);
    gr_eff->SetMaximum(1.05);

    gr_eff->Draw("APL");

    c_eff->SaveAs("efficiencies_per_cut.pdf");
    c_eff->SaveAs("efficiencies_per_cut.C");

    // --- Efficienze bin-by-bin (vs energia) con bin logaritmici ---
    std::vector<TH1D*> eff_hists;
    eff_hists.reserve(cut_stages.size());

    for (size_t i = 0; i < histograms.size(); ++i) {
        if (i == 0) {
            // Primo cut: efficienza = 1 in tutti i bin
            TH1D* h_eff = new TH1D(Form("eff_%zu", i), Form("Efficiency after cut %s", cut_stages[i].first.c_str()),
                                   N_BINS, log_bins.data());
            for (int bin=1; bin <= N_BINS; ++bin) {
                h_eff->SetBinContent(bin, 1.0);
                h_eff->SetBinError(bin, 0);
            }
            eff_hists.push_back(h_eff);
        } else {
            TH1D* h_prev = histograms[i-1].GetPtr();
            TH1D* h_curr = histograms[i].GetPtr();

            TH1D* h_eff = new TH1D(Form("eff_%zu", i), Form("Efficiency after cut %s", cut_stages[i].first.c_str()),
                                   N_BINS, log_bins.data());

            for (int bin=1; bin <= N_BINS; ++bin) {
                double prev_val = h_prev->GetBinContent(bin);
                double curr_val = h_curr->GetBinContent(bin);
                double eff = (prev_val > 0) ? curr_val / prev_val : 0;
                h_eff->SetBinContent(bin, eff);
                // errore binomiale approssimato:
                double err = (prev_val > 0) ? sqrt(eff * (1 - eff) / prev_val) : 0;
                h_eff->SetBinError(bin, err);
            }
            eff_hists.push_back(h_eff);
        }
    }

    // Canvas con asse X in scala logaritmica per efficienze vs energia
    TCanvas *c_eff_bin = new TCanvas("c_eff_bin", "Efficiency per Cut vs Energy", 1200, 800);
    c_eff_bin->Divide(3, 4); // adatta il layout al numero di cut
    c_eff_bin->SetLogx();

    for (size_t i = 0; i < eff_hists.size(); ++i) {
        c_eff_bin->cd(i+1);
        c_eff_bin->cd(i+1)->SetLogx();
        eff_hists[i]->SetMinimum(0);
        eff_hists[i]->SetMaximum(1.1);
        eff_hists[i]->SetTitle(Form("Efficiency after cut %s", cut_stages[i].first.c_str()));
        eff_hists[i]->GetXaxis()->SetTitle("Energy [GeV]");
        eff_hists[i]->GetYaxis()->SetTitle("Efficiency");
        eff_hists[i]->Draw("E");
    }

    c_eff_bin->SaveAs("efficiency_per_cut_vs_energy.pdf");
    c_eff_bin->SaveAs("efficiency_per_cut_vs_energy.C");


    // === Efficienze cumulative bin-by-bin (vs energia) ===
std::vector<TH1D*> eff_cumulative_hists;
eff_cumulative_hists.reserve(cut_stages.size());

TH1D* h_initial = histograms[0].GetPtr(); // riferimento ai conteggi iniziali

for (size_t i = 0; i < histograms.size(); ++i) {
    TH1D* h_cut = histograms[i].GetPtr();

    TH1D* h_eff_cum = new TH1D(Form("eff_cum_%zu", i),
                               Form("Cumulative Efficiency - %s", cut_stages[i].first.c_str()),
                               N_BINS, log_bins.data());

    for (int bin = 1; bin <= N_BINS; ++bin) {
        double initial_val = h_initial->GetBinContent(bin);
        double cut_val = h_cut->GetBinContent(bin);

        double eff = (initial_val > 0) ? cut_val / initial_val : 0.0;
        h_eff_cum->SetBinContent(bin, eff);

        double err = (initial_val > 0) ? sqrt(eff * (1 - eff) / initial_val) : 0;
        h_eff_cum->SetBinError(bin, err);
    }

    eff_cumulative_hists.push_back(h_eff_cum);
}

// === Canvas: efficienza cumulativa per cut vs energia ===
TCanvas *c_eff_cumulative = new TCanvas("c_eff_cumulative", "Cumulative Efficiency per Cut vs Energy", 1200, 800);
c_eff_cumulative->Divide(3, 4); // stesso layout

for (size_t i = 0; i < eff_cumulative_hists.size(); ++i) {
    c_eff_cumulative->cd(i+1)->SetLogx();  // logaritmico solo sull’asse X
    c_eff_cumulative->cd(i+1)->SetLogy();  // logaritmico solo sull’asse X

    eff_cumulative_hists[i]->SetMinimum(0.0001);
    eff_cumulative_hists[i]->SetMaximum(1.1);
    eff_cumulative_hists[i]->SetTitle(Form("Cumulative Eff. - %s", cut_stages[i].first.c_str()));
    eff_cumulative_hists[i]->GetXaxis()->SetTitle("Energy [GeV]");
    eff_cumulative_hists[i]->GetYaxis()->SetTitle("Efficiency");
    eff_cumulative_hists[i]->Draw("E");
}

c_eff_cumulative->SaveAs("efficiency_cumulative_vs_energy.pdf");
 c_eff_cumulative->SaveAs("efficiency_cumulative_vs_energy.C");


    // === Disegno spettri dopo i tagli ===
    auto canvas = new TCanvas("c", "Spectra after cuts", 1000, 800);
    canvas->SetLogx();
    canvas->SetLogy();

    auto legend = new TLegend(0.25, 0.1, 0.75, 0.45);

    double max_y = 0;
    for (auto& h : histograms) {
        if (h.GetPtr()->GetMaximum() > max_y)
            max_y = h.GetPtr()->GetMaximum();
    }

    histograms[0].GetPtr()->SetMaximum(max_y * 1.2);
    histograms[0].GetPtr()->SetMinimum(1e-3);

    for (size_t i = 0; i < histograms.size(); ++i) {
        histograms[i]->Draw(i == 0 ? "HIST" : "HIST SAME");
        legend->AddEntry(histograms[i].GetPtr(), cut_stages[i].first.c_str(), "l");
    }

    legend->Draw();
    canvas->SaveAs("spectra_single_file1.pdf");
    canvas->SaveAs("spectra_single_file1.C");

    std::cout << "Figures are saved!" << std::endl;
}
