// Efficiency of each selection cut on the weighted MC, as a function of energy.
//
// Usage:
//   root -l -b -q 'macros/cutflow_efficiency.C("/path/to/tt_scaled_cos_theta_bins/exp0.0/")'
//
// Produces efficiencies_per_cut.pdf, efficiency_per_cut_vs_energy.pdf,
// efficiency_cumulative_vs_energy.pdf and spectra_per_cut.pdf in the working directory.
// The cut list mirrors an earlier ordering of src/dampe_gamma/selection/cuts.py with a
// reduced Sanity cut; see REPORT.md.

#include <ROOT/RDataFrame.hxx>
#include <TCanvas.h>
#include <TGraph.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TList.h>
#include <TStyle.h>
#include <TSystemDirectory.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

const std::string kTreeName = "MC";
constexpr int kNBins = 50;
constexpr double kEMin = 1;
constexpr double kEMax = 1e4;
constexpr double kPsdX2 = -298.5;
constexpr double kPsdY2 = -324.7;

using CutList = std::vector<std::pair<std::string, std::string>>;

std::vector<std::string> get_root_files(const std::string& directory) {
    std::vector<std::string> paths;
    TSystemDirectory dir("dataDir", directory.c_str());
    TList* files = dir.GetListOfFiles();
    if (!files) return paths;
    TIter next(files);
    while (TObject* obj = next()) {
        TString name = obj->GetName();
        if (name.EndsWith(".root")) paths.emplace_back(directory + name.Data());
    }
    return paths;
}

std::vector<double> log_bins(double min_val, double max_val, int n_bins) {
    std::vector<double> edges(n_bins + 1);
    const double step = (std::log10(max_val) - std::log10(min_val)) / n_bins;
    for (int i = 0; i <= n_bins; ++i) edges[i] = std::pow(10, std::log10(min_val) + i * step);
    return edges;
}

CutList cut_stages() {
    return {
        {"No cut", ""},
        {"HET", "tt_isHET == 1"},
        {"Geometry",
         "tt_F0 < 0.35 && tt_F1 < 0.45 && CUT_TRACK_NO_HOLES_Best == 1 && theta_BestSTK < 60 && "
         "tt_Xtrl < 100 && STKPlaneCut_Best == 1 && fabs(X_ProjectionBest_X1) < 392 && "
         "fabs(X_ProjectionBest_X2) < 392 && fabs(Y_ProjectionBest_Y1) < 392 && "
         "fabs(Y_ProjectionBest_Y2) < 392"},
        {"BGO trajectory",
         Form("fabs(tt_bgoRecSlopeX * %.1f + tt_bgoRecInterceptX) < 420 && "
              "fabs(tt_bgoRecSlopeY * %.1f + tt_bgoRecInterceptY) < 420",
              kPsdX2, kPsdY2)},
        {"Tracks", "nnTracks > 0"},
        {"Ecore", "tt_Ecore3 <= 1 && tt_Ecore3 > 0.9"},
        {"Charge",
         "(ProjSTKPSD_countX_Best<=1 && ProjSTKPSD_countY_Best<=1) || "
         "(ProjSTKPSD_count_Best/log(tt_bgoTotalE_GeV)<=0.65)"},
        {"CNN", "CNN_linear > 1.84"},
        {"Sanity",
         "tt_maxEX1Psd/tt_bgoTotalE_GeV<1.5 && tt_maxEX2Psd/tt_bgoTotalE_GeV<1.5 && "
         "tt_maxEY1Psd/tt_bgoTotalE_GeV<1.5 && tt_maxEY2Psd/tt_bgoTotalE_GeV<1.5"},
        {"BDT",
         "(tt_bgoTotalE_GeV<=100 && BDT_cleaned>0.18-0.038*log(tt_bgoTotalE_GeV)) || "
         "((tt_bgoTotalE_GeV>100 && tt_bgoTotalE_GeV<=1000) && "
         "BDT_cleaned>0.05-0.035*log(0.01*tt_bgoTotalE_GeV)) || "
         "(tt_bgoTotalE_GeV>1000 && BDT_cleaned>0.05-0.02*log(0.001*tt_bgoTotalE_GeV))"},
    };
}

// Bin by bin ratio num / den with binomial errors; a null den gives a flat efficiency of 1.
TH1D* ratio_histogram(const char* name, const char* title, const TH1D* num, const TH1D* den,
                      const std::vector<double>& edges) {
    auto* h = new TH1D(name, title, kNBins, edges.data());
    for (int bin = 1; bin <= kNBins; ++bin) {
        if (!den) {
            h->SetBinContent(bin, 1.0);
            continue;
        }
        const double d = den->GetBinContent(bin);
        const double eff = d > 0 ? num->GetBinContent(bin) / d : 0.0;
        h->SetBinContent(bin, eff);
        h->SetBinError(bin, d > 0 ? std::sqrt(eff * (1 - eff) / d) : 0.0);
    }
    return h;
}

void draw_grid(const char* name, const char* title, std::vector<TH1D*>& hists, double y_min,
               double y_max, const char* output) {
    auto* canvas = new TCanvas(name, title, 1200, 800);
    canvas->Divide(3, 4);
    for (size_t i = 0; i < hists.size(); ++i) {
        canvas->cd(i + 1)->SetLogx();
        hists[i]->SetMinimum(y_min);
        if (y_max > 0) hists[i]->SetMaximum(y_max);
        hists[i]->GetXaxis()->SetTitle("Energy [GeV]");
        hists[i]->GetYaxis()->SetTitle("Efficiency");
        hists[i]->Draw("E");
    }
    canvas->SaveAs(output);
}

}  // namespace

void cutflow_efficiency(const std::string& data_dir) {
    ROOT::EnableImplicitMT();
    gStyle->SetOptStat(0);

    const auto files = get_root_files(data_dir);
    if (files.empty()) {
        std::cerr << "No ROOT files found in " << data_dir << std::endl;
        return;
    }

    ROOT::RDataFrame rdf(kTreeName, files);
    const auto edges = log_bins(kEMin, kEMax, kNBins);
    const auto stages = cut_stages();
    const std::vector<int> colors = {kBlack,      kRed,       kBlue,    kGreen + 2, kMagenta,
                                     kOrange + 7, kCyan + 2, kViolet, kTeal + 3,  kGray + 2};

    std::vector<ROOT::RDF::RNode> nodes{rdf};
    std::vector<ROOT::RDF::RResultPtr<TH1D>> spectra;
    for (size_t i = 0; i < stages.size(); ++i) {
        const auto& [label, cut] = stages[i];
        nodes.push_back(cut.empty() ? nodes.back() : nodes.back().Filter(cut));
        auto h = nodes.back().Histo1D<double>(
            {Form("hist_%zu", i), "Spectra of successive cuts", kNBins, edges.data()}, "E_true");
        h->SetLineColor(colors[i % colors.size()]);
        h->SetLineWidth(2);
        spectra.push_back(h);
        std::cout << "Stage " << i << ": " << label << ", entries " << h->GetEntries() << std::endl;
    }

    const int n = static_cast<int>(stages.size());
    auto* graph = new TGraph(n);
    for (int i = 0; i < n; ++i) {
        const double prev = i == 0 ? 1.0 : spectra[i - 1]->GetEntries();
        const double eff = i == 0 ? 1.0 : (prev > 0 ? spectra[i]->GetEntries() / prev : 0.0);
        graph->SetPoint(i, i, eff);
    }
    auto* c_eff = new TCanvas("c_eff", "Cut Efficiencies", 800, 600);
    c_eff->SetGrid();
    graph->SetTitle("Relative Efficiency per Cut;Cut Index;Efficiency");
    graph->SetMarkerStyle(20);
    graph->SetMarkerColor(kBlue);
    graph->SetLineColor(kBlue);
    graph->SetMinimum(0);
    graph->SetMaximum(1.05);
    graph->Draw("APL");
    c_eff->SaveAs("efficiencies_per_cut.pdf");

    std::vector<TH1D*> relative, cumulative;
    for (size_t i = 0; i < spectra.size(); ++i) {
        const auto& label = stages[i].first;
        const TH1D* previous = i == 0 ? nullptr : spectra[i - 1].GetPtr();
        relative.push_back(ratio_histogram(Form("eff_%zu", i),
                                           Form("Efficiency after cut %s", label.c_str()),
                                           spectra[i].GetPtr(), previous, edges));
        cumulative.push_back(ratio_histogram(Form("eff_cum_%zu", i),
                                             Form("Cumulative efficiency, %s", label.c_str()),
                                             spectra[i].GetPtr(), spectra[0].GetPtr(), edges));
    }
    draw_grid("c_eff_bin", "Efficiency per Cut vs Energy", relative, 0.0, 1.1,
              "efficiency_per_cut_vs_energy.pdf");
    draw_grid("c_eff_cumulative", "Cumulative Efficiency per Cut vs Energy", cumulative, 1e-4, -1,
              "efficiency_cumulative_vs_energy.pdf");

    auto* canvas = new TCanvas("c", "Spectra after cuts", 1000, 800);
    canvas->SetLogx();
    auto* legend = new TLegend(0.25, 0.1, 0.75, 0.45);
    double max_y = 0;
    for (auto& h : spectra) max_y = std::max(max_y, h->GetMaximum());
    spectra[0]->SetMaximum(max_y * 1.2);
    spectra[0]->SetMinimum(1e-3);
    for (size_t i = 0; i < spectra.size(); ++i) {
        spectra[i]->Draw(i == 0 ? "HIST" : "HIST SAME");
        legend->AddEntry(spectra[i].GetPtr(), stages[i].first.c_str(), "l");
    }
    legend->Draw();
    canvas->SaveAs("spectra_per_cut.pdf");
}
