#ifdef __CLING__
#pragma cling optimize(0)
#endif
void efficiencies_per_cut()
{
//=========Macro generated from canvas: c_eff/Cut Efficiencies
//=========  (Fri Jun 13 13:14:57 2025) by ROOT version 6.34.02
   TCanvas *c_eff = new TCanvas("c_eff", "Cut Efficiencies",0,0,800,600);
   gStyle->SetOptStat(0);
   c_eff->Range(-1.2375,-0.13125,11.1375,1.18125);
   c_eff->SetFillColor(0);
   c_eff->SetBorderMode(0);
   c_eff->SetBorderSize(2);
   c_eff->SetGridx();
   c_eff->SetGridy();
   c_eff->SetFrameBorderMode(0);
   c_eff->SetFrameBorderMode(0);
   
   Double_t Graph0_fx1[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
   Double_t Graph0_fy1[10] = { 1, 0.2581046, 0.3970897, 0.8278741, 0.8355256, 0.9939024, 0.9645025, 0.9844941, 0.9994098, 0.9594822 };
   TGraph *graph = new TGraph(10,Graph0_fx1,Graph0_fy1);
   graph->SetName("Graph0");
   graph->SetTitle("Relative Efficiency per Cut");
   graph->SetFillStyle(1000);

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#0000ff");
   graph->SetLineColor(ci);

   ci = TColor::GetColor("#0000ff");
   graph->SetMarkerColor(ci);
   graph->SetMarkerStyle(20);
   
   TH1F *Graph_Graph01 = new TH1F("Graph_Graph01","Relative Efficiency per Cut",100,0,9.9);
   Graph_Graph01->SetMinimum(0);
   Graph_Graph01->SetMaximum(1.05);
   Graph_Graph01->SetDirectory(nullptr);
   Graph_Graph01->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_Graph01->SetLineColor(ci);
   Graph_Graph01->GetXaxis()->SetTitle("Cut Index");
   Graph_Graph01->GetXaxis()->SetLabelFont(42);
   Graph_Graph01->GetXaxis()->SetTitleOffset(1);
   Graph_Graph01->GetXaxis()->SetTitleFont(42);
   Graph_Graph01->GetYaxis()->SetTitle("Efficiency");
   Graph_Graph01->GetYaxis()->SetLabelFont(42);
   Graph_Graph01->GetYaxis()->SetTitleFont(42);
   Graph_Graph01->GetZaxis()->SetLabelFont(42);
   Graph_Graph01->GetZaxis()->SetTitleOffset(1);
   Graph_Graph01->GetZaxis()->SetTitleFont(42);
   graph->SetHistogram(Graph_Graph01);
   
   graph->Draw("apl");
   
   TPaveText *pt = new TPaveText(0.2607286,0.9341608,0.7392714,0.995,"blNDC");
   pt->SetName("title");
   pt->SetBorderSize(0);
   pt->SetFillColor(0);
   pt->SetFillStyle(0);
   pt->SetTextFont(42);
   TText *pt_LaTex = pt->AddText("Relative Efficiency per Cut");
   pt->Draw();
   c_eff->Modified();
   c_eff->SetSelected(c_eff);
}
