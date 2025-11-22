from __future__ import print_function, division, absolute_import
#from ROOT import gSystem
import sys
import os
import math

#gSystem.Load('libDmpEvent.so')
from ROOT import *
#gSystem.Load('libDmpEvent.so')
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#import matplotlib

matplotlib.rc('xtick', labelsize=26)
matplotlib.rc('ytick', labelsize=26)

####################################
### TO COMPLETE ###

e_min = 1.
e_max = 10000.
n_points = 100

var_name = "BDT"
cond = -0.01


PSD_X1=-284.5
PSD_Y1=-310.7
PSD_X2=-298.5
PSD_Y2=-324.7

BGO_Z = 58.5
BGOz=np.array([58.5,87.5,116.5,145.5,174.5,203.5,232.5,261.5,290.5,319.5,348.5,377.5,406.5,435.5])
BGOzTop = 58.5
psd_y1=-310.7
psd_y2=-324.7
psd_x1=-284.5
psd_x2=-298.5

###################################

ene = np.logspace(1, 5, n_points+1) / 10.
print(ene)

nentries_tot = []
temp_file = open('TotalEntriesPerBin_gamma_012_100Bins_10TeV.txt', 'r')
for line in temp_file:
        nentries_tot.append(float(line))

skim1GeV_p = []
temp_file_2 = open('NEntries1GeVPerBin_gamma_012_100Bins_10TeV.txt', 'r')
for line in temp_file_2:
        #nentries_tot.append(float(line))
        skim1GeV_p.append(float(line))


######################################################

#f = "/srv/beegfs/scratch/users/f/friedenj/out_MCESep/DmlNtup_gamma_part00Full_MPhotonSelProj_TQ_1GeV_1TeV.root"
#f = "/srv/beegfs/scratch/users/f/friedenj/DmlNtup_gamma_Full_part00Full_TQ_FullSel.root"
f = "/srv/beegfs/scratch/users/f/friedenj/DmlNtup_gamma_Full_part00Full_TQ_FullSel_dummy.root"


TF = TFile(f,'READ')
TT = TF.Get("DmlNtup")



print(str(TT.GetEntries()))

nevents = 1000000

#ene = np.linspace(e_min, e_max, n_points)

trigger_p = [0]*n_points
bgoFid_p = [0]*n_points
ecore_p = [0]*n_points
sanity_p = [0]*n_points
track_p = [0]*n_points
stkFid_p = [0]*n_points
DNN_p = [0]*n_points
BDT_p = [0]*n_points
consist_p = [0]*n_points
CNNlin_p = [0]*n_points
TQ_p = [0]*n_points
geom_p = [0]*n_points
charge_p = [0]*n_points

#for n in range(0,8000):
for n in range(1,TT.GetEntries()):
        TT.GetEntry(n)
        ekin = TT.tt_ekin/1000.
        Erec = TT.tt_bgoTotalE_GeV
        #ekin = Erec
        HET = getattr(TT, "tt_isHET")
        LET = getattr(TT, "tt_isLET")
        UB = getattr(TT, "UB_trigger")
        MIP1 = getattr(TT, "MIP1_trigger_gen")
        MIP2 = getattr(TT, "MIP2_trigger_gen")
        Ecore3 = getattr(TT, "tt_Ecore3")
        tt_QcloClusterXTrackCorr = getattr(TT, "tt_QcloClusterXTrackCorr")
        tt_QcloClusterYTrackCorr = getattr(TT, "tt_QcloClusterYTrackCorr")
        tt_maxEX1Psd = getattr(TT, "tt_maxEX1Psd")
        tt_maxEX2Psd = getattr(TT, "tt_maxEX2Psd")
        tt_maxEY1Psd = getattr(TT, "tt_maxEY1Psd")
        tt_maxEY2Psd = getattr(TT, "tt_maxEY2Psd")
        nnTracks = getattr(TT, "nnTracks")
        X_ProjectionBest_X1 = getattr(TT, "X_ProjectionBest_X1")
        X_ProjectionBest_X2 = getattr(TT, "X_ProjectionBest_X2")
        Y_ProjectionBest_Y1 = getattr(TT, "Y_ProjectionBest_Y1")
        Y_ProjectionBest_Y2 = getattr(TT, "Y_ProjectionBest_Y2")
        X_ProjectionBest_X1_E = getattr(TT, "X_ProjectionBest_X1_E")
        X_ProjectionBest_X2_E = getattr(TT, "X_ProjectionBest_X2_E")
        Y_ProjectionBest_Y1_E = getattr(TT, "Y_ProjectionBest_Y1_E")
        Y_ProjectionBest_Y2_E = getattr(TT, "Y_ProjectionBest_Y2_E")

        X_ProjectionBest_X1_E_1bar = getattr(TT, "X_ProjectionBest_X1_E_1bar")
        X_ProjectionBest_X2_E_1bar = getattr(TT, "X_ProjectionBest_X2_E_1bar")
        Y_ProjectionBest_Y1_E_1bar = getattr(TT, "Y_ProjectionBest_Y1_E_1bar")
        Y_ProjectionBest_Y2_E_1bar = getattr(TT, "Y_ProjectionBest_Y2_E_1bar")
        X_ProjectionBest_X1_E_3bar = getattr(TT, "X_ProjectionBest_X1_E_3bar")
        X_ProjectionBest_X2_E_3bar = getattr(TT, "X_ProjectionBest_X2_E_3bar")
        Y_ProjectionBest_Y1_E_3bar = getattr(TT, "Y_ProjectionBest_Y1_E_3bar")
        Y_ProjectionBest_Y2_E_3bar = getattr(TT, "Y_ProjectionBest_Y2_E_3bar")

        Etrack_STK_Lay0_02mm_Best = getattr(TT, "Etrack_STK_Lay0_02mm_Best")
        Etrack_STK_Lay1_02mm_Best = getattr(TT, "Etrack_STK_Lay1_02mm_Best")
        Etrack_STK_Lay0_1mm_Best = getattr(TT, "Etrack_STK_Lay0_1mm_Best")
        Etrack_STK_Lay1_1mm_Best = getattr(TT, "Etrack_STK_Lay1_1mm_Best")
        Etrack_STK_Lay0_5mm_Best = getattr(TT, "Etrack_STK_Lay0_5mm_Best")
        Etrack_STK_Lay1_5mm_Best = getattr(TT, "Etrack_STK_Lay1_5mm_Best")
        Etrack_STK_Lay0_10mm_Best = getattr(TT, "Etrack_STK_Lay0_10mm_Best")
        Etrack_STK_Lay1_10mm_Best = getattr(TT, "Etrack_STK_Lay1_10mm_Best")
        #DNN = getattr(TT, "DNN_score")
        BDT = getattr(TT, "BDT")
        BDT_new = getattr(TT, "BDT_new")
        dxCloCluX1_track = getattr(TT, "dxCloCluX1_track")
        dxCloCluX2_track = getattr(TT, "dxCloCluX2_track")
        dxCloCluY1_track = getattr(TT, "dxCloCluY1_track")
        dxCloCluY2_track = getattr(TT, "dxCloCluY2_track")
        TQ_bestT = getattr(TT, "TQ_bestT")
        trackCHI2NDOF = getattr(TT, "trackCHI2NDOF")
        TQ_bestT_index = getattr(TT, "TQ_bestT_index")
        theta_BestSTK = getattr(TT, "theta_BestSTK")
        tt_F0 = getattr(TT, "tt_F0")
        tt_F1 = getattr(TT, "tt_F1")
        tt_Xtrl = getattr(TT, "tt_Xtrl")
        STKPlaneCut_Best = getattr(TT,"STKPlaneCut_Best")
        ProjSTKPSD_count_Best = getattr(TT, "ProjSTKPSD_count_Best")
        ProjSTKPSD_countX_Best = getattr(TT, "ProjSTKPSD_countX_Best")
        ProjSTKPSD_countY_Best = getattr(TT, "ProjSTKPSD_countY_Best")
        CUT_TRACK_NO_HOLES_Best = getattr(TT, "CUT_TRACK_NO_HOLES_Best")
        tt_nPsdHits = getattr(TT, "tt_nPsdHits")
        ECloCluX1_track = getattr(TT, "ECloCluX1_track")
        ECloCluX2_track = getattr(TT, "ECloCluX2_track")
        ECloCluY1_track = getattr(TT, "ECloCluY1_track")
        ECloCluY2_track = getattr(TT, "ECloCluY2_track")

        BGO_slopeX = TT.tt_bgoRecSlopeX
        BGO_slopeY = TT.tt_bgoRecSlopeY
        BGO_interceptX = TT.tt_bgoRecInterceptX
        BGO_interceptY = TT.tt_bgoRecInterceptY

        BGO_ProjX2 = BGO_slopeX*psd_x2 + BGO_interceptX
        BGO_ProjY2 = BGO_slopeY*psd_y2 + BGO_interceptY

        incl_x = (X_ProjectionBest_X1 - X_ProjectionBest_X2)/(PSD_X1 - PSD_X2)
        incl_y = (Y_ProjectionBest_Y1 - Y_ProjectionBest_Y2)/(PSD_Y1 - PSD_Y2)
        

        #if(ekin > 10000): continue

        bin_i = 0
        for i in range(0, len(ene)):
                if(ekin > ene [i]):
                        bin_i = i

        TriggerPassed = HET==1
        if(TriggerPassed):
                trigger_p[bin_i] += 1

        BgoFidPassed = TriggerPassed and abs(BGO_ProjX2)<420 and abs(BGO_ProjY2)<420
        if(BgoFidPassed):
                bgoFid_p[bin_i] +=1

        EcorePassed = BgoFidPassed and Ecore3<=1 and Ecore3>=0.9
        if(EcorePassed):
                ecore_p[bin_i] +=1

        TrackPassed = EcorePassed and nnTracks>0
        if(TrackPassed):
                track_p[bin_i] +=1

        GeomPassed = TrackPassed and tt_F0<0.35 and tt_F1<0.45 and CUT_TRACK_NO_HOLES_Best==1 and theta_BestSTK<60. and tt_Xtrl<100 and STKPlaneCut_Best==1 and abs(X_ProjectionBest_X1)<392 and abs(X_ProjectionBest_X2)<392 and abs(Y_ProjectionBest_Y1)<392 and abs(Y_ProjectionBest_Y2)<392
        if(GeomPassed):
                geom_p[bin_i] +=1
        '''
        ConsistPassed = GeomPassed and ((ProjSTKPSD_countX_Best<=1 and ProjSTKPSD_countY_Best<=1) or ProjSTKPSD_count_Best/log(Erec)<=0.65) and (Etrack_STK_Lay0_10mm_Best+Etrack_STK_Lay1_10mm_Best)/Erec<1000 and (tt_maxEX1Psd+tt_maxEX2Psd+tt_maxEY1Psd+tt_maxEY2Psd)/Erec<=3 and (ECloCluX1_track+ECloCluX2_track+ECloCluY1_track+ECloCluY2_track)/Erec<3.
        if(ConsistPassed):
                consist_p[bin_i] +=1

        TQPassed = ConsistPassed and TQ_bestT>0.32
        if(TQPassed):
                TQ_p[bin_i] +=1
        '''

        ChargePassed = GeomPassed and ((ProjSTKPSD_countX_Best<=1 and ProjSTKPSD_countY_Best<=1) or ProjSTKPSD_count_Best/log(Erec)<=0.65) #and tt_nPsdHits/Erec<1.8 and (X_ProjectionBest_X1_E_1bar+X_ProjectionBest_X2_E_1bar+Y_ProjectionBest_Y1_E_1bar+Y_ProjectionBest_Y2_E_1bar)/Erec<0.8
        if(ChargePassed):
                charge_p[bin_i] +=1

        CNNPassed = ChargePassed and TT.CNN_linear>1.84 #and TT.CNN_sigmoid>0.58
        if(CNNPassed):
                CNNlin_p[bin_i] +=1
        

        #StkFidPassed = TrackPassed and abs(X_ProjectionBest_X1)<420 and abs(X_ProjectionBest_X2)<420 and abs(Y_ProjectionBest_Y1)<420 and abs(Y_ProjectionBest_Y2)<420
        #if(StkFidPassed):
        #        stkFid_p[bin_i] +=1


        SanityPassed = CNNPassed and tt_maxEX1Psd/Erec<1.5 and tt_maxEX2Psd/Erec<1.5 and tt_maxEY1Psd/Erec<1.5 and TT.tt_maxEY2Psd/Erec<1.5 and tt_QcloClusterXTrackCorr/Erec<0.5 and tt_QcloClusterYTrackCorr/Erec<0.5 and (X_ProjectionBest_X1_E+X_ProjectionBest_X2_E)/Erec<1. and (Y_ProjectionBest_Y1_E+Y_ProjectionBest_Y2_E)/Erec<1. and (X_ProjectionBest_X1_E_1bar+X_ProjectionBest_X2_E_1bar)/Erec<1. and (Y_ProjectionBest_Y1_E_1bar+Y_ProjectionBest_Y2_E_1bar)/Erec<1. and (X_ProjectionBest_X1_E_3bar+X_ProjectionBest_X2_E_3bar)/Erec<1. and (Y_ProjectionBest_Y1_E_3bar+Y_ProjectionBest_Y2_E_3bar)/Erec<1. and Etrack_STK_Lay0_02mm_Best<np.exp(2.8*Erec) and Etrack_STK_Lay1_02mm_Best<np.exp(2.8*Erec) and Etrack_STK_Lay0_1mm_Best<np.exp(2.8*Erec) and Etrack_STK_Lay1_1mm_Best<np.exp(2.8*Erec) and Etrack_STK_Lay0_5mm_Best<np.exp(3.8*Erec) and Etrack_STK_Lay1_5mm_Best<np.exp(3.8*Erec) and Etrack_STK_Lay0_10mm_Best<np.exp(4.0*Erec) and Etrack_STK_Lay1_10mm_Best<np.exp(4.0*Erec)
        if(SanityPassed):
                sanity_p[bin_i] +=1

                
        #BDTPassed = SanityPassed and ( (Erec<=100 and BDT_new>0.2-0.038*log(Erec) and BDT_new>0.02) or (Erec>100 and Erec<=1000 and TT.BDT_new>0.1-0.06*log(0.01*Erec) and TT.BDT_new>-0.06) or (Erec>1000 and BDT_new>0.05-0.05*log(0.001*Erec) and BDT_new>-0.03) )
        BDTPassed = SanityPassed and ( (Erec<=100 and BDT_new>0.18-0.038*log(Erec) ) or (Erec>100 and Erec<=1000 and TT.BDT_new>0.05-0.035*log(0.01*Erec) ) or (Erec>1000 and BDT_new>0.05-0.02*log(0.001*Erec) ) )
        if(BDTPassed):
                BDT_p[bin_i] +=1




                 
vec_eff_skim1GeV = []
vec_eff_trigger = []
vec_eff_bgoFid = []
vec_eff_ecore = []
vec_eff_geom = []
vec_eff_charge = []
vec_eff_sanity = []
vec_eff_track = []
vec_eff_stkFid = []
vec_eff_DNN = []
vec_eff_BDT = []
vec_eff_consist = []
vec_eff_CNNlin = []
vec_eff_TQ = []

err_eff_skim1GeV = []
err_eff_trigger = []
err_eff_bgoFid = []
err_eff_ecore = []
err_eff_geom = []
err_eff_charge = []
err_eff_sanity = []
err_eff_track = []
err_eff_stkFid = []
err_eff_DNN = []
err_eff_BDT = []
err_eff_consist = []
err_eff_CNNlin = []
err_eff_TQ = []

Energy_vec = []
err_ene = [0]*(n_points)

for e in range(0, n_points):
        vec_eff_skim1GeV.append(skim1GeV_p[e]/nentries_tot[e] * 37.59)
        vec_eff_trigger.append(trigger_p[e]/nentries_tot[e] * 37.59)
        vec_eff_bgoFid.append(bgoFid_p[e]/nentries_tot[e] * 37.59)
        vec_eff_ecore.append(ecore_p[e]/nentries_tot[e] * 37.59)
        vec_eff_geom.append(geom_p[e]/nentries_tot[e] * 37.59)
        vec_eff_charge.append(charge_p[e]/nentries_tot[e] * 37.59)
        vec_eff_sanity.append(sanity_p[e]/nentries_tot[e] * 37.59)
        vec_eff_track.append(track_p[e]/nentries_tot[e] * 37.59)
        #vec_eff_stkFid.append(stkFid_p[e]/nentries_tot[e] * 37.59)
        #vec_eff_DNN.append(DNN_p[e]/nentries_tot[e] * 37.59)
        vec_eff_BDT.append(BDT_p[e]/nentries_tot[e] * 37.59)
        vec_eff_CNNlin.append(CNNlin_p[e]/nentries_tot[e] * 37.59)
        vec_eff_consist.append(consist_p[e]/nentries_tot[e] * 37.59)
        vec_eff_TQ.append(TQ_p[e]/nentries_tot[e] * 37.59)

        err_eff_skim1GeV.append( (math.sqrt(skim1GeV_p[e])/nentries_tot[e] + ((skim1GeV_p[e]*math.sqrt(nentries_tot[e]))/(nentries_tot[e]*nentries_tot[e])) ) * 37.59)
        err_eff_trigger.append( (math.sqrt(trigger_p[e])/nentries_tot[e] + ((trigger_p[e]*math.sqrt(nentries_tot[e]))/(nentries_tot[e]*nentries_tot[e])) ) * 37.59)
        err_eff_bgoFid.append( (math.sqrt(bgoFid_p[e])/nentries_tot[e] + ((bgoFid_p[e]*math.sqrt(nentries_tot[e]))/(nentries_tot[e]*nentries_tot[e])) ) * 37.59)
        err_eff_ecore.append( (math.sqrt(ecore_p[e])/nentries_tot[e] + ((ecore_p[e]*math.sqrt(nentries_tot[e]))/(nentries_tot[e]*nentries_tot[e])) ) * 37.59)
        err_eff_geom.append( (math.sqrt(geom_p[e])/nentries_tot[e] + ((geom_p[e]*math.sqrt(nentries_tot[e]))/(nentries_tot[e]*nentries_tot[e])) ) * 37.59)
        err_eff_charge.append( (math.sqrt(charge_p[e])/nentries_tot[e] + ((charge_p[e]*math.sqrt(nentries_tot[e]))/(nentries_tot[e]*nentries_tot[e])) ) * 37.59)
        err_eff_sanity.append( (math.sqrt(sanity_p[e])/nentries_tot[e] + ((sanity_p[e]*math.sqrt(nentries_tot[e]))/(nentries_tot[e]*nentries_tot[e])) ) * 37.59)
        err_eff_track.append( (math.sqrt(track_p[e])/nentries_tot[e] + ((track_p[e]*math.sqrt(nentries_tot[e]))/(nentries_tot[e]*nentries_tot[e])) ) * 37.59)
        #err_eff_stkFid.append( (math.sqrt(stkFid_p[e])/nentries_tot[e] + ((stkFid_p[e]*math.sqrt(nentries_tot[e]))/(nentries_tot[e]*nentries_tot[e])) ) * 37.59)
        #err_eff_DNN.append( (math.sqrt(DNN_p[e])/nentries_tot[e] + ((DNN_p[e]*math.sqrt(nentries_tot[e]))/(nentries_tot[e]*nentries_tot[e])) ) * 37.59)
        err_eff_BDT.append( (math.sqrt(BDT_p[e])/nentries_tot[e] + ((BDT_p[e]*math.sqrt(nentries_tot[e]))/(nentries_tot[e]*nentries_tot[e])) ) * 37.59)
        err_eff_consist.append( (math.sqrt(consist_p[e])/nentries_tot[e] + ((consist_p[e]*math.sqrt(nentries_tot[e]))/(nentries_tot[e]*nentries_tot[e])) ) * 37.59)
        err_eff_CNNlin.append( (math.sqrt(CNNlin_p[e])/nentries_tot[e] + ((CNNlin_p[e]*math.sqrt(nentries_tot[e]))/(nentries_tot[e]*nentries_tot[e])) ) * 37.59)
        err_eff_TQ.append( (math.sqrt(TQ_p[e])/nentries_tot[e] + ((TQ_p[e]*math.sqrt(nentries_tot[e]))/(nentries_tot[e]*nentries_tot[e])) ) * 37.59)

        Energy_vec.append((ene[e]+ene[e+1])/2)


# Plot the efficiency curves
fig = plt.figure(figsize=(12,8))
ax1=fig.add_subplot(111)
ax1.errorbar(Energy_vec, vec_eff_skim1GeV, xerr=err_ene, yerr=err_eff_skim1GeV, color='cyan', marker='o', label="Skim Erec >= 1GeV")
ax1.errorbar(Energy_vec, vec_eff_trigger, xerr=err_ene, yerr=err_eff_trigger, color='black', marker='o', label="HET Trigger")
ax1.errorbar(Energy_vec, vec_eff_bgoFid, xerr=err_ene, yerr=err_eff_bgoFid, color='blue', marker='o', label="BgoFid")
ax1.errorbar(Energy_vec, vec_eff_ecore, xerr=err_ene, yerr=err_eff_ecore, color='red', marker='o', label="BgoFid+Ecore")
ax1.errorbar(Energy_vec, vec_eff_geom, xerr=err_ene, yerr=err_eff_geom, color='darkgreen', marker='o', label="BgoFid+Geom")
ax1.errorbar(Energy_vec, vec_eff_charge, xerr=err_ene, yerr=err_eff_charge, color='darkcyan', marker='o', label="BgoFid+NoCharge")
ax1.errorbar(Energy_vec, vec_eff_CNNlin, xerr=err_ene, yerr=err_eff_CNNlin, color='pink', marker='o', label="Skim+CNN")
ax1.errorbar(Energy_vec, vec_eff_track, xerr=err_ene, yerr=err_eff_track, color='green', marker='o', label="Skim+CNN+STK track")
#ax1.errorbar(Energy_vec, vec_eff_stkFid, xerr=err_ene, yerr=err_eff_stkFid, color='lightblue', marker='o', label="Skim+CNN+STK Fid")
ax1.errorbar(Energy_vec, vec_eff_sanity, xerr=err_ene, yerr=err_eff_sanity, color='yellow', marker='o', label="Skim+CNN+STK+Sanity")
ax1.errorbar(Energy_vec, vec_eff_BDT, xerr=err_ene, yerr=err_eff_BDT, color='violet', marker='o', label="Skim+CNN+STK+Sanity+BDT")
ax1.errorbar(Energy_vec, vec_eff_consist, xerr=err_ene, yerr=err_eff_consist, color='brown', marker='o', label="Skim+CNN+STK+BDT+Consistency")
ax1.errorbar(Energy_vec, vec_eff_TQ, xerr=err_ene, yerr=err_eff_TQ, color='orange', marker='o', label="Skim+STK+BDT+CNN+TQ")

ax1.set_xscale('log')
ax1.set_ylim([-0.1, 3.2])
#ax1.set_yscale('log')
plt.grid(True)
plt.title("Efficiency as a function of the kinetic energy", fontsize=18)
plt.ylabel("Effective acceptance [m$^{2}$ sr]", fontsize=15)
plt.xlabel("Kinetic energy [GeV]", fontsize=15)
plt.legend(loc='upper left', fontsize=15)
fig.savefig("eff_plots/Acceptance_VS_energy_withCNNlin_gamma_100bins_v26.pdf")



acc_file = open("acceptance_values/acc_gamma_withCNN_100bins_TQ35_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = vec_eff_TQ[i]
        acc_file.write(str(acc_bin)+"\n")
acc_file.close()


acc_Ecut = open("curves_acceptance/acceptance_vs_energy_cutE1GeV_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = vec_eff_skim1GeV[i]
        acc_Ecut.write(str(acc_bin)+"\n")
acc_Ecut.close()

acc_trigger = open("curves_acceptance/acceptance_vs_energy_cutTrigger_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = vec_eff_trigger[i]
        acc_trigger.write(str(acc_bin)+"\n")
acc_trigger.close()

acc_BgoFid = open("curves_acceptance/acceptance_vs_energy_cutBgoFid_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = vec_eff_bgoFid[i]
        acc_BgoFid.write(str(acc_bin)+"\n")
acc_BgoFid.close()

acc_Ecore = open("curves_acceptance/acceptance_vs_energy_cutEcore_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = vec_eff_ecore[i]
        acc_Ecore.write(str(acc_bin)+"\n")
acc_Ecore.close()

acc_Geom = open("curves_acceptance/acceptance_vs_energy_cutGeom_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = vec_eff_geom[i]
        acc_Geom.write(str(acc_bin)+"\n")
acc_Geom.close()

acc_Charge = open("curves_acceptance/acceptance_vs_energy_cutCharge_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = vec_eff_charge[i]
        acc_Charge.write(str(acc_bin)+"\n")
acc_Charge.close()

acc_StkTrk = open("curves_acceptance/acceptance_vs_energy_cutStkTrk_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = vec_eff_track[i]
        acc_StkTrk.write(str(acc_bin)+"\n")
acc_StkTrk.close()

'''
acc_StkFid = open("curves_acceptance/acceptance_vs_energy_cutStkFid_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = vec_eff_stkFid[i]
        acc_StkFid.write(str(acc_bin)+"\n")
acc_StkFid.close()
'''

acc_BDT = open("curves_acceptance/acceptance_vs_energy_cutBDT_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = vec_eff_BDT[i]
        acc_BDT.write(str(acc_bin)+"\n")
acc_BDT.close()

acc_Sanity = open("curves_acceptance/acceptance_vs_energy_cutSanity_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = vec_eff_sanity[i]
        acc_Sanity.write(str(acc_bin)+"\n")
acc_Sanity.close()

acc_Consist = open("curves_acceptance/acceptance_vs_energy_cutConsist_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = vec_eff_consist[i]
        acc_Consist.write(str(acc_bin)+"\n")
acc_Consist.close()

acc_CNN = open("curves_acceptance/acceptance_vs_energy_cutCNN_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = vec_eff_CNNlin[i]
        acc_CNN.write(str(acc_bin)+"\n")
acc_CNN.close()

acc_TQ = open("curves_acceptance/acceptance_vs_energy_cutTQ_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = vec_eff_TQ[i]
        acc_TQ.write(str(acc_bin)+"\n")
acc_TQ.close()






acc_trigger_err = open("curves_acceptance/acceptance_vs_energy_cutTrigger_err_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = err_eff_trigger[i]
        acc_trigger_err.write(str(acc_bin)+"\n")
acc_trigger_err.close()

acc_BgoFid_err = open("curves_acceptance/acceptance_vs_energy_cutBgoFid_err_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = err_eff_bgoFid[i]
        acc_BgoFid_err.write(str(acc_bin)+"\n")
acc_BgoFid_err.close()

acc_Ecore_err = open("curves_acceptance/acceptance_vs_energy_cutEcore_err_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = err_eff_ecore[i]
        acc_Ecore_err.write(str(acc_bin)+"\n")
acc_Ecore_err.close()

acc_Geom_err = open("curves_acceptance/acceptance_vs_energy_cutGeom_err_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = err_eff_geom[i]
        acc_Geom_err.write(str(acc_bin)+"\n")
acc_Geom_err.close()


acc_Charge_err = open("curves_acceptance/acceptance_vs_energy_cutCharge_err_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = err_eff_charge[i]
        acc_Charge_err.write(str(acc_bin)+"\n")
acc_Charge_err.close()

acc_StkTrk_err = open("curves_acceptance/acceptance_vs_energy_cutStkTrk_err_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = err_eff_track[i]
        acc_StkTrk_err.write(str(acc_bin)+"\n")
acc_StkTrk_err.close()


acc_BDT_err = open("curves_acceptance/acceptance_vs_energy_cutBDT_err_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = err_eff_BDT[i]
        acc_BDT_err.write(str(acc_bin)+"\n")
acc_BDT_err.close()

acc_Sanity_err = open("curves_acceptance/acceptance_vs_energy_cutSanity_err_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = err_eff_sanity[i]
        acc_Sanity_err.write(str(acc_bin)+"\n")
acc_Sanity_err.close()

acc_CNN_err = open("curves_acceptance/acceptance_vs_energy_cutCNN_err_v26.txt", "w")
for i in range(0, n_points):
        acc_bin = err_eff_CNNlin[i]
        acc_CNN_err.write(str(acc_bin)+"\n")
acc_CNN_err.close()
