import numpy as np
import awkward as ak
from yahist import Hist1D, Hist2D

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.CMS])

pres = preselection.Preselection("./metadata/chunklist_2018D_zerobias.csv", "./metadata/runs_from_2544bunches.npy")
obj_list = ["wire", "comparator"]
events, lumi = pres.prepareInputs(pres.chunks[1], obj_list, addLumi=True)

cscIDs = preselection.cscIDs()

comb_s_r = [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2)]

def get_s_r(fullCSCID):
    l_fullCSCID = fullCSCID.split("_")
    s_r = l_fullCSCID[1] + "_" + l_fullCSCID[2]
    return s_r

def hists_dict_to_list(hists_dict):
    l_hists = []
    for s_r in comb_s_r:
        key_s_r = str(s_r[0]) + "_" + str(s_r[1])
        l_hists.append(hists_dict[key_s_r])
    return l_hists
    
def mergeHist(hists, savename, option="station_ring", log=False, show_counts=False):
    ## hists is a dictionary, its key is "endcap_station_ring_chamber"
    comb_s_r_tostring = [str(s_r[0]) + "_" + str(s_r[1]) for s_r in comb_s_r]
    
    #import matplotlib as mpl
    #COLOR = 'black'
    #mpl.rcParams['text.color'] = COLOR    
    
    if option == "station_ring":
        hists_s_r = {}
        for s_r in comb_s_r_tostring:
            hists_s_r[s_r] = np.sum( [ hists[key_s_r] for key_s_r in hists.keys() if get_s_r(key_s_r) == s_r ] ) 
        
        l_hists_s_r = hists_dict_to_list(hists_s_r)
        
        fig, axs = plt.subplots(4,4, figsize=(25,25))
        if type(l_hists_s_r[0]) == Hist2D:
            [l_hists_s_r[i].plot(axs[comb_s_r[i][0]-1][comb_s_r[i][1]-1], show_counts=show_counts, cmap='Blues') for i in range(len(comb_s_r))]
        else:
            [l_hists_s_r[i].plot(axs[comb_s_r[i][0]-1][comb_s_r[i][1]-1], show_counts=show_counts) for i in range(len(comb_s_r))]
        [axs[comb_s_r[i][0]-1][comb_s_r[i][1]-1].set_title("station " + str(comb_s_r[i][0]) + " ring " + str(comb_s_r[i][1])) for i in range(len(comb_s_r))]
        if log == True:
            [axs[comb_s_r[i][0]-1][comb_s_r[i][1]-1].set_yscale('log') for i in range(len(comb_s_r)) ]
        
        fig.savefig(savename+".pdf", bbox_inches='tight')
        fig.savefig(savename+".png", bbox_inches='tight')


def mask_by_cscID(objs, cscid_s):
    cscid = cscid_s.split("_")
    mask_endcap = objs.ID_endcap == int(cscid[0])
    mask_station = objs.ID_station == int(cscid[1])
    mask_ring = objs.ID_ring == int(cscid[2])
    mask_chamber = objs.ID_chamber == int(cscid[3])
    mask_cscid = (mask_endcap & mask_station & mask_ring & mask_chamber)
    return mask_cscid


def doComparators(comparators):
    h_comparator_time = {}
    for cscID in cscIDs:
        mask_cscid = mask_by_cscID(comparators, cscID)
        h_comparator_time[cscID] = Hist1D(ak.flatten(comparators.timeBin[mask_cscid]), bins = np.linspace(0,10,11), label="comparator time")  
    return h_comparator_time
    
def dowires(wires):
    h_wire_time = {}
    for cscID in cscIDs:
        mask_cscid = mask_by_cscID(wires, cscID)
        h_wire_time[cscID] = Hist1D(ak.flatten(wires.timeBin[mask_cscid]), bins = np.linspace(0,16,17), label="wire time")  
    return h_wire_time

%time
import numba as nb

@nb.jit
def do_wire_strip(wires, strips):
    '''
    for each layer that has one wire hits in a time window [2-10]
    how many comparator hits got in the same layer in a time window [1-9]
    '''
    nbins = 11
    nStripHits_cntr = np.zeros(nbins, dtype=np.int64)
    
    nEvts = len(wires)
    for i in range(nEvts):
        nWireHits = np.zeros(6, dtype = np.int64)
        nStripHits = np.zeros(6, dtype = np.int64)
        # wire loop
        for j in range(len(wires[i])):
            if (wires[i][j].timeBin >= 2) and (wires[i][j].timeBin <= 10):
                nWireHits[wires[i][j].ID_layer - 1] += 1
        for j in range(len(strips[i])):
            if nWireHits[strips[i][j].ID_layer - 1] == 1:
                if strips[i][j].timeBin < 1: continue
                #if (strips[i][j].timeBin >= 2) and (strips[i][j].timeBin <= 10):
                nStripHits[strips[i][j].ID_layer - 1] += 1
        for j in range(6):
            # check wire hit container one by one
            # if for layer x, number of wire hit == 1
            if nWireHits[j] == 1:
                # check number of strip hits, if it is n, then n-1 th bin of nStripHits_cntr++
                nStripHits_cntr[nStripHits[j]] += 1
    return nStripHits_cntr
    
nbins = 11
n_striphits_cntr = {
   "1_1": np.zeros(nbins, dtype = np.int64),
   "1_2": np.zeros(nbins, dtype = np.int64), 
   "1_3": np.zeros(nbins, dtype = np.int64), 
   "2_1": np.zeros(nbins, dtype = np.int64), 
   "2_2": np.zeros(nbins, dtype = np.int64), 
   "3_1": np.zeros(nbins, dtype = np.int64), 
   "3_2": np.zeros(nbins, dtype = np.int64), 
   "4_1": np.zeros(nbins, dtype = np.int64), 
   "4_2": np.zeros(nbins, dtype = np.int64) 
}
    
#cscID = "1_1_1_5"
for cscID in cscIDs:
    station = cscID.split("_")[1]
    ring = cscID.split("_")[2]

    if ring == '4': continue
    
    selectedWires = events.firedWireDigis[mask_by_cscID(events.firedWireDigis, cscID)]
    selectedStrips = events.comparatorDigis[mask_by_cscID(events.comparatorDigis, cscID)]

    n_cntr = do_wire_strip(selectedWires, selectedStrips)

    
    n_striphits_cntr[station + "_" + ring] += n_cntr


fig, axs = plt.subplots(4,4, figsize=(25,25))
for key in n_striphits_cntr.keys():
    x,y = key.split("_")
    hist = Hist1D(np.array([]), bins=np.linspace(0,10,11))
    #print (hist.counts)
    for i in range(len(hist.counts)):
        hist.counts[i] = n_striphits_cntr[key][i]
    hist.plot(axs[int(x)-1][int(y)-1], show_counts=True)
    
plt.savefig("/home/users/hmei/public_html/cscbkg/ncomparatorHits_if_has_wHit.pdf", bbox_inches='tight')
plt.savefig("/home/users/hmei/public_html/cscbkg/ncomparatorHits_if_has_wHit.png", bbox_inches='tight')
