import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak
from yahist import Hist1D,Hist2D

import mplhep as hep
plt.style.use([hep.style.CMS])

import csv
import ast

import logging

import decorators
import helper_nb

logging.basicConfig(filename='pylogs/preselection.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def cscIDs():
    
    ncsc_per_type = {"1_1": 36, "1_2": 36, "1_3": 36, "1_4": 36,
                     "2_1": 18, "3_1": 18, "4_1": 18,
                     "2_2": 36, "3_2": 36, "4_2": 36}
    
    #masks_endcap = [cscObj.ID_endcap == i for i in range(1,3)]
    #masks_station = [cscObj.ID_station == i for i in range(1,5)]
    #masks_ring = [cscObj.ID_ring == i for i in range(1,5)]
    #masks_chamber = [cscObj.ID_chamber == i for i in range(1,37)]
   
    # get all cscID (e,s,r,c) combination
    cscID_comb = []
    for endcap in range(1,3):
        for station in range(1,5):
            for ring in range(1,5):
                if station > 1 and ring > 2: continue
                for chamber in range(1, ncsc_per_type[str(station) + "_" + str(ring)]+1):
                                     #cscID_comb.append((endcap, station, ring, chamber))
                                     cscID_comb.append('_'.join([str(endcap), str(station), str(ring), str(chamber)]))

    return cscID_comb #, mask_cscID

class Preselection:

    station_rings = [(1,1), (1,2), (1,3), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2)]
    fn_lumi_2018D = "./metadata/output_byls_2018ABCD_hzub_clean.csv"

    def __init__(self, chunklist, runlist):
        #'./metadata/chunklist_2018D_zerobias.csv'
        self.chunklist = chunklist
        #"./metadata/runs_from_2544bunches.npy", 'rb'
        self.runlist = runlist 
        self.df_lumi = pd.read_csv(self.fn_lumi_2018D) 

        logging.info("Start preselection")
        logging.info('chunklist: {}'.format(self.chunklist))
        logging.info('total number of chunks {}, total number of events {}'.format(len(chunklist), np.sum([chunk[2]-chunk[1] for chunk in self.chunks])))
        logging.info('runlist: {}'.format(self.runlist))

    @property
    def chunks(self):
        with open(self.chunklist, newline='') as f:
            reader = csv.reader(f)
            chunks = list(reader)
        f.close()

        return [ast.literal_eval(i) for i in chunks[0]]

    @chunks.setter
    def chunks(self, chunklist): 
        self.chunklist = chunklist

    @property
    def selected_runs(self):
        with open(self.runlist, "rb") as f:
            selected_runs = np.load(f)
        f.close()
        return selected_runs
        
    @selected_runs.setter
    def selected_runs(self, runlist): 
        self.runlist = runlist 

    @decorators.my_timer
    def prepareInputs(self, chunk, obj_list, addLumi=False):
        fname,entrystart,entrystop = chunk 
        f = uproot.open(fname)
        t = f["cscRootMaker/Events"]
        
        keys_to_save = ["Run", "Event", "LumiSect", "BunchCrossing"]
        muon_keys = t.keys(filter_name="muons_*")
        
        keys_for_filter = keys_to_save + muon_keys
        
        if "muon" in obj_list:
            keys_to_save = keys_to_save + muon_keys
        
        if "segment" in obj_list:
            seg_keys = t.keys(filter_name="cscSegments_*")
            keys_to_save = keys_to_save + seg_keys
        
        if "rechit" in obj_list:
            rh_keys = t.keys(filter_name="recHits2D_*")
            keys_to_save = keys_to_save + rh_keys
            
        if "wire" in obj_list:
            wire_keys = t.keys(filter_name="firedWireDigis_*")
            keys_to_save = keys_to_save + wire_keys
            
        if "strip" in obj_list:
            strip_keys = t.keys(filter_name="firedStripDigis_*")
            keys_to_save = keys_to_save + strip_keys

        if "comparator" in obj_list:
            comparator_keys = t.keys(filter_name="comparatorDigis_*")
            keys_to_save = keys_to_save + comparator_keys

        if "alct" in obj_list:
            alct_keys = t.keys(filter_name="alct_*")
            keys_to_save = keys_to_save + alct_keys
            
        if "clct" in obj_list:
            clct_keys = t.keys(filter_name="clct_*")
            keys_to_save = keys_to_save + clct_keys

        if "lct" in obj_list:
            lct_keys = t.keys(filter_name="correlatedLct_*")
            keys_to_save = keys_to_save + lct_keys
            
        if "tmb" in obj_list:
            tmb_keys = t.keys(filter_name="tmb_*")
            keys_to_save = keys_to_save + tmb_keys
            
        events_for_filter = t.arrays(keys_for_filter, library="ak", how="zip", entry_start=entrystart, entry_stop=entrystop) 
        events = t.arrays(keys_to_save, library="ak", how="zip", entry_start=entrystart, entry_stop=entrystop) 

        logging.debug("number of events before selection: {}".format(len(events))) 
        logging.debug("number of run ranges selected: {}".format(len(self.selected_runs)))

        nevt = len(events)
        
        if addLumi:
            df_lumi_select = self.df_lumi.loc[ (self.df_lumi.run <= ak.max(events.Run)) & (self.df_lumi.run >= ak.min(events.Run))]
            instLumi = [df_lumi_select.loc[ (df_lumi_select["run"] == events.Run[i]) & 
                                            (df_lumi_select["lumisect"] == events.LumiSect[i] ) ]["deliver"].to_numpy() for i in range(nevt)]

            ## this is protect against missing lumi in json file
            ## 1000 is to change to 1e33
            instLumi_array = np.array([l[0]/1000.0 if len(l) == 1 else -1 for l in instLumi])
            #print (instLumi_array)
            events.InstLumi = ak.Array(instLumi_array)
        
        mask_run = np.array([False]*nevt)
        masks = [(events_for_filter.Run >= self.selected_runs[i][0]) & (events_for_filter.Run <= self.selected_runs[i][1]) for i in range(len(self.selected_runs))]
        for mask in masks:
            mask_run = np.any([mask_run, mask],axis=0)
            
        # mask events without muon segment in CSC
        museg_endcap = events_for_filter.muons_cscSegmentRecord_endcap
        mask_noMuCSCSeg = ak.sum( ak.sum(museg_endcap, axis=2) , axis=1) == 0
        ##nmucscseg = ak.num(events.muons[mask_muCSCSeg])
        #evts_noMuCSCSeg = events[mask_noMuCSCSeg]
        
        if addLumi:
            return events[mask_run & mask_noMuCSCSeg], instLumi_array[mask_run & mask_noMuCSCSeg]
            #return events[mask_run], instLumi_array[mask_run]
        else:
            return events[mask_run & mask_noMuCSCSeg]



    def makeData(self, events, lumis, cscID_s_r):
        '''
        1. find for each cscID (unique in each event), the keyWG of alct0
        2. save all WG info for this chamber
        3. also event level id: event/run/bx/instLumi
        '''
        
        #mask_cscid_wg = mask_by_cscID(events.firedWireDigis, cscID)
        mask_cscid_wg = self.mask_by_station_ring(events.firedWireDigis, cscID_s_r)
        wgs_select = events.firedWireDigis[mask_cscid_wg]
        
        #mask_cscid_alct = mask_by_cscID(events.alct, cscID)
        mask_cscid_alct = self.mask_by_station_ring(events.alct, cscID_s_r)
        alcts_select = events.alct[mask_cscid_alct]
        
        mask_evt = ak.num(alcts_select) >= 1
        
        data_perEvent, lumi_perEvent = helper_nb.makeData_perEvent(events[mask_evt], wgs_select[mask_evt], alcts_select[mask_evt], lumis[mask_evt])

        column_names = ["event", "run", "bx", "endcap", "chamber", "wgNum", "wgLayer", "wgTime", "wgNWireTimeBins", "nalct", "alct0_keyWG", "adjacentWG"]
        row_names = [i+1 for i in range(len(data_perEvent))]
        
        df_per_event = pd.DataFrame(data=data_perEvent, index=row_names, columns=column_names)
        df_per_event["instLumi"] = lumi_perEvent
        df_per_event = df_per_event.set_index("event")
        
        return df_per_event

    def mask_by_cscID(self, objs, cscid_s):
        cscid = cscid_s.split("_")
        mask_endcap = objs.ID_endcap == int(cscid[0])
        mask_station = objs.ID_station == int(cscid[1])
        mask_ring = objs.ID_ring == int(cscid[2])
        mask_chamber = objs.ID_chamber == int(cscid[3])
        mask_cscid = (mask_endcap & mask_station & mask_ring & mask_chamber)
        return mask_cscid


    def mask_by_station_ring(self, objs, s_r):
        station, ring = s_r[0], s_r[1]
        mask_station = objs.ID_station == station
        mask_ring = objs.ID_ring == ring
        mask_station_ring = (mask_station & mask_ring)
        return mask_station_ring

#pres = Preselection("./metadata/chunklist_2018D_zerobias.csv","./metadata/runs_from_2544bunches.npy")
#logging.debug("number of runs selected: {}".format(len(pres.selected_runs)))
#
#obj_list = ["wire", "strip", "alct", "clct"]
#
#test_chunk = (pres.chunks[1][0], 0, 100)
#events, lumis = pres.prepareInputs(test_chunk, obj_list, addLumi=True)
#logging.debug("dump events")
#logging.debug("number of events preselected: {}".format(len(events)))
#df_per_event = pres.makeData(events, lumis, (1,1))
#print (df_per_event.head())


#def doCSCAll(args): #, runs_selected, index):
#    
#    index = args[0].split(".")[1].split('_')[-1]
#    events, lumis = prepareInputs(args, runs_selected, ["wire", "alct"], addLumi=True)
#    for cscID_s_r in station_rings:
#        df_per_event = makeData(events, lumis, cscID_s_r)
#        
#        csc_type = str(cscID_s_r[0]) + "_" + str(cscID_s_r[1])
#        savename = "./data/intermediateData_cleanTrack_singleMu/ME_" + csc_type + "_WG_index_" + str(index) + ".csv" 
#        print (savename)
#        from subprocess import call
#        import os
#        if os.path.exists(savename):
#            os.remove(savename)
#        df_per_event.to_csv(savename)
#
#
#
