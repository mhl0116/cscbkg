import pandas as pd
import dask.dataframe as dd
from yahist import Hist1D
from yahist import Hist2D
import numpy as np
import os
from dask.distributed import Client

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.CMS])

import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('pylogs/plotter.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

import decorators
import helper_plotter

#logging.basicConfig(filename='pylogs/plotter.log', level=logging.INFO,
#                    format='%(asctime)s:%(levelname)s:%(message)s')

class Plotter:

    station_rings = [(1,1), (1,2), (1,3), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2)]

    wg_bound = {(1,1): 16, (1,2): 24, (1,3): 16,
                (2,1): 24, (2,2): 24,
                (3,1): 24, (3,2): 24, 
                (4,1): 24, (4,2): 24 }

    bxs = (100, 3500)
    #bxs = (62,62)
    areas_up = {(1,1): 3965.0, (1,2): 9060.0, (1,3): 7392.0,
                (2,1): 14946.0, (2,2): 22344, 
                (3,1): 13474.0, (3,2): 22344,
                (4,1): 11992.0, (4,2): 22344}

    areas_dn = {(1,1): 1228.0, (1,2): 3024, (1,3): 5392,
                (2,1): 2187.0, (2,2): 9168,
                (3,1): 2448, (3,2): 9172,
                (4,1): 2714, (4,2): 9176}

    wTimeBins = {
        (1,1): np.arange(1,15),
        (1,2): np.array([1,2,3,13,14]),
        (1,3): np.array([1,2,3,13,14]),
        (2,1): np.arange(1,14),
        (2,2): np.array([1,2,3,4,13,14]),
        (3,1): np.arange(1,15),
        (3,2): np.array([1,2,3,4,13,14]),
        (4,1): np.arange(1,15),
        (4,2): np.array([1,2,3,4,13,14])
    }

    k = {(1,1): 0.75, (1,2): 1, (1,3): 1,
         (2,1): 0.78, (2,2): 1,
         (3,1): 0.66, (3,2): 1,
         (4,1): 0.6, (4,2): 0.6}

    def __init__(self, filepath, debug):

        logger.info("Start set info")
        self.getBXs()
        logger.info("N colliding bx: {}".format(len(self.col_bx)))
        logger.info("N non colliding bx: {}".format(len(self.non_col_bx)))
        logger.info("N non colliding bx active: {}".format(len(self.noncol_active_bx)))
        logger.info("N non colliding bx non active: {}".format(len(self.noncol_nonactive_bx)))

        self.filepath = filepath
        self.debug = debug
        self.special_selection_configs = {}

        self.hists_upper = {}
        self.hists_lower = {}

        self.h_rate_vs_lumi = {}
        self.h_rate_vs_lumi_colbx = {}
        self.h_rate_vs_lumi_noncolbx = {}
        self.h_rate_vs_lumi_weighted = {}
        #filepath = "data/intermediateData/"
        #filepath = "data/intermediateData_cleanTrack/"
        #filepath = "data/intermediateData_cleanTrack_singleMu/"

    def getBXs(self):

        with open("./data/collidingBunches_singleMu_2544.npy", 'rb') as f:
            self.col_bx = np.load(f)

        allbx = np.arange(1,3565)
        self.non_col_bx = allbx[~np.in1d(allbx, self.col_bx)]

        with open("./data/noncollidingBunches_active_2544.npy", 'rb') as f:
            self.noncol_active_bx = np.load(f)
        with open("./data/noncollidingBunches_nonactive_2544.npy", 'rb') as f:
            self.noncol_nonactive_bx = np.load(f)

        '''
        non_col_bx_measure = []
        non_col_bx_nomeasure = []

        for i in range(len(non_col_bx)):
            bx = non_col_bx[i]
            if ((bx-7) in load_bx) or ((bx+7) in load_bx):
                non_col_bx_measure.append(bx)
            else:
                non_col_bx_nomeasure.append(bx)

        print (len(non_col_bx_measure), len(non_col_bx_nomeasure))

        with open('./metadata/noncollidingBunches_active_2544.npy', 'wb') as f:
            np.save(f, np.array(non_col_bx_measure))
        with open('./metadata/noncollidingBunches_nonactive_2544.npy', 'wb') as f:
            np.save(f, np.array(non_col_bx_nomeasure))
        '''

    def getHist(self, histname):
        if histname == "h_rate_vs_lumi_weighted":
            return self.h_rate_vs_lumi_weighted

    def fitHist(self, **kwargs):
        if "histname" not in kwargs:
            logger.debug("histname is not in kwargs")
        if "csctype" not in kwargs:
            logger.debug("csctype is not in kwargs")
        if "fitrange" not in kwargs:
            logger.debug("fitrange is not in kwargs")
        if "func" not in kwargs:
            logger.debug("func is not in kwargs")
        if "ax" not in kwargs:
            logger.debug("ax is not in kwargs")

        histname = kwargs["histname"]
        csctype = kwargs["csctype"]
        xmin, xmax = kwargs["fitrange"] 
        func = kwargs["func"]
        ax = kwargs["ax"]

        fitres = self.getHist(histname)[csctype].restrict(xmin,xmax).fit(func, ax=ax)
        return fitres

    def makeHists_upper_lower(self):
        logger.info("make hists for all CSC types")
        for s_r in self.station_rings:
            csc_type = str(s_r[0]) + "_" + str(s_r[1])
            filename = os.path.join(self.filepath, '*ME_' + csc_type + '_*csv')
            
            ddf = dd.read_csv(filename)

            if self.debug == True:
                ddf = dd.from_pandas(ddf.head(), npartitions=1)
            #bx_min, bx_max = self.bxs 

            logger.info("make hists for ME{} in upper region".format(csc_type))
            
            # trigger on lower region, check on higher region
            wgs_trigger = (0, self.wg_bound[s_r] - 2) # protect against alct close to bound
            wgs_check = (self.wg_bound[s_r], 200) 
            area = self.areas_up[s_r]
            wTBins = self.wTimeBins[s_r]
            self.hists_upper[s_r] = helper_plotter.makeHists((ddf, self.bxs, wgs_trigger, wgs_check, area, wTBins), **self.special_selection_configs)

            logger.info("make hists for ME{} in lower region".format(csc_type))
            
            # trigger on upper region, check on lower region
            wgs_trigger = (self.wg_bound[s_r] + 2, 200) 
            wgs_check = (0, self.wg_bound[s_r]) # protect against alct close to bound
            area = self.areas_dn[s_r]
            wTBins = self.wTimeBins[s_r]
            self.hists_lower[s_r] = helper_plotter.makeHists((ddf, self.bxs, wgs_trigger, wgs_check, area, wTBins), **self.special_selection_configs)

    def assembleHists_upper_lower(self):
        logger.info("assemble plots from upper and lower part of csc")
        for s_r in self.station_rings:
            n_upper = self.hists_upper[s_r]["h_lumi_allEvts"].integral
            n_lower = self.hists_lower[s_r]["h_lumi_allEvts"].integral
            #print (n_upper, n_lower, self.areas_up[s_r], self.areas_dn[s_r])
            self.h_rate_vs_lumi[s_r] = (self.hists_upper[s_r]["h_rate_vs_lumi"]*self.areas_up[s_r] + self.hists_lower[s_r]["h_rate_vs_lumi"]*self.areas_dn[s_r])/(self.areas_up[s_r]+self.areas_dn[s_r])
            self.h_rate_vs_lumi_colbx[s_r] = (self.hists_upper[s_r]["h_rate_vs_lumi_colbx"]*self.areas_up[s_r] + self.hists_lower[s_r]["h_rate_vs_lumi_colbx"]*self.areas_dn[s_r])/(self.areas_up[s_r]+self.areas_dn[s_r])
            self.h_rate_vs_lumi_noncolbx[s_r] = (self.hists_upper[s_r]["h_rate_vs_lumi_noncolbx"]*self.areas_up[s_r] + self.hists_lower[s_r]["h_rate_vs_lumi_noncolbx"]*self.areas_dn[s_r])/(self.areas_up[s_r]+self.areas_dn[s_r])
            self.h_rate_vs_lumi_weighted[s_r] = (self.h_rate_vs_lumi_colbx[s_r]*len(self.col_bx) + self.h_rate_vs_lumi_noncolbx[s_r]*len(self.noncol_active_bx) + self.h_rate_vs_lumi_colbx[s_r]*self.k[s_r]*len(self.noncol_nonactive_bx))/3564
            #h_rate_vs_lumi_weighted[s_r] = (h_rate_vs_lumi_colbx[s_r]*2544 + h_rate_vs_lumi_noncolbx[s_r]*504 + h_rate_vs_lumi_colbx[s_r]*k[s_r]*516)/(2544+504+516)

    def plot(self, plotConfig, doFit=False):

        from datetime import date
        today = date.today()
            
        #endcap_dict = {"1":"+", "2":"-"}
        fig1, axs1 = plt.subplots(4,4, figsize=(25,25))
        histnames = self.hists_lower[(1,1)].keys()

        for s_r in self.station_rings:
            #csctype = str(comb[0]) + "_" + str(comb[1])
            
            s = s_r[0]
            r = s_r[1]
            self.getHist(plotConfig["histname"])[s_r].plot(axs1[s-1][r-1], show_errors=True)

            plotConfig["csctype"] = s_r
            plotConfig["ax"] = axs1[s-1][r-1] 

            csctypelabel = "station " + str(s) + " ring " + str(r)
            savename_inner = plotConfig["histname"] + "_" + csctypelabel.replace(" ", "_") + "_" + plotConfig["savetag"] + "_" + str(today) + ".json"
            hist_json = self.getHist(plotConfig["histname"])[s_r].to_json("./hists/" + savename_inner)
            if doFit == True:
                fitres = self.fitHist(**plotConfig)
                with open('./fitresults/' + savename_inner, 'w') as fp:
                    json.dump(fitres['params'], fp)
            #self.h_rate_vs_lumi_weighted[s_r].plot(axs1[s-1][r-1], show_errors=True)
            #self.h_rate_vs_lumi_weighted[s_r].restrict(6,18).fit("a*x+b", ax=axs1[s-1][r-1])
            #h_rate_vs_lumi_colbx[s_r].plot(axs1[s-1][r-1], show_errors=True)
            #h_rate_vs_lumi_colbx[s_r].restrict(6,18).fit("a*x+b", ax=axs1[s-1][r-1])
            axs1[s-1][r-1].set_title(csctypelabel)
            #axs1[s-1][r-1].legend()

            
        if "xlabel" in plotConfig.keys():
            axs1[3][0].set_xlabel(plotConfig["xlabel"])
            axs1[3][1].set_xlabel(plotConfig["xlabel"])
            axs1[3][2].set_xlabel(plotConfig["xlabel"])
            axs1[3][3].set_xlabel(plotConfig["xlabel"])
        if "ylabel" in plotConfig.keys():
            axs1[0][0].set_ylabel(plotConfig["ylabel"])
            axs1[1][0].set_ylabel(plotConfig["ylabel"])
            axs1[2][0].set_ylabel(plotConfig["ylabel"])
            axs1[3][0].set_ylabel(plotConfig["ylabel"])

        savename = plotConfig["histname"] + "_" + plotConfig["savetag"] + "_" + str(today)
        savedir = "/home/users/hmei/public_html/cscbkg/" 
        logger.info("Save {} in path: {}".format(savename, savedir))

        fig1.savefig(savedir + savename + ".pdf", bbox_inches='tight')
        fig1.savefig(savedir + savename + ".png", bbox_inches='tight')
        #fig1.savefig("/home/users/hmei/public_html/cscbkg/wHit_vs_lumi_colbx_" + str(today) + ".pdf", bbox_inches='tight')
        #fig1.savefig("/home/users/hmei/public_html/cscbkg/wHit_vs_lumi_colbx_" + str(today) + ".png", bbox_inches='tight')
        plt.close()

    def plot_composite(self, whichplot, plotConfig):

        for s_r in self.station_rings:
            s = s_r[0]
            r = s_r[1]

            csc_name = "ME" + str(s) + "/" + str(r)

            if whichplot == "rate_vs_bxID":
                hist1_lower = self.hists_lower[s_r]["h_rate_vs_bxID"]
                hist1_upper = self.hists_upper[s_r]["h_rate_vs_bxID"]
                hist1 = (hist1_upper*self.areas_up[s_r] + hist1_lower*self.areas_dn[s_r])/(self.areas_up[s_r]+self.areas_dn[s_r])
                #hist2 = hists_lower[s_r]["h_bx_allEvts"]
                hist2_lower = self.hists_lower[s_r]["h_bx_perEvts"]
                hist2_upper = self.hists_upper[s_r]["h_bx_perEvts"]
                hist2 = (hist2_upper*self.areas_up[s_r] + hist2_lower*self.areas_dn[s_r])/(self.areas_up[s_r]+self.areas_dn[s_r])

                helper_plotter.plot_rate_vs_bx((hist1,hist2), csc_name, **plotConfig)


#c = Client(memory_limit='4GB', n_workers=20, threads_per_worker=1)
c = Client("tcp://127.0.0.1:3755")
logger.info("connect to client {}".format(c))
plotter = Plotter("data/intermediateData_cleanTrack/", debug=False)

doParticleRate = True

plotter.special_selection_configs["doCluster"] = doParticleRate 

plotter.makeHists_upper_lower()
plotter.assembleHists_upper_lower()

if doParticleRate:
    special_savetag = "particlerate"
    plotter.plot_composite("rate_vs_bxID", {"ylabel":"particle rate ($Hz/cm^2$)"})
else:
    special_savetag = "hitrate"
    plotter.plot_composite("rate_vs_bxID", {"ylabel":"hit rate ($Hz/cm^2$)"})

plotConfig = {"histname":"h_rate_vs_lumi_weighted", "fitrange": [6,18], "func": "a*x+b", "savetag": special_savetag, "xlabel": "$10^{33} Hz/cm^{2}$", "ylabel": "$Hz/cm^{2}$"}
plotter.plot(plotConfig, doFit=True)

'''
###
# for rate_vs_lumi
###

for i in range(1, 7):
#    for pos in ["isForward", "isBackward"]:
#        special_savetag = "layer_" + str(i) + "_" + pos
        special_savetag = "layer_" + str(i)

        plotter = Plotter("data/intermediateData_cleanTrack/", debug=False)
        plotter.special_selection_configs["layer"] = i 
        #plotter.special_selection_configs[pos] = True 
        plotter.makeHists_upper_lower()
        plotter.assembleHists_upper_lower()

        plotConfig = {"histname":"h_rate_vs_lumi_weighted", "fitrange": [6,18], "func": "a*x+b", "savetag": special_savetag}
        plotter.plot(plotConfig, doFit=True)
'''


