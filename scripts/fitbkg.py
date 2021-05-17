from yahist import Hist1D
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.CMS])

import json
import logging

logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('../pylogs/bkgfit.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

class BackgroundFit():

    fullbx = 3564
    from datetime import date
    today = date.today()

    def __init__(self, histname, colbxs):

        self.hist = Hist1D.from_json(histname)
        with open(colbxs, 'rb') as f:
            self.col_bx = np.load(f)

        logger.info("read hist: \"{}\" and colbxs: \"{}\"".format(histname, colbxs))
        logger.info("{}".format(self.hist))
        logger.info("{}".format(self.col_bx))

        self.smallGaps = self.findGap("small")
        self.largeGaps = self.findGap("large")
        self.trains = self.findTrain()

        self.fullOrbitCorr = (0,0)

    def findTrain(self):
        '''
        save all bunch trains in a format [(train_bx_start, train_bx_end), (), ...]
        '''
        alltrains = [(self.col_bx[0], self.col_bx[0])]
        for i in range(1, len(self.col_bx)):
            this_bx_id = self.col_bx[i]
            if this_bx_id - alltrains[-1][1] == 1:
                alltrains[-1] = (alltrains[-1][0], this_bx_id)
            else:
                alltrains.append((this_bx_id, this_bx_id))

        logger.info("number of trains: {}".format(len(alltrains)))
        logger.info("all trains: {}".format(alltrains))
        return alltrains

    def findGap(self, gaptype):
        '''
        find fill gap that is either small or large (in view of CSC anode readout)
        small: gap size <= 16, large: gap size > 16
        '''

        # 62 is because that is the first filled bx in LHC 2018 Run 
        firstColBX = 62
        allgaps = []
        if gaptype == "large":
            allgaps.append((0, firstColBX-1))
        logger.info("start to find {} gap".format(gaptype))

        gapEdge_low, gapEdge_high = firstColBX, firstColBX 
        for i in range(len(self.col_bx)-1):
            bxdiff = self.col_bx[i+1] - self.col_bx[i]
            if (bxdiff > 1):

                if (gaptype == "small") and (bxdiff <= 16):
                    newgap = (self.col_bx[i]+1, self.col_bx[i+1]-1)
                    allgaps.append(newgap)
                if (gaptype == "large") and (bxdiff > 16):
                    newgap = (self.col_bx[i]+1, self.col_bx[i+1]-1)
                    allgaps.append(newgap)

        logger.info("number of {} gaps: {}".format(gaptype, len(allgaps)))
        logger.info("all {} gaps: {}".format(gaptype, allgaps))
        return allgaps

    def plot_averaged_rate_per_train(self, savedir, tag):
        
        import copy
        hist_copy = copy.deepcopy(self.hist)
        for i in range(len(hist_copy.counts)):
            if i not in self.col_bx:
                hist_copy.counts[i]  =  0

        fig, ax = plt.subplots(1,1, figsize=(30,10))
        self.hist.plot(ax, show_errors = True, color="orange")
        hist_copy.plot(ax, show_errors = True)
        fitres = hist_copy.fit("a*x+b", ax=ax)

        logger.info("full orbit correction, fit result: {}".format(fitres))

        self.fullOrbitCorr = (fitres['params']['a']['value'], fitres['params']['b']['value'])
        logger.info("full orbit correction, a: {}, b: {}".format(self.fullOrbitCorr[0], self.fullOrbitCorr[1]))

        ax.set_xlabel('BX ID')
        ax.set_ylabel('rate')
        ax.set_ylim(0,800)
        #ax.set_xlim(200,600)

        savename = "averaged_rate_per_train_" + str(self.today) + "_" + tag
        self.savefit(plt, savename, savedir)

    def plot_gaps_merge(self, savedir, tag):
        '''
        plot rate vs bxID, in a bxID window defined by gap range [bx1,bx2]
        plot range is: [bx1-16, bx+16], plot for both small and big gap
        merge all gaps in one plot, after correcting rate vs BXID trend
        '''
        fig, ax = plt.subplots(2,1,figsize=(12,20))

        '''
        1. make separate hist for all gaps
        2. correct
        3. merge 
        4. plot
        5. fit
        '''
        self.hist._counts = np.nan_to_num(self.hist.counts)
        hists_smallgap = [self.hist.restrict(self.smallGaps[i][0] - 16, self.smallGaps[i][1] + 16) for i in range(1, len(self.smallGaps))]
        hists_largegap = [self.hist.restrict(self.largeGaps[i][0] - 16, self.largeGaps[i][1] + 16) for i in range(1, len(self.largeGaps))]

        for i in range(len(hists_smallgap)):
            for j in range(len(hists_smallgap[i].counts)):
                hists_smallgap[i].counts[j] += -1*hists_smallgap[i].edges[j]*self.fullOrbitCorr[0] 
                hists_smallgap[i].edges[j+1] += -1*hists_smallgap[i].edges[0] 
            hists_smallgap[i].edges[0] += -1*hists_smallgap[i].edges[0] 

        for i in range(len(hists_largegap)):
            for j in range(len(hists_largegap[i].counts)):
                hists_largegap[i].counts[j] += -1*hists_largegap[i].edges[j]*self.fullOrbitCorr[0] 
                hists_largegap[i].edges[j+1] += -1*hists_largegap[i].edges[0] 
            hists_largegap[i].edges[0] += -1*hists_largegap[i].edges[0] 

        hist_smallgap = sum(hists_smallgap)/len(hists_smallgap)
        hists_largegap.pop(3) #this gap is 4 BX wider
        hists_largegap.pop(7) #this gap is 4 BX wider
        hists_largegap.pop(11) #this gap is 4 BX wider
        hist_largegap = sum(hists_largegap)/len(hists_largegap)

        hist_smallgap.plot(ax[0], histtype='step', show_errors = True, color="blue")
        hist_largegap.plot(ax[1], histtype='step', show_errors = True, color="blue")
        hist_largegap._counts[np.isnan(hist_largegap.errors)] = 0 
        hist_largegap._errors[np.isnan(hist_largegap.errors)] = 0 
        #hist_largegap.restrict(17,47).fit("m1 * t1 * np.exp(-t1 * (x-17)) + b1", ax=ax[1],  curve_fit_kwargs = {"p0":[100000,0.1,100]} )
        ## for ME31
        #fitres1 = hist_largegap.restrict(19,47).fit("m1 * t1 * np.exp(-t1 * (x-19)) + b1", ax=ax[1],  curve_fit_kwargs = {"p0":[100,0.1,100]} )
        #print (fitres1["params"]["m1"], fitres1["params"]["t1"], fitres1["params"]["b1"] )
        def f_linear(x,a,b):
            return a*x+b

        def convolve(arr, kernel):
            """Simple convolution of two arrays."""
            npts = min(arr.size, kernel.size)
            pad = np.ones(npts)
            tmp = np.concatenate((pad*arr[0], arr, pad*arr[-1]))
            out = np.convolve(tmp, kernel, mode='valid')
            noff = int((len(out) - npts) / 2)
            return out[noff:noff+npts]

        def f_composite(x, A, t):
            from scipy import signal
            #y1 = A*np.exp(-(x)/t) 
            y1 = A*np.exp(-(x)/t) 
            return signal.convolve(y1, np.heaviside(x-13,0), "valid") +200 

        def sigmoid(x, a, f, xp, k1, k2, c):
            y = a*f / (1 + np.exp(-k1*(-x+xp))) + a*(1-f)/(1 + np.exp(-k2*(-x+xp))) + c #b*np.exp(-t*x)/(1+np.exp(-k2*(x-(x0+2)))) + c
            return y

        a0 = 200 #max(hist_largegap.counts)
        f0 = 0.8
        xp0 = 15 #np.median(hist_largegap.edges)
        k10 = 1
        k20 = 0.5
        c0 = 200
        params_low = [100,0.1,14,0.5,0.04,100]
        params_high = [500,0.9,17,5,1,500]

        fitres2 = hist_largegap.restrict(1,47).fit(sigmoid, ax=ax[1], curve_fit_kwargs = {"p0":[a0,f0,xp0,k10,k20,c0], "bounds":(params_low, params_high)})

        ax[0].set_xlabel('BX ID')
        ax[0].set_ylabel('rate ($Hz/cm^2$)')
        ax[1].set_xlabel('BX ID')
        ax[1].set_ylabel('rate ($Hz/cm^2$)')

        median_rate = np.median(np.nan_to_num(self.hist.counts))
        ax[0].set_ylim(0,median_rate*2)
        ax[1].set_ylim(0,median_rate*2)

        savename = "fillgaps_merged_" + str(self.today) + "_" + tag
        self.savefit(plt, savename, savedir)

    def plot_gaps(self, smallGapIndex, largeGapIndex, savedir, tag):
        '''
        plot rate vs bxID, in a bxID window defined by gap range [bx1,bx2]
        plot range is: [bx1-16, bx+16], plot for both small and big gap
        '''
        fig, ax = plt.subplots(2,1,figsize=(12,20))

        logger.info("plot small fill gap, use small gap #{}".format(smallGapIndex))

        self.hist.plot(ax[0], histtype='step', show_errors = True, color="blue")
        small_gap_low, small_gap_high =  self.smallGaps[smallGapIndex] 
        ax[0].axvline(small_gap_low, ymin = ax[0].get_ylim()[0], ymax = ax[0].get_ylim()[1], linewidth=4, color='r')
        ax[0].axvline(small_gap_high, ymin = ax[0].get_ylim()[0], ymax = ax[0].get_ylim()[1], linewidth=4, color='r')

        logger.info("plot large fill gap, use large gap #{}".format(largeGapIndex))

        self.hist.plot(ax[1], histtype='step', show_errors = True, color="blue")
        large_gap_low, large_gap_high =  self.largeGaps[largeGapIndex] 
        ax[1].axvline(large_gap_low, ymin = ax[1].get_ylim()[0], ymax = ax[1].get_ylim()[1], linewidth=4, color='r')
        ax[1].axvline(large_gap_high, ymin = ax[1].get_ylim()[0], ymax = ax[1].get_ylim()[1], linewidth=4, color='r')
        
        ax[0].set_xlabel('BX ID')
        ax[0].set_ylabel('rate ($Hz/cm^2$)')
        ax[1].set_xlabel('BX ID')
        ax[1].set_ylabel('rate ($Hz/cm^2$)')

        ax[0].set_xlim(self.smallGaps[smallGapIndex][0] - 16, self.smallGaps[smallGapIndex][1] + 16)
        ax[1].set_xlim(self.largeGaps[largeGapIndex][0] - 16, self.largeGaps[largeGapIndex][1] + 16)

        median_rate = np.median(np.nan_to_num(self.hist.counts))
        ax[0].set_ylim(0,median_rate*1.2)
        ax[1].set_ylim(0,median_rate*1.2)

        savename = "fillgaps_smallGap_" + str(smallGapIndex) + "_largeGap_" + str(largeGapIndex) + "_" + str(self.today) + "_" + tag
        self.savefit(plt, savename, savedir)

    def savefit(self, plt, savename, savedir):

   
        dirname = "/home/users/hmei/public_html/cscbkg/" + savedir + "_" + str(self.today) + "/"
        from subprocess import call
        call('mkdir -p ' + dirname, shell=True)

        plt.savefig(dirname + savename + ".pdf", bbox_inches='tight')
        plt.savefig(dirname + savename + ".png", bbox_inches='tight')

        logger.info("plots saved at {}".format(dirname))

        call('cp /home/users/hmei/public_html/cscbkg/index.php ' + dirname, shell=True)
        call('chmod -R 755 ' + dirname, shell=True)

#histname = "../hists/rate_vs_bx_0_3564_ME1_1_2021-03-03.json"
#histname = "../hists/rate_vs_bx_0_3564_ME2_1_2021-03-03.json"
#for tag in ["ME1_1", "ME2_1", "ME3_1", "ME4_1", "ME4_2"]:
#for tag in ["ME1_1", "ME2_1", "ME4_1", "ME4_2"]:
for tag in ["ME1_1"]:
    #tag = "ME4_2"
    histname = "../hists/rate_vs_bx_0_3564_" + tag + "_2021-04-14.json"
    colbxs = "../data/collidingBunches_singleMu_2544.npy" 
    bkgfit = BackgroundFit(histname, colbxs)

    bkgfit.plot_averaged_rate_per_train("testfit", tag)
    bkgfit.plot_gaps_merge("testfit", tag)

#for i in range(19):
#    bkgfit.plot_gaps(i,i, "testfit", tag)



