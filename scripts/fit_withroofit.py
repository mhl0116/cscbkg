import ROOT
import numpy as np
import root_numpy

import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('../pylogs/bkgfit_roofit.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

class BackgroundFitRF():

    fullbx = 3564

    def __init__(self, histname, fitrange):

        with open(histname) as f:
              hist_json = json.load(f)

        self.hist_counts = hist_json["_counts"]
        self.hist_edges = hist_json["_edges"]
        self.fitrange = fitrange
        
        logger.info("read hist: \"{}\"".format(histname))
        logger.debug("type of counts and edges are: {}, {}".format(type(self.hist_counts), type(self.hist_edges)))
        logger.info("counts: {}".format(self.hist_counts[:100]))
        logger.info("edges: {}".format(self.hist_edges[:100]))
        logger.info("fitrange is: {}".format(self.fitrange))


    @property
    def nphist(self):
        np_counts = np.array(self.hist_counts[self.fitrange[0]:self.fitrange[1]], dtype=np.float64)
        np_edges = np.array(self.hist_edges[self.fitrange[0]:self.fitrange[1]], dtype=np.float64)
        return np.array(list(zip(np_counts, np_edges)))
        #return np.array([np_counts, np_edges]) #, dtype=[('counts', np.float64),('edges', np.float64)]) 

    def roodatahist(self, blindRange):

        nbins = self.fitrange[1] - self.fitrange[0] + 1
        h = ROOT.TH1F("rate","", nbins, self.fitrange[0], self.fitrange[1])
        for i in range(len(self.nphist)):
            h.SetBinContent(i+1, self.nphist[i][0])

        w = ROOT.RooWorkspace("w")
        w.factory("bxID[" + str(self.fitrange[0]) + "," + str(self.fitrange[1]-1)+ "]")
        h_data = ROOT.RooDataHist("h_data","", ROOT.RooArgList(w.var("bxID")), h)
        h_data.Print()

        w.var("bxID").setRange("SL", self.fitrange[0], blindRange[0])
        w.var("bxID").setRange("SU", blindRange[1], self.fitrange[1])
        w.var("bxID").setRange("blind",blindRange[0],blindRange[1])
        w.var("bxID").setRange("full", self.fitrange[0], self.fitrange[1])

        w.factory("Exponential:shape(bxID, tau[-2,-10,0])")
        w.factory("ExtendPdf:shape_ext(shape, nevt[100,0,10000], 'full')")

        w.pdf("shape_ext").fitTo(h_data, ROOT.RooFit.Range("SL,SU"), ROOT.RooFit.Extended(True), ROOT.RooFit.PrintLevel(-1))

        frame = w.var("bxID").frame()

        h_data.plotOn(frame, ROOT.RooFit.CutRange("SL"));
        h_data.plotOn(frame, ROOT.RooFit.CutRange("SU"));
        w.pdf("shape_ext").plotOn(frame, ROOT.RooFit.Range(self.fitrange[0], self.fitrange[1]))

        c1 = ROOT.TCanvas("c1", "c1", 800, 800)
        dummy = ROOT.TH1D("dummy","dummy",1,self.fitrange[0], self.fitrange[1])
        dummy.SetMinimum(0)
        yMax = np.max(self.nphist[:,0])*1.5
        dummy.SetMaximum(yMax)
        dummy.SetLineColor(0)
        dummy.SetMarkerColor(0)
        dummy.SetLineWidth(0)
        dummy.SetMarkerSize(0)
        dummy.GetYaxis().SetTitle("Rate")
        dummy.GetYaxis().SetTitleOffset(1.3)
        dummy.GetXaxis().SetTitle("BX ID")
        dummy.Draw()

        frame.Draw("same")

        #latex = TLatex()
        #latex.SetNDC()
        #latex.SetTextSize(0.6*c1.GetTopMargin())
        #latex.SetTextFont(42)
        #latex.SetTextAlign(11)
        #latex.SetTextColor(1)

        #latex.DrawLatex(0.4, 0.85, (tag.split("_"))[-2] + "_" + (tag.split("_"))[-1])
        #latex.DrawLatex(0.4, 0.78, "nEvents = " + str(round(w.var("nevt").getVal(), 3) ) + " #pm " + str(round(w.var("nevt").getError(), 3) ))
        #latex.DrawLatex(0.4, 0.71, "nEvents_blind = " + str(round(nevt_sigR.getVal()*w.var("nevt").getVal(), 3) ) )
        #latex.DrawLatex(0.4, 0.64, "#tau = " + str(round(w.var("tau").getVal(), 3) ) + " #pm " + str(round(w.var("tau").getError(), 3) ))

        c1.SaveAs("/home/users/hmei/public_html/cscbkg/fit2.png")
        c1.SaveAs("/home/users/hmei/public_html/cscbkg/fit2.pdf")

histname = "../hists/rate_vs_bx_0_3564_ME1_1_2021-03-03.json"

bkgfitrf = BackgroundFitRF(histname, (165,195))
#bkgfitrf = BackgroundFitRF(histname, (165,172))
nphist = bkgfitrf.nphist
bkgfitrf.roodatahist([173,189])
#logger.debug("2D numpy array to be turned into root tree: {}".format(nphist))
#nphist.dtype = [('counts', np.float64),('edges', np.float64)]
#tree = root_numpy.array2tree(nphist)
#tree.Print()
#tree.Scan("counts:edges")
