import numpy as np
from yahist import Hist1D, Hist2D
import matplotlib.pyplot as plt

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('pylogs/helper_plotter.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def get_num_bx():

    allbx = np.arange(1,3565)
    with open("./metadata/collidingBunches_singleMu_2544.npy", 'rb') as f:
        col_bx = np.load(f)
    non_col_bx = allbx[~np.in1d(allbx, col_bx)]

    def get_num_collidBX(i):
        return np.count_nonzero( np.in1d( adjust_bx(np.array([i])) , col_bx) )
    def get_num_noncollidBX(i):
        return np.count_nonzero( np.in1d( adjust_bx(np.array([i])) , non_col_bx) )

    collbxs_perbx = np.array( [get_num_collidBX(i) for i in range(1, 3565)] )
    noncollbxs_perbx = np.array( [get_num_noncollidBX(i) for i in range(1, 3565)] )

    h_collbxs_perbx = Hist1D(np.array([]), bins=np.linspace(0,3600,3601))
    h_noncollbxs_perbx = Hist1D(np.array([]), bins=np.linspace(0,3600,3601))

    for i in range(3564):
        h_collbxs_perbx.counts[i+1] = collbxs_perbx[i]
        h_noncollbxs_perbx.counts[i+1] = noncollbxs_perbx[i]

    #h_collbxs_perbx
    #h_noncollbxs_perbx.counts.mean()
    return h_collbxs_perbx, h_noncollbxs_perbx

def getNWTimeBins(bx, nbxs):
    '''
    bxIDs can be colliding bunches and non-colliding bunches
    '''
    return nbxs[bx]

def get_n_bx(wTimeBins, colbxs, noncolbxs):
    
    def get_num_collidBX(i):
        return np.count_nonzero( np.in1d( adjust_bx_activeBins(np.array([i]), wTimeBins) , colbxs) )
    def get_num_noncollidBX(i):
        return np.count_nonzero( np.in1d( adjust_bx_activeBins(np.array([i]), wTimeBins) , noncolbxs) )
    
    collbxs_perbx = np.array( [get_num_collidBX(i) for i in range(1, 3565)] )
    noncollbxs_perbx = np.array( [get_num_noncollidBX(i) for i in range(1, 3565)] )
    
    return collbxs_perbx, noncollbxs_perbx

def adjust_bx(bxs):
    
    # bx_adjusted = bx + wTime - 8
    # for each event (readout), there are 16 wire time bins, with value 0-15
    # this block is to map them into bx-ID, for each event there are 16 value of readjusted (remapped) bx
    bx_adjuster = np.arange(0,16) - 8
    # https://numpy.org/doc/stable/user/basics.broadcasting.html
    bxs_adjusted = bxs[:, np.newaxis] + bx_adjuster
    
    return bxs_adjusted

def adjust_bx_activeBins(bxs, activeBins):
    
    # bx_adjusted = bx + wTime - 8
    # for each event (readout), there are 16 wire time bins, with value 0-15
    # this block is to map them into bx-ID, for each event there are 16 value of readjusted (remapped) bx
    bx_adjuster = activeBins - 8
    # https://numpy.org/doc/stable/user/basics.broadcasting.html
    bxs_adjusted = bxs[:, np.newaxis] + bx_adjuster
    
    return bxs_adjusted

def makeHists(args, **kwargs):

    with open("./metadata/collidingBunches_singleMu_2544.npy", 'rb') as f:
        col_bx = np.load(f)
    with open("./data/noncollidingBunches_active_2544.npy", 'rb') as f:
        noncol_active_bx = np.load(f)
    with open("./data/noncollidingBunches_nonactive_2544.npy", 'rb') as f:
        noncol_nonactive_bx = np.load(f)
    
    ddf = args[0]
    bx_min, bx_max = args[1] 
    wg_min_trigger, wg_max_trigger = args[2]
    wg_min_check, wg_max_check = args[3] 
    area = args[4]
    wTimeBins = args[5]
    #wTime_min, wTime_max = 
    
    logger.info("head of daskdataframe:")
    logger.info("{}".format(ddf.head()))
    logger.info("bx min: {}, bx max: {}".format(bx_min, bx_max))
    logger.info("[trigger]: wire min: {}, wire max: {}".format(wg_min_trigger, wg_max_trigger))
    logger.info("[check]: wire min: {}, wire max: {}".format(wg_min_check, wg_max_check))
    logger.info("area of this region: {}".format(area))
    logger.info("wire time bins to consider: {}".format(wTimeBins))

    mask_bx = (ddf.bx >= bx_min) & (ddf.bx <= bx_max) #& (ddf.wgLayer == 1) #& (ddf.endcap == 1) & (ddf.chamber == 1)

    mask_trigger_extra = mask_bx
    mask_check_extra = mask_bx
    if "layer" in kwargs:
        logger.info("consider only layer {}".format(kwargs["layer"]))
        mask_check_extra = mask_bx & (ddf.wgLayer == kwargs["layer"])
    if ("isForward" in kwargs) and (kwargs["isForward"] == True):
        logger.info("consider only forward chamber, use even chamber ID")
        #mask_base = mask_base & (ddf.chamber % 2 == 0)
        mask_trigger_extra = mask_bx & (ddf.chamber % 2 == 0)
        mask_check_extra = mask_check_extra & (ddf.chamber % 2 == 0)
    if ("isBackward" in kwargs) and (kwargs["isBackward"] == True):
        logger.info("consider only backward chamber, use odd chamber ID")
        mask_trigger_extra = mask_bx & (ddf.chamber % 2 != 0)
        mask_check_extra = mask_check_extra & (ddf.chamber % 2 != 0)

    #print ("print")
    #print (kwargs)
    mask_trigger = (ddf.alct0_keyWG >= wg_min_trigger) & (ddf.alct0_keyWG <= wg_max_trigger) & mask_trigger_extra #& (ddf.wgLayer == kwargs["layer"])
    mask_check = (ddf.wgNum >= wg_min_check) & (ddf.wgNum <= wg_max_check) & mask_trigger & mask_check_extra #& (ddf.adjacentWG == 0)

    if ( "doCluster" in kwargs) and kwargs["doCluster"] == True:
        mask_check = mask_check & (ddf.adjacentWG == 0)
    #mask_check = (ddf.wgNum >= wg_min_check) & (ddf.wgNum <= wg_max_check) & ((ddf.wgLayer %2) + (ddf.wgNum % 2) == 1) & mask_trigger #& (ddf.nalct < 2)
    #mask_check = (ddf.wgNum >= wg_min_check) & (ddf.wgNum <= wg_max_check) & (((ddf.wgLayer %2) + (ddf.wgNum % 2)) % 2 == 0) & mask_trigger #& (ddf.nalct < 2)

    ddf_trigger = ddf[mask_trigger]
    ddf_check = ddf[mask_check]

    hists = {}
    
    df_trigger = ddf_trigger.compute()
    gb_trigger = df_trigger.groupby(['event','chamber','endcap'])
    
    lumi_all = gb_trigger.instLumi.mean()
    bx_all = gb_trigger.bx.mean()
    
    ncolbxs,nnoncolbxs = get_n_bx(wTimeBins, col_bx, noncol_active_bx)
    df_trigger['nColBXs'] = getNWTimeBins(df_trigger['bx'].values, ncolbxs)
    df_trigger['nNonColBXs'] = getNWTimeBins(df_trigger['bx'].values, nnoncolbxs)
    
    ncolbx_all = gb_trigger.nColBXs.mean()
    nnoncolbx_all = gb_trigger.nNonColBXs.mean()
    
    bin_lumi = np.linspace(0,20,21)
    hists["h_lumi_allEvts"] = Hist1D(lumi_all.to_numpy(), bins=bin_lumi)
    hists["h_lumi_allEvts_colbx"] = Hist1D(lumi_all.to_numpy(), weights=ncolbx_all.to_numpy(), bins=bin_lumi)
    hists["h_lumi_allEvts_noncolbx"] = Hist1D(lumi_all.to_numpy(), weights=nnoncolbx_all.to_numpy(), bins=bin_lumi)
    
    bin_bx = np.linspace(0,3600,3601)
    ## [:,1:] is to not count wTimeBin = 0
    hists["h_bx_allEvts"] = Hist1D(adjust_bx(bx_all.to_numpy())[:,1:].flatten(), bins=bin_bx)
    hists["h_bx_perEvts"] = Hist1D(bx_all.to_numpy(), bins=bin_bx)
    
    #gb_bx = df.trigger.groupby(['bx'])
    #hists["h_lumi_at_bx"] = Hist1D(df_trigger.groupby(['bx'].))
    hists["h_nWHits_all"] = Hist1D(df_trigger.groupby(['event','chamber','endcap']).wgTime.count().to_numpy(), bins=np.linspace(0,20,21))
    
    df_check = ddf_check.compute()
    bx_count = df_check[df_check.wgTime > 0].bx.to_numpy() + df_check[df_check.wgTime > 0].wgTime.to_numpy() - 8
    
    hists["h_wTime"] = Hist1D(df_check.wgTime.to_numpy(), bins=np.linspace(0,16,17))
    hists["h_nWHits"] = Hist1D(df_check.groupby(['event','chamber','endcap']).wgTime.count().to_numpy(), bins=np.linspace(0,20,21))
    #mask_inTime = (df_check.wgTime >= 1) & (df_check.wgTime <= 4)
    #mask_inTime = (df_check.wgTime >= 1) #& (df_check.wgTime <= 4)
    mask_inTime = df_check.wgTime.isin(wTimeBins) #& (df_check.wgTime <= 4)
    mask_colbx = (df_check.bx+df_check.wgTime-8).isin(col_bx)
    mask_noncolbx = (df_check.bx+df_check.wgTime-8).isin(noncol_active_bx)
    hists["h_lumi_check"] = Hist1D(df_check[mask_inTime].instLumi.to_numpy(), bins=np.linspace(0,20,21))
    hists["h_lumi_check_colbx"] = Hist1D(df_check[mask_inTime & mask_colbx].instLumi.to_numpy(), bins=np.linspace(0,20,21))
    hists["h_lumi_check_noncolbx"] = Hist1D(df_check[mask_inTime & mask_noncolbx].instLumi.to_numpy(), bins=np.linspace(0,20,21))
    hists["h_bx_check"] = Hist1D(bx_count, bins=bin_bx)
    hists["h_rate_vs_lumi"] = hists["h_lumi_check"]/hists["h_lumi_allEvts"]/25/len(wTimeBins)/area*1e9/6
    hists["h_rate_vs_lumi_colbx"] = hists["h_lumi_check_colbx"]/hists["h_lumi_allEvts_colbx"]/25/area*1e9/6
    hists["h_rate_vs_lumi_noncolbx"] = hists["h_lumi_check_noncolbx"]/hists["h_lumi_allEvts_colbx"]/25/area*1e9/6
    #hists["h_rate_vs_lumi"] = hists["h_lumi_check"]/hists["h_lumi_allEvts"]/25/4/area*1e9
    hists["h_rate_vs_bxID"] = hists["h_bx_check"]/hists["h_bx_allEvts"]/25/area*1e9/6

    if ( "doCluster" in kwargs) and kwargs["doCluster"] == True:
        hists["h_rate_vs_lumi"] = hists["h_rate_vs_lumi"]*6
        hists["h_rate_vs_lumi_colbx"] = hists["h_rate_vs_lumi_colbx"]*6
        hists["h_rate_vs_lumi_noncolbx"] = hists["h_rate_vs_lumi_noncolbx"]*6
        hists["h_rate_vs_bxID"] = hists["h_rate_vs_bxID"]*6
    
    return hists
    #return h_rate_vs_lumi, h_wTime, h_nWHits, h_lumi_allEvts


def plot_rate_vs_bx(hists, tag, **kwargs):
    
    rate_vs_bx, bx_profile = hists

    bx_min = 0
    bx_max = 3564 
    if "bxrange" in kwargs:
        logger.info("consider only bx from {} to {}".format(kwargs["bxrange"][0], kwargs["bxrange"][1]))
        bx_min, bx_max = kwargs["bxrange"]

    from datetime import date
    today = date.today()
    savename = "rate_vs_bx_" + str(bx_min) + "_" + str(bx_max) + "_" + tag.replace("/", "_") + "_" + str(today)
    
    color1 = "blue"
    color2 = "orange"

    fig, ax = plt.subplots(1,1,figsize=(30,9))
    #hists_upper[(1,1)]["h_rate_vs_bxID"].plot(ax, histtype='step', show_errors=True, color=color1)
    rate_vs_bx.plot(ax, histtype='step', show_errors=True, color=color1)
    rate_vs_bx.to_json("./hists/" + savename + ".json")
    
    ax.tick_params(axis="y", labelcolor=color1)
    ax.set_xlabel('BX ID')
    #ax.set_ylabel('particle rate ($Hz/cm^2$)', color=color1)
    ax.set_ylabel(kwargs["ylabel"], color=color1)
    median_rate = np.median(np.nan_to_num(rate_vs_bx.counts))
    ax.hlines(y=median_rate, xmin=0, xmax=3600, linewidth=4, color='r')
    ax.set_title(tag)
    ax.set_xlim(bx_min,bx_max)
    if tag.split("/")[1] == '2' and tag.split("/")[0] != '4':
        ax.set_ylim(0,median_rate*5)
    else:
        ax.set_ylim(0,median_rate*2)
        #ax.set_ylim(0,850)
        
    h_collbxs_perbx, h_noncollbxs_perbx = get_num_bx()
    h_collbxs_perbx = h_collbxs_perbx/16.0*median_rate/2
    h_collbxs_perbx.plot(ax, histtype='step', color="red")
    h_noncollbxs_perbx = h_noncollbxs_perbx/16.0*median_rate/2
    h_noncollbxs_perbx.plot(ax, histtype='step', color="blue")

    ax2 = ax.twinx()
    bx_profile.plot(ax2, histtype='step', color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylabel('Counts', color=color2)
    plt.savefig("/home/users/hmei/public_html/cscbkg/" + savename + ".pdf", bbox_inches='tight')
    plt.savefig("/home/users/hmei/public_html/cscbkg/" + savename + ".png", bbox_inches='tight')
    #plt.savefig("/home/users/hmei/public_html/cscbkg/rate_vs_bx_" + tag.replace("/", "_") + "_" + str(today) + ".pdf", bbox_inches='tight')
    #plt.savefig("/home/users/hmei/public_html/cscbkg/rate_vs_bx_" + tag.replace("/", "_") + "_" + str(today) + ".png", bbox_inches='tight')
    plt.close()
