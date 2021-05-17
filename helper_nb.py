import numba as nb
import numpy as np

@nb.jit
def makeData_perEvent(events, wgs, alcts, lumis):
    
    nevt = np.int64(len(events))
    nwg = np.int64(0) 
    for i in range(np.int64(nevt)):
        nwg_this_evt = len(wgs[i])
        nwg += nwg_this_evt
    
    # run,evt,bx,endcap,chamber,wgNumber,wgLayer,wireTime,nWireTimeBins,nalct,alct0_keyWG,adjacentWG
    data_perEvent = np.zeros((nwg, 12), dtype=np.int64) 
    # instLumi
    lumi_perEvent = np.zeros(nwg, dtype=np.float64)
    #
    cnt_nwg = np.int64(0) 
    for i in range(nevt):
        # one TYPE of chamber per event to look at
        run = events[i].Run
        lumi = events[i].LumiSect
        evt = events[i].Event
        bx = events[i].BunchCrossing
        instLumi = lumis[i] 
        
        for j in range(len(wgs[i])):
            #alct = len(alcts[i]) 
            wg = wgs[i][j]
            wgNum = wg.ID_wire
            wgLayer = wg.ID_layer
            wgTime = wg.timeBin
            wgNWireTimeBins = wg.numberWireTimeBins
            endcap = wg.ID_endcap
            chamber = wg.ID_chamber
            #wgAFEB = wg.AFEB
            alct0_keyWG = -1
            nalct = 0
            for k in range(len(alcts[i])):
                alct = alcts[i][k]
                if (alct.ID_endcap == endcap) and (alct.ID_chamber == chamber):
                    alct0_keyWG = alct.ID_keyWG
                    #break
                    nalct += 1
            
            adjacentWG = 0
            ## check against wg that has been processed that are in the same chamber
            ## if they are too close, or form a wire cluster, add a flag (closeWG)
            for k in range(0, j):
                wg_ = wgs[i][k]
                wgNum_ = wg_.ID_wire
                wgLayer_ = wg_.ID_layer
                wgTime_ = wg_.timeBin
                endcap_ = wg_.ID_endcap
                station_ = wg_.ID_station
                ring_ = wg_.ID_ring
                chamber_ = wg_.ID_chamber
                if (endcap_ != endcap) or (chamber_ != chamber): continue
                ### same layer, adjacent wg, close in time
                wgDiff = abs(wgNum_ - wgNum)
                layerDiff = abs(wgLayer_ - wgLayer)
                wTimeDiff = abs(wgTime_ - wgTime)
                
                adjacent = (wgDiff + layerDiff <= 2) and wTimeDiff <= 2
                sameWG = (wgDiff ==0)
                track_veto = (layerDiff > 0) and (wgDiff/layerDiff <= 1)
                if (station_ == 1 & ring_ > 1):
                    track_veto = (layerDiff > 0) and (wgDiff/layerDiff <= 3)
                if (station_ == 2 & ring_ > 1):
                    track_veto = (layerDiff > 0) and (wgDiff/layerDiff <= 2)
                if (station_ > 2 & ring_ > 1):
                    track_veto = (layerDiff > 0) and (wgDiff/layerDiff <= 1.5)
                if adjacent or sameWG:
                    adjacentWG = 1
                    break
                if track_veto:
                    adjacentWG = 2 
                    break
                    
            #print ([evt, run, bx, wgNum, wgLayer, wgTime, wgNWireTimeBins, nalct, alct0_keyWG]) 
            data_perEvent[cnt_nwg] = np.array([evt, run, bx, endcap, chamber, wgNum, wgLayer, wgTime, wgNWireTimeBins, nalct, alct0_keyWG, adjacentWG]) 
            lumi_perEvent[cnt_nwg] = instLumi
            cnt_nwg += 1
    #    
    #
    return (data_perEvent, lumi_perEvent)
