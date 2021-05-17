import json
import numpy as np

def get_rate_at_15e34(fitresult_name):

    #with open('fitresults/h_rate_vs_lumi_weighted_station_1_ring_1_layer_1_isBackward_2021-03-02.json') as f:
    with open('../fitresults/' + fitresult_name) as f:
        data = json.load(f)

    rate = data['a']['value']*15 + data['b']['value']
    err = np.sqrt((15*data['a']['error'])**2 + data['b']['error']**2) 
    return round(rate,2), round(err,2)


chamber_types = [(1,1), (1,2), (1,3), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2)]
for i in range(1, 7):
    rates = []
    errors = []
    for chamber_type in chamber_types:
        station = str(chamber_type[0])
        ring = str(chamber_type[1])
        #name = "h_rate_vs_lumi_weighted_station_" + station + "_ring_" + ring + "_particlerate_2021-03-15.json"
        name = "h_rate_vs_lumi_weighted_station_" + station + "_ring_" + ring + "_hitrate_2021-03-15.json"
        #name = "h_rate_vs_lumi_weighted_station_" + station + "_ring_" + ring + "_layer_" + str(i) + "_2021-03-03.json"
        #name = "h_rate_vs_lumi_weighted_station_" + station + "_ring_" + ring + "_layer_" + str(i) + "_isBackward_2021-03-03.json"
        #name = "h_rate_vs_lumi_weighted_station_" + station + "_ring_" + ring + "_layer_" + str(i) + "_isForward_2021-03-02.json"
        rate,error = get_rate_at_15e34(name) 
        rates.append(str(rate) + "\pm" + str(error))
        errors.append(str(error))
    print ("|" + str(i) + "|", "|".join(rates))
         
