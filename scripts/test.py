from yahist import Hist1D
import numpy as np

from yahist import Hist1D
import matplotlib.pyplot as plt
import mplhep as hep
#hist = Hist1D(np.array([1]), bins=np.linspace(10,0,11))
#hist.counts[7] = 7
#hist.counts[2] = 2 
#hist2 = Hist1D(np.array([2]), bins=np.linspace(10,0,11))
#hist2.counts[7] = 8  
#hist2.counts[3] = 2 
#
#print (hist + hist2)
#print (np.sum(np.array([hist, hist2])))

yraw=np.array([346,275.528421,0.,262.68383844
,253.05590186,243.53541651,251.25105771,227.75029781,289.68367486
,228.48339104,255.14372174,249.87959987,0.,169.19195045
,203.13819034,0.,154.58392898,272.00570895,192.07943228
,226.16530887,0.,197.87005158,293.07657225,274.81439099
,223.31993322,0.,227.58146098,222.26583523,215.92066155])
xraw=np.array([16,17.,18.,19.,20.,21.,22.,23.,24.,25.,26.,27.,28.,29.,30.,31.,32.
,33.,34.,35.,36.,37.,38.,39.,40.,41.,42.,43.,44.])
from scipy.optimize import curve_fit
def func(x, a, b, c):
    return a * b * np.exp(-b * x) + c

mask = (yraw == 0)

y = yraw[~mask]
x = xraw[~mask]

plt.scatter(x,y)
popt, pcov = curve_fit(func, x, y, p0=[100,1,300])
plt.plot(x, func(x, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.legend()
plt.savefig("/home/users/hmei/public_html/cscbkg/testfit_2021-04-14/test.png")
plt.savefig("/home/users/hmei/public_html/cscbkg/testfit_2021-04-14/test.pdf")
from subprocess import call
call("chmod -R 755  /home/users/hmei/public_html/cscbkg/testfit_2021-04-14/", shell=True)
