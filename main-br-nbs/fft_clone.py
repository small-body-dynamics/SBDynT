import numpy as np
import commentjson as json
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
#sys.path.insert(0, '/Users/kvolk/Documents/GitHub/SBDynT/src')
sys.path.insert(0, '../src')
import run_reb
import rebound
import numpy as np
import horizons_api
import tools
from bin_to_df import bin_to_df

import scipy.signal as signal

class ReadJson(object):
    def __init__(self, filename):
        #print('Read the runprops.txt file')
        self.data = json.load(open(filename))
    def outProps(self):
        return self.data


plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["figure.titlesize"] = 25

folder = sys.argv[1]

astdys = pd.read_csv('data_files/'+str(folder)+'_data.csv')
pe_cols = ['Name','sma','obs_ecc','obs_sinI','calc_ecc','calc_sinI','err_ecc','err_sinI','ast_ecc','ast_sinI']
filename = str(astdys['Name'].iloc[1])
fileloc = 'Sims/'+str(folder)+ '/' + str(filename)
fullfile = 'Sims/'+str(folder)+ '/' + str(filename)+'/archive.bin'
print(fullfile)
#arc1 = rebound.SimulationArchive(fullfile)
#series = bin_to_df(folder,filename,arc1)
#print(series)
#series = pd.read_csv('Sims/' + str(folder) + '/' + str(filename) + '/series.csv')

#series = series[:500]

allplan = pd.read_csv('../test-notebooks/series_3.csv',index_col=0)

dt = allplan['t'][1]
n = len(allplan)
freqtot = np.fft.rfftfreq(n,dt)

gp_vals = np.zeros((len(astdys),10))
pe_df = pd.DataFrame(gp_vals,columns = pe_cols)

pYpu = np.abs(np.fft.rfft(allplan['pu'].values))**2
pYpn = np.abs(np.fft.rfft(allplan['pn'].values))**2
pYpj = np.abs(np.fft.rfft(allplan['pj'].values))**2
pYps = np.abs(np.fft.rfft(allplan['ps'].values))**2
pYqu = np.abs(np.fft.rfft(allplan['qu'].values))**2
pYqn = np.abs(np.fft.rfft(allplan['qn'].values))**2
pYqj = np.abs(np.fft.rfft(allplan['qj'].values))**2
pYqs = np.abs(np.fft.rfft(allplan['qs'].values))**2
pYhu = np.abs(np.fft.rfft(allplan['hu'].values))**2
pYhn = np.abs(np.fft.rfft(allplan['hn'].values))**2
pYhj = np.abs(np.fft.rfft(allplan['hj'].values))**2
pYhs = np.abs(np.fft.rfft(allplan['hs'].values))**2
pYku = np.abs(np.fft.rfft(allplan['ku'].values))**2
pYkn = np.abs(np.fft.rfft(allplan['kn'].values))**2
pYkj = np.abs(np.fft.rfft(allplan['kj'].values))**2
pYks = np.abs(np.fft.rfft(allplan['ks'].values))**2

pYpmc = np.abs(np.fft.rfft(allplan['pmc'].values))**2
pYpv = np.abs(np.fft.rfft(allplan['pv'].values))**2
pYpe = np.abs(np.fft.rfft(allplan['pe'].values))**2
pYpmr = np.abs(np.fft.rfft(allplan['pmr'].values))**2
pYqmc = np.abs(np.fft.rfft(allplan['qmc'].values))**2
pYqv = np.abs(np.fft.rfft(allplan['qv'].values))**2
pYqe = np.abs(np.fft.rfft(allplan['qe'].values))**2
pYqmr = np.abs(np.fft.rfft(allplan['qmr'].values))**2
pYhmc = np.abs(np.fft.rfft(allplan['hmc'].values))**2
pYhv = np.abs(np.fft.rfft(allplan['hv'].values))**2
pYhe = np.abs(np.fft.rfft(allplan['he'].values))**2
pYhmr = np.abs(np.fft.rfft(allplan['hmr'].values))**2
pYkmc = np.abs(np.fft.rfft(allplan['kmc'].values))**2
pYkv = np.abs(np.fft.rfft(allplan['kv'].values))**2
pYke = np.abs(np.fft.rfft(allplan['ke'].values))**2
pYkmr = np.abs(np.fft.rfft(allplan['kmr'].values))**2

numfreqs = 3
pumax = np.zeros(numfreqs)
pnmax = np.zeros(numfreqs)
pjmax = np.zeros(numfreqs)
psmax = np.zeros(numfreqs)
qumax = np.zeros(numfreqs)
qnmax = np.zeros(numfreqs)
qjmax = np.zeros(numfreqs)
qsmax = np.zeros(numfreqs)
humax = np.zeros(numfreqs)
hnmax = np.zeros(numfreqs)
hjmax = np.zeros(numfreqs)
hsmax = np.zeros(numfreqs)
kumax = np.zeros(numfreqs)
knmax = np.zeros(numfreqs)
kjmax = np.zeros(numfreqs)
ksmax = np.zeros(numfreqs)

pmcmax = np.zeros(numfreqs)
pvmax = np.zeros(numfreqs)
pemax = np.zeros(numfreqs)
pmrmax = np.zeros(numfreqs)
qmcmax = np.zeros(numfreqs)
qvmax = np.zeros(numfreqs)
qemax = np.zeros(numfreqs)
qmrmax = np.zeros(numfreqs)
hmcmax = np.zeros(numfreqs)
hvmax = np.zeros(numfreqs)
hemax = np.zeros(numfreqs)
hmrmax = np.zeros(numfreqs)
kmcmax = np.zeros(numfreqs)
kvmax = np.zeros(numfreqs)
kemax = np.zeros(numfreqs)
kmrmax = np.zeros(numfreqs)

pYpuc = np.copy(pYpu[1:])
pYpnc = np.copy(pYpn[1:])
pYpjc = np.copy(pYpj[1:])
pYpsc = np.copy(pYps[1:])
pYquc = np.copy(pYqu[1:])
pYqnc = np.copy(pYqn[1:])
pYqjc = np.copy(pYqj[1:])
pYqsc = np.copy(pYqs[1:])
pYhuc = np.copy(pYhu[1:])
pYhnc = np.copy(pYhn[1:])
pYhjc = np.copy(pYhj[1:])
pYhsc = np.copy(pYhs[1:])
pYkuc = np.copy(pYku[1:])
pYknc = np.copy(pYkn[1:])
pYkjc = np.copy(pYkj[1:])
pYksc = np.copy(pYks[1:])

pYpmcc = np.copy(pYpmc[1:])
pYpvc = np.copy(pYpv[1:])
pYpec = np.copy(pYpe[1:])
pYpmrc = np.copy(pYpmr[1:])
pYqmcc = np.copy(pYqmc[1:])
pYqvc = np.copy(pYqv[1:])
pYqec = np.copy(pYqe[1:])
pYqmrc = np.copy(pYqmr[1:])
pYhmcc = np.copy(pYhmc[1:])
pYhvc = np.copy(pYhv[1:])
pYhec = np.copy(pYhe[1:])
pYhmrc = np.copy(pYhmr[1:])
pYkmcc = np.copy(pYkmc[1:])
pYkvc = np.copy(pYkv[1:])
pYkec = np.copy(pYke[1:])
pYkmrc = np.copy(pYkmr[1:])

for i in range(numfreqs):
    pumax[i] = np.max(pYpuc)
    pnmax[i] = np.max(pYpnc)
    pjmax[i] = np.max(pYpjc)
    psmax[i] = np.max(pYpsc)
    qumax[i] = np.max(pYquc)
    qnmax[i] = np.max(pYqnc)
    qjmax[i] = np.max(pYqjc)
    qsmax[i] = np.max(pYqsc)
    humax[i] = np.max(pYhuc)
    hnmax[i] = np.max(pYhnc)
    hjmax[i] = np.max(pYhjc)
    hsmax[i] = np.max(pYhsc)
    kumax[i] = np.max(pYkuc)
    knmax[i] = np.max(pYknc)
    kjmax[i] = np.max(pYkjc)
    ksmax[i] = np.max(pYksc)
    
    pmcmax[i] = np.max(pYpmcc)
    pvmax[i] = np.max(pYpvc)
    pemax[i] = np.max(pYpec)
    pmrmax[i] = np.max(pYpmrc)
    qmcmax[i] = np.max(pYqmcc)
    qvmax[i] = np.max(pYqvc)
    qemax[i] = np.max(pYqec)
    qmrmax[i] = np.max(pYqmrc)
    hmcmax[i] = np.max(pYhmcc)
    hvmax[i] = np.max(pYhvc)
    hemax[i] = np.max(pYhec)
    hmrmax[i] = np.max(pYhmrc)
    kmcmax[i] = np.max(pYkmcc)
    kvmax[i] = np.max(pYkvc)
    kemax[i] = np.max(pYkec)
    kmrmax[i] = np.max(pYkmrc)

    pYpuc[np.argmax(pYpuc)] = 0
    pYpnc[np.argmax(pYpnc)] = 0
    pYpjc[np.argmax(pYpjc)] = 0
    pYpsc[np.argmax(pYpsc)] = 0
    pYquc[np.argmax(pYquc)] = 0
    pYqnc[np.argmax(pYqnc)] = 0
    pYqjc[np.argmax(pYqjc)] = 0
    pYqsc[np.argmax(pYqsc)] = 0
    pYhuc[np.argmax(pYhuc)] = 0
    pYhnc[np.argmax(pYhnc)] = 0
    pYhjc[np.argmax(pYhjc)] = 0
    pYhsc[np.argmax(pYhsc)] = 0
    pYkuc[np.argmax(pYkuc)] = 0
    pYknc[np.argmax(pYknc)] = 0
    pYkjc[np.argmax(pYkjc)] = 0
    pYksc[np.argmax(pYksc)] = 0
    
    pYpmcc[np.argmax(pYpmcc)] = 0
    pYpvc[np.argmax(pYpvc)] = 0
    pYpec[np.argmax(pYpec)] = 0
    pYpmrc[np.argmax(pYpmrc)] = 0
    pYqmcc[np.argmax(pYqmcc)] = 0
    pYqvc[np.argmax(pYqvc)] = 0
    pYqec[np.argmax(pYqec)] = 0
    pYqmrc[np.argmax(pYqmrc)] = 0
    pYhmcc[np.argmax(pYhmcc)] = 0
    pYhvc[np.argmax(pYhvc)] = 0
    pYhec[np.argmax(pYhec)] = 0
    pYhmrc[np.argmax(pYhmrc)] = 0
    pYkmcc[np.argmax(pYkmcc)] = 0
    pYkvc[np.argmax(pYkvc)] = 0
    pYkec[np.argmax(pYkec)] = 0
    pYkmrc[np.argmax(pYkmrc)] = 0

   
hj = allplan['hj'].values
kj = allplan['kj'].values
pj = allplan['pj'].values
qj = allplan['qj'].values
    
hs = allplan['hs'].values
ks = allplan['ks'].values
ps = allplan['ps'].values
qs = allplan['qs'].values

hu = allplan['hu'].values
ku = allplan['ku'].values
pu = allplan['pu'].values
qu = allplan['qu'].values
    
hn = allplan['hn'].values
kn = allplan['kn'].values
pn = allplan['pn'].values
qn = allplan['qn'].values

hmc = allplan['hmc'].values
kmc = allplan['kmc'].values
pmc = allplan['pmc'].values
qmc = allplan['qmc'].values

hv = allplan['hv'].values
kv = allplan['kv'].values
pv = allplan['pv'].values
qv = allplan['qv'].values

he = allplan['he'].values
ke = allplan['ke'].values
pe = allplan['pe'].values
qe = allplan['qe'].values

hmr = allplan['hmr'].values
kmr = allplan['kmr'].values
pmr = allplan['pmr'].values
qmr = allplan['qmr'].values

ihumax = np.argmax(pYhu[1:])+1
ihnmax = np.argmax(pYhn[1:])+1
ihjmax = np.argmax(pYhj[1:])+1
ihsmax = np.argmax(pYhs[1:])+1
ikumax = np.argmax(pYku[1:])+1
iknmax = np.argmax(pYkn[1:])+1
ikjmax = np.argmax(pYkj[1:])+1
iksmax = np.argmax(pYks[1:])+1
ipumax = np.argmax(pYpu[1:])+1
ipnmax = np.argmax(pYpn[1:])+1
ipjmax = np.argmax(pYpj[1:])+1
ipsmax = np.argmax(pYps[1:])+1
iqumax = np.argmax(pYqu[1:])+1
iqnmax = np.argmax(pYqn[1:])+1
iqjmax = np.argmax(pYqj[1:])+1
iqsmax = np.argmax(pYqs[1:])+1


#arange = range(173,174)
for j in range(len(astdys)):
#for j in arange:
    #print(j)
    objname = astdys['Name'].iloc[j]
    print(objname)
    numclone = 0
    #print(filename)
    horizon = pd.read_csv(str(fileloc)+'/horizon_data.csv')
#    prnt(objname)
    sini_pe = np.zeros(numclone+1)
    ecc_pe = np.zeros(numclone+1)
    try:
        fileloc = 'Sims/' + str(folder) + '/' + str(objname)
        archive = rebound.SimulationArchive(fileloc +'/archive.bin')
        
        for w in range(numclone + 1):
            series = bin_to_df(folder,objname,archive, w)
            #print(series)
            #if horizon['flag'][0] == 0:
            #    continue
            t = series['t'].values
            a = series['a'].values
            an = series['an'].values
        #    print(series)
            e = series['ecc'].values
            en = series['eccn'].values
            inc = series['inc'].values
            
            h = series['h'].values
            k = series['k'].values
            p = series['p'].values
            q = series['q'].values
            
            dt = t[1]
            n = len(h)
            #print('dt = ', dt)
            #print('n = ', n)
            freq = np.fft.rfftfreq(n,d=dt)
            
            #particle eccentricity vectors
            Yh= np.fft.rfft(h)
            Yk = np.fft.rfft(k)
            Yp= np.fft.rfft(p)
            Yq = np.fft.rfft(q)
            
            #giant planets
            Yp_f = Yp.copy()
            Yq_f = Yq.copy()
            Yh_f = Yh.copy()
            Yk_f = Yk.copy()
          
            imax = len(Yp)
            #disregard antyhing with a period shorter than 5000 years
            freqlim = 1./1000.
            #disregard frequencies for which any planet has power at higher than 10% the max
            pth = 0.1

            spread = 3
    
        
            #print(hk_ind,pq_ind)
            pYh = np.abs(Yh)**2
            pYk = np.abs(Yk)**2
            pYp = np.abs(Yp)**2
            pYq = np.abs(Yq)**2

            ihmax = np.argmax(pYh)+1
            ipmax = np.argmax(pYp)+1

            rev = 1296000
            g = freqtot[ihmax]
            s = freqtot[ipmax]
            g5 = freqtot[ihjmax]
            g6 = freqtot[ihsmax]
            g7 = freqtot[ihumax]
            g8 = freqtot[ihnmax]
            s5 = 0
            s6 = freqtot[ipsmax]
            s7 = freqtot[ipumax]
            s8 = freqtot[ipnmax]
            
            v1 = g - g5
            v2 = g - g6
            v3 = s - s6

            z1 = abs(g+s-g6-s6) # G and S
            z2 = abs(g+s-g5-s7) # G and S
            z3 = abs(g+s-g5-s6) # G and S
            z4 = abs(g-2*g6+g5) # G
            z5 = abs(g-2*g6+g7) # G
            z6 = abs(s-s6-g5+g6) # S
            z7 = abs(g-3*g6+2*g5) # G
            z8 = abs(2*(g-g6)+s-s6) # G and S
            z9 = abs(3*(g-g6)+s-s6) # G and S
            
            g5 = 4.24/rev
            g6 = 28.22/rev
            g7 = 3.08/rev
            g8 = 0.67/rev
            s6 = 26.34/rev
            s7 = 2.99/rev
            s8 = 0.69/rev
            
            z1 = abs(g+s-g6-s6)
            z2 = abs(g+s-g5-s7)
            z3 = abs(g+s-g5-s6)
            z4 = abs(g-2*g6+g5)
            z5 = abs(g-2*g6+g7)
            z6 = abs(s-s6-g5+g6)
            z7 = abs(g-3*g6+2*g5)
            z8 = abs(2*(g-g6)+s-s6)
            z9 = abs(3*(g-g6)+s-s6)
            
            g5 = 4.25749319/rev
            g6 = 28.24552984/rev
            g7 = 3.08675577/rev
            g8 = 0.67255084/rev
            s6 = -26.34496354/rev
            s7 = -2.99266093/rev
            s8 = -0.69251386/rev
            #'''
            #'''
            
            z1 = 2*g6-g5
            z2 = 2*g6-g7
            z3 = -2*g5+3*g6
            z4 = -g5+g6+g7
            z5 = g5+g6-g7
            z6 = -g5+2*g6+s6-s7
            z7 = g5-s6+s7
            z8 = 2*g5-g7
            
            correction = np.log10(g5*0.1)
            secresind = [np.where(freq >= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0],np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0],np.where(freq >= z1)[0][0],np.where(freq >= z2)[0][0],np.where(freq >= z3)[0][0],np.where(freq >= z4)[0][0],np.where(freq >= z5)[0][0],np.where(freq >= z6)[0][0],np.where(freq >= z7)[0][0],np.where(freq >= z8)[0][0]]
            
            secresind1 = [np.where(freq >= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0],np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0],np.where(freq >= z1)[0][0],np.where(freq >= z2)[0][0],np.where(freq >= z3)[0][0],np.where(freq >= z4)[0][0],np.where(freq >= z5)[0][0],np.where(freq >= z6)[0][0],np.where(freq >= z7)[0][0],np.where(freq >= z8)[0][0]]
            secresind2 = [np.where(freq >= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0],np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0],np.where(freq >= z1)[0][0],np.where(freq >= z2)[0][0],np.where(freq >= z3)[0][0]-6,np.where(freq >= z4)[0][0]-6,np.where(freq >= z5)[0][0]-6,np.where(freq >= z6)[0][0]-6,np.where(freq >= z7)[0][0]-6,np.where(freq >= z8)[0][0]]

            #secresind1 = [np.where(freq >= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0],np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0],np.where(freq >= z1)[0][0],np.where(freq >= z2)[0][0],np.where(freq >= z3)[0][0],np.where(freq >= z4)[0][0],np.where(freq >= z5)[0][0],np.where(freq >= z6)[0][0],np.where(freq >= z7)[0][0],np.where(freq >= z8)[0][0],np.where(freq >= z9)[0][0]]
            #secresind2 = [np.where(freq >= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0],np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0],np.where(freq >= z1)[0][0],np.where(freq >= z2)[0][0],np.where(freq >= z3)[0][0],np.where(freq >= z4)[0][0],np.where(freq >= z5)[0][0],np.where(freq >= z6)[0][0],np.where(freq >= z7)[0][0],np.where(freq >= z8)[0][0],np.where(freq >= z9)[0][0]]
            
            #secresind1 = [np.where(freq >= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0],np.where(freq >= z1)[0][0],np.where(freq >= z2)[0][0],np.where(freq >= z3)[0][0],np.where(freq >= z4)[0][0],np.where(freq >= z5)[0][0],np.where(freq >= z6)[0][0],np.where(freq >= z7)[0][0],np.where(freq >= z8)[0][0],np.where(freq >= z9)[0][0]]
            #secresind2 = [np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0],np.where(freq >= z1)[0][0],np.where(freq >= z2)[0][0],np.where(freq >= z3)[0][0],np.where(freq >= z4)[0][0],np.where(freq >= z5)[0][0],np.where(freq >= z6)[0][0],np.where(freq >= z7)[0][0],np.where(freq >= z8)[0][0],np.where(freq >= z9)[0][0]]
            
            #secresind1 = [np.where(freq >= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0]]
            #secresind2 = [np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0]]

            #secresind1 = [np.where(freq >= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0],np.where(freq >= z1)[0][0],np.where(freq >= z2)[0][0],np.where(freq >= z3)[0][0],np.where(freq >= z4)[0][0],np.where(freq >= z5)[0][0],np.where(freq >= z7)[0][0],np.where(freq >= z8)[0][0],np.where(freq >= z9)[0][0],np.where(freq >= v1)[0][0],np.where(freq >= v2)[0][0]]
            #secresind2 = [np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0],np.where(freq >= z1)[0][0],np.where(freq >= z2)[0][0],np.where(freq >= z3)[0][0],np.where(freq >= z6)[0][0],np.where(freq >= z8)[0][0],np.where(freq >= z9)[0][0],np.where(freq >= v3)[0][0]]

            
            '''
            for i in range(len(secresind)):
                if spread > 0:
                    Yh_f[secresind[i]-spread:secresind[i]+spread] = 0
                    Yk_f[secresind[i]-spread:secresind[i]+spread] = 0
                    Yp_f[secresind[i]-spread:secresind[i]+spread] = 0
                    Yq_f[secresind[i]-spread:secresind[i]+spread] = 0
                else:
                    Yh_f[secresind[i]] = 0
                    Yk_f[secresind[i]] = 0
                    Yp_f[secresind[i]] = 0
                    Yq_f[secresind[i]] = 0
            '''       
            for i in range(len(secresind1)):
                if spread > 0:
                    Yh_f[secresind1[i]-spread:secresind1[i]+spread] = 0
                    Yk_f[secresind1[i]-spread:secresind1[i]+spread] = 0
                else:
                    Yh_f[secresind1[i]] = 0
                    Yk_f[secresind1[i]] = 0
                    
            for i in range(len(secresind2)):
                if spread > 0:
                    Yp_f[secresind2[i]-spread:secresind2[i]+spread] = 0
                    Yq_f[secresind2[i]-spread:secresind2[i]+spread] = 0
                else:
                    Yp_f[secresind2[i]] = 0
                    Yq_f[secresind2[i]] = 0
            
            g=g*rev
            s=s*rev
            g5=g5*rev
            g6=g6*rev
            g7=g7*rev
            g8=g8*rev
            s6=s6*rev
            s7=s7*rev
            s8=s8*rev

            de = np.abs(np.array([g-g5,g-g6,g5-g6,s-s7,s-s6,s7-s6,g+s-s7-g5,g+s-s7-g6,g+s-s6-g5,g+s-s6-g6,2*g-2*s,g-2*g5+g6,g+g5-2*g6,2*g-g5-g6,-g+s+g5-s7,-g+s+g6-s7,-g+s+g5-s6,-g+s+g6-s6,g-g5+s7-s6,g-g5-s7+s6,g-g6+s7-s6,g-g6-s7+s6,2*g-s-s7,2*g-s-s6,-g+2*s-g5,-g+2*s-g6,2*g-2*s7,2*g-2*s6,2*g-s7-s6,g-s+g5-s7,g-s+g5-s6,g-s+g6-s7,g-s+g6-s6,g+g5-2*s7,g+g6-2*s7,g+g5-2*s6,g+g6-2*s6,g+g5-s7-s6,g+g6-s7-s6,s-2*s7+s6,s+s7-2*s6,2*s-s7-s6,s+g5-g6-s7,s-g5+g6-s7,s+g5-g6-s6,s-g5+g6-s6,2*s-2*g5,2*s-2*g6,2*s-g5-g6,s-2*g5+s7,s-2*g5+s6,s-2*g6+s7,s-2*g6+s6,s-g5-g6+s7,s-g5-g6+s6,2*g-2*g5,2*g-2*g6,2*s-2*s7,2*s-2*s6,g-2*g6+g7,g-3*g6+2*g5,2*(g-g6)+(s-s6),g+g5-g6-g7,g-g5-g6+g7,g+g5-2*g6-s6+s7,3*(g-g6)+(s-s6)]))/rev
            #print('de', de)
            secresde = np.zeros(len(de))
            
            for i in range(len(de)):
                secresde[i] = int(np.where(freq >= de[i])[0][0])
            #print('484', secresde)
            
            for i in range(len(de)):
                ind = int(secresde[i])
                if spread > 0:
                    Yh_f[ind-spread:ind+spread] = 0
                    Yk_f[ind-spread:ind+spread] = 0
                    Yp_f[ind-spread:ind+spread] = 0
                    Yq_f[ind-spread:ind+spread] = 0
                else:
                    Yh_f[ind] = 0
                    Yk_f[ind] = 0
                    Yp_f[ind] = 0
                    Yq_f[ind] = 0
            #print('497')       
            '''
            freqind = np.where(freq < freqlim)[0]
            Yh_f[freqind] = 0
            Yk_f[freqind] = 0
            Yp_f[freqind] = 0
            Yq_f[freqind] = 0
            #'''
            
            
            '''
            plt.scatter(1/freq[1:],pYh[1:])
            
            plt.vlines(1/g5,ymin=1e-3,ymax=1e3)
            plt.vlines(1/g6,ymin=1e-3,ymax=1e3)
            plt.vlines(1/g7,ymin=1e-3,ymax=1e3)
            plt.vlines(1/g8,ymin=1e-3,ymax=1e3)

            plt.vlines(1/s6,ymin=1e-3,ymax=1e3)
            plt.vlines(1/s7,ymin=1e-3,ymax=1e3)
            plt.vlines(1/s8,ymin=1e-3,ymax=1e3)

            plt.vlines(1/z1,ymin=1e-3,ymax=1e3)
            plt.vlines(1/z2,ymin=1e-3,ymax=1e3)
            plt.vlines(1/z3,ymin=1e-3,ymax=1e3)
            plt.vlines(1/z4,ymin=1e-3,ymax=1e3)
            plt.vlines(1/z5,ymin=1e-3,ymax=1e3)
            plt.vlines(1/z6,ymin=1e-3,ymax=1e3)
            plt.vlines(1/z7,ymin=1e-3,ymax=1e3)
            plt.vlines(1/z8,ymin=1e-3,ymax=1e3)
            plt.vlines(1/z9,ymin=1e-3,ymax=1e3)
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig('h_vec.png')
            plt.close()
            #'''

            '''
            for i in range(0,imax-1):
                m = 1.02496658e26
                M = 1.98969175e30
                for z in range(numfreqs):
                    if (pYpu[i]>pth*pumax[z] or pYpj[i]>pth*pjmax[z] or pYps[i]>pth*psmax[z] 
                       or pYpn[i]>pth*pnmax[z] or pYpmc[i]>pth*pmcmax[z] or pYpv[i]>pth*pvmax[z]
                        or pYpe[i]>pth*pemax[z] or pYpmr[i]>pth*pmrmax[z] or freq[i]>freqlim):
                        Yp_f[i-spread:i+spread]=0
                        if spread == 0:
                            Yp_f[i]=0
        #        else:
        #            p_transfer_f[j][i] = 1
                    if (pYqu[i]>pth*qumax[z] or pYqj[i]>pth*qjmax[z] or pYqs[i]>pth*qsmax[z] 
                       or pYqn[i]>pth*qnmax[z] or pYqmc[i]>pth*qmcmax[z] or pYqv[i]>pth*qvmax[z]
                        or pYqe[i]>pth*qemax[z] or pYqmr[i]>pth*qmrmax[z] or freq[i]>freqlim):
                        Yq_f[i-spread:i+spread]=0
                        if spread == 0:
                            Yq_f[i]=0
            #        else:
            #            q_transfer_f[j][i] = 1
                    if (pYhu[i]>pth*humax[z] or pYhj[i]>pth*hjmax[z] or pYhs[i]>pth*hsmax[z] 
                       or pYhn[i]>pth*hnmax[z] or pYhmc[i]>pth*hmcmax[z] or pYhv[i]>pth*hvmax[z]
                        or pYhe[i]>pth*hemax[z] or pYhmr[i]>pth*hmrmax[z] or freq[i]>freqlim):
                        Yh_f[i-spread:i+spread]=0
                        if spread == 0:
                            Yh_f[i]=0
            #        else:
            #            h_transfer_f[j][i] = 1
                    if (pYku[i]>pth*kumax[z] or pYkj[i]>pth*kjmax[z] or pYks[i]>pth*ksmax[z] 
                       or pYkn[i]>pth*knmax[z] or pYkmc[i]>pth*kmcmax[z] or pYkv[i]>pth*kvmax[z]
                        or pYke[i]>pth*kemax[z] or pYkmr[i]>pth*kmrmax[z] or freq[i]>freqlim):
                        Yk_f[i-spread:i+spread]=0
                        if spread == 0:
                            Yk_f[i]=0
            #        else:
        #            k_transfer_f[j][i] = 1    
            '''
            
            '''
            plt.scatter(1/freq[1:],np.abs(Yh_f[1:]**2))
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig('h_vec_filt.png')
            plt.close()
            #'''
            
            p_f = np.fft.irfft(Yp_f,len(p))
            q_f = np.fft.irfft(Yq_f,len(q))
            h_f = np.fft.irfft(Yh_f,len(h))
            k_f = np.fft.irfft(Yk_f,len(k))
            
            #print(p_f,q_f)
        
            #print(astdys['sinI'],j)
            sini_f = np.sqrt(p_f*p_f + q_f*q_f)
            ecc_f = np.sqrt(h_f*h_f + k_f*k_f)
            sini_pe[w] = np.mean(sini_f)
            ecc_pe[w] = np.mean(ecc_f)
            
       
    
    except:
        print(folder, objname, filename)
        print('Error occurred while producing proper elements')
        continue
        
    #series = pd.read_csv(filename+'/series.csv')
    #print('Objname: ', objname)print
    #print('Cal sinI: ' , np.mean(sini_f))
    #print('Cal e: ', np.mean(ecc_f))
    pe_df['Name'][j] = objname
    pe_df['sma'][j] = np.mean(a)
    pe_df['obs_ecc'][j] = np.mean(e)
    pe_df['obs_sinI'][j] = np.mean(np.sin(inc))

    pe_df['calc_sinI'][j] = np.mean(sini_pe)
    pe_df['calc_ecc'][j] = np.mean(ecc_pe)
    pe_df['err_sinI'][j] = np.std(sini_pe)
    pe_df['err_ecc'][j] = np.std(ecc_pe)


pe_df.to_csv('data_files/prop_elem_'+folder+'_secres_orbfit_correct.csv')
