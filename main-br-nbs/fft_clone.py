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
arc1 = rebound.SimulationArchive(fullfile)
series = bin_to_df(folder,filename,arc1)
#print(series)
#series = pd.read_csv('Sims/' + str(folder) + '/' + str(filename) + '/series.csv')

series = series[:500]

allplan = pd.read_csv('../test-notebooks/series_3.csv',index_col=0)

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

numfreqs = 12
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

arange = range(1,2)
#for j in range(len(astdys)):
for j in arange:
    print(j)
    objname = astdys['Name'].iloc[j]
    numclone = 20
    print(filename)
    horizon = pd.read_csv(str(fileloc)+'/horizon_data.csv')
#    prnt(objname)
    sini_pe = np.zeros(numclone+1)
    ecc_pe = np.zeros(numclone+1)
    try:
        fileloc = 'Sims/' + str(folder) + '/' + str(objname)
        archive = rebound.SimulationArchive(fileloc +'/archive.bin')
        
        for w in range(numclone + 1):
            series = bin_to_df(folder,objname,archive, w)

            series = series[:250]

            if horizon['flag'][0] == 0:
                continue
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
            freqlim = 1./10000.
            #disregard frequencies for which any planet has power at higher than 10% the max
            pth = 0.1
            
            spread = 1
    
        
            #print(hk_ind,pq_ind)
            pYh = np.abs(Yh)**2
            pYk = np.abs(Yk)**2
            pYp = np.abs(Yp)**2
            pYq = np.abs(Yq)**2
            for i in range(0,imax-1):
                m = 1.02496658e26
                M = 1.98969175e30

            for i in range(0,imax-1):
                m = 1.02496658e26
                M = 1.98969175e30
                   
                for z in range(numfreqs):
                    if (pYpu[i]>pth*pumax[z] or pYpj[i]>pth*pjmax[z] or pYps[i]>pth*psmax[z] 
                       or pYpn[i]>pth*pnmax[z] or pYpmc[i]>pth*pmcmax[z] or pYpv[i]>pth*pvmax[z]
                        or pYpe[i]>pth*pemax[z] or pYpmr[i]>pth*pmrmax[z] or freq[i]>freqlim):
                        Yp_f[i-3:i+3]=0
        #        else:
        #            p_transfer_f[j][i] = 1
                    if (pYqu[i]>pth*qumax[z] or pYqj[i]>pth*qjmax[z] or pYqs[i]>pth*qsmax[z] 
                       or pYqn[i]>pth*qnmax[z] or pYqmc[i]>pth*qmcmax[z] or pYqv[i]>pth*qvmax[z]
                        or pYqe[i]>pth*qemax[z] or pYqmr[i]>pth*qmrmax[z] or freq[i]>freqlim):
                        Yq_f[i-3:i+3]=0
            #        else:
            #            q_transfer_f[j][i] = 1
                    if (pYhu[i]>pth*humax[z] or pYhj[i]>pth*hjmax[z] or pYhs[i]>pth*hsmax[z] 
                       or pYhn[i]>pth*hnmax[z] or pYhmc[i]>pth*hmcmax[z] or pYhv[i]>pth*hvmax[z]
                        or pYhe[i]>pth*hemax[z] or pYhmr[i]>pth*hmrmax[z] or freq[i]>freqlim):
                        Yh_f[i-3:i+3]=0
            #        else:
            #            h_transfer_f[j][i] = 1
                    if (pYku[i]>pth*kumax[z] or pYkj[i]>pth*kjmax[z] or pYks[i]>pth*ksmax[z] 
                       or pYkn[i]>pth*knmax[z] or pYkmc[i]>pth*kmcmax[z] or pYkv[i]>pth*kvmax[z]
                        or pYke[i]>pth*kemax[z] or pYkmr[i]>pth*kmrmax[z] or freq[i]>freqlim):
                        Yk_f[i-3:i+3]=0
            #        else:
        #            k_transfer_f[j][i] = 1    
 
            p_f = np.fft.irfft(Yp_f,len(p))
            q_f = np.fft.irfft(Yq_f,len(q))
            h_f = np.fft.irfft(Yh_f,len(h))
            k_f = np.fft.irfft(Yk_f,len(k))
        
            #print(astdys['sinI'],j)
            sini_f = np.sqrt(p_f*p_f + q_f*q_f)
            ecc_f = np.sqrt(h_f*h_f + k_f*k_f)

            sini_pe = np.append(sini_pe,np.mean(sini_f))
            ecc_pe = np.append(ecc_pe,np.mean(ecc_f))
            
            
    
    except:
        print(folder, objname, filename)
        print('Error occurred while producing proper elements')
        continue
    #series = pd.read_csv(filename+'/series.csv')
    

    
    #print('Objname: ', objname)
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
    
    print(sini_pe)


pe_df.to_csv('data_files/prop_elem_'+folder+'_8p.csv')
