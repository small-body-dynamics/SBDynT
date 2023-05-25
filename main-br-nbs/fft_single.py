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
import bin_to_df

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

astdys = pd.read_csv('data_files/astdys_tnos.csv')
pe_cols = ['Name','obs_ecc','obs_sinI','calc_ecc','calc_sinI','ast_ecc','ast_sinI','megno','lyapunov']
filename = astdys['Name'].iloc[0]
#series = pd.read_csv('TNOs/'+str(filename)+'/series.csv')

#series = series[:500]

allplan = pd.read_csv('../test-notebooks/series_2.csv',index_col=0)

#p_transfer_f = np.zeros((len(astdys),len(np.fft.rfft(allplan['t'].values))))
#q_transfer_f = np.zeros((len(astdys),len(np.fft.rfft(allplan['t'].values))))
#h_transfer_f = np.zeros((len(astdys),len(np.fft.rfft(allplan['t'].values))))
#k_transfer_f = np.zeros((len(astdys),len(np.fft.rfft(allplan['t'].values))))

gp_vals = np.zeros((len(astdys),9))
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

ihjmax = np.argmax(pYhj)
ihsmax = np.argmax(pYhs)
ihumax = np.argmax(pYhu)
ihnmax = np.argmax(pYhn)
ipjmax = np.argmax(pYpj)
ipsmax = np.argmax(pYps)
ipumax = np.argmax(pYpu)
ipnmax = np.argmax(pYpn)

   
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

arange = range(500,520)
#for j in range(len(astdys)):
for j in arange:
    print(j)
    
    objname = astdys['Name'].iloc[j]
#    print(objname)
    filename = 'TNOs/' + objname
    
    archive = rebound.SimulationArchive(filename+'/archive.bin')
    #print(len(archive),'len archive')
    series = bin_to_df.bin_to_df(objname,archive)
    #print(series)

    horizon = pd.read_csv(filename+'/horizon_data.csv')
    if horizon['flag'][0] == 0:
        continue
    t = series['t'].values
    a = series['a'].values
    #an = series['an'].values
#    print(series)
    e = series['ecc'].values
    inc = series['inc'].values
    
    h = series['h'].values
    k = series['k'].values
    p = series['p'].values
    q = series['q'].values
    
    
    dt = int(t[1])/20
    n = len(h)
    #print('dt = ', dt)
    #print('n = ', n)
    freq = np.fft.rfftfreq(n,d=dt)

    #particle eccentricity vectors
    Yh = np.fft.rfft(h)
    Yk = np.fft.rfft(k)
    Yp = np.fft.rfft(p)
    Yq = np.fft.rfft(q)

    Yp[0]=0
    Yq[0]=0
    Yh[0]=0
    Yk[0]=0

    gind = np.argmax(np.abs(Yh))    
    sind = np.argmax(np.abs(Yp))

    Yh_sin = np.zeros(len(Yh))
    Yk_sin = np.zeros(len(Yk))
    Yp_sin = np.zeros(len(Yp))
    Yq_sin = np.zeros(len(Yq))
    
    Yh_sin[gind] = Yh[gind]
    Yk_sin[gind] = Yk[gind]
    Yp_sin[sind] = Yp[sind]
    Yq_sin[sind] = Yq[sind]
    
    p_f = np.fft.irfft(Yp_sin,len(p))
    q_f = np.fft.irfft(Yq_sin,len(q))
    h_f = np.fft.irfft(Yh_sin,len(h))
    k_f = np.fft.irfft(Yk_sin,len(k))
    
    if objname=='2004 PY107':
        print(dt)
        print(len(Yp),len(Yh),len(h))  
        print(gind,sind)
        print(Yh[gind],Yp[sind])
        print(np.max(p_f))
        print(np.max(q_f))
        print(np.max(h_f))
        print(np.max(k_f))
        
    
    #print(astdys['sinI'],j)
    sini_f = np.sqrt(p_f*p_f + q_f*q_f)
    ecc_f = np.sqrt(h_f*h_f + k_f*k_f)
    astsinI = astdys['sinI'][j]
    astecc = astdys['e'][j]    

    pe_df['Name'][j] = objname
    pe_df['obs_ecc'][j] = np.mean(e)
    pe_df['obs_sinI'][j] = np.mean(np.sin(inc))

    pe_df['calc_sinI'][j] = np.mean(sini_f)
    pe_df['calc_ecc'][j] = np.mean(ecc_f)
    pe_df['calc_sinI'][j] = np.max(sini_f)
    pe_df['calc_ecc'][j] = np.max(ecc_f)
    print(np.max(sini_f), np.max(ecc_f))
    
    pe_df['ast_sinI'][j] = astsinI
    pe_df['ast_ecc'][j] = astecc
    #pe_df['megno'][j] = np.mean(series['megno'].values)
    #pe_df['lyapunov'][j] = np.mean(series['lyapunov'].values)
    #plt.figure()
    #plt.scatter(t,inc)
    #plt.savefig(filename+'/inc.png')

        

pe_df.to_csv('data_files/prop_elem_tnos_singlefreq.csv')
