import numpy as np

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

import scipy.signal as signal


plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["figure.titlesize"] = 25

astdys = pd.read_csv('TNOs/astdys_tnos.csv')
gp_cols = ['h_j_max','k_j_max','p_j_max','q_j_max','h_s_max','k_s_max','p_s_max','q_s_max','h_u_max','k_u_max','p_u_max','q_u_max','h_n_max','k_n_max','p_n_max','q_n_max']
freq_cols = ['h_j_freq','k_j_freq','p_j_freq','q_j_freq','h_s_freq','k_s_freq','p_s_freq','q_s_freq','h_u_freq','k_u_freq','p_u_freq','q_u_freq','h_n_freq','k_n_freq','p_n_freq','q_n_freq']

for i in range(16):
    gp_cols.append(freq_cols[i])

print(gp_cols)

gp_vals = np.zeros((len(astdys),32))
gp_freqs = pd.DataFrame(gp_vals,columns = gp_cols)
for i in range(len(astdys)):
    print(i)
    objname = astdys['Name'].iloc[i]
    filename = 'TNOs/' + objname

    series = pd.read_csv(filename+'/series.csv')
    t = series['t'].values
    a = series['a'].values
    e = series['ecc'].values
    inc = series['inc'].values
    #omega = series['omega'].values
    #Omega = series['Omega'].values
    #M = series['M'].values
    h = series['h'].values
    k = series['k'].values
    p = series['p'].values
    q = series['q'].values
    
    hj = series['hj'].values
    kj = series['kj'].values
    pj = series['pj'].values
    qj = series['qj'].values
    
    hs = series['hs'].values
    ks = series['ks'].values
    ps = series['ps'].values
    qs = series['qs'].values

    hu = series['hu'].values
    ku = series['ku'].values
    pu = series['pu'].values
    qu = series['qu'].values
    
    hn = series['hn'].values
    kn = series['kn'].values
    pn = series['pn'].values
    qn = series['qn'].values
    
    dt = t[1]
    n = len(h)
    freq = np.fft.rfftfreq(n,d=dt)

    #particle eccentricity vectors
    Yh= np.fft.rfft(h)
    Yk = np.fft.rfft(k)
    Yp= np.fft.rfft(p)
    Yq = np.fft.rfft(q)
    
    #giant planets
    Yhj = np.abs(np.fft.rfft(hj))
    Yhs = np.abs(np.fft.rfft(hs))
    Yhu = np.abs(np.fft.rfft(hu))
    Yhn = np.abs(np.fft.rfft(hn))
    Ykj = np.abs(np.fft.rfft(kj))
    Yks = np.abs(np.fft.rfft(ks))
    Yku = np.abs(np.fft.rfft(ku))
    Ykn = np.abs(np.fft.rfft(kn))
    Ypj = np.abs(np.fft.rfft(pj))
    Yps = np.abs(np.fft.rfft(ps))
    Ypu = np.abs(np.fft.rfft(pu))
    Ypn = np.abs(np.fft.rfft(pn))
    Yqj = np.abs(np.fft.rfft(qj))
    Yqs = np.abs(np.fft.rfft(qs))
    Yqu = np.abs(np.fft.rfft(qu))
    Yqn = np.abs(np.fft.rfft(qn))
    
    #find the max power and indexes of that max power
    #(disregarding the frequency=0 terms)
    kumax = Yku[1:].max()
    knmax = Ykn[1:].max()
    ksmax = Yks[1:].max()
    kjmax = Ykj[1:].max()
    humax = Yhu[1:].max()
    hnmax = Yhn[1:].max()
    hsmax = Yhs[1:].max()
    hjmax = Yhj[1:].max()
    pumax = Ypu[1:].max()
    pnmax = Ypn[1:].max()
    psmax = Yps[1:].max()
    pjmax = Ypj[1:].max()
    qumax = Yqu[1:].max()
    qnmax = Yqn[1:].max()
    qsmax = Yqs[1:].max()
    qjmax = Yqj[1:].max()

    ihmax = freq[np.argmax(Yh[1:])+1]
    ikmax = freq[np.argmax(Yk[1:])+1]
    ipmax = freq[np.argmax(Yp[1:])+1]
    iqmax = freq[np.argmax(Yq[1:])+1]
    
    ihjmax = freq[np.argmax(Yhj[1:])+1]
    ikjmax = freq[np.argmax(Ykj[1:])+1]
    ipjmax = freq[np.argmax(Ypj[1:])+1]
    iqjmax = freq[np.argmax(Yqj[1:])+1]
    
    ihsmax = freq[np.argmax(Yhs[1:])+1]
    iksmax = freq[np.argmax(Yks[1:])+1]
    ipsmax = freq[np.argmax(Yps[1:])+1]
    iqsmax = freq[np.argmax(Yqs[1:])+1]
    
    ihumax = freq[np.argmax(Yhu[1:])+1]
    ikumax = freq[np.argmax(Yku[1:])+1]
    ipumax = freq[np.argmax(Ypu[1:])+1]
    iqumax = freq[np.argmax(Yqu[1:])+1]
    
    ihnmax = freq[np.argmax(Yhn[1:])+1]
    iknmax = freq[np.argmax(Ykn[1:])+1]
    ipnmax = freq[np.argmax(Ypn[1:])+1]
    iqnmax = freq[np.argmax(Yqn[1:])+1]
    #(these need the plus 1 to account for neglecting the f=0 term)
    gp_freqs['h_j_max'][i] = hjmax
    gp_freqs['k_j_max'][i] = kjmax
    gp_freqs['p_j_max'][i] = pjmax
    gp_freqs['q_j_max'][i] = qjmax
    gp_freqs['h_s_max'][i] = hsmax
    gp_freqs['k_s_max'][i] = ksmax
    gp_freqs['p_s_max'][i] = psmax
    gp_freqs['q_s_max'][i] = qsmax
    gp_freqs['h_u_max'][i] = humax
    gp_freqs['k_u_max'][i] = kumax
    gp_freqs['p_u_max'][i] = pumax
    gp_freqs['q_u_max'][i] = qumax
    gp_freqs['h_n_max'][i] = hnmax
    gp_freqs['k_n_max'][i] = knmax
    gp_freqs['p_n_max'][i] = pnmax
    gp_freqs['q_n_max'][i] = qnmax
    
    gp_freqs['h_j_freq'][i] = ihjmax
    gp_freqs['k_j_freq'][i] = ikjmax
    gp_freqs['p_j_freq'][i] = ipjmax
    gp_freqs['q_j_freq'][i] = iqjmax
    gp_freqs['h_s_freq'][i] = ihsmax
    gp_freqs['k_s_freq'][i] = iksmax
    gp_freqs['p_s_freq'][i] = ipsmax
    gp_freqs['q_s_freq'][i] = iqsmax
    gp_freqs['h_u_freq'][i] = ihumax
    gp_freqs['k_u_freq'][i] = ikumax
    gp_freqs['p_u_freq'][i] = ipumax
    gp_freqs['q_u_freq'][i] = iqumax
    gp_freqs['h_n_freq'][i] = ihnmax
    gp_freqs['k_n_freq'][i] = iknmax
    gp_freqs['p_n_freq'][i] = ipnmax
    gp_freqs['q_n_freq'][i] = iqnmax

gp_freqs.to_csv('gp_freqs.csv')
