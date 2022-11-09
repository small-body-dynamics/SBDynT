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
pe_cols = ['calc_ecc','calc_sinI','ast_ecc','ast_sinI']

gp_vals = np.zeros((len(astdys),4))
pe_df = pd.DataFrame(gp_vals,columns = pe_cols)
for j in range(len(astdys)):
    print(j)
    objname = astdys['Name'].iloc[j]
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
    Yhj = (np.fft.rfft(hj))
    Yhs = (np.fft.rfft(hs))
    Yhu = (np.fft.rfft(hu))
    Yhn = (np.fft.rfft(hn))
    Ykj = (np.fft.rfft(kj))
    Yks = (np.fft.rfft(ks))
    Yku = (np.fft.rfft(ku))
    Ykn = (np.fft.rfft(kn))
    Ypj = (np.fft.rfft(pj))
    Yps = (np.fft.rfft(ps))
    Ypu = (np.fft.rfft(pu))
    Ypn = (np.fft.rfft(pn))
    Yqj = (np.fft.rfft(qj))
    Yqs = (np.fft.rfft(qs))
    Yqu = (np.fft.rfft(qu))
    Yqn = (np.fft.rfft(qn))
    
    pYpj = np.abs(Ypj)
    pYqj = np.abs(Yqj)
    pYhj = np.abs(Yhj)
    pYkj = np.abs(Ykj)
    pYps = np.abs(Yps)
    pYqs = np.abs(Yqs)
    pYhs = np.abs(Yhs)
    pYks = np.abs(Yks)
    pYpu = np.abs(Ypu)
    pYqu = np.abs(Yqu)
    pYhu = np.abs(Yhu)
    pYku = np.abs(Yku)
    pYpn = np.abs(Ypn)
    pYqn = np.abs(Yqn)
    pYhn = np.abs(Yhn)
    pYkn = np.abs(Ykn)
    
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

    ihmax = np.argmax(Yh[1:]+1)
    ikmax = np.argmax(Yk[1:]+1)
    ipmax = np.argmax(Yp[1:]+1)
    iqmax = np.argmax(Yq[1:]+1)
    
    ihjmax = np.argmax(Yhj[1:]+1)
    ikjmax = np.argmax(Ykj[1:]+1)
    ipjmax = np.argmax(Ypj[1:]+1)
    iqjmax = np.argmax(Yqj[1:]+1)
    
    ihsmax = np.argmax(Yhs[1:]+1)
    iksmax = np.argmax(Yks[1:]+1)
    ipsmax = np.argmax(Yps[1:]+1)
    iqsmax = np.argmax(Yqs[1:]+1)
    
    ihumax = np.argmax(Yhu[1:]+1)
    ikumax = np.argmax(Yku[1:]+1)
    ipumax = np.argmax(Ypu[1:]+1)
    iqumax = np.argmax(Yqu[1:]+1)
    
    ihnmax = np.argmax(Yhn[1:]+1)
    iknmax = np.argmax(Ykn[1:]+1)
    ipnmax = np.argmax(Ypn[1:]+1)
    iqnmax = np.argmax(Yqn[1:]+1)
    #(these need the plus 1 to account for neglecting the f=0 term)
    #make copies of the FFT outputs
    Yp_f = Yp.copy()
    Yq_f = Yq.copy()
    Yh_f = Yh.copy()
    Yk_f = Yk.copy()
    
    imax = len(Yp)
    #disregard antyhing with a period shorter than 5000 years
    freqlim = 1./5000.
    #disregard frequencies for which any planet has power at higher than 10% the max
    pth = 0.25
    
    spread = 1
    #'''
    test = np.zeros(len(Yp_f))
    test[ipmax-spread:ipmax+spread] = Yp[ipmax-spread:ipmax+spread]
    test_2 = np.zeros(len(Yq_f))
    test_2[iqmax-spread:iqmax+spread] = Yq[iqmax-spread:iqmax+spread]
    test_3 = np.zeros(len(Yh_f))
    test_3[ihmax-spread:ihmax+spread] = Yh[ihmax-spread:ihmax+spread]
    test_4 = np.zeros(len(Yk_f))
    test_4[ikmax-spread:ikmax+spread] = Yk[ikmax-spread:ikmax+spread]
    
    for i in range(0,imax-1):
        if (pYpu[i]>pth*pumax or pYpj[i]>pth*pjmax or pYps[i]>pth*psmax 
           or pYpn[i]>pth*pnmax or freq[i]>freqlim):
            Yp_f[i]=0
        if (pYqu[i]>pth*qumax or pYqj[i]>pth*qjmax or pYqs[i]>pth*qsmax 
           or pYqn[i]>pth*qnmax or freq[i]>freqlim):
            Yq_f[i]=0
        if (pYhu[i]>pth*humax or pYhj[i]>pth*hjmax or pYhs[i]>pth*hsmax 
           or pYhn[i]>pth*hnmax or freq[i]>freqlim):
            Yh_f[i]=0
        if (pYku[i]>pth*kumax or pYkj[i]>pth*kjmax or pYks[i]>pth*ksmax 
           or pYkn[i]>pth*knmax or freq[i]>freqlim):
            Yk_f[i]=0
    
    
        
    p_f = np.fft.irfft(Yp_f,len(p))
    q_f = np.fft.irfft(Yq_f,len(q))
    h_f = np.fft.irfft(Yh_f,len(h))
    k_f = np.fft.irfft(Yk_f,len(k))
    '''
    i_p = np.fft.irfft(test,len(p))
    i_q = np.fft.irfft(test_2,len(q))
    i_h = np.fft.irfft(test_3,len(h))
    i_k = np.fft.irfft(test_4,len(k))
    '''
    #print(astdys['sinI'],j)
    sini_f = np.sqrt(p_f*p_f + q_f*q_f)
    ecc_f = np.sqrt(h_f*h_f + k_f*k_f)
    astsinI = astdys['sinI'][j]
    astecc = astdys['sinI'][j]    
    
    #print(astecc)
    pe_df['calc_sinI'][j] = np.mean(sini_f)
    pe_df['calc_ecc'][j] = np.mean(ecc_f)
    pe_df['ast_sinI'][j] = astsinI
    pe_df['ast_ecc'][j] = astecc
    
pe_df.to_csv('prop_elem.csv')
