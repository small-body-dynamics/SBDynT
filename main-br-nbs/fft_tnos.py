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
pe_cols = ['Name','obs_ecc','obs_sinI','calc_ecc','calc_sinI','ast_ecc','ast_sinI']

filename = astdys['Name'].iloc[0]
series = pd.read_csv('TNOs/'+str(filename)+'/series.csv')

p_transfer_f = np.zeros((len(astdys),len(series['t'].values)))
q_transfer_f = np.zeros((len(astdys),len(series['t'].values)))
h_transfer_f = np.zeros((len(astdys),len(series['t'].values)))
k_transfer_f = np.zeros((len(astdys),len(series['t'].values)))

gp_vals = np.zeros((len(astdys),7))
pe_df = pd.DataFrame(gp_vals,columns = pe_cols)
arange = range(250,300)
for j in range(len(astdys)):
#for j in arange:
    print(j)
    objname = astdys['Name'].iloc[j]
   # print(objname)
    filename = 'TNOs/' + objname

    series = pd.read_csv(filename+'/series.csv')
    horizon = pd.read_csv(filename+'/horizon_data.csv')
    if horizon['flag'][0] == 0:
        continue
    t = series['t'].values
    a = series['a'].values
    e = series['ecc'].values
    inc = series['inc'].values
    
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
    
    pYh = np.abs(Yh)**2
    pYk = np.abs(Yk)**2
    pYp = np.abs(Yp)**2
    pYq = np.abs(Yq)**2
    
    pYpj = np.abs(Ypj)**2
    pYqj = np.abs(Yqj)**2
    pYhj = np.abs(Yhj)**2
    pYkj = np.abs(Ykj)**2
    pYps = np.abs(Yps)**2
    pYqs = np.abs(Yqs)**2
    pYhs = np.abs(Yhs)**2
    pYks = np.abs(Yks)**2
    pYpu = np.abs(Ypu)**2
    pYqu = np.abs(Yqu)**2
    pYhu = np.abs(Yhu)**2
    pYku = np.abs(Yku)**2
    pYpn = np.abs(Ypn)**2
    pYqn = np.abs(Yqn)**2
    pYhn = np.abs(Yhn)**2
    pYkn = np.abs(Ykn)**2
    
    #find the max power and indexes of that max power
    #(disregarding the frequency=0 terms)
    kumax = pYku[1:].max()
    knmax = pYkn[1:].max()
    ksmax = pYks[1:].max()
    kjmax = pYkj[1:].max()
    humax = pYhu[1:].max()
    hnmax = pYhn[1:].max()
    hsmax = pYhs[1:].max()
    hjmax = pYhj[1:].max()
    pumax = pYpu[1:].max()
    pnmax = pYpn[1:].max()
    psmax = pYps[1:].max()
    pjmax = pYpj[1:].max()
    qumax = pYqu[1:].max()
    qnmax = pYqn[1:].max()
    qsmax = pYqs[1:].max()
    qjmax = pYqj[1:].max()

    ihmax = np.argmax(pYh[1:])+1
    ikmax = np.argmax(pYk[1:])+1
    ipmax = np.argmax(pYp[1:])+1
    iqmax = np.argmax(pYq[1:])+1
    
    ihjmax = np.argmax(pYhj[1:])+1
    ikjmax = np.argmax(pYkj[1:])+1
    ipjmax = np.argmax(pYpj[1:])+1
    iqjmax = np.argmax(pYqj[1:])+1
    
    ihsmax = np.argmax(pYhs[1:])+1
    iksmax = np.argmax(pYks[1:])+1
    ipsmax = np.argmax(pYps[1:])+1
    iqsmax = np.argmax(pYqs[1:])+1
    
    ihumax = np.argmax(pYhu[1:])+1
    ikumax = np.argmax(pYku[1:])+1
    ipumax = np.argmax(pYpu[1:])+1
    iqumax = np.argmax(pYqu[1:])+1
    
    ihnmax = np.argmax(pYhn[1:])+1
    iknmax = np.argmax(pYkn[1:])+1
    ipnmax = np.argmax(pYpn[1:])+1
    iqnmax = np.argmax(pYqn[1:])+1

    print(ihjmax,pjmax)
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
        else:
            p_transfer_f[j][i] = 1
        if (pYqu[i]>pth*qumax or pYqj[i]>pth*qjmax or pYqs[i]>pth*qsmax 
           or pYqn[i]>pth*qnmax or freq[i]>freqlim):
            Yq_f[i]=0
        else:
            q_transfer_f[j][i] = 1
        if (pYhu[i]>pth*humax or pYhj[i]>pth*hjmax or pYhs[i]>pth*hsmax 
           or pYhn[i]>pth*hnmax or freq[i]>freqlim):
            Yh_f[i]=0
        else:
            h_transfer_f[j][i] = 1
        if (pYku[i]>pth*kumax or pYkj[i]>pth*kjmax or pYks[i]>pth*ksmax 
           or pYkn[i]>pth*knmax or freq[i]>freqlim):
            Yk_f[i]=0
        else:
            k_transfer_f[j][i] = 1    
    
        
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
    astecc = astdys['e'][j]    
    
    #print('Objname: ', objname)
    #print('Cal sinI: ' , np.mean(sini_f))
    #print('Cal e: ', np.mean(ecc_f))
    pe_df['Name'][j] = objname
    pe_df['obs_ecc'][j] = np.mean(e)
    pe_df['obs_sinI'][j] = np.mean(np.sin(inc))

    pe_df['calc_sinI'][j] = np.mean(sini_f)
    pe_df['calc_ecc'][j] = np.mean(ecc_f)
    pe_df['ast_sinI'][j] = astsinI
    pe_df['ast_ecc'][j] = astecc
    #plt.figure()
    #plt.scatter(t,inc)
    #plt.savefig(filename+'/inc.png')
    
pe_df.to_csv('prop_elem_tnos.csv')
np.savetxt('data_files/p_tnos_transfer.txt',p_transfer_f)
np.savetxt('data_files/q_tnos_transfer.txt',q_transfer_f)
np.savetxt('data_files/h_tnos_transfer.txt',h_transfer_f)
np.savetxt('data_files/k_tnos_transfer.txt',k_transfer_f)

