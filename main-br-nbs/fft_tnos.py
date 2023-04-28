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

astdys = pd.read_csv('TNOs/astdys_tnos.csv')
pe_cols = ['Name','obs_ecc','obs_sinI','calc_ecc','calc_sinI','ast_ecc','ast_sinI','megno','lyapunov']
filename = astdys['Name'].iloc[0]
series = pd.read_csv('TNOs/'+str(filename)+'/series.csv')

series = series[:500]

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

#arange = range(1174,1183)
for j in range(len(astdys)):
#for j in arange:
    print(j)
    objname = astdys['Name'].iloc[j]
#    print(objname)
    filename = 'TNOs/' + objname
    
    series = pd.read_csv(filename+'/series.csv')
    series = series[:2000]
    getData = ReadJson(str(filename)+'/runprops.txt')
    runprops = getData.outProps()
    runprops = {"3_Hill__Neptune": False}
    runprops = {"2_Hill__Neptune": False}
    runprops = {"1_Hill__Neptune": False}

    if runprops.get('run_success') == False:
        print(Objname +" failed in it's simulation. Will be skipped.")
    horizon = pd.read_csv(filename+'/horizon_data.csv')
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
    '''
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
    '''
    '''    
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
    '''  
    Yp_f = Yp.copy()
    Yq_f = Yq.copy()
    Yh_f = Yh.copy()
    Yk_f = Yk.copy()
  
    imax = len(Yp)
    #disregard antyhing with a period shorter than 5000 years
    freqlim = 1./5000.
    #disregard frequencies for which any planet has power at higher than 10% the max
    pth = 0.1
    
    spread = 1

    #'''
    #test = np.zeros(len(Yp_f))
    #test[ipmax-spread:ipmax+spread] = Yp[ipmax-spread:ipmax+spread]
    #test_2 = np.zeros(len(Yq_f))
    #test_2[iqmax-spread:iqmax+spread] = Yq[iqmax-spread:iqmax+spread]
    #test_3 = np.zeros(len(Yh_f))
    #test_3[ihmax-spread:ihmax+spread] = Yh[ihmax-spread:ihmax+spread]
    #test_4 = np.zeros(len(Yk_f))
    #test_4[ikmax-spread:ikmax+spread] = Yk[ikmax-spread:ikmax+spread]
    #'''
    hk_freqs = np.loadtxt('hires_hk_freqs.txt')
    pq_freqs = np.loadtxt('hires_pq_freqs.txt')
    hk_ind = []
    pq_ind = []
    #print('Length ',len(h))
    #print(freq,hk_freqs,pq_freqs)
    for i in hk_freqs:
        hk_ind.append(np.where(freq >= i)[0][0])
        #print(np.where(freq >= i)[0])
    for i in pq_freqs:
        pq_ind.append(np.where(freq >= i)[0][0])
        
    #print(hk_ind,pq_ind)
    pYh = np.abs(Yh)**2
    pYk = np.abs(Yk)**2
    pYp = np.abs(Yp)**2
    pYq = np.abs(Yq)**2
    for i in range(0,imax-1):
        m = 1.02496658e26
        M = 1.98969175e30
        if abs(an[i]*(1+en[i]) - a[i]*(1-e[i])) < 3*an[i]*(m/3/M)**(1/3):
            runprops['3_Hill_Neptune'] = True
            #print('Within 3 Hill sphere\'s of Neptune with Neptune at ' + str(an[i]*(1+en$
            #print('Obj ecc: ', e[i])
        if abs(an[i]*(1+en[i]) - a[i]*(1-e[i])) < 2*an[i]*(m/3/M)**(1/3):
            runprops['2_Hill_Neptune'] = True
            #print('Within 3 Hill sphere\'s of Neptune with Neptune at ' + str(an[i]*(1+en$
            #print('Obj ecc: ', e[i])
        if abs(an[i]*(1+en[i]) - a[i]*(1-e[i])) < 1*an[i]*(m/3/M)**(1/3):
            runprops['1_Hill_Neptune'] = True
            #print('Within 3 Hill sphere\'s of Neptune with Neptune at ' + str(an[i]*(1+en$
            #print('Obj ecc: ', e[i])
    '''
        if freq[i] > freqlim:
            Yp_f[i] = 0
            Yq_f[i] = 0
            Yh_f[i] = 0
            Yk_f[i] = 0
        for l in pq_ind:
            if pYp[i]>pth*pYp[l]:
                Yp_f[i-1:i+1]=0
            if pYq[i]>pth*pYq[l]:
                Yq_f[i-1:i+1]=0
        for l in hk_ind:
            if pYh[i]>pth*pYh[l]:
                Yh_f[i-1:i+1]=0
            if pYk[i]>pth*pYk[l]:
                Yk_f[i-1:i+1]=0

    '''
    for i in range(0,imax-1):
        m = 1.02496658e26
        M = 1.98969175e30
        if abs(an[i]*(1+en[i]) - a[i]*(1-e[i])) < 3*an[i]*(m/3/M)**(1/3):
            runprops['3_Hill_Neptune'] = True
            #print('Within 3 Hill sphere\'s of Neptune with Neptune at ' + str(an[i]*(1+en[i])) + ' AU and object at ' + str(a[i]*(1-e[i])) + ' AU')
            #print('Obj ecc: ', e[i])
        if abs(an[i]*(1+en[i]) - a[i]*(1-e[i])) < 2*an[i]*(m/3/M)**(1/3):
            runprops['2_Hill_Neptune'] = True
            #print('Within 3 Hill sphere\'s of Neptune with Neptune at ' + str(an[i]*(1+en$
            #print('Obj ecc: ', e[i])
        if abs(an[i]*(1+en[i]) - a[i]*(1-e[i])) < 1*an[i]*(m/3/M)**(1/3):
            runprops['1_Hill_Neptune'] = True
            #print('Within 3 Hill sphere\'s of Neptune with Neptune at ' + str(an[i]*(1+en$
            #print('Obj ecc: ', e[i])
            
        for z in range(numfreqs):
            if (pYpu[i]>pth*pumax[z] or pYpj[i]>pth*pjmax[z] or pYps[i]>pth*psmax[z] 
               or pYpn[i]>pth*pnmax[z] or freq[i]>freqlim):
                Yp_f[i-3:i+3]=0
    #        else:
    #            p_transfer_f[j][i] = 1
            if (pYqu[i]>pth*qumax[z] or pYqj[i]>pth*qjmax[z] or pYqs[i]>pth*qsmax[z] 
               or pYqn[i]>pth*qnmax[z] or freq[i]>freqlim):
                Yq_f[i-3:i+3]=0
    #        else:
    #            q_transfer_f[j][i] = 1
            if (pYhu[i]>pth*humax[z] or pYhj[i]>pth*hjmax[z] or pYhs[i]>pth*hsmax[z] 
               or pYhn[i]>pth*hnmax[z] or freq[i]>freqlim):
                Yh_f[i-3:i+3]=0
    #        else:
    #            h_transfer_f[j][i] = 1
            if (pYku[i]>pth*kumax[z] or pYkj[i]>pth*kjmax[z] or pYks[i]>pth*ksmax[z] 
               or pYkn[i]>pth*knmax[z] or freq[i]>freqlim):
                Yk_f[i-3:i+3]=0
    #        else:
    #            k_transfer_f[j][i] = 1    
        
            
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
    #pe_df['megno'][j] = np.mean(series['megno'].values)
    #pe_df['lyapunov'][j] = np.mean(series['lyapunov'].values)
    #plt.figure()
    #plt.scatter(t,inc)
    #plt.savefig(filename+'/inc.png')

    runpath = str(filename)+'/runprops.txt'
    with open(runpath, 'w') as file:
        file.write(json.dumps(runprops, indent = 4))
    

pe_df.to_csv('prop_elem_tnos_20mil.csv')
