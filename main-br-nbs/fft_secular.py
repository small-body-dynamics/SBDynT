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

rev = 1296000

#arange = range(1174,1183)
for j in range(len(astdys)):
#for j in arange:
    print(j)
    
    objname = astdys['Name'].iloc[j]
#    print(objname)
    filename = 'TNOs/' + objname
    
    series = pd.read_csv(filename+'/series.csv')
    #series = series[:250]
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
    #an = series['an'].values
#    print(series)
    e = series['ecc'].values
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
    #print(len(freq), 'L: length of frequency array')
    g5 = freq[ihjmax]
    g6 = freq[ihsmax]
    g7 = freq[ihumax]
    g8 = freq[ihnmax]
    s5 = 0
    s6 = freq[ipsmax]
    s7 = freq[ipumax]
    s8 = freq[ipnmax]
    #particle eccentricity vectors
    Yh= np.fft.rfft(h)
    Yk = np.fft.rfft(k)
    Yp= np.fft.rfft(p)
    Yq = np.fft.rfft(q)
    
    Yp[0]=0
    Yq[0]=0
    Yh[0]=0
    Yk[0]=0
  
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
    
    #make copies of the FFT outputs
    Yp_f = Yp.copy()
    Yq_f = Yq.copy()
    Yh_f = Yh.copy()
    Yk_f = Yk.copy()
    
    #z10,z11,s7,z6
    
    Yp_f[0]=0
    Yq_f[0]=0
    Yh_f[0]=0
    Yk_f[0]=0

        
    gind = np.argmax(Yh)    
    sind = np.argmax(Yp)
    g = freq[gind]  
    s = freq[sind]
    z1 = abs(g+s-g6-s6)
    z2 = abs(g+s-g5-s7)
    z3 = abs(g+s-g5-s6)
    z4 = abs(g-2*g6+g5)
    z5 = abs(g-2*g6+g7)
    z6 = abs(s-s6-g5+g6)
    z7 = abs(g-3*g6+2*g5)
    z8 = abs(2*(g-g6)+s-s6)
    z9 = abs(3*(g-g6)+s-s6)
    

    #print(np.where(freq>=g5)[0][0])

    #Include all linear coupled frequencies
    #secresind1 = [np.where(freq >= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0],np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0],np.where(freq >= z1)[0][0],np.where(freq >= z2)[0][0],np.where(freq >= z3)[0][0],np.where(freq >= z4)[0][0],np.where(freq >= z5)[0][0],np.where(freq >= z6)[0][0],np.where(freq >= z7)[0][0],np.where(freq >= z8)[0][0],np.where(freq >= z9)[0][0]]
    #secresind2 = [np.where(freq >= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0],np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0],np.where(freq >= z1)[0][0],np.where(freq >= z2)[0][0],np.where(freq >= z3)[0][0],np.where(freq >= z4)[0][0],np.where(freq >= z5)[0][0],np.where(freq >= z6)[0][0],np.where(freq >= z7)[0][0],np.where(freq >= z8)[0][0]]

    #Identical to above method
    #secresind1 = [np.where(freq <= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0],np.where(freq >= z1)[0][0],np.where(freq >= z4)[0][0],np.where(freq >= z5)[0][0],np.where(freq >= z7)[0][0],np.where(freq >= z8)[0][0],np.where(freq >= z9)[0][0]]
    #secresind2 = [np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0],np.where(freq >= z1)[0][0],np.where(freq >= z4)[0][0],np.where(freq >= z5)[0][0],np.where(freq >= z7)[0][0],np.where(freq >= z8)[0][0],np.where(freq >= z9)[0][0]]

#Include all g and s secular frequencies
    #secresind1 = [np.where(freq >= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0],np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0]]
    #secresind2 = [np.where(freq >= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0],np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0]]

#Specific g and s frequencies
    #secresind1 = [np.where(freq >= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0]]
    #secresind2 = [np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0]]

#Knezevic and Milani frequencies
    secresind1 = [np.where(freq >= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0],np.where(freq >= z4)[0][0]]
    secresind2 = [np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0],np.where(freq >= z4)[0][0]]


    limit_ind = np.where(freq >= freqlim)[0]

    spread = 0

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
        
    Yp_f[limit_ind] = 0
    Yq_f[limit_ind] = 0
    Yk_f[limit_ind] = 0
    Yh_f[limit_ind] = 0
    
    p_f = np.fft.irfft(Yp_f,len(p))
    q_f = np.fft.irfft(Yq_f,len(q))
    h_f = np.fft.irfft(Yh_f,len(h))
    k_f = np.fft.irfft(Yk_f,len(k))

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

    runpath = str(filename)+'/runprops.txt'
    with open(runpath, 'w') as file:
        file.write(json.dumps(runprops, indent = 4))
    

pe_df.to_csv('data_files/prop_elem_tnos_singlefreq.csv')