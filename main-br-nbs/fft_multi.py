import numpy as np
import commentjson as json
import pandas as pd
import bin_to_df
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
import warnings
warnings.filterwarnings("ignore")
import schwimmbad
import functools

import scipy.signal as signal

class ReadJson(object):
    def __init__(self, filename):
        #print('Read the runprops.txt file')
        self.data = json.load(open(filename))
    def outProps(self):
        return self.data

def prop_calc(j, astdys):
    print(j)
    
    objname = astdys['Name'].iloc[j]
#    print(objname)
    filename = 'AstFam_families/' + str(j)
    #filename = '~/../../../hdd/haumea-data/djspenc/SBDynT_Sims/TNOs_new/' + str(j)

    try:
        #fullfile = '~/../../../hdd/haumea-data/djspenc/SBDynT_Sims/TNOs_new/'+str(j)+'/archive_hires.bin'
        fullfile = 'Sims/AstFam_families/'+str(j)+'/archive_hires.bin'
        print(fullfile)
        arc1 = rebound.SimulationArchive(fullfile)
        #print(arc1,j)
        series = bin_to_df.bin_to_df('AstFam_families',str(j),arc1,astdys,'8planet')
        #archive = rebound.SimulationArchive(filename+'/archive.bin')
        #print(len(archive),'len archive')
        #series = bin_to_df.bin_to_df(objname,archive)
        
    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print('Failed with error:', error)
        return [objname,0,0,0,0,0,0,0]
    #series = pd.read_csv(filename+'/series.csv')
    #series = series[:250]
    print('Series read, doing fft now')
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
    #print(t)
    
    dt = t[1]
    n = len(h)
    #print('dt = ', dt)
    #print('n = ', n)
    freq = np.fft.rfftfreq(n,d=dt)
    #print(len(freq), 'L: length of frequency array')
    rev = 1296000
    g5 = 4.24/rev
    g6 = 28.22/rev
    g7 = 3.08/rev
    g8 = 0.67/rev
    s6 = 26.34/rev
    s7 = 2.99/rev
    s8 = 0.69/rev
    '''
    #g5 = freq[ihjmax]
    #g6 = freq[ihsmax]
    #g7 = freq[ihumax]
    #g8 = freq[ihnmax]
    #s5 = 0
    #s6 = freq[ipsmax]
    #s7 = freq[ipumax]
    #s8 = freq[ipnmax]
    '''
    #particle eccentricity vectors
    Yh= np.fft.rfft(h)
    Yk = np.fft.rfft(k)
    Yp= np.fft.rfft(p)
    Yq = np.fft.rfft(q)
    Ya_f = np.fft.rfft(a)
    
    Yp[0]=0
    Yq[0]=0
    Yh[0]=0
    Yk[0]=0
  
    imax = len(Yp)
    #disregard antyhing with a period shorter than 5000 years
    freqlim = 1./1000.
    #disregard frequencies for which any planet has power at higher than 10% the max
    pth = 0.25
    
    spread = 3
       
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
    
    #Yp_f[0]=0
    #Yq_f[0]=0
    #Yh_f[0]=0
    #Yk_f[0]=0

        
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
    
        
    z1_g = (-s+g6+s6)
    z2_g = (-s+g5+s7)
    z3_g = (-s+g5+s6)
    z4_g = (2*g6-g5)
    z5_g = (2*g6-g7)
    z7_g = (3*g6-2*g5)
    z8_g = (2*g6-s+s6)/2
    z9_g = (3*g6-s+s6)/3
        
        
    z1_s = (-g+g6+s6)
    z2_s = (-g+g5+s7)
    z3_s = (-g+g5+s6)
    z6_s = (s6+g5-g6)
    z8_s = (-2*(g-g6)+s6)
    z9_s = (-3*(g-g6)+s6)
    
        

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

    g=g*rev
    s=s*rev
    g5=g5*rev
    g6=g6*rev
    g7=g7*rev
    g8=g8*rev
    s6=s6*rev
    s7=s7*rev
    s8=s8*rev

    de = np.abs(np.array([g5,g6,g7,g8,s6,s7,s8,g-g5,g-g6,g5-g6,s-s7,s-s6,s7-s6,g+s-s7-g5,g+s-s7-g6,g+s-s6-g5,g+s-s6-g6,2*g-2*s,g-2*g5+g6,g+g5-2*g6,2*g-g5-g6,-g+s+g5-s7,-g+s+g6-s7,-g+s+g5-s6,-g+s+g6-s6,g-g5+s7-s6,g-g5-s7+s6,g-g6+s7-s6,g-g6-s7+s6,2*g-s-s7,2*g-s-s6,-g+2*s-g5,-g+2*s-g6,2*g-2*s7,2*g-2*s6,2*g-s7-s6,g-s+g5-s7,g-s+g5-s6,g-s+g6-s7,g-s+g6-s6,g+g5-2*s7,g+g6-2*s7,g+g5-2*s6,g+g6-2*s6,g+g5-s7-s6,g+g6-s7-s6,s-2*s7+s6,s+s7-2*s6,2*s-s7-s6,s+g5-g6-s7,s-g5+g6-s7,s+g5-g6-s6,s-g5+g6-s6,2*s-2*g5,2*s-2*g6,2*s-g5-g6,s-2*g5+s7,s-2*g5+s6,s-2*g6+s7,s-2*g6+s6,s-g5-g6+s7,s-g5-g6+s6,2*g-2*g5,2*g-2*g6,2*s-2*s7,2*s-2*s6,g-2*g6+g7,g-3*g6+2*g5,2*(g-g6)+(s-s6),g+g5-g6-g7,g-g5-g6+g7,g+g5-2*g6-s6+s7,3*(g-g6)+(s-s6)]))/rev
    #Knezevic and Milani frequencies
    secresind1 = []
    secresind2 = []
    
    secresde = []
            
    for i in range(len(de)):
        if de[i] < 10000:
            continue
        secresde.append(int(np.where(freq >= de[i])[0][0]))
    #z7_g,
    #z1_s,z8_s,z9_s
    freq1 = [g5,g6,g7,g8,z1_g,z2_g,z3_g,z4_g,z5_g,z8_g,z9_g]
    freq2 = [s6,s7,s8,z2_s,z3_s,z6_s]
    
    freq1 = [g5,g6,g7,g8,z1,z2,z3,z4,z5,z7,z8,z9]
    freq2 = [s6,s7,s8,z1,z2,z3,z6,z8,z9]
    
    #freq1 = [g5,g6,g7,g8,s6,s7,s8,z1,z2,z3,z4,z5,z7,z8,z9]
    #freq2 = [g5,g6,g7,g8,s6,s7,s8,z1,z2,z3,z6,z8,z9]
    
    #freq1 = [g5,g6,g7,g8,z4]
    #freq2 = [s6,s7,s8,z4]
    
    for i in freq1:
        if 1/i < 10000:
            continue
        secresind1.append(np.where(freq>=i)[0][0])
    for i in freq2:
        if 1/i < 10000:
            continue
        secresind2.append(np.where(freq>=i)[0][0])
    
    #secresind1 = [np.where(freq >= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0],np.where(freq >= z1_g)[0][0],np.where(freq >= z2_g)[0][0],np.where(freq >= z3_g)[0][0],np.where(freq >= z4_g)[0][0],np.where(freq >= z5_g)[0][0],np.where(freq >= z7_g)[0][0],np.where(freq >= z8_g)[0][0],np.where(freq >= z9_g)[0][0]]
    #secresind2 = [np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0],np.where(freq >= z1_s)[0][0],np.where(freq >= z2_s)[0][0],np.where(freq >= z3_s)[0][0],np.where(freq >= z6_s)[0][0],np.where(freq >= z8_s)[0][0],np.where(freq >= z9_s)[0][0]]

    limit_ind = np.where(freq >= freqlim)[0]

    spread = 3
    #'''
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
    '''     
    for i in range(len(secresde)):
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
    #'''
        
    Yp_f[limit_ind] = 0
    Yq_f[limit_ind] = 0
    Yk_f[limit_ind] = 0
    Yh_f[limit_ind] = 0
    Ya_f[limit_ind] = 0
    
    p_f = np.fft.irfft(Yp_f,len(p))
    q_f = np.fft.irfft(Yq_f,len(q))
    h_f = np.fft.irfft(Yh_f,len(h))
    k_f = np.fft.irfft(Yk_f,len(k))
    a_f = np.fft.irfft(Ya_f,len(a))

    #print(astdys['sinI'],j)
    sini_f = np.sqrt(p_f*p_f + q_f*q_f)
    ecc_f = np.sqrt(h_f*h_f + k_f*k_f)
    astsinI = astdys['sinI'][j]
    astecc = astdys['e'][j]    
    
    
    #pe_df['megno'][j] = np.mean(series['megno'].values)
    #pe_df['lyapunov'][j] = np.mean(series['lyapunov'].values)
    #plt.figure()
    #plt.scatter(t,inc)
    #plt.savefig(filename+'/inc.png')

    return [objname,np.mean(e),np.mean(np.sin(inc)),np.mean(ecc_f),np.mean(sini_f),astecc,astsinI,np.mean(a_f)]

plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["figure.titlesize"] = 25


if __name__ == '__main__':

    from schwimmbad import MPIPool
    #print('schwimmbad in')
    with MPIPool() as pool:
        #print('mpipool pooled')    
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
            
        astdys = pd.read_csv('data_files/AstFam_families_data.csv')
        pe_cols = ['Name','obs_ecc','obs_sinI','calc_ecc','calc_sinI','ast_ecc','ast_sinI','calc_sma']
        filename = astdys['Name'].iloc[0]
        #series = pd.read_csv('TNOs/'+str(filename)+'/series.csv')

        allplan = pd.read_csv('../test-notebooks/series_2.csv',index_col=0)

        #p_transfer_f = np.zeros((len(astdys),len(np.fft.rfft(allplan['t'].values))))
        #q_transfer_f = np.zeros((len(astdys),len(np.fft.rfft(allplan['t'].values))))
        #h_transfer_f = np.zeros((len(astdys),len(np.fft.rfft(allplan['t'].values))))
        #k_transfer_f = np.zeros((len(astdys),len(np.fft.rfft(allplan['t'].values))))

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
        g5 = 4.24/rev
        g6 = 28.22/rev
        g7 = 3.08/rev
        g8 = 0.67/rev
        s6 = 26.34/rev
        s7 = 2.99/rev
        s8 = 0.69/rev

        multi_prop = functools.partial(prop_calc, astdys=astdys)
        j = range(len(astdys))
        #j = range(600,650)
        #begin = datetime.now()
        data = pool.map(multi_prop, j)
        gp_vals = np.zeros((len(astdys),9))
        print(data)
        pe_df = pd.DataFrame(data,columns = pe_cols)
        print(pe_df)

        pe_df.to_csv('data_files/prop_elem_AstFam_families_multi_sec_hires.csv')
