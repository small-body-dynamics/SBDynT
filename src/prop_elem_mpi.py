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

def prop_calc(j, names_file, datatype):
    print(j)
    
    objname = names_file['Name'].iloc[j]
#    print(objname)
    filename = datatype + '/' + str(j)

    try:
        #fullfile = 'Sims/'+datatype+'/'+str(j)+'/archive_hires.bin'
        fullfile = '../data/'+datatype+'/'+str(j)+'/archive.bin'
        print(fullfile)
        arc1 = rebound.SimulationArchive(fullfile)
        
        nump = len(arc1[0].particles)
        print(nump)
        if nump == 6:
            series = bin_to_df.bin_to_df(datatype,str(j),arc1,names_file,'4planet')
        elif nump == 10:
            series = bin_to_df.bin_to_df(datatype,str(j),arc1,names_file,'8planet')
        ds = int(len(series)/10)

#        series = series[int(5*ds):int(7*ds)]

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
    t = t - t[0]
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
    freqlim = 1./2000.
    #disregard frequencies for which any planet has power at higher than 10% the max
    pth = 0.25
    
    spread = 2
       
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
  
    gind = np.argmax(np.abs(Yh[1:]))+1    
    sind = np.argmax(np.abs(Yp[1:]))+1
    g = freq[gind]  
    s = freq[sind]

    g5 = 4.25749319/rev
    g6 = 28.24552984/rev
    g7 = 3.08675577/rev
    g8 = 0.67255084/rev
    s6 = -26.34496354/rev
    s7 = -2.99266093/rev
    s8 = -0.69251386/rev

    #print(g,s,g6,s6)
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
    #print('freq min/max:', freq[0],freq[-1])
    
    #print(g5,g6,g7,g8,z1,z2,z3,z4,z5,z7,z8,z9)
    #print(s6,s7,s8,z1,z2,z3,z6,z8,z9)
    #secresind1 = [np.where(freq >= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0],np.where(freq >= z4)[0][0]]
    #secresind2 = [np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0],np.where(freq >= z4)[0][0]]

        
    #Actual Milani and Knezevic
    #secresind1 = [np.where(freq >= g5)[0][0],np.where(freq >= g6)[0][0],np.where(freq >= g7)[0][0],np.where(freq >= g8)[0][0],np.where(freq >= z1)[0][0],np.where(freq >= z2)[0][0],np.where(freq >= z3)[0][0],np.where(freq >= z4)[0][0],np.where(freq >= z5)[0][0],np.where(freq >= z7)[0][0],np.where(freq >= z8)[0][0],np.where(freq >= z9)[0][0]]
    #secresind2 = [np.where(freq >= s6)[0][0],np.where(freq >= s7)[0][0],np.where(freq >= s8)[0][0],np.where(freq >= z1)[0][0],np.where(freq >= z2)[0][0],np.where(freq >= z3)[0][0],np.where(freq >= z6)[0][0],np.where(freq >= z8)[0][0],np.where(freq >= z9)[0][0]]
    
            #'''
            #'''


    #freq1 = [g5,g6,g7,g8,z1_g,z2_g,z3_g,z4_g,z5_g,z8_g,z9_g]
    #freq2 = [s6,s7,s8,z2_s,z3_s,z6_s]

    freq1 = [g5,g6,g7,g8,z1,z2,z3,z4,z5,z7,z8,z9]
    freq2 = [s6,s7,s8,z1,z2,z3,z6,z8,z9]

    #freq1 = [g5,g6,g7,g8,s6,s7,s8,z1,z2,z3,z4,z5,z7,z8,z9]
    #freq2 = [g5,g6,g7,g8,s6,s7,s8,z1,z2,z3,z6,z8,z9]

    #print('1:',secresind2)
    secresind1 = []
    secresind2 = []
    for i in freq1:
        #print(i)
        try:
            secresind1.append(np.where(freq>=i)[0][0])
        except:
            continue
    for i in freq2:
        #if 1/i < 10000:
        #    continue
        try:
            secresind2.append(np.where(freq>=i)[0][0])
        except:
            continue


    limit_ind = np.where(freq >= freqlim)[0]

    for i in range(len(secresind1)):
        if secresind1[i] == gind:
            continue
        if abs(secresind1[i] - gind) < 4:
            continue

        if spread > 0:
            Yh_f[secresind1[i]-spread:secresind1[i]+spread] = 0
            Yk_f[secresind1[i]-spread:secresind1[i]+spread] = 0
        else:
            Yh_f[secresind1[i]] = 0
            Yk_f[secresind1[i]] = 0
            
    for i in range(len(secresind2)):
        if secresind2[i] == sind:
            continue
        if abs(secresind2[i] - sind) < 4:
            continue

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
    Ya_f[limit_ind] = 0
    
    p_f = np.fft.irfft(Yp_f,len(p))
    q_f = np.fft.irfft(Yq_f,len(q))
    h_f = np.fft.irfft(Yh_f,len(h))
    k_f = np.fft.irfft(Yk_f,len(k))
    a_f = np.fft.irfft(Ya_f,len(a))

    #print(names_file['sinI'],j)
    sini_f = np.sqrt(p_f*p_f + q_f*q_f)
    ecc_f = np.sqrt(h_f*h_f + k_f*k_f)
    #astsinI = names_file['sinI'][j]
    #astecc = names_file['e'][j]    

    
    outputs =  [objname,np.mean(e),np.mean(np.sin(inc)),np.mean(ecc_f),np.mean(sini_f),np.mean(a_f)]
    
    series_perm = series.copy()
    for i in range(9):
        series=series_perm[int(i*ds):int((i+2)*ds)]
        t = series['t'].values
        t = t - t[0]
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
        freqlim = 1./2000.
        #disregard frequencies for which any planet has power at higher than 10% the max
        pth = 0.25
        
        spread = 2
           
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
      
        gind = np.argmax(np.abs(Yh[1:]))+1    
        sind = np.argmax(np.abs(Yp[1:]))+1
        g = freq[gind]  
        s = freq[sind]
    
        g5 = 4.25749319/rev
        g6 = 28.24552984/rev
        g7 = 3.08675577/rev
        g8 = 0.67255084/rev
        s6 = -26.34496354/rev
        s7 = -2.99266093/rev
        s8 = -0.69251386/rev
    
        #print(g,s,g6,s6)
        z1 = abs(g+s-g6-s6)
        z2 = abs(g+s-g5-s7)
        z3 = abs(g+s-g5-s6)
        z4 = abs(g-2*g6+g5)
        z5 = abs(g-2*g6+g7)
        z6 = abs(s-s6-g5+g6)
        z7 = abs(g-3*g6+2*g5)
        z8 = abs(2*(g-g6)+s-s6)
        z9 = abs(3*(g-g6)+s-s6)
    
    
        freq1 = [g5,g6,g7,g8,z1,z2,z3,z4,z5,z7,z8,z9]
        freq2 = [s6,s7,s8,z1,z2,z3,z6,z8,z9]
    
        secresind1 = []
        secresind2 = []
        for i in freq1:
            #print(i)
            try:
                secresind1.append(np.where(freq>=i)[0][0])
            except:
                continue
        for i in freq2:
            #if 1/i < 10000:
            #    continue
            try:
                secresind2.append(np.where(freq>=i)[0][0])
            except:
                continue
    
    
        limit_ind = np.where(freq >= freqlim)[0]
    
        for i in range(len(secresind1)):
            if secresind1[i] == gind:
                continue
            if abs(secresind1[i] - gind) < 4:
                continue
    
            if spread > 0:
                Yh_f[secresind1[i]-spread:secresind1[i]+spread] = 0
                Yk_f[secresind1[i]-spread:secresind1[i]+spread] = 0
            else:
                Yh_f[secresind1[i]] = 0
                Yk_f[secresind1[i]] = 0
                
        for i in range(len(secresind2)):
            if secresind2[i] == sind:
                continue
            if abs(secresind2[i] - sind) < 4:
                continue
    
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
        Ya_f[limit_ind] = 0
        
        p_f = np.fft.irfft(Yp_f,len(p))
        q_f = np.fft.irfft(Yq_f,len(q))
        h_f = np.fft.irfft(Yh_f,len(h))
        k_f = np.fft.irfft(Yk_f,len(k))
        a_f = np.fft.irfft(Ya_f,len(a))
    
        #print(names_file['sinI'],j)
        sini_f = np.sqrt(p_f*p_f + q_f*q_f)
        ecc_f = np.sqrt(h_f*h_f + k_f*k_f)
   
        outputs.append([np.mean(ecc_f),np.mean(sini_f),np.mean(a_f)])

    return outputs

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
        datatype = sys.argv[1]   
        names_file = pd.read_csv('../data/data_files/'+datatype+'_data.csv')
        pe_cols = ['Name','obs_ecc','obs_sinI','calc_ecc','calc_sinI','calc_sma']
        for i in range(9):
            string = str(i)+str(i+2)
            pe_cols.append([string+'_ecc',string+'_sinI',string+'_sma'])
        filename = names_file['Name'].iloc[0]
        #series = pd.read_csv('TNOs/'+str(filename)+'/series.csv')

        multi_prop = functools.partial(prop_calc, names_file=names_file, datatype=datatype)
        j = range(len(names_file))
        #j = range(22,23)
        #begin = datetime.now()
        data = pool.map(multi_prop, j)
        gp_vals = np.zeros((len(names_file),9))
        print(data,pe_cols)
        pe_df = pd.DataFrame(data,columns = pe_cols)
        #print(pe_df)


        pe_df.to_csv('../data/data_files/prop_elem_'+datatype+'.csv')

