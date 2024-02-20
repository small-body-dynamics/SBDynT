import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, '../src')
import rebound
import numpy as np
import horizons_api
import tools
import warnings
warnings.filterwarnings("ignore")
import scipy.signal as signal

import run_reb
import tools

import functools
import schwimmbad

def prop_multi(filename):
    names_df = pd.read_csv('../data/data_files/'+filename+'_data.csv')
    data = []
    for i,objname in enumerate(names_df['Name']):
        fullfile = '../data/'+filename+'/'+str(objname)+'/archive.bin'
        #archive = rebound.SimulationArchive(fullfile)
        data_line = prop_calc(str(objname),filename)
        #print(data_line)
        data.append(data_line)
    column_names = ['Objname','ObsEcc','ObsSin(Inc)','PropEcc','PropSin(Inc)','PropSMA','0_2PE','1_3PE','2_4PE','3_5PE','4_6PE','5_7PE','6_8PE','7_9PE','8_10PE']
    #print(data)
    data_df = pd.DataFrame(data,columns=column_names)
    data_df.to_csv('../data/results/'+filename+'_prop_elem.csv')
        



def prop_calc(objname, filename='Single',objdes=None):
    
    """
    Calculate prop elements of small celestial bodies from simulation archive files, using a given file list of names.

    Parameters:
        objname (int): Index of the celestial body in the dataset.
        datatype (str): Name of the file containing the list of names, and the directory containing the arxhive.bin files. 

    Returns:
        outputs: A list containing calculated proper elements, or  
        - objname
        - Observed Eccentricity
        - Observed Sin(Inclination)
        - Calculated Proper Eccentricity
        - Calculated Proper Sin(Inclination)
        - Calculated Proper Semimajor Axis 
        - Running Block Proper Elements [Calc Proper Ecc, Calc Proper Sin(Inc), Calc Proper SMA]
        
        The Running Block calculations are used to calculate a mean error for the proper elements. The default run will be a 100 Myr integration, with running blocks from 0-20,10-30,...80-100 Myr, producing 9 running blocks total. So your default run will produce an outputs list consisting of 33 values total.  
        
        If an error occurs in the code, then outputs is instead returned as [objname,0,0,0,0,0,0,0].
        
    """    
#    print(objname)
    try:       
        fullfile = '../data/'+filename+'/'+str(objname)+'/archive.bin'
        #print(fullfile)
        archive = rebound.SimulationArchive(fullfile)
        nump = len(archive[0].particles)
        flag, a_init, e_init, inc_init, lan_init, aop_init, M_init, t_init = tools.read_sa_for_sbody(sbody = str(objname), archivefile=fullfile,nclones=0,tmin=0.,tmax=archive[-1].t)

        
    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno

        error_message = f"An error occurred in {fname} at line {line_number}: {error}"
    
        # Print the error message
        print(error_message)
        return [objname,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    p_init = np.sin(inc_init)*np.sin(lan_init)
    q_init = np.sin(inc_init)*np.cos(lan_init)
    h_init = (e_init)*np.sin(lan_init+aop_init)
    k_init = (e_init)*np.cos(lan_init+aop_init)
    #print(t_init)
    dt = t_init[1]
    n = len(h_init)
    freq = np.fft.rfftfreq(n,d=dt)
    rev = 1296000

    #particle eccentricity vectors
    Yh= np.fft.rfft(h_init)
    Yk = np.fft.rfft(k_init)
    Yp= np.fft.rfft(p_init)
    Yq = np.fft.rfft(q_init)
    Ya_f = np.fft.rfft(a_init)
    
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
        try:
            secresind1.append(np.where(freq>=i)[0][0])
        except:
            continue
    for i in freq2:
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
    
    p_f = np.fft.irfft(Yp_f,len(p_init))
    q_f = np.fft.irfft(Yq_f,len(q_init))
    h_f = np.fft.irfft(Yh_f,len(h_init))
    k_f = np.fft.irfft(Yk_f,len(k_init))
    a_f = np.fft.irfft(Ya_f,len(a_init))

    sini_f = np.sqrt(p_f*p_f + q_f*q_f)
    ecc_f = np.sqrt(h_f*h_f + k_f*k_f) 

    
    outputs =  [objname,np.mean(e_init),np.mean(np.sin(inc_init)),np.mean(ecc_f),np.mean(sini_f),np.mean(a_f)]
    
    ds = int(len(t_init)/10)
    for i in range(9):
        t = t_init.copy()[int(i*ds):int((i+2)*ds)]
        t = t - t[0]
        
        a = a_init.copy()[int(i*ds):int((i+2)*ds)]
        #an = series['an'].values
    #    print(series)
        e = e_init.copy()[int(i*ds):int((i+2)*ds)]
        inc = inc_init.copy()[int(i*ds):int((i+2)*ds)]
        
        h = h_init.copy()[int(i*ds):int((i+2)*ds)]
        k = k_init.copy()[int(i*ds):int((i+2)*ds)]
        p = p_init.copy()[int(i*ds):int((i+2)*ds)]
        q = q_init.copy()[int(i*ds):int((i+2)*ds)]
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

        sini_f = np.sqrt(p_f*p_f + q_f*q_f)
        ecc_f = np.sqrt(h_f*h_f + k_f*k_f)
        
        
   
        outputs.append([np.mean(ecc_f),np.mean(sini_f),np.mean(a_f)])

    return outputs

if __name__ == "__main__":
    filename = str(sys.argv[1])
    
    from schwimmbad import MPIPool
    #print('schwimmbad in')
    with MPIPool() as pool:
        #print('mpipool pooled')    
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        names_df = pd.read_csv('../data/data_files/'+filename+'.csv')
        data = []
        
        objname = np.array(names_df['Name'])
        run = functools.partial(prop_calc, filename=filename)

        #j = range(len(names_df))
        #begin = datetime.now()
        data = pool.map(run, objname)
        #print(data)
        column_names = ['Objname','ObsEcc','ObsSin(Inc)','PropEcc','PropSin(Inc)','PropSMA','0_2PE','1_3PE','2_4PE','3_5PE','4_6PE','5_7PE','6_8PE','7_9PE','8_10PE']
        data_df = pd.DataFrame(data,columns=column_names)
        data_df.to_csv('../data/results/'+filename+'_prop_elem.csv')
        
    