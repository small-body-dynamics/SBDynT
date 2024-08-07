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

def prop_calc(objname, filename='Single',objdes=None):
    
    """
    Calculate prop elements of small celestial bodies from simulation archive files, using a given file list of names.

    Parameters:
        objname (int): Name/designation of the celestial body in the dataset.
        filename (str): Name of the file containing the list of names, and the directory containing the arxhive.bin files. 

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
        print(fullfile)
        archive = rebound.Simulationarchive(fullfile)
        
        try:
            earth = archive[0].particles['earth']
            small_planets_flag = True
        except:
            small_planets_flag = False
        
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
    
    try:
        dt = t_init[1]
    
        n = len(h_init)
        if n < 10001:
            print(n)
        freq = np.fft.fftfreq(n,d=dt)
        freqn = np.fft.rfftfreq(len(a_init),d=dt)
        rev = 1296000
    
        #particle eccentricity vectors
        Yhk= np.fft.fft(k_init+1j*h_init)
        Ypq = np.fft.fft(q_init+1j*p_init)
        Ya_f = np.fft.rfft(a_init)
        
        Ypq[0]=0
        #Yq[0]=0
        #Yh[0]=0
        #Yk[0]=0
      
        imax = len(Ypq)
        #disregard antyhing with a period shorter than 5000 years
        freqlim = 1./2000.
        #disregard frequencies for which any planet has power at higher than 10% the max
        pth = 0.25
           
        #print(hk_ind,pq_ind)
        pYhk = np.abs(Yhk)**2
        pYpq = np.abs(Ypq)**2
        
        #make copies of the FFT outputs
        Ypq_f = Ypq.copy()
        Yhk_f = Yhk.copy()
      
        gind = np.argmax(np.abs(Yhk[1:])**2)+1    
        sind = np.argmax(np.abs(Ypq[1:])**2)+1
        g = freq[gind]  
        s = freq[sind]
        
    except:
        return [objname,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    # g1s1 -> g4s4 taken from Murray and Dermott SSD Table 7.1
    g1 = 5.46326/rev
    s1 = -5.20154/rev
    g2 = 7.34474/rev
    s2 = -6.57080/rev
    g3 = 17.32832/rev
    s3 = -18.74359/rev
    g4 = 18.00233/rev
    s4 = -17.63331/rev
    
    # g5s6 -> g8s8 taken directly from OrbFit software.
    g5 = 4.25749319/rev
    g6 = 28.24552984/rev
    g7 = 3.08675577/rev
    g8 = 0.67255084/rev
    s6 = -26.34496354/rev
    s7 = -2.99266093/rev
    s8 = -0.69251386/rev

    #print(g,s,g6,s6)
    z1 = (g+s-g6-s6)
    z2 = (g+s-g5-s7)
    z3 = (g+s-g5-s6)
    z4 = (2*g6-g5)
    z5 = (2*g6-g7)
    z6 = (s-s6-g5+g6)
    z7 = (g-3*g6+2*g5)
    
    z8 = (2*(g-g6)+s-s6)
    z9 = (3*(g-g6)+s-s6)
    z10 = ((g-g6)+s-s6)

    z11 = g-2*g7+g6
    z12 = 2*g-2*g5
    z13 = -4*g+4*g7
    z14 = -2*s-s6

    #if small_planets_flag:
    #   freq1 = [(g1),(g2),(g3),(g4),(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,z8,z9]
    #    freq2 = [(s1),(s2),(s3),(s4),(s6),(s7),(s8),z1,z2,z3,z6,z8,z9]
    #else:
    #    freq1 = [(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,z8,z9]
    #    freq2 = [(s6),(s7),(s8),z1,z2,z3,z6,z8,z9]
        
    if small_planets_flag:
        freq1 = [(g1),(g2),(g3),(g4),(g5),(g6),(g7),(g8)]
        freq2 = [(s1),(s2),(s3),(s4),(s6),(s7),(s8)]
    else:
        freq1 = [(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,z8,z9]
        freq2 = [(s6),(s7),(s8),z1,z2,z3,z6,z8,z9]
               
    #if small_planets_flag:
    #    freq1 = [(g1),(g2),(g3),(g4),(g5),(g6),(g7),(g8),g-z8,g-z9,g-z10,z11,z12,z13]
    #    freq2 = [(s1),(s2),(s3),(s4),(s6),(s7),(s8),s-z8,s-z9,s-z10,z14]
    #else:
    #    freq1 = [(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,g-z8,g-z9,g-z10,z11,z12,z13]
    #    freq2 = [(s6),(s7),(s8),z1,z2,z3,z6,z8,z9,s-z8,s-z9,s-z10,z14]
    
    secresind1 = []
    secresind2 = []
    for i in freq1:
        try:
            secresind1.append((np.abs(freq - i)).argmin())
        except:
            continue
    for i in freq2:
        try:
            secresind2.append((np.abs(freq - i)).argmin())
        except:
            continue

    ############################################################################
    #Testg find_peaks
        
    #from scipy.signal import find_peaks
    #secresind1, _ = find_peaks((np.abs(Yh[:200])**2), distance=8,threshold=40)
    #secresind2, _ = find_peaks((np.abs(Yp[:200])**2), distance=8,threshold=40)
    ############################################################################
    
    #spread = 1
    #while int(1/(freq[gind+spread]-freq[gind-spread])/dt) > 2500:
    #    spread = spread+1
      
    #freq_dist_lim = 1
    #while int(1/(freq[gind+freq_dist_lim]-freq[gind-freq_dist_lim])/dt) > 1250:
    #    freq_dist_lim = freq_dist_lim+1
    
    spread_dist = 1.10
    freq_dist = 1.15
    spreads = np.zeros(len(secresind1)+len(secresind2))
    spreads = spreads.astype(int)
    freq_dist_lims = np.zeros(len(secresind1)+len(secresind2))
    freq_dist_lims = freq_dist_lims.astype(int)
    
    for i in range(len(secresind1)): 
        if freq[secresind1[i]] > 0:
            while ((freq[secresind1[i]+spreads[i]]/freq[secresind1[i]-spreads[i]])) < spread_dist:
                spreads[i] = spreads[i]+1   
                if secresind1[i]+spreads[i] >= len(freq):
                    break
            while ((freq[secresind1[i]+freq_dist_lims[i]]/freq[secresind1[i]-freq_dist_lims[i]])) < freq_dist:
                freq_dist_lims[i] = freq_dist_lims[i]+1
                if secresind1[i]+freq_dist_lims[i] >= len(freq):
                    break
        else:
            while ((freq[secresind1[i]-spreads[i]]/freq[secresind1[i]+spreads[i]])) < spread_dist:
                spreads[i] = spreads[i]+1   
                if secresind1[i]+spreads[i] >= len(freq):
                    break               
            while ((freq[secresind1[i]-freq_dist_lims[i]]/freq[secresind1[i]+freq_dist_lims[i]])) < freq_dist:
                freq_dist_lims[i] = freq_dist_lims[i]+1
                if secresind1[i]+freq_dist_lims[i] >= len(freq):
                     break
                
    for i in range(len(secresind2)):
        if freq[secresind2[i]] > 0:
            while ((freq[secresind2[i]+spreads[i+len(secresind1)]]/freq[secresind2[i]-spreads[i+len(secresind1)]])) < spread_dist:
            #print(i,((freqn[secresind2[i]-spreads[i+len(secresind1)]])/(freqn[secresind2[i]+spreads[i+len(secresind1)]])))
                spreads[i+len(secresind1)] = spreads[i+len(secresind1)]+1
                if secresind2[i]+spreads[i+len(secresind1)] >= len(freq):
                    break
            while ((freq[secresind2[i]+freq_dist_lims[i+len(secresind1)]]/freq[secresind2[i]-freq_dist_lims[i+len(secresind1)]])) < freq_dist:
                freq_dist_lims[i+len(secresind1)] = freq_dist_lims[i+len(secresind1)]+1
                if secresind2[i]+freq_dist_lims[i+len(secresind1)] >= len(freq):
                    break
        else:
            while ((freq[secresind2[i]-spreads[i+len(secresind1)]]/freq[secresind2[i]+spreads[i+len(secresind1)]])) < spread_dist:
            #print(i,((freqn[secresind2[i]-spreads[i+len(secresind1)]])/(freqn[secresind2[i]+spreads[i+len(secresind1)]])))
                spreads[i+len(secresind1)] = spreads[i+len(secresind1)]+1
                if secresind2[i]+spreads[i+len(secresind1)] >= len(freq):
                    break
            while ((freq[secresind2[i]-freq_dist_lims[i+len(secresind1)]]/freq[secresind2[i]+freq_dist_lims[i+len(secresind1)]])) < freq_dist:
                freq_dist_lims[i+len(secresind1)] = freq_dist_lims[i+len(secresind1)]+1
                if secresind2[i]+freq_dist_lims[i+len(secresind1)] >= len(freq):
                    break

    limit_ind = np.where(freq >= freqlim)[0]
    limit_indr = np.where(freqn >= freqlim)[0]

    '''
    for i in range(len(secresind1)):
        if secresind1[i] == gind:
            continue
        if abs(secresind1[i] - gind) < freq_dist_lim:
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
        if abs(secresind2[i] - sind) < freq_dist_lim:
            continue

        if spread > 0:
            Yp_f[secresind2[i]-spread:secresind2[i]+spread] = 0
            Yq_f[secresind2[i]-spread:secresind2[i]+spread] = 0
        else:
            Yp_f[secresind2[i]] = 0
            Yq_f[secresind2[i]] = 0
    '''        
    ##########################################################################################
    #Using different spreads for different values
    for i in range(len(secresind1)):
        if secresind1[i] == gind:
            continue
        if abs(secresind1[i] - gind) < freq_dist_lims[i]:
            continue
    
        if spreads[i] > 0:
            Yhk_f[secresind1[i]-spreads[i]:secresind1[i]+spreads[i]] = 0
        else:
            Yhk_f[secresind1[i]] = 0
                
    for i in range(len(secresind2)):
        if secresind2[i] == sind:
            continue
        if abs(secresind2[i] - sind) < freq_dist_lims[i+len(secresind1)]:
            continue
    
        if spreads[i+len(secresind1)] > 0:
            Ypq_f[secresind2[i]-spreads[len(secresind1)+i]:secresind2[i]+spreads[len(secresind1)+i]] = 0
        else:
            Ypq_f[secresind2[i]] = 0
    ##########################################################################################
    Ypq_f[limit_ind] = 0
    Yhk_f[limit_ind] = 0
    Ya_f[limit_indr] = 0
    
    pq_f = np.fft.ifft(Ypq_f,len(p_init))
    hk_f = np.fft.ifft(Yhk_f,len(h_init))
    a_f = np.fft.irfft(Ya_f,len(a_init))

    sini_f = np.abs(pq_f)
    ecc_f = np.abs(hk_f) 

    h_mom = np.sqrt(a_f*(1-ecc_f**2))*np.cos(np.arcsin(sini_f))+1/2/a_f
    
    outputs =  np.array([np.nanmean(a_init),np.nanmean(e_init),np.nanmean(np.sin(inc_init)),np.nanmean(np.sqrt(a_init*(1-e_init**2))),np.nanmean(a_f),np.mean(ecc_f),np.nanmean(sini_f),np.nanmean(h_mom)])
    
    error_list = np.zeros((9,4))
    ds = int(len(t_init)/10)
    for j in range(9):
        t = t_init.copy()[int(j*ds):int((j+2)*ds)]
        t = t - t[0]
        
        a = a_init.copy()[int(j*ds):int((j+2)*ds)]
        #an = series['an'].values
    #    print(series)
        e = e_init.copy()[int(j*ds):int((j+2)*ds)]
        inc = inc_init.copy()[int(j*ds):int((i+2)*ds)]
        
        h = h_init.copy()[int(j*ds):int((j+2)*ds)]
        k = k_init.copy()[int(j*ds):int((j+2)*ds)]
        p = p_init.copy()[int(j*ds):int((j+2)*ds)]
        q = q_init.copy()[int(j*ds):int((j+2)*ds)]
        #print(t)
        
        dt = t[1]
        n = len(h)
        #print('dt = ', dt)
        #print('n = ', n)
        freq = np.fft.fftfreq(n,d=dt)
        freqn = np.fft.fftfreq(n,d=dt)
        #print(len(freq), 'L: length of frequency array')
        rev = 1296000
    
        #particle eccentricity vectors
        Yhk= np.fft.fft(k+1j*h)
        Ypq = np.fft.fft(q+1j*p)
        Ya_f = np.fft.rfft(a)
        
        #Ypq[0]=0
        #Yhk[0]=0
        
      
        imax = len(Ypq)
        #disregard antyhing with a period shorter than 5000 years
        #freqlim = 1./2000.
        #disregard frequencies for which any planet has power at higher than 10% the max
        pth = 0.25
        
        spread = 2
           
        #print(hk_ind,pq_ind)
        pYhk = np.abs(Yhk)**2
        pYpq = np.abs(Ypq)**2
        
        #make copies of the FFT outputs
        Ypq_f = Ypq.copy()
        Yhk_f = Yhk.copy()
      
        gind = np.argmax(np.abs(Yhk[1:]))+1    
        sind = np.argmax(np.abs(Ypq[1:]))+1
        g = freq[gind]  
        s = freq[sind]
        
        #print(g,s,g6,s6)
        z1 = (g+s-g6-s6)
        z2 = (g+s-g5-s7)
        z3 = (g+s-g5-s6)
        z4 = (2*g6-g5)
        z5 = (2*g6-g7)
        z6 = (s-s6-g5+g6)
        z7 = (g-3*g6+2*g5)
        z8 = (2*(g-g6)+s-s6)
        z9 = (3*(g-g6)+s-s6)
        z10 = ((g-g6)+s-s6)
    
        z11 = g-2*g7+g6
        z12 = 2*g-2*g5
        z13 = -4*g+4*g7
        z14 = -2*s-s6
    
        #if small_planets_flag:
        #    freq1 = [(g1),(g2),(g3),(g4),(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,z8,z9]
        #    freq2 = [(s1),(s2),(s3),(s4),(s6),(s7),(s8),z1,z2,z3,z6,z8,z9]
        #else:
        #    freq1 = [(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,z8,z9]
        #    freq2 = [(s6),(s7),(s8),z1,z2,z3,z6,z8,z9]
            
        if small_planets_flag:
            freq1 = [(g1),(g2),(g3),(g4),(g5),(g6),(g7),(g8)]
            freq2 = [(s1),(s2),(s3),(s4),(s6),(s7),(s8)]
        else:
            freq1 = [(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,z8,z9]
            freq2 = [(s6),(s7),(s8),z1,z2,z3,z6,z8,z9]
                   
        #if small_planets_flag:
        #    freq1 = [(g1),(g2),(g3),(g4),(g5),(g6),(g7),(g8),g-z8,g-z9,g-z10,z11,z12,z13]
        #    freq2 = [(s1),(s2),(s3),(s4),(s6),(s7),(s8),s-z8,s-z9,s-z10,z14]
        #else:
        #    freq1 = [(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,g-z8,g-z9,g-z10,z11,z12,z13]
        #    freq2 = [(s6),(s7),(s8),z1,z2,z3,z6,z8,z9,s-z8,s-z9,s-z10,z14]
    
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
    
        spreads = np.ones(len(secresind1)+len(secresind2))
        spreads = spreads.astype(int)
        freq_dist_lims = np.ones(len(secresind1)+len(secresind2))
        freq_dist_lims = freq_dist_lims.astype(int)
    
        spreads = np.zeros(len(secresind1)+len(secresind2))
        spreads = spreads.astype(int)
        freq_dist_lims = np.zeros(len(secresind1)+len(secresind2))
        freq_dist_lims = freq_dist_lims.astype(int)
    
        for i in range(len(secresind1)): 
            if freq[secresind1[i]] > 0:
                while ((freq[secresind1[i]+spreads[i]]/freq[secresind1[i]-spreads[i]])) < spread_dist:
                    spreads[i] = spreads[i]+1   
                    if secresind1[i]+spreads[i] >= len(freq):
                        break
                while ((freq[secresind1[i]+freq_dist_lims[i]]/freq[secresind1[i]-freq_dist_lims[i]])) < freq_dist:
                    freq_dist_lims[i] = freq_dist_lims[i]+1
                    if secresind1[i]+freq_dist_lims[i] >= len(freq):
                        break
            else:
                while ((freq[secresind1[i]-spreads[i]]/freq[secresind1[i]+spreads[i]])) < spread_dist:
                    spreads[i] = spreads[i]+1   
                    if secresind1[i]+spreads[i] >= len(freq):
                        break               
                while ((freq[secresind1[i]-freq_dist_lims[i]]/freq[secresind1[i]+freq_dist_lims[i]])) < freq_dist:
                    freq_dist_lims[i] = freq_dist_lims[i]+1
                    if secresind1[i]+freq_dist_lims[i] >= len(freq):
                         break
                    
        for i in range(len(secresind2)):
            if freq[secresind2[i]] > 0:
                while ((freq[secresind2[i]+spreads[i+len(secresind1)]]/freq[secresind2[i]-spreads[i+len(secresind1)]])) < spread_dist:
                #print(i,((freqn[secresind2[i]-spreads[i+len(secresind1)]])/(freqn[secresind2[i]+spreads[i+len(secresind1)]])))
                    spreads[i+len(secresind1)] = spreads[i+len(secresind1)]+1
                    if secresind2[i]+spreads[i+len(secresind1)] >= len(freq):
                        break
                while ((freq[secresind2[i]+freq_dist_lims[i+len(secresind1)]]/freq[secresind2[i]-freq_dist_lims[i+len(secresind1)]])) < 1.250:
                    freq_dist_lims[i+len(secresind1)] = freq_dist_lims[i+len(secresind1)]+1
                    if secresind2[i]+freq_dist_lims[i+len(secresind1)] >= len(freq):
                        break
            else:
                while ((freq[secresind2[i]-spreads[i+len(secresind1)]]/freq[secresind2[i]+spreads[i+len(secresind1)]])) < spread_dist:
                #print(i,((freqn[secresind2[i]-spreads[i+len(secresind1)]])/(freqn[secresind2[i]+spreads[i+len(secresind1)]])))
                    spreads[i+len(secresind1)] = spreads[i+len(secresind1)]+1
                    if secresind2[i]+spreads[i+len(secresind1)] >= len(freq):
                        break
                while ((freq[secresind2[i]-freq_dist_lims[i+len(secresind1)]]/freq[secresind2[i]+freq_dist_lims[i+len(secresind1)]])) < 1.250:
                    freq_dist_lims[i+len(secresind1)] = freq_dist_lims[i+len(secresind1)]+1
                    if secresind2[i]+freq_dist_lims[i+len(secresind1)] >= len(freq):
                        break
                        
        limit_ind = np.where(freq >= freqlim)[0]
        limit_indr = np.where(freqn >= freqlim)[0]
     
        ############################################################################
        #Testg find_peaks
        
        #from scipy.signal import find_peaks
        #secresind1, _ = find_peaks((np.abs(Yh[:200])**2), distance=8,threshold=40)
        #secresind2, _ = find_peaks((np.abs(Yp[:200])**2), distance=8,threshold=40)
        ##########################################################################3
        '''
        for i in range(len(secresind1)):
            if secresind1[i] == gind:
                continue
            if abs(secresind1[i] - gind) < freq_dist_lim:
                continue
    
            if spread > 0:
                Yhk_f[secresind1[i]-spread:secresind1[i]+spread] = 0
            else:
                Yhk_f[secresind1[i]] = 0
                
        for i in range(len(secresind2)):
            if secresind2[i] == sind:
                continue
            if abs(secresind2[i] - sind) < freq_dist_lim:
                continue
    
            if spread > 0:
                Ypq_f[secresind2[i]-spread:secresind2[i]+spread] = 0
            else:
                Ypq_f[secresind2[i]] = 0
        '''
        ##########################################################################################
        #Using different spreads for different values
        for i in range(len(secresind1)):
            if secresind1[i] == gind:
                continue
            if abs(secresind1[i] - gind) < freq_dist_lims[i]:
                continue
        
            if spreads[i] > 0:
                Yhk_f[secresind1[i]-spreads[i]:secresind1[i]+spreads[i]] = 0
            else:
                Yhk_f[secresind1[i]] = 0
                    
        for i in range(len(secresind2)):
            if secresind2[i] == sind:
                continue
            if abs(secresind2[i] - sind) < freq_dist_lims[i+len(secresind1)]:
                continue
        
            if spreads[i+len(secresind1)] > 0:
                Ypq_f[secresind2[i]-spreads[len(secresind1)+i]:secresind2[i]+spreads[len(secresind1)+i]] = 0
            else:
                Ypq_f[secresind2[i]] = 0
        ##########################################################################################
        Ypq_f[limit_ind] = 0
        Yhk_f[limit_ind] = 0
        Ya_f[limit_indr] = 0
        
        pq_f = np.fft.ifft(Ypq_f,len(p))
        hk_f = np.fft.ifft(Yhk_f,len(h))
        a_f = np.fft.irfft(Ya_f,len(a))

        sini_f = np.abs(pq_f)
        ecc_f = np.abs(hk_f)
        
        h_mom = np.sqrt(a_f*(1-ecc_f**2))*np.cos(np.arcsin(sini_f))+1/2/a_f
       
        error_list[j][0] = np.nanmean(a_f)
        error_list[j][1] = np.nanmean(ecc_f)
        error_list[j][2] = np.nanmean(sini_f)
        error_list[j][3] = np.nanmean(h_mom)
    #print(outputs)
    #errors = outputs[7:]
    #print(error_list,outputs[-3:])
    #print(np.array(error_list)-np.array(outputs[-3:]))
    #print(
          
    rms = np.sqrt(np.nanmean((np.array(error_list)-np.array(outputs[-4:]))**2,axis=0))
    
    maxvals = np.max(np.array(error_list)-np.array(outputs[-4:]),axis=0)
    #outputs.append(outputs,np.array([rms,maxvals]))
    #print(outputs,error_list,rms,maxvals)
    return_data = [objname]
    for i in range(len(outputs)):
        return_data.append(outputs[i])
    for i in range(len(error_list)):
        for j in range(len(error_list[0])):
            return_data.append(error_list[i][j])
    return_data.append(rms[0])
    return_data.append(rms[1])
    return_data.append(rms[2])
    return_data.append(rms[3])
    return_data.append(maxvals[0])
    return_data.append(maxvals[1])
    return_data.append(maxvals[2])
    return_data.append(maxvals[3])
    return return_data

def prop_multi(filename):
    names_df = pd.read_csv('../data/data_files/'+filename+'.csv')
    data = []
    for i,objname in enumerate(names_df['Name']):
        #archive = rebound.SimulationArchive(fullfile)
        data_line = prop_calc(objname,filename)
        #print(data_line)
        data.append(data_line)
    column_names = ['Objname','ObsSMA','ObsEcc','ObsSin(Inc)','Obs_h','PropSMA','PropEcc','PropSin(Inc)','Prop_h']
    for i in range(9):
        numrange = str(i)+'_'+str(i+2)+'PE'
        column_names.append(numrange+'_a')
        column_names.append(numrange+'_e')
        column_names.append(numrange+'_sinI')
        column_names.append(numrange+'_h')
            #print(numrange)
    column_names.append('RMS_err_a')
    column_names.append('RMS_err_e')
    column_names.append('RMS_err_sinI')
    column_names.append('RMS_err_h')
    column_names.append('Delta_a')
    column_names.append('Delta_e')
    column_names.append('Delta_sinI')
    column_names.append('Delta_h')
    data_df = pd.DataFrame(data,columns=column_names)
    data_df.to_csv('../data/results/'+filename+'_prop_elem.csv')
    return data

if __name__ == "__main__":
    filename = str(sys.argv[1])
    
    if filename != 'Single':
        data = prop_multi(filename)  
    else:
        column_names = ['Objname','ObsSMA','ObsEcc','ObsSin(Inc)','Obs_h','PropSMA','PropEcc','PropSin(Inc)','Prop_h']
        for i in range(9):
            numrange = str(i)+'_'+str(i+2)+'PE'
            column_names.append(numrange+'_a')
            column_names.append(numrange+'_e')
            column_names.append(numrange+'_sinI')
            column_names.append(numrange+'_h')
            #print(numrange)
        column_names.append('RMS_err_a')
        column_names.append('RMS_err_e')
        column_names.append('RMS_err_sinI')
        column_names.append('RMS_err_h')
        column_names.append('Delta_a')
        column_names.append('Delta_e')
        column_names.append('Delta_sinI')
        column_names.append('Delta_h')
        objname = str(sys.argv[2])
        fullfile = '../data/'+filename+'/'+objname+'/archive.bin'
        #archive = rebound.SimulationArchive(fullfile)
        data_line = prop_calc(objname,fullfile)
        data_df = pd.DataFrame(data_line,columns = column_names)
        data_df.to_csv('../data/Single/'+objname+'/'+objname+'_prop_elem_prop.csv')
        
                       

        
