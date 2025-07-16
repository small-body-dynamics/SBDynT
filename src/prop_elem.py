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

from scipy import signal
import run_reb
import tools

av_init = 0;ev_init = 0;incv_init = 0;lanv_init = 0;aopv_init = 0;Mv_init = 0
ae_init = 0;ee_init = 0;ince_init = 0;lane_init = 0;aope_init = 0;Me_init = 0
am_init = 0;e_init = 0;incm_init = 0;lanm_init = 0;aopm_init = 0;Mm_init = 0
aj_init = 0;ej_init = 0;incj_init = 0;lanj_init = 0;aopj_init = 0;Mj_init = 0
as_init = 0;es_init = 0;incs_init = 0;lans_init = 0;aops_init = 0;Ms_init = 0
au_init = 0;eu_init = 0;incu_init = 0;lanu_init = 0;aopu_init = 0;Mu_init = 0
an_init = 0;en_init = 0;incn_init = 0;lann_init = 0;aopn_init = 0;Mn_init = 0


g2 = 0;g3 = 0;g4 = 0;g5 = 0;g6 = 0;g7 = 0;g8 = 0
s2 = 0;s3 = 0;s4 = 0;s6 = 0;s7 = 0;s8 = 0
g_arr = []
s_arr=[]
#small_planets_flag=False

def pe_vals(t,a,p,q,h,k,g_arr,s_arr,small_planets_flag,debug=False):
    
    global g1;global g2;global g3;global g4;global g5;global g6;global g7;global g8    
    global s1;global s2;global s3;global s4;global s6;global s7;global s8
    
    try:
        dt = abs(t[1])
    
        n = len(h)
        #if n < 10001:
        #    print(n)
        freq = np.fft.fftfreq(n,d=dt)
        freqn = np.fft.rfftfreq(len(a),d=dt)
        rev = 1296000
    
        #particle eccentricity vectors
        Yhk= np.fft.fft(k+1j*h)
        Ypq = np.fft.fft(q+1j*p)
        Ya_f = np.fft.rfft(a)
        window = signal.windows.hamming(n)
        #window = signal.windows.general_hamming(n,0.1)
        Yhk_win = Yhk*window
        Ypq_win = Ypq*window
        
        #Yhk_win = Yhk
        #Ypq_win = Ypq
        Ypq[0]=0
        #Yhk[0]=0
        #Yq[0]=0
        #Yh[0]=0
        #Yk[0]=0
      
        imax = len(Ypq)
        #disregard antyhing with a period shorter than 5000 years
        freqlim = 1./100.
        #disregard frequencies for which any planet has power at higher than 10% the max
        pth = 0.25
           
        #print(hk_ind,pq_ind)
        pYhk = np.abs(Yhk)**2
        pYpq = np.abs(Ypq)**2
        
        #make copies of the FFT outputs
        Ypq_f = Ypq.copy()
        Yhk_f = Yhk.copy()
      
        #gind = np.argmax(np.abs(Yhk[1:])**2)+1    
        #sind = np.argmax(np.abs(Ypq[1:])**2)+1
        gind = np.argmax(np.abs(Yhk_win[1:])**2)+1    
        sind = np.argmax(np.abs(Ypq_win[1:])**2)+1
        
        temp1 = Yhk_win[gind]
        temp2 = Ypq_win[sind]
        Yhk_win[gind] = 0
        Ypq_win[sind] = 0
        
        gind2 = np.argmax(np.abs(Yhk_win[1:])**2)+1    
        sind2 = np.argmax(np.abs(Ypq_win[1:])**2)+1
        '''
        if freq[gind]/freq[gind2] > 0:
        
            Yhk_win[gind] = temp1
            summax1hk = np.sum(Yhk_win[gind-3:gind+4])
            summax2hk = np.sum(Yhk_win[gind2-3:gind2+4])
            if summax2hk > summax1hk:
                gind = gind2
                
        if freq[sind]/freq[sind2] > 0:
            Ypq_win[sind] = temp2

            summax1pq = np.sum(Ypq_win[gind-3:gind+4])
            summax2pq = np.sum(Ypq_win[gind2-3:gind2+4])
            if summax2pq > summax1pq:
                sind = sind2
        '''
        g = freq[gind]  
        s = freq[sind]
        '''
        if small_planets_flag:
            g2=g_arr[5],g3=g_arr[4],g4=g_arr[6]
            s2=s_arr[4],s3=s_arr[3],s4=s_arr[5]
            
        g5=g_arr[0],g6=g_arr[1],g7=g_arr[2],g8=g_arr[3]
        s6=s_arr[0],s7=s_arr[1],s8=s_arr[2]
        '''

    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno

        error_message = "An error occurred in at line "+str(line_number)
    
        # Print the error message
        print(error_message)
        print(error)
        return[0,0,0]
    
    z1 = (g+s-g6-s6); z2 = (g+s-g5-s7); z3 = (g+s-g5-s6); z4 = (2*g6-g5); z5 = (2*g6-g7); z6 = (s-s6-g5+g6); z7 = (g-3*g6+2*g5)
        
    z8 = (2*(g-g6)+s-s6); z9 = (3*(g-g6)+s-s6); z10 = ((g-g6)+s-s6)
    
    z11 = g-2*g7+g6; z12 = 2*g-2*g5; z13 = -4*g+4*g7; z14 = -2*s-s6

    if small_planets_flag:
        #freq1 = [(g1),(g2),(g3),(g4),(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,z8,z9,g-z8,g-z9,g-z10,z11,z12,z13]
        #freq2 = [(s1),(s2),(s3),(s4),(s6),(s7),(s8),z1,z2,z3,z6,z8,z9,s-z8,s-z9,s-z10,z14]
        freq1 = [(g2),(g3),(g4),(g5),(g6),(g7),(g8),g-z8,g-z9,g-z10,z11,z12,z13,-g+2*s-g5]
        freq2 = [(s2),(s3),(s4),(s6),(s7),(s8),s-z8,s-z9,s-z10,z14,g-s+g5-s7,g+g5-2*s6,2*g-2*s6]
    else:
        freq1 = [(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,g-z8,g-z9,g-z10,z11,z12]
        freq2 = [(s6),(s7),(s8),z1,z2,z3,z6,z8,z9,s-z8,s-z9,s-z10,z14]
      
    de = [g-g5,g-g6,g5-g6,s-s7,s-s6,s7-s6,g+s-s7-g5,g+s-s7-g6,g+s-s6-g5,g+s-s6-g6,2*g-2*s,g-2*g5+g6,g+g5-2*g6,2*g-g5-g6,-g+s+g5-s7,-g+s+g6-s7,-g+s+g5-s6,-g+s+g6-s6,g-g5+s7-s6, g-g5-s7+s6,g-g6+s7-s6,g-g6-s7+s6,2*g-s-s7,2*g-s-s6,-g+2*s-g5,-g+2*s-g6, 2*g-2*s7,2*g-2*s6, 2*g-s7-s6,g-s+g5-s7,g-s+g5-s6,g-s+g6-s7,g-s+g6-s6,g+g5-2*s7,g+g6-2*s7,g+g5-2*s6,g+g6-2*s6,g+g5-s7-s6,g+g6-s7-s6,s-2*s7+s6,s+s7-2*s6,2*s-s7-s6,s+g5-g6-s7,s-g5+g6-s7,s+g5-g6-s6, s-g5+g6-s6,2*s-2*g5, 2*s-2*g6, 2*s-g5-g6,s-2*g5+s7, s-2*g5+s6,s-2*g6+s7, s-2*g6+s6,s-g5-g6+s7, s-g5-g6+s6,2*g-2*g5,2*g-2*g6, 2*s-2*s7, 2*s-2*s6,g-2*g6+g7,g-3*g6+2*g5, 2*(g-g6)+(s-s6),g+g5-g6-g7,g-g5-g6+g7,g+g5-2*g6-s6+s7,3*(g-g6)+(s-s6)]
    
    #freq1.extend(de)
    #freq2.extend(de)
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
    
    spread_dist1 = 1.05
    spread_dist2 = 1.05
    freq_dist1 = 1.1
    freq_dist2 = 1.1
    
    spread_dist1 = 1.1
    spread_dist2 = 1.1
    freq_dist1 = 1.15
    freq_dist2 = 1.15
    spreads = np.zeros(len(secresind1)+len(secresind2))
    spreads = spreads.astype(int)
    freq_dist_lims = np.zeros(len(secresind1)+len(secresind2))
    freq_dist_lims = freq_dist_lims.astype(int)
    
    ##########################################################################################################################
    #'''
    for i in range(len(secresind1)): 
        if freq[secresind1[i]] > 0:
            
            while ((freq[secresind1[i]+spreads[i]]/freq[secresind1[i]-spreads[i]])) < spread_dist1:
                spreads[i] = spreads[i]+1   
                if secresind1[i]-spreads[i] == 0:
                    break
                if secresind1[i]+spreads[i]>=len(freq):
                    break
                if freq[secresind1[i]+spreads[i]] < 0:
                    #print(freq[secresind1[i]+spreads[i]],secresind1[i],spreads[i])
                    break
            while ((freq[secresind1[i]+freq_dist_lims[i]]/freq[secresind1[i]-freq_dist_lims[i]])) < freq_dist1:
                freq_dist_lims[i] = freq_dist_lims[i]+1
                if secresind1[i]+freq_dist_lims[i] >= len(freq):
                    break
        else:
            while ((freq[secresind1[i]-spreads[i]]/freq[secresind1[i]+spreads[i]])) < spread_dist1:
                spreads[i] = spreads[i]+1   
                if secresind1[i]+spreads[i] >= len(freq):
                    break    
                if freq[secresind1[i]-spreads[i]] > 0:
                    break
            while ((freq[secresind1[i]-freq_dist_lims[i]]/freq[secresind1[i]+freq_dist_lims[i]])) < freq_dist1:
                freq_dist_lims[i] = freq_dist_lims[i]+1
                if secresind1[i]+freq_dist_lims[i] >= len(freq):
                     break
                
    for i in range(len(secresind2)):
        if freq[secresind2[i]] > 0:
            while ((freq[secresind2[i]+spreads[i+len(secresind1)]]/freq[secresind2[i]-spreads[i+len(secresind1)]])) < spread_dist2:
            #print(i,((freqn[secresind2[i]-spreads[i+len(secresind1)]])/(freqn[secresind2[i]+spreads[i+len(secresind1)]])))
                spreads[i+len(secresind1)] = spreads[i+len(secresind1)]+1
                if secresind2[i]-spreads[i+len(secresind1)] == 0:
                    break
                if secresind2[i]+spreads[i+len(secresind1)]>=len(freq):
                    break
                if freq[secresind2[i]+spreads[i+len(secresind1)]] < 0:
                    break
            while ((freq[secresind2[i]+freq_dist_lims[i+len(secresind1)]]/freq[secresind2[i]-freq_dist_lims[i+len(secresind1)]])) < freq_dist2:
                freq_dist_lims[i+len(secresind1)] = freq_dist_lims[i+len(secresind1)]+1
                if secresind2[i]+freq_dist_lims[i+len(secresind1)] >= len(freq):
                    break
        else:
            while ((freq[secresind2[i]-spreads[i+len(secresind1)]]/freq[secresind2[i]+spreads[i+len(secresind1)]])) < spread_dist2:
            #print(i,((freqn[secresind2[i]-spreads[i+len(secresind1)]])/(freqn[secresind2[i]+spreads[i+len(secresind1)]])))
                spreads[i+len(secresind1)] = spreads[i+len(secresind1)]+1
                if secresind2[i]+spreads[i+len(secresind1)] >= len(freq):
                    break
                if freq[secresind2[i]+spreads[i+len(secresind1)]] > 0:
                    break
            while ((freq[secresind2[i]-freq_dist_lims[i+len(secresind1)]]/freq[secresind2[i]+freq_dist_lims[i+len(secresind1)]])) < freq_dist2:
                freq_dist_lims[i+len(secresind1)] = freq_dist_lims[i+len(secresind1)]+1
                if secresind2[i]+freq_dist_lims[i+len(secresind1)] >= len(freq):
                    break
    #''' 
    
##########################################################################################################################
    Yhk_copy = Yhk_f.copy()    
    Ypq_copy = Ypq_f.copy()    
    
    for i in range(len(secresind1)):
        if secresind1[i] == gind:
            #Yhk_f[gind] = 0
            #Yhk_f[gind] = np.nanmean(np.real(Yhk_copy[gind-2:gind+3]))+1j*np.nanmean(np.imag(Yhk_copy[gind-2:gind+3]))
            continue
            
        if abs(secresind1[i] - gind) <= freq_dist_lims[i]:
            #Yhk_f[gind] = np.nanmean(np.real(Yhk_copy[gind-1:gind+2]))+1j*np.nanmean(np.imag(Yhk_copy[gind-1:gind+2]))
            continue
            
        if spreads[i] > 0:
            if secresind1[i] <= 3:
                continue
            if len(Yhk_copy) - secresind1[i] <= 3:
                continue
                
            if secresind1[i]-2*spreads[i]-1 <= 0:
                minarr = Yhk_copy[secresind1[i]+spreads[i]+1:secresind1[i]+2*spreads[i]+1]
            elif secresind1[i]+2*spreads[i]+1 > len(freq):
                minarr = Yhk_copy[secresind1[i]-2*spreads[i]-1:secresind1[i]-spreads[i]-1]
            else:
                minarr = np.array([(Yhk_copy[secresind1[i]-2*spreads[i]-1:secresind1[i]-spreads[i]-1]),(Yhk_copy[secresind1[i]+spreads[i]+1:secresind1[i]+2*spreads[i]+1])],dtype=np.float64)
            #Yhk_f[secresind1[i]-spreads[i]:secresind1[i]+spreads[i]] = 10**(np.nanmedian(np.log10(np.real(minarr))))+1j*10**(np.nanmedian(np.log10(np.imag(minarr))))
            Yhk_f[secresind1[i]-spreads[i]:secresind1[i]+spreads[i]] = np.nanmean(np.real(minarr))+1j*np.nanmean(np.imag(minarr))
            
            #if secresind1[i]-2*spreads[i]-1 <= 0:
            #    minarr = Yhk_copy[secresind1[i]+spreads[i]+1:secresind1[i]+2*spreads[i]+1]
            #elif secresind1[i]+2*spreads[i]+1 > len(freq):
            #    minarr = Yhk_copy[secresind1[i]-2*spreads[i]-1:secresind1[i]-spreads[i]-1]
            #else:
            #    minarr = np.array([(Yhk_copy[secresind1[i]-2*spreads[i]-1:secresind1[i]+2*spreads[i]+1])])
            #bump_start = 
            #Yhk_f[secresind1[i]-spreads[i]:secresind1[i]+spreads[i]] = smooth(minarr, bump_start, bump_end, log_power=True)
            #Yhk_f[secresind1[i]-spreads[i]:secresind1[i]+spreads[i]] = 0
        else:
            if secresind1[i] <= 3:
                continue
            if len(Yhk_copy) - secresind1[i] <= 3:
                continue
            else:
                Yhk_f[secresind1[i]] = np.nanmean(np.real(Yhk_copy[secresind1[i]-1:secresind1[i]+2].astype(float)))+1j*np.nanmean(np.imag(Yhk_copy[secresind1[i]-1:secresind1[i]+2].astype(float)))
                
    for i in range(len(secresind2)):
        if secresind2[i] == sind:
            continue
        if abs(secresind2[i] - sind) < freq_dist_lims[i+len(secresind1)]:
            continue
    
        if spreads[i] > 0:
            if secresind2[i] <= 3:
                continue
            if len(Ypq_copy) - secresind2[i] <= 3:
                continue
            
            if secresind2[i]-2*spreads[i+len(secresind1)]-1 <= 0:
                minarr = Ypq_copy[secresind2[i]+spreads[i+len(secresind1)]+1:secresind2[i]+2*spreads[i+len(secresind1)]+1]
            elif secresind2[i]+2*spreads[i+len(secresind1)]+1 > len(freq):
                minarr = Ypq_copy[secresind2[i]-2*spreads[i+len(secresind1)]-1:secresind2[i]-spreads[i+len(secresind1)]-1]
            else:
                minarr = np.array([(Ypq_copy[secresind2[i]-2*spreads[i+len(secresind1)]-1:secresind2[i]-spreads[i+len(secresind1)]-1]),(Ypq_copy[secresind2[i]+spreads[i+len(secresind1)]+1:secresind2[i]+2*spreads[i+len(secresind1)]+1])],dtype=np.float64)
            #Ypq_f[secresind2[i]-spreads[i+len(secresind1)]:secresind2[i]+spreads[i+len(secresind1)]] = 10**(np.nanmedian(np.log10(np.real(minarr))))+1j*10**(np.nanmedian(np.log10(np.imag(minarr))))
            Ypq_f[secresind2[i]-spreads[i+len(secresind1)]:secresind2[i]+spreads[i+len(secresind1)]] = np.nanmean(np.real(minarr))+1j*np.nanmean(np.imag(minarr))
            #Ypq_f[secresind2[i]-spreads[i+len(secresind1)]:secresind2[i]+spreads[i+len(secresind1)]] = 0
        else:
            if secresind2[i] <= 3:
                continue
            if len(Ypq_copy) - secresind2[i] <= 3:
                continue
            else:
                Ypq_f[secresind2[i]] = np.nanmean(np.real(Ypq_copy[secresind2[i]-1:secresind2[i]+2].astype(float)))+1j*np.nanmean(np.imag(Ypq_copy[secresind2[i]-1:secresind2[i]+2].astype(float)))
######################################################################################################################                
         
    #Yhk_f = smooth_all_regions(
    #fft_signal=Yhk_copy,
    #freqs=freq,
    #bad_freqs=freq1,  # frequencies you want to smooth
    #keep_freqs=[g],              # primary frequency to protect
    #db_thresh=1.8,                 # how far above baseline is "real"
    #max_width=10                  # how far to walk for wings
    #)
    
    #Ypq_f = smooth_all_regions(
    #fft_signal=Ypq_copy,
    #freqs=freq,
    #bad_freqs=freq2,  # frequencies you want to smooth
    #keep_freqs=[s],              # primary frequency to protect
    #db_thresh=1.8,                 # how far above baseline is "real"
    #max_width=10                   # how far to walk for wings
    #)
    
    
    limit_ind = np.where(freq >= freqlim)[0]
    limit_indr = np.where(freqn >= freqlim)[0]
    
    Ypq_f[limit_ind] = 0
    Yhk_f[limit_ind] = 0
    Ya_f[limit_indr] = 0
    
    pq_f = np.fft.ifft(Ypq_f,len(p))
    hk_f = np.fft.ifft(Yhk_f,len(h))
    a_f = np.fft.irfft(Ya_f,len(a))
    sini_f = np.abs(pq_f)
    ecc_f = np.abs(hk_f) 

    pc5 = int(n*0.1)
    
    #outputs =  np.array(np.nanmean(a),np.mean(ecc_f[pc5:-pc5]),np.nanmean(sini_f[pc5:-pc5])])
    #outputs =  np.array([np.nanmedian(a),np.nanmedian(e),np.nanmedian(np.sin(inc)),np.nanmedian(a),np.median(ecc_f[pc5:-pc5]),np.nanmedian(sini_f[pc5:-pc5])])
    if debug==True:
        return [np.nanmean(a),np.nanmean(ecc_f[pc5:-pc5]),np.nanmean(sini_f[pc5:-pc5])],Yhk_f,Ypq_f,h,k,p,q,freq1,freq2,g,s,freq
        #return [np.nanmedian(a),np.median(ecc_f[pc5:-pc5]),np.nanmedian(sini_f[pc5:-pc5])],Yhk_f,Ypq_f,h,k,p,q,freq1,freq2,g,s,freq
    
    return np.nanmean(a),np.nanmean(ecc_f[pc5:-pc5]),np.nanmean(sini_f[pc5:-pc5])
    #return np.nanmean(a),np.median(ecc_f[pc5:-pc5]),np.nanmedian(sini_f[pc5:-pc5])

    
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.ndimage import convolve1d, gaussian_filter1d


def smooth_fft_convolution(fft_signal, freqs, primary_freqs, time, protect_radius_bins_init=3, kernel_size=15, method="gaussian", inc_filt = False, known_planet_freqs = [],freq_tol=2e-7, win=False):
    """
    Fast convolutional smoothing of FFT log-power spectrum, excluding primary peaks.
    """
    fft_signal = np.asarray(fft_signal)
    log_power = 10 * np.log10(np.abs(fft_signal)**2 + 1e-10)

    #print(protect_radius_bins_init)
    #print(kernel_size)
    
    log_power[np.isnan(log_power)] = 1e-10

    #print('has nan',np.any(np.isnan(log_power)))
    phase = np.angle(fft_signal)

    N = len(fft_signal)
    mask = np.ones(N, dtype=bool)
    mask_opp = np.ones(N, dtype=bool)
    mask_long = np.ones(N, dtype=bool)
    mask_short = np.ones(N, dtype=bool)
    #mask_double = np.ones(N, dtype=bool)
    #mask_double_opp = np.ones(N, dtype=bool)
    
    # Mask protected bins around each primary frequency

    first = True
    protect_radius_bins = protect_radius_bins_init
    
    dt = time[1]-time[0]
    
    for pf in primary_freqs:
        
        dex_protect = 0.01
        #if win:
        #    dex_protect = 0.015

        idx = np.argmin(np.abs(freqs - pf))
        if first: 
            p_idx = idx
            #if inc_filt == False:
            #    dex_protect = 0.06
            #first = False
            f_pf_idx = np.argmin(abs(pf-freqs))
            dex_protect = dex_protect*20
        else:
            protect_radius_bins = round(protect_radius_bins_init/2)



        extra = 1
        if pf < 0:
            min_logf = -10**(np.log10(abs(pf)) - dex_protect*extra)
            max_logf = -10**(np.log10(abs(pf)) + dex_protect)
            in_window = np.where((freqs >= max_logf) & (freqs <= min_logf))[0]
            if len(in_window) <= 2 and (len(log_power) - idx) > 1 and idx > 1:
                in_window = np.array([idx-1,idx,idx+1])
            min_logf = -10**(np.log10(abs(pf)) - dex_protect*extra)
            max_logf = -10**(np.log10(abs(pf)) + dex_protect)
            in_window_opp = np.where((freqs <= -max_logf) & (freqs >= -min_logf))[0]
            if len(in_window_opp) <= 2 and (len(log_power) - idx) > 1 and idx > 1:
                in_window_opp = np.array([idx-1,idx,idx+1])
            
        else:
            min_logf = 10**(np.log10(abs(pf)) - dex_protect*extra)
            max_logf = 10**(np.log10(abs(pf)) + dex_protect)
            in_window = np.where((freqs >= min_logf) & (freqs <= max_logf))[0]
            if len(in_window) <= 2 and (len(log_power) - idx) > 1 and idx > 1:
                in_window = np.array([idx-1,idx,idx+1])
            min_logf = 10**(np.log10(abs(pf)) - dex_protect*extra)
            max_logf = 10**(np.log10(abs(pf)) + dex_protect)
            in_window_opp = np.where((freqs <= -min_logf) & (freqs >= -max_logf))[0]
            idx_opp = np.argmin(abs(-freqs-pf))
            if len(in_window_opp) <= 2 and (len(log_power) - idx_opp) > 1 and idx_opp > 1:
                idx_opp = np.argmin(abs(-freqs-pf))
                in_window_opp = np.array([idx_opp-1,idx_opp,idx_opp+1])
            
        if first:
            first = False
            dex_extra = 0.1/5
            if pf < 0:
                min_logf = -10**(np.log10(abs(pf)) - dex_extra)
                max_logf = -10**(np.log10(abs(pf)) + dex_extra)
                p_window = np.where((freqs >= max_logf) & (freqs <= min_logf))[0]
                if len(p_window) <= 2 and (len(log_power) - idx) > 2 and idx > 2:
                    p_window = np.array([idx-2,idx-1,idx,idx+1,idx+2])
            else:
                min_logf = 10**(np.log10(abs(pf)) - dex_extra)
                max_logf = 10**(np.log10(abs(pf)) + dex_extra)
                p_window = np.where((freqs <= max_logf) & (freqs >= min_logf))[0]
                if len(p_window) <= 2 and (len(log_power) - idx) > 2 and idx > 2:
                    p_window = np.array([idx-2,idx-1,idx,idx+1,idx+2])
                
        mask[in_window] = False
        
        #idx_opp = np.argmin(np.abs(- pf - freqs))
        mask[in_window_opp] = False

    #inds_long = np.where(1/np.abs(freqs) > abs(time[-1]-time[0])+abs(2*dt))[0]
    #mask_long[inds_long] = False  

    
    #inds_short = np.where(1/np.abs(freqs) <= 2.5*abs(dt))[0]
    #log_power[inds_short] = -8

    lowest_per_p = np.min(abs(1/np.array(known_planet_freqs)))
    lowest_per_gs = np.min(abs(1/np.array(primary_freqs)))
    lowest_period = min(lowest_per_p,lowest_per_gs)
        
    shortperiod = np.where(1/abs(freqs) < lowest_period/4)[0]
    #if len(shortperiod) 
    #mask_short[shortperiod]
    
    
    
    new_spec = log_power.copy()
    dt = abs(time[1]-time[0])
    tol_bins = abs(round(freq_tol*len(fft_signal)*dt))
    if tol_bins < 1:
        tol_bins = 1
    
    #if inc_filt == False:
    
    mask_filt = np.ones(N, dtype=bool)
    
    masked_log_power = log_power.copy()
    
    if inc_filt != None:
        for pf in known_planet_freqs:

            dex_filt = dex_protect/2
            if win:
                dex_filt = dex_filt*4/3
            
            idx = np.nanargmin(abs(freqs - pf))
            if idx-tol_bins <= 0 or idx+tol_bins >= len(log_power):
                continue

            dist_freq = np.abs(primary_freqs[0] - pf)
            
            if dist_freq <= freq_tol*4:
                continue

            if dist_freq <= freq_tol*10:
                dex_filt=dex_filt/2
                continue

            if pf < 0:
                min_logf = -10**(np.log10(abs(pf)) - dex_filt*extra)
                max_logf = -10**(np.log10(abs(pf)) + dex_filt)
                in_window = np.where((freqs >= max_logf) & (freqs <= min_logf))[0]
                if len(in_window) <= 2:
                    in_window = [idx-1,idx,idx+1]
            
            else:
                min_logf = 10**(np.log10(abs(pf)) - dex_filt*extra)
                max_logf = 10**(np.log10(abs(pf)) + dex_filt)
                in_window = np.where((freqs >= min_logf) & (freqs <= max_logf))[0]
                if len(in_window) <= 2 and (len(log_power) - idx) > 1 and idx >1:
                    
                    in_window = [idx-1,idx,idx+1]

            
                
            try:
                if dist_freq <= freq_tol*10:
                    mask_filt[idx] = False
                    masked_log_power[idx] = log_power[idx]
                    continue
                else:
                    mask_filt[in_window] = False
                    masked_log_power[in_window] = log_power[in_window]
                    continue
            except Exception as e:
                print('521',e)
                mask_filt[idx] = False
                masked_log_power[idx] = log_power[idx]
                continue
                
            '''
            try:
                if dist_freq <= freq_tol*4:
                    #mask_filt[idx] = False
                    #masked_log_power[idx] = log_power[idx]
                    continue
                elif dist_freq <= freq_tol*10:
                    mask_filt[idx] = False
                    masked_log_power[idx] = log_power[idx]
                    continue
                elif dist_freq <= freq_tol*20 and (idx-tol_bins) > 0 and (idx+tol_bins+1) < len(log_power):
                    mask_filt[idx-tol_bins:idx+tol_bins+1] = False
                    masked_log_power[idx-tol_bins:idx+tol_bins+1] = log_power[idx-tol_bins:idx+tol_bins+1]
                    continue
                elif dist_freq <= freq_tol*40 and (idx-tol_bins*2) > 0 and (idx+2*tol_bins+1) < len(log_power):
                    mask_filt[idx-2*tol_bins:idx+2*tol_bins+1]= False
                    masked_log_power[idx-tol_bins*2:idx+2*tol_bins+1] = log_power[idx-2*tol_bins:idx+2*tol_bins+1]
                    continue
                elif dist_freq <= freq_tol*80 and (idx-3*tol_bins) > 0 and (idx+3*tol_bins+1) < len(log_power):
                    mask_filt[idx-3*tol_bins:idx+3*tol_bins+1] = False
                    masked_log_power[idx-3*tol_bins:idx+3*tol_bins+1] = log_power[idx-3*tol_bins:idx+3*tol_bins+1]
                    continue
                elif dist_freq <= freq_tol*160 and (idx-4*tol_bins) > 0 and (idx+4*tol_bins+1) < len(log_power):
                    mask_filt[idx-4*tol_bins:idx+4*tol_bins+1] = False
                    masked_log_power[idx-4*tol_bins:idx+4*tol_bins+1] = log_power[idx-4*tol_bins:idx+4*tol_bins+1]
                    continue

                elif dist_p_freq <= freq_tol*4:
                    mask_filt[idx] = False
                    masked_log_power[idx] = log_power[idx]
                    continue
                elif dist_p_freq <= freq_tol*10  and (idx-tol_bins) > 0 and (idx+tol_bins+1) < len(log_power):
                    mask_filt[idx-tol_bins:idx+tol_bins+1] = False
                    masked_log_power[idx-tol_bins:idx+tol_bins+1] = log_power[idx-tol_bins:idx+tol_bins+1]
                    continue
                elif dist_p_freq <= freq_tol*20 and (idx-2*tol_bins) > 0 and (idx+2*tol_bins+1) < len(log_power):
                    mask_filt[idx-2*tol_bins:idx+2*tol_bins+1]= False
                    masked_log_power[idx-tol_bins*2:idx+2*tol_bins+1] = log_power[idx-2*tol_bins:idx+2*tol_bins+1]
                    continue
                elif dist_p_freq <= freq_tol*40 and (idx-3*tol_bins) > 0 and (idx+3*tol_bins+1) < len(log_power):
                    mmask_filt[idx-3*tol_bins:idx+3*tol_bins+1] = False
                    masked_log_power[idx-3*tol_bins:idx+3*tol_bins+1] = log_power[idx-3*tol_bins:idx+3*tol_bins+1]
                    continue
                elif dist_p_freq <= freq_tol*80 and (idx-5*tol_bins) > 0 and (idx+5*tol_bins+1) < len(log_power):
                    mask_filt[idx-5*tol_bins:idx+5*tol_bins+1] = False
                    masked_log_power[idx-5*tol_bins:idx+5*tol_bins+1] = log_power[idx-5*tol_bins:idx+5*tol_bins+1]
                    continue
                elif dist_p_freq <= freq_tol*160 and (idx-6*tol_bins) > 0 and (idx+6*tol_bins+1) < len(log_power):
                    mask_filt[idx-6*tol_bins:idx+6*tol_bins+1] = False
                    masked_log_power[idx-6*tol_bins:idx+6*tol_bins+1] = log_power[idx-6*tol_bins:idx+6*tol_bins+1]
                    continue
                elif (idx-15*tol_bins) > 0 and (idx+15*tol_bins+1) < len(log_power):
                    mask_filt[idx-8*tol_bins:idx+8*tol_bins+1] = False
                    masked_log_power[idx-8*tol_bins:idx+8*tol_bins+1] = log_power[idx-8*tol_bins:idx+8*tol_bins+1]
                    continue
            except:
                mask_filt[idx] = False
                masked_log_power[idx] = log_power[idx]/2
                continue
            '''


    #mask_filt
    long_protect = False
    if abs(1/primary_freqs[0]) > abs(time[-1]/20):
        long_protect = True
        long_inds = np.where(abs(1/freqs) > abs(time[-1]/20))[0]
        mask_long[long_inds] = False

    
    peak = log_power[idx]

    mask_filt[p_window] = True
    # Prepare data and mask for convolution
    #masked_log_power[~mask] = log_power[~mask]/4  # zero out protected region before smoothing
    #masked_log_power[~mask_opp] = log_power[~mask]/4  # zero out protected region before smoothing
    #masked_log_power[~mask_long] = log_power[~mask]/4  # zero out protected region before smoothing

    #masked_log_power[~mask_filt] = log_power[~mask_filt]
    

    #rmasked_log_power = log_power.copy()
    #rmasked_log_power[mask] = np.mean(log_power) # zero out protected region before smoothing
    #rmasked_log_power[mask_opp] = np.mean(log_power)  # zero out protected region before smoothing
    #rmasked_log_power[mask_long] = np.mean(log_power[mask_long])  # zero out protected region before smoothing
    #masked_log_power[~mask_double_opp] = 0  # zero out protected region before smoothing
    #masked_log_power[~mask_double] = 0  # zero out protected region before smoothing
    
    #masked_log_power[np.where(masked_log_power == np.nan)[0]] = 0
    #masked_log_power[~mask_long] = 0  # zero out protected region before smoothing
    #masked_log_power[~mask_half] = 0  # zero out protected region before smoothing
    #masked_log_power[~mask_double] = 0  # zero out protected region before smoothing
    mask_float = mask.astype(float)
    mask_opp_float = mask_opp.astype(float)
    #mask_long_float = mask_long.astype(float)
    mask_filt_float = mask_filt.astype(float)
    #rmasked_low_power = log_power.copy()
    # Convolve both signal and mask to normalize after convolution

    if method == "gaussian":
        #smoothed_log_power = gaussian_filter1d(masked_log_power, sigma=kernel_size, mode="constant",cval = np.nanmedian(log_power))
        #smoothed_mask = gaussian_filter1d(mask_float, sigma=kernel_size, mode="constant",cval = np.nanmedian(log_power))
        smoothed_log_power = gaussian_filter1d(masked_log_power, sigma=kernel_size, mode="mirror")
        smoothed_mask = gaussian_filter1d(mask_float, sigma=kernel_size, mode="mirror")
        #smoothed_mask_opp = gaussian_filter1d(mask_opp_float, sigma=kernel_size, mode="mirror")
        #smoothed_mask_long = gaussian_filter1d(mask_long_float, sigma=kernel_size, mode="mirror")
        smoothed_mask_filt = gaussian_filter1d(mask_filt_float, sigma=kernel_size, mode="mirror")

        #log_power_smooth = gaussian_filter1d(rmasked_log_power, sigma=tol_bins*5, mode="mirror")
    else:  # fallback to uniform boxcar
        kernel = np.ones(kernel_size) / kernel_size
        smoothed_log_power = convolve1d(masked_log_power, kernel, mode="nearest")
        smoothed_mask = convolve1d(mask_float, kernel, mode="nearest")
        smoothed_mask_opp = convolve1d(mask_opp_float, kernel, mode="nearest")
        smoothed_mask_long = convolve1d(mask_long_float, kernel, mode="nearest")
        smoothed_mask_filt = convolve1d(mask_filt_float, kernel, mode="nearest")

    # Normalize result: divide smoothed signal by smoothed mask
    #smoothed_mask[int(N/2)-3:int(N/2)+3] = 1
    #smoothed_mask[N-6:int(N)-1] = 0
    #smoothed_mask[0] = 0
    with np.errstate(invalid='ignore', divide='ignore'):
        #normalized_log_power = np.where(smoothed_mask_filt < 1e-6 , smoothed_log_power / smoothed_mask_filt, log_power)
        normalized_log_power = np.where(smoothed_mask > 1e-6 , smoothed_log_power, log_power)
        #normalized_log_power = np.where(smoothed_mask_opp > 1e-10 , smoothed_log_power / smoothed_mask_opp, log_power)
        #normalized_log_power = np.where(smoothed_mask_long > 1e-6 , smoothed_log_power / smoothed_mask_long, log_power)

    # Restore original log_power in protected regions
    #normalized_log_power = smoothed_log_power.copy()
    norm_og = normalized_log_power.copy()
    
    #print(np.max(log_power))
    #print(np.max(smoothed_log_power))
    #print(np.max(normalized_log_power))
    #normalized_log_power[np.where(np.isnan(normalized_log_power))[0]] = 0
    #normalized_log_power[normalized_log_power > 1e-5*np.max(log_power)] = 1e-3*np.max(log_power)
    normalized_log_power[~mask] = log_power[~mask]
    #normalized_log_power[~mask_opp] = log_power[~mask_opp]
    if long_protect:
        normalized_log_power[~mask_long] = log_power[~mask_long]

    #print(np.sum(mask_filt),len(norm_og))
    #print(len(norm_og[~mask_filt]))
    #print(len(normalized_log_power[~mask_filt]))
    normalized_log_power[~mask_filt] = norm_og[~mask_filt]

    #normalized_log_power[~mask] = log_power_smooth[~mask]
    #normalized_log_power[~mask_opp] = log_power_smooth[~mask_opp]
    #normalized_log_power[~mask_long] = log_power_smooth[~mask_long]
    
    #normalized_log_power[~mask_double_opp] = log_power[~mask_double_opp]
    #normalized_log_power[~mask_double] = log_power[~mask_double]
    #normalized_log_power[~mask_long] = log_power[~mask_long]
    #normalized_log_power[~mask_half] = log_power[~mask_half]
    #normalized_log_power[~mask_double] = log_power[~mask_double]

    #print('428',log_power)
    #print('429',smoothed_log_power)
    #print('430',masked_log_power)
    #print('431',normalized_log_power)
    #print('432',kernel_size)
    #print('433',idx)
    
    

    # Back to magnitude
    magnitude = 10 ** (0.5 * normalized_log_power / 10) 

    # Normalize power to match original
    orig_power = np.nansum(np.abs(fft_signal)**2)
    new_power = np.nansum(np.abs(magnitude)**2)
    scale = np.sqrt(orig_power / new_power)

    smoothed_fft = magnitude * np.exp(1j * phase) * scale
    #print('446',smoothed_fft)
    #smoothed_fft[idx] = np.median(smoothed_fft[idx-1:idx+2])
    #smoothed_fft[idx] = 0
    
    #scaling_factor = np.sqrt(1/(np.nanmax(np.abs(smoothed_fft)**2) / np.nanmax(np.abs(fft_signal)**2)))
    scaling_factor = np.sqrt(1/((np.abs(smoothed_fft[f_pf_idx])**2) / (np.abs(fft_signal[f_pf_idx])**2)))
    #scaling_factor=1
    #smoothed_fft[0] = np.median(smoothed_fft[1:10])
    return smoothed_fft * scaling_factor

def argmedian(x):
    """
    Returns the index of the median element for odd-length arrays,
    or one of the middle elements for even-length arrays.
    """
    return np.argpartition(x, len(x) // 2)[len(x) // 2]

def extract_proper_mode(signal, time, known_planet_freqs, freq_tol=2e-7, protect_bins=None,kernel=60, proper_freq = None, inc_filt = False, win=False):
    """
    Extract the 'free' (proper) frequency from an asteroid signal by:
    1. Identifying frequencies not associated with planetary forcing.
    2. Filtering all but the closest match in frequency space (band-pass).
    
    Parameters:
    - signal: array-like, real or complex signal (e.g., k + i*h or q + i*p)
    - time: array-like, same length as signal, evenly spaced time values
    - known_planet_freqs: list of planetary secular frequencies to remove (in same units as 1/time)
    - freq_tol: float, tolerance for deciding if a frequency matches a known planetary mode
    - keep_width: float, half-width of the frequency band to preserve around the free frequency
    
    Returns:
    - proper_signal: the filtered signal with only the proper mode preserved
    - proper_freq: the frequency identified as the free/proper mode
    """

    try:
        N = len(signal)
        dt = np.abs(time[1] - time[0])
        freqs = fftfreq(N, d=dt)
        spectrum = np.fft.fft(signal)

        freqs = np.fft.fftshift(freqs)
        spectrum = np.fft.fftshift(spectrum)

        #print('516', signal)
        #print('517', spectrum)
        tol_bins = round(freq_tol*N*dt)

        ref_spec = spectrum.copy()

        ind_0 = np.where(freqs == 0)[0][0]
        if inc_filt:
            spectrum[ind_0] = np.nanmean(np.array([ref_spec[ind_0-1],ref_spec[ind_0+1]]),dtype=np.float64)
        #    spectrum[ind_0] = 0

        lowest_per_p = np.min(abs(1/np.array(known_planet_freqs)))
        lowest_per_gs = np.min(abs(1/np.array(proper_freq)))
        #low_2g2s = 1/(2*g_arr[1]-2*s_arr[0])

        lowest_period = min(lowest_per_p,lowest_per_gs)
        #lowest_period = min(lowest_period,low_2g2s)
        
        shortperiod = np.where(1/abs(freqs) < lowest_period/4)[0]
        #if abs(1/proper_freq[0]) > abs(10*dt):
        if len(shortperiod) > 0:    
            short_ref = ref_spec[shortperiod]
            spectrum[shortperiod] = short_ref[argmedian(np.abs(short_ref)**2)]
        #    spectrum[shortperiod] = 0
        #print('527', spectrum[0:200])
    # Find the frequency with the highest power that is NOT near any planetary frequency
        power = np.abs(spectrum)**2
        sorted_indices = np.argsort(power)[::-1]  # sort descending by power

        dt = time[1]-time[0]

        proper_idx = np.nanargmin(abs(freqs - proper_freq[0]))
        new_spec = spectrum.copy()


    #if inc_filt == False:
    #print('536')
        #'''
        
        if proper_freq is None:
            raise ValueError("No proper (free) frequency found distinct from planetary modes.")

        if protect_bins == None:
            protect_bins = 6*tol_bins

        if protect_bins < 1:
            protect_bins = 1
        #print('protect_bins:',protect_bins)
        #print('741',spectrum[0:200])

        nanvals = np.where(np.isnan(spectrum))[0]
        #spectrum[nanvals] = np.nanmedian(spectrum)
        #print('nanvals', len(nanvals), nanvals)
        spectrum[nanvals] = np.nanmean(ref_spec)

        '''
        plt.scatter(1/freqs,spectrum,s=5)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        
        plt.scatter(1/np.flip(freqs),spectrum,s=5)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        '''
       
        filt_signal = smooth_fft_convolution(spectrum, freqs, proper_freq, 
                                         time, protect_radius_bins_init=protect_bins, 
                                         kernel_size=kernel, method="gaussian", inc_filt=inc_filt,
                                         known_planet_freqs = known_planet_freqs, freq_tol=freq_tol,
                                         win=win)

        
        nan_inds = np.where(np.isnan(filt_signal))[0]
        filt_signal[nan_inds] = np.nanmedian(filt_signal)
        
        shortperiod = np.where(1/abs(freqs) < abs(3*dt))[0]
        #if abs(1/proper_freq[0]) > abs(10*dt):
            #filt_signal[shortperiod] = ref_spec[np.argmin(np.abs(ref_spec)**2)]
        #    filt_signal[shortperiod] = 0

        
        ind_0 = np.where(freqs == 0)[0][0]
        #if inc_filt:
            #filt_signal[ind_0] = np.nanmean(np.array([filt_signal[ind_0-1],filt_signal[ind_0+1]]))
        #    filt_signal[ind_0] = 0
            
        filt_signal = np.fft.ifftshift(filt_signal)

            
        
        #nan_inds = np.where(filt_signal == np.nan+1j*np.nan)[0]
        #filt_signal[nan_inds] = np.nanmedian(filt_signal)
        proper_signal = np.fft.ifft(filt_signal)

    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno

        error_message = "An error occurred in at line "+str(line_number)
    
        # Print the error message
        print(error_message)
        print(error)

    if all(proper_signal == 0):
        proper_signal = signal

    
        

    return proper_signal, proper_freq, protect_bins, filt_signal

def find_local_max_windowed(freqs, powers, window_half_dex=0.05, window_protect_dex=0.1):
    """
    Identify local max frequency in windowed summed power spectrum, split separately
    for positive and negative frequencies.
    """
    # Separate positive and negative frequencies
    pos_mask = freqs > 0
    neg_mask = freqs < 0

    freqs_pos = freqs[pos_mask]
    powers_pos = powers[pos_mask]
    freqs_neg = freqs[neg_mask]
    powers_neg = powers[neg_mask]

    # Sort and log-transform positive frequencies
    logf_pos = np.log10(freqs_pos)
    sort_idx_pos = np.argsort(logf_pos)
    logf_pos = logf_pos[sort_idx_pos]
    powers_pos = powers_pos[sort_idx_pos]
    freqs_pos = freqs_pos[sort_idx_pos]

    # Sort and log-transform negative frequencies
    logf_neg = np.log10(-freqs_neg)
    sort_idx_neg = np.argsort(logf_neg)
    logf_neg = logf_neg[sort_idx_neg]
    powers_neg = powers_neg[sort_idx_neg]
    freqs_neg = freqs_neg[sort_idx_neg]

    # Cumulative sums
    cumsum_pos = np.cumsum(powers_pos)
    cumsum_neg = np.cumsum(powers_neg)

    # Initialize local power arrays
    local_power_pos = np.zeros_like(powers_pos)
    local_power_neg = np.zeros_like(powers_neg)

    # Compute windowed sums for positive
    for i in range(len(logf_pos)):
        center = logf_pos[i]
        min_logf = center - window_half_dex
        max_logf = center + window_half_dex

        left = np.searchsorted(logf_pos, min_logf, side='left')
        right = np.searchsorted(logf_pos, max_logf, side='right') - 1

        if left == 0:
            local_power_pos[i] = cumsum_pos[right]
        else:
            local_power_pos[i] = cumsum_pos[right] - cumsum_pos[left - 1]

    # Compute windowed sums for negative
    for i in range(len(logf_neg)):
        center = logf_neg[i]
        min_logf = center - window_half_dex
        max_logf = center + window_half_dex

        left = np.searchsorted(logf_neg, min_logf, side='left')
        right = np.searchsorted(logf_neg, max_logf, side='right') - 1

        if left == 0:
            local_power_neg[i] = cumsum_neg[right]
        else:
            local_power_neg[i] = cumsum_neg[right] - cumsum_neg[left - 1]

    # Recombine into full array
    local_power_all = np.zeros_like(freqs)
    unsort_idx_pos = np.argsort(sort_idx_pos)
    unsort_idx_neg = np.argsort(sort_idx_neg)
    local_power_all[pos_mask] = local_power_pos[unsort_idx_pos]
    local_power_all[neg_mask] = local_power_neg[unsort_idx_neg]

    # Find global maximum
    max_idx = np.argmax(local_power_all)
    max_freq = freqs[max_idx]
    #dominant_freq = freqs[max_idx]

    #if dominant_freq < 0:
    #    in_window = np.where((np.log10(-freqs) >= min_logf) & (np.log10(-freqs) <= max_logf))[0]
    #else:
    #    in_window = np.where((np.log10(freqs) >= min_logf) & (np.log10(freqs) <= max_logf))[0]
        
    #plt.plot(-1/freqs,local_power_all)
    #plt.vlines(abs(1/dominant_freq),ymin=1e-3,ymax=1e7)
    #plt.vlines(abs(1/max_freq),ymin=1e-3,ymax=1e7)
    #plt.vlines([abs(1/freqs[np.max(in_window)]),abs(1/freqs[np.min(in_window)])],ymin=1e-3,ymax=1e7,colors='r')
    #plt.yscale('log')
    #plt.xscale('log')
    #plt.xlim(abs(0.5/freqs[np.min(in_window)]),abs(2/freqs[np.max(in_window)]))
    #plt.ylim(5e6,8e6)
    #plt.show()

    min_logf = np.log10(abs(max_freq)) - window_half_dex
    max_logf = np.log10(abs(max_freq)) + window_half_dex

    
    ind_low = np.argmin(abs(freqs-10**min_logf))
    ind_high = np.argmin(abs(freqs-10**max_logf))

    if max_freq < 0:
        ind_high = np.argmin(abs(freqs+10**min_logf))
        ind_low = np.argmin(abs(freqs+10**max_logf))

    #max_idx = np.argmax(powers[ind_low:ind_high])+ind_low
    #print(max_freq,max_freq*2*np.pi*206265)

    if ind_high - ind_low <= 1:
        ind_low = max_idx - 1
        ind_high = max_idx + 1
    
    dominant_freq = np.sum(freqs[ind_low:ind_high+1]*(powers[ind_low:ind_high+1])**2)/np.sum((powers[ind_low:ind_high+1])**2)
    #print(dominant_freq,dominant_freq*2*np.pi*206265)
    max_idx = np.argmin(np.abs(freqs-dominant_freq))
    dominant_freq = freqs[max_idx]
    
    #min_logf = np.log10(abs(dominant_freq)) - window_protect_dex
    #max_logf = np.log10(abs(dominant_freq)) + window_protect_dex

    #if dominant_freq < 0:
    #    in_window = np.where((np.log10(-freqs) >= min_logf) & (np.log10(-freqs) <= max_logf))[0]
    #else:
    #    in_window = np.where((np.log10(freqs) >= min_logf) & (np.log10(freqs) <= max_logf))[0]
 
    protect_bins = 10
    '''
    print(ind_high-ind_low)
    print(1/abs(dominant_freq))
    fig,ax = plt.subplots(1,2,figsize=(9,3))
    ax[0].scatter(-1/freqs,local_power_all,s=5)
    ax[0].vlines(abs(1/dominant_freq),ymin=1e-3,ymax=1e7,colors='r')
    ax[0].vlines([abs(1/freqs[ind_high]),abs(1/freqs[ind_low])],ymin=1e-3,ymax=1e7,colors='r')
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    #ax[0].set_xlim(2e4,2e5)
    #plt.set_ylim(5e6,8e6)
    #plt.show()
    ax[1].scatter(-1/freqs,powers,s=5)
    ax[1].vlines(abs(1/dominant_freq),ymin=1e-3,ymax=1e7,colors='r')
    ax[1].vlines([abs(1/freqs[ind_high]),abs(1/freqs[ind_low])],ymin=1e-3,ymax=1e7,colors='r')
    #ax[1].vlines([abs(1/freqs[np.max(in_window)]),abs(1/freqs[np.min(in_window)])],ymin=1e-3,ymax=1e7,colors='r')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[0].set_title('Positive')
    #ax[1].set_xlim(2e4,2e5)
    plt.show()

    fig,ax = plt.subplots(1,2,figsize=(9,3))
    ax[0].scatter(1/freqs,local_power_all,s=5)
    ax[0].vlines(abs(1/dominant_freq),ymin=1e-3,ymax=1e7,colors='r')
    ax[0].vlines([abs(1/freqs[ind_high]),abs(1/freqs[ind_low])],ymin=1e-3,ymax=1e7,colors='r')
    #ax[0].vlines([abs(1/freqs[np.max(in_window)]),abs(1/freqs[np.min(in_window)])],ymin=1e-3,ymax=1e7,colors='r')
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    #ax[0].set_xlim(2e4,2e5)
    #plt.set_ylim(5e6,8e6)
    #plt.show()
    ax[1].scatter(1/freqs,powers,s=5)
    ax[1].vlines(abs(1/dominant_freq),ymin=1e-3,ymax=1e7,colors='r')
    ax[1].vlines([abs(1/freqs[ind_high]),abs(1/freqs[ind_low])],ymin=1e-3,ymax=1e7,colors='r')
    #ax[1].vlines([abs(1/freqs[np.max(in_window)]),abs(1/freqs[np.min(in_window)])],ymin=1e-3,ymax=1e7,colors='r')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[0].set_title('Negative')
    #ax[1].set_xlim(2e4,2e5)
    plt.show()
    #'''

    return max_idx, dominant_freq, local_power_all, protect_bins


def prop_calc(objname, filename='Single',windows=9,debug=False):
    
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
        #print(fullfile)
        home = str(os.path.expanduser("~"))
        #fullfile = home+'/nobackup/archive/SBDynT_sims/'+filename+'/'+str(objname)+'/archive.bin'
        #print(os.listdir(home+'/../../../hdd/haumea-data/djspenc/SBDynT_sims/NesvornyAst/13/'))
        #fullfile=home+'/../../../hdd/haumea-data/djspenc/SBDynT_sims/'+filename+'/'+str(objname)+'/archive.bin'
        #fullfile = '../data/NesvornyAst_full/'++'/archive.bin'
        #fullfile = '../data/results/2009_archive.bin'
        #fullfile = '../data/results/9997_archive.bin'
        #fullfile = '../data/results/40_archive.bin'
        #fullfile = '../data/results/9985_archive.bin'
        print(fullfile)
        archive = rebound.Simulationarchive(fullfile)
        #archive = archive[::6]
        
        try:
            earth = archive[0].particles['earth']
            small_planets_flag = True
        except:
            small_planets_flag = False

        nump = len(archive[0].particles)
        
        time_run = archive[-1].t

        #if abs(time_run)<=9e6:
        #    print('Integration not long enough for accurate proper element calculation')
            
        #    return [objname,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        
        #time_run = 6e6
        
        #print('small planets flag:', small_planets_flag)
        if archive[-1].t > 0:
            flag, a_init, e_init, inc_init, lan_init, aop_init, M_init, t_init = tools.read_sa_for_sbody(des = str(objname), archivefile=fullfile,clones=0,tmin=0.,tmax=time_run,center='helio',s=archive)
        else:
            flag, a_init, e_init, inc_init, lan_init, aop_init, M_init, t_init = tools.read_sa_for_sbody(des = str(objname), archivefile=fullfile,clones=0,tmax=0.,tmin=time_run,center='helio',s=archive)
        
        #global av_init; global ev_init; global incv_init; global lanv_init; global aopv_init; global Mv_init
        #global ae_init; global ee_init; global ince_init; global lane_init; global aope_init; global Me_init        
        #global am_init; global em_init; global incm_init; global lanm_init; global aopm_init; global Mm_init                    
        #global aj_init; global ej_init; global incj_init; global lanj_init; global aopj_init; global Mj_init
        #global as_init; global es_init; global incs_init; global lans_init; global aops_init; global Ms_init
        #global au_init; global eu_init; global incu_init; global lanu_init; global aopu_init; global Mu_init        
        #global an_init; global en_init; global incn_init; global lann_init; global aopn_init; global Mn_init        
        #global g1; global g2; global g3; global g4; global g5; global g6; global g7; global g8        
        #global s1; global s2; global s3; global s4; global s6; global s7; global s8
        #global g_arr; global s_arr

        g1 = 0; g2 = 0; g3 = 0; g4 = 0; g5 = 0; g6 = 0; g7 = 0; g8 = 0
        s1 = 0; s2 = 0; s3 = 0; s4 = 0; s5 = 0; s6 = 0; s7 = 0; s8 = 0
        av_init = 0
        aj_init = 0

        p_init = np.sin(inc_init)*np.sin(lan_init)
        q_init = np.sin(inc_init)*np.cos(lan_init)
        h_init = (e_init)*np.sin(lan_init+aop_init)
        k_init = (e_init)*np.cos(lan_init+aop_init)

        hk_arr = k_init+1j*h_init
        pq_arr = q_init+1j*p_init

        goal = 2**16
        goal = len(hk_arr)
        j=10
        goal_p = 0
        while goal_p < len(hk_arr):
            goal_p = 2**j
            j += 1
        #goal_p = 2**j
        #goal = goal_p
        goal_p = len(hk_arr)
        #goal = len(hk_arr)
        #goal_p = 2**15
        N_og = len(hk_arr)
        dt = abs(t_init[1] - t_init[0])
        fbin_og = 1/N_og/dt

        if goal > N_og:
            hk_arr_s = np.append(hk_arr, np.zeros(goal-N_og))
            pq_arr_s = np.append(pq_arr, np.zeros(goal-N_og))
        else:
            hk_arr_s = hk_arr
            pq_arr_s = pq_arr
            
        Yhk = np.fft.fft(hk_arr)
        Ypq = np.fft.fft(pq_arr)

        #Ypq[0] = 0

        
        if small_planets_flag:
            
            if isinstance(av_init, int):
                if archive[-1].t > 0:
                    flag, av_init, ev_init, incv_init, lanv_init, aopv_init, Mv_init, t_init = tools.read_sa_for_sbody(des = str('venus'), archivefile=fullfile,clones=0,tmax=(time_run),tmin=0,center='helio',s=archive)
                    flag, ae_init, ee_init, ince_init, lane_init, aope_init, Me_init, t_init = tools.read_sa_for_sbody(des = str('earth'), archivefile=fullfile,clones=0,tmax=(time_run),tmin=0,center='helio',s=archive)
                    flag, am_init, em_init, incm_init, lanm_init, aopm_init, Mm_init, t_init = tools.read_sa_for_sbody(des = str('mars'), archivefile=fullfile,clones=0,tmax=(time_run),tmin=0,center='helio',s=archive)
                else:
                    flag, av_init, ev_init, incv_init, lanv_init, aopv_init, Mv_init, t_init = tools.read_sa_for_sbody(des = str('venus'), archivefile=fullfile,clones=0,tmin=(time_run),tmax=0,center='helio',s=archive)
                    flag, ae_init, ee_init, ince_init, lane_init, aope_init, Me_init, t_init = tools.read_sa_for_sbody(des = str('earth'), archivefile=fullfile,clones=0,tmin=(time_run),tmax=0,center='helio',s=archive)
                    flag, am_init, em_init, incm_init, lanm_init, aopm_init, Mm_init, t_init = tools.read_sa_for_sbody(des = str('mars'), archivefile=fullfile,clones=0,tmin=(time_run),tmax=0,center='helio',s=archive)
            
                hv = ev_init*np.sin(lanv_init+aopv_init); he = ee_init*np.sin(lane_init+aope_init); hm = em_init*np.sin(lanm_init+aopm_init)
                kv = ev_init*np.cos(lanv_init+aopv_init); ke = ee_init*np.cos(lane_init+aope_init); km = em_init*np.cos(lanm_init+aopm_init)                
                pv = np.sin(incv_init)*np.sin(lanv_init); pe = np.sin(ince_init)*np.sin(lane_init); pm = np.sin(incm_init)*np.sin(lanm_init)                
                qv = np.sin(incv_init)*np.cos(lanv_init); qe = np.sin(ince_init)*np.cos(lane_init); qm = np.sin(incm_init)*np.cos(lanm_init)

                hv = np.append(hv,np.zeros(goal_p-N_og)); he = np.append(he,np.zeros(goal_p-N_og)); hm = np.append(hm,np.zeros(goal_p-N_og))
                kv = np.append(kv,np.zeros(goal_p-N_og)); ke = np.append(ke,np.zeros(goal_p-N_og)); km = np.append(km,np.zeros(goal_p-N_og))
                pv = np.append(pv,np.zeros(goal_p-N_og)); pe = np.append(pe,np.zeros(goal_p-N_og)); pm = np.append(pm,np.zeros(goal_p-N_og))
                qv = np.append(qv,np.zeros(goal_p-N_og)); qe = np.append(qe,np.zeros(goal_p-N_og)); qm = np.append(qm,np.zeros(goal_p-N_og))
                            
                Yhkv = np.fft.fft(kv+1j*hv); Yhke = np.fft.fft(ke+1j*he); Yhkm = np.fft.fft(km+1j*hm)
                Ypqv = np.fft.fft(qv+1j*pv); Ypqe = np.fft.fft(qe+1j*pe); Ypqm = np.fft.fft(qm+1j*pm)
        
        if isinstance(aj_init, int):
            if archive[-1].t > 0 :
                flag, aj_init, ej_init, incj_init, lanj_init, aopj_init, Mj_init, t_init = tools.read_sa_for_sbody(des = str('jupiter'), archivefile=fullfile,clones=0,tmax=(time_run),tmin=0,center='helio',s=archive)
                flag, as_init, es_init, incs_init, lans_init, aops_init, Ms_init, t_init = tools.read_sa_for_sbody(des = str('saturn'), archivefile=fullfile,clones=0,tmax=(time_run),tmin=0,center='helio',s=archive)
                flag, au_init, eu_init, incu_init, lanu_init, aopu_init, Mu_init, t_init = tools.read_sa_for_sbody(des = str('uranus'), archivefile=fullfile,clones=0,tmax=(time_run),tmin=0,center='helio',s=archive)
                flag, an_init, en_init, incn_init, lann_init, aopn_init, Mn_init, t_init = tools.read_sa_for_sbody(des = str('neptune'), archivefile=fullfile,clones=0,tmax=(time_run),tmin=0,center='helio',s=archive)
            else:
                flag, aj_init, ej_init, incj_init, lanj_init, aopj_init, Mj_init, t_init = tools.read_sa_for_sbody(des = str('jupiter'), archivefile=fullfile,clones=0,tmin=(time_run),tmax=0,center='helio',s=archive)
                flag, as_init, es_init, incs_init, lans_init, aops_init, Ms_init, t_init = tools.read_sa_for_sbody(des = str('saturn'), archivefile=fullfile,clones=0,tmin=(time_run),tmax=0,center='helio',s=archive)
                flag, au_init, eu_init, incu_init, lanu_init, aopu_init, Mu_init, t_init = tools.read_sa_for_sbody(des = str('uranus'), archivefile=fullfile,clones=0,tmin=(time_run),tmax=0,center='helio',s=archive)
                flag, an_init, en_init, incn_init, lann_init, aopn_init, Mn_init, t_init = tools.read_sa_for_sbody(des = str('neptune'), archivefile=fullfile,clones=0,tmin=(time_run),tmax=0,center='helio',s=archive)
            
            hj = ej_init*np.sin(lanj_init+aopj_init); hs = es_init*np.sin(lans_init+aops_init); hu = eu_init*np.sin(lanu_init+aopu_init); hn = en_init*np.sin(lann_init+aopn_init)
            kj = ej_init*np.cos(lanj_init+aopj_init); ks = es_init*np.cos(lans_init+aops_init); ku = eu_init*np.cos(lanu_init+aopu_init); kn = en_init*np.cos(lann_init+aopn_init)
            pj = np.sin(incj_init)*np.sin(lanj_init); ps = np.sin(incs_init)*np.sin(lans_init); pu = np.sin(incu_init)*np.sin(lanu_init); pn = np.sin(incn_init)*np.sin(lann_init)
            qj = np.sin(incj_init)*np.cos(lanj_init); qs = np.sin(incs_init)*np.cos(lans_init); qu = np.sin(incu_init)*np.cos(lanu_init); qn = np.sin(incn_init)*np.cos(lann_init)

            hj = np.append(hj,np.zeros(goal_p-N_og)); hs = np.append(hs,np.zeros(goal_p-N_og)); hu = np.append(hu,np.zeros(goal_p-N_og)); hn = np.append(hn,np.zeros(goal_p-N_og))
            kj = np.append(kj,np.zeros(goal_p-N_og)); ks = np.append(ks,np.zeros(goal_p-N_og)); ku = np.append(ku,np.zeros(goal_p-N_og)); kn = np.append(kn,np.zeros(goal_p-N_og))
            pj = np.append(pj,np.zeros(goal_p-N_og)); ps = np.append(ps,np.zeros(goal_p-N_og)); pu = np.append(pu,np.zeros(goal_p-N_og)); pn = np.append(pn,np.zeros(goal_p-N_og))
            qj = np.append(qj,np.zeros(goal_p-N_og)); qs = np.append(qs,np.zeros(goal_p-N_og)); qu = np.append(qu,np.zeros(goal_p-N_og)); qn = np.append(qn,np.zeros(goal_p-N_og))
    
            Yhkj = np.fft.fft(kj+1j*hj); Yhks = np.fft.fft(ks+1j*hs); Yhku = np.fft.fft(ku+1j*hu); Yhkn = np.fft.fft(kn+1j*hn)
                    
            Ypqj = np.fft.fft(qj+1j*pj); Ypqs = np.fft.fft(qs+1j*ps); Ypqu = np.fft.fft(qu+1j*pu); Ypqn = np.fft.fft(qn+1j*pn)
            
    
            g_arr = []
            s_arr = []

            g_inds = []
            s_inds = []
            
            n = len(hj)
            dt = abs(archive[1].t - archive[0].t)
            #if n < 10001:
                #print(n)
            freq = np.fft.fftfreq(n,d=dt)
            #freqn = np.fft.rfftfreq(len(aj_init),d=dt)

            half = int(len(freq)/2)

            good_freq_inds = np.where(abs(1/freq) <= abs(t_init[-1]))[0]
            bad_freq_inds = np.where(abs(1/freq) > abs(t_init[-1]))[0]
            
            g5 = freq[np.argmax(np.abs(Yhkj[good_freq_inds])**2)]

            dex = 0.02
            dexl = dex/2
            div_num = 500
            
            powers = np.abs(Yhkj)**2
            powers[bad_freq_inds] = 0
            #print('Jupiter g')
            max_idx, g5, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
            g_arr.append(g5)
            g_inds.append(max_idx)

            powers = np.abs(Yhks)**2
            powers[bad_freq_inds] = 0


            #print('Saturn g')
            for i in g_inds:
                gfreq = freq[i]
                ind_low = np.argmin(abs(freq - 10**(np.log10(abs(gfreq))-dexl)))
                ind_high = np.argmin(abs(freq - 10**(np.log10(abs(gfreq))+dexl)))

                if gfreq < 0:
                    ind_high = np.argmin(abs(freq+10**(np.log10(abs(gfreq))-dexl)))
                    ind_low = np.argmin(abs(freq+10**(np.log10(abs(gfreq))+dexl)))

                if ind_high - ind_low <= 1:
                    ind_high = i + 1
                    ind_low = i - 1

                powers[ind_low:ind_high] = powers[ind_low:ind_high]/div_num
                
            max_idx, g6, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
            g_arr.append(g6)
            g_inds.append(max_idx)

            #print('Uranus g')
            powers = np.abs(Yhku)**2
            #powers[bad_freq_inds] = 0
            powers[0] = 0

            for i in g_inds:
                gfreq = freq[i]
                ind_low = np.argmin(abs(freq - 10**(np.log10(abs(gfreq))-dexl)))
                ind_high = np.argmin(abs(freq - 10**(np.log10(abs(gfreq))+dexl)))

                if gfreq < 0:
                    ind_high = np.argmin(abs(freq+10**(np.log10(abs(gfreq))-dexl)))
                    ind_low = np.argmin(abs(freq+10**(np.log10(abs(gfreq))+dexl)))

                if ind_high - ind_low <= 1:
                    ind_high = i + 1
                    ind_low = i - 1

                powers[ind_low:ind_high] = powers[ind_low:ind_high]/div_num
 
                           
            max_idx, g7, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
            g_arr.append(g7)
            g_inds.append(max_idx)

            #print('Neptune g')
            powers = np.abs(Yhkn)**2
            #powers[bad_freq_inds] = 0
            powers[0] = 0

            for i in g_inds:
                gfreq = freq[i]
                ind_low = np.argmin(abs(freq - 10**(np.log10(abs(gfreq))-dexl)))
                ind_high = np.argmin(abs(freq - 10**(np.log10(abs(gfreq))+dexl)))

                if gfreq < 0:
                    ind_high = np.argmin(abs(freq+10**(np.log10(abs(gfreq))-dexl)))
                    ind_low = np.argmin(abs(freq+10**(np.log10(abs(gfreq))+dexl)))

                if ind_high - ind_low <= 1:
                    ind_high = i + 1
                    ind_low = i - 1

                powers[ind_low:ind_high] = powers[ind_low:ind_high]/div_num
                
            max_idx, g8, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
            g_arr.append(g8)
            g_inds.append(max_idx)

            '''
            gu_power = np.abs(Yhku[good_freq_inds])**2
            sorted = np.argsort(gu_power)[::-1]
            ind = 0
            while good_freq[sorted[ind]] in g_arr:
                ind += 1
            g7 = good_freq[sorted[ind]]

            #g7 = freq[np.argmax(np.abs(Yhku[1:half])**2)+1]
            #g7_neg = freq[np.argmax(np.abs(Yhku[half:-2])**2)+half]
            
            g_arr.append(g7)
            #g_arr.append(g7_neg)
            
            g_power = np.abs(Yhku[good_freq_inds])**2
            sorted = np.argsort(g_power)[::-1]
            ind = 0
            while good_freq[sorted[ind]] in g_arr:
                ind += 1
            g8 = good_freq[sorted[ind]]
            
            #g8 = freq[np.argmax(np.abs(Yhkn[1:half])**2)+1]
            #g8_neg = freq[np.argmax(np.abs(Yhkn[half:-2])**2)+half]
            
            g_arr.append(g8)
            #g_arr.append(g8_neg)
            '''

            powers = np.abs(Ypqs)**2
            powers[bad_freq_inds] = 0


            #print('Saturn s')
            max_idx, s6, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
            s_arr.append(s6)
            s_inds.append(max_idx)

            powers = np.abs(Ypqu)**2
            powers[bad_freq_inds] = 0

            #print('Uranus s')
            for i in s_inds:
                sfreq = freq[i]
                ind_low = np.argmin(abs(freq - 10**(np.log10(abs(sfreq))-dexl)))
                ind_high = np.argmin(abs(freq - 10**(np.log10(abs(sfreq))+dexl)))

                if sfreq < 0:
                    ind_high = np.argmin(abs(freq+10**(np.log10(abs(sfreq))-dexl)))
                    ind_low = np.argmin(abs(freq+10**(np.log10(abs(sfreq))+dexl)))

                if ind_high - ind_low <= 1:
                    ind_high = i + 1
                    ind_low = i - 1

                powers[ind_low:ind_high] = powers[ind_low:ind_high]/div_num
                
            max_idx, s7, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
            s_arr.append(s7)
            s_inds.append(max_idx)

            powers = np.abs(Ypqn)**2
            powers[bad_freq_inds] = 0

            #print('Neptune s')
            for i in s_inds:
                sfreq = freq[i]
                ind_low = np.argmin(abs(freq - 10**(np.log10(abs(sfreq))-dexl)))
                ind_high = np.argmin(abs(freq - 10**(np.log10(abs(sfreq))+dexl)))

                if sfreq < 0:
                    ind_high = np.argmin(abs(freq+10**(np.log10(abs(sfreq))-dexl)))
                    ind_low = np.argmin(abs(freq+10**(np.log10(abs(sfreq))+dexl)))

                if ind_high - ind_low <= 1:
                    ind_high = i + 1
                    ind_low = i - 1

                powers[ind_low:ind_high] = powers[ind_low:ind_high]/div_num
                
            max_idx, s8, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
            s_arr.append(s8)
            s_inds.append(max_idx)

            '''
            s6 = freq[np.argmax(np.abs(Ypqs[good_freq_inds])**2)]
            #s6 = freq[np.argmax(np.abs(Ypqs[1:half])**2)+1]
            #s6_neg = freq[np.argmax(np.abs(Ypqs[half:-2])**2)+half]
            
            s_arr.append(s6)
            #s_arr.append(s6_neg)

            s_power = np.abs(Ypqu[good_freq_inds])**2
            sorted = np.argsort(s_power)[::-1]
            ind = 0
            while good_freq[sorted[ind]] in s_arr:
                ind += 1
            s7 = good_freq[sorted[ind]]
            

            #s7 = freq[np.argmax(np.abs(Ypqu[1:half])**2)+1]
            #s7_neg = freq[np.argmax(np.abs(Ypqu[half:-2])**2)+half]
            
            s_arr.append(s7)
            #s_arr.append(s7_neg)
            #s_arr.append(s7)
            s_power = np.abs(Ypqn[good_freq_inds])**2
            sorted = np.argsort(s_power)[::-1]
            ind = 0
            while good_freq[sorted[ind]] in s_arr:
                ind += 1
            s8 = good_freq[sorted[ind]]
            #s8 = freq[np.argmax(np.abs(Ypqn[1:half])**2)+1]
            #s8_neg = freq[np.argmax(np.abs(Ypqn[half:-2])**2)+half]
            
            s_arr.append(s8)

            #print(1/s8,'s8 period')
            #s_arr.append(s8_neg)
            
            #s_arr.append(s8)
            '''
            
            if small_planets_flag:
                #print('small planets freqs')
                powers = np.abs(Yhkv)**2
                powers[bad_freq_inds] = 0

                #print('Venus g')
                for i in g_inds:
                    gfreq = freq[i]
                    ind_low = np.argmin(abs(freq - 10**(np.log10(abs(gfreq))-dexl)))
                    ind_high = np.argmin(abs(freq - 10**(np.log10(abs(gfreq))+dexl)))

                    if gfreq < 0:
                        ind_high = np.argmin(abs(freq+10**(np.log10(abs(gfreq))-dexl)))
                        ind_low = np.argmin(abs(freq+10**(np.log10(abs(gfreq))+dexl)))

                    if ind_high - ind_low <= 1:
                        ind_high = i + 1
                        ind_low = i - 1

                    powers[ind_low:ind_high] = powers[ind_low:ind_high]/div_num
                
                max_idx, g2, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
                g_arr.append(g2)
                g_inds.append(max_idx)
                
                powers = np.abs(Yhke)**2
                powers[bad_freq_inds] = 0

                #print('Earth g')
                for i in g_inds:
                    gfreq = freq[i]
                    ind_low = np.argmin(abs(freq - 10**(np.log10(abs(gfreq))-dexl)))
                    ind_high = np.argmin(abs(freq - 10**(np.log10(abs(gfreq))+dexl)))

                    if gfreq < 0:
                        ind_high = np.argmin(abs(freq+10**(np.log10(abs(gfreq))-dexl)))
                        ind_low = np.argmin(abs(freq+10**(np.log10(abs(gfreq))+dexl)))

                    if ind_high - ind_low <= 1:
                        ind_high = i + 1
                        ind_low = i - 1

                    powers[ind_low:ind_high] = powers[ind_low:ind_high]/div_num
                
                max_idx, g3, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
                g_arr.append(g3)
                g_inds.append(max_idx)


                #print('Mars g')
                powers = np.abs(Yhkm)**2
                powers[bad_freq_inds] = 0

                for i in g_inds:
                    gfreq = freq[i]
                    ind_low = np.argmin(abs(freq - 10**(np.log10(abs(gfreq))-dexl)))
                    ind_high = np.argmin(abs(freq - 10**(np.log10(abs(gfreq))+dexl)))

                    if gfreq < 0:
                        ind_high = np.argmin(abs(freq+10**(np.log10(abs(gfreq))-dexl)))
                        ind_low = np.argmin(abs(freq+10**(np.log10(abs(gfreq))+dexl)))

                    if ind_high - ind_low <= 1:
                        ind_high = i + 1
                        ind_low = i - 1

                    powers[ind_low:ind_high] = powers[ind_low:ind_high]/div_num
                
                max_idx, g4, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
                g_arr.append(g4)
                g_inds.append(max_idx)

                
                powers = np.abs(Ypqe)**2
                powers[bad_freq_inds] = 0

                #print('Earth s')
                for i in s_inds:
                    sfreq = freq[i]
                    ind_low = np.argmin(abs(freq - 10**(np.log10(abs(sfreq))-dexl)))
                    ind_high = np.argmin(abs(freq - 10**(np.log10(abs(sfreq))+dexl)))

                    if sfreq < 0:
                        ind_high = np.argmin(abs(freq+10**(np.log10(abs(sfreq))-dexl)))
                        ind_low = np.argmin(abs(freq+10**(np.log10(abs(sfreq))+dexl)))
    
                    if ind_high - ind_low <= 1:
                        ind_high = i + 1
                        ind_low = i - 1

                    powers[ind_low:ind_high] = powers[ind_low:ind_high]/div_num
                
                max_idx, s3, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
                s_arr.append(s3)
                s_inds.append(max_idx)

                powers = np.abs(Ypqm)**2
                powers[bad_freq_inds] = 0

                #print('Mars s')
                for i in s_inds:
                    sfreq = freq[i]
                    ind_low = np.argmin(abs(freq - 10**(np.log10(abs(sfreq))-dexl)))
                    ind_high = np.argmin(abs(freq - 10**(np.log10(abs(sfreq))+dexl)))

                    if sfreq < 0:
                        ind_high = np.argmin(abs(freq+10**(np.log10(abs(sfreq))-dexl)))
                        ind_low = np.argmin(abs(freq+10**(np.log10(abs(sfreq))+dexl)))
    
                    if ind_high - ind_low <= 1:
                        ind_high = i + 1
                        ind_low = i - 1

                    powers[ind_low:ind_high] = powers[ind_low:ind_high]/div_num
                
                max_idx, s4, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
                s_arr.append(s4)
                s_inds.append(max_idx)

                powers = np.abs(Ypqv)**2
                powers[bad_freq_inds] = 0

                for i in s_inds:
                    sfreq = freq[i]
                    ind_low = np.argmin(abs(freq - 10**(np.log10(abs(sfreq))-dexl)))
                    ind_high = np.argmin(abs(freq - 10**(np.log10(abs(sfreq))+dexl)))

                    if sfreq < 0:
                        ind_high = np.argmin(abs(freq+10**(np.log10(abs(sfreq))-dexl)))
                        ind_low = np.argmin(abs(freq+10**(np.log10(abs(sfreq))+dexl)))
    
                    if ind_high - ind_low <= 1:
                        ind_high = i + 1
                        ind_low = i - 1

                    powers[ind_low:ind_high] = powers[ind_low:ind_high]/div_num

                #print('Venus g')                
                max_idx, s2, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
                s_arr.append(s2)
                s_inds.append(max_idx)

                '''
                g_power = np.abs(Yhke[good_freq_inds])**2
                sorted = np.argsort(g_power)[::-1]
                ind = 0
                while good_freq[sorted[ind]] in g_arr:
                    ind += 1
                g3 = good_freq[sorted[ind]]
                
                #g3 = freq[np.argmax(np.abs(Ypqe[1:half])**2)+1]
                #g3_neg = freq[np.argmax(np.abs(Ypqe[half:-2])**2)+half]
            
                g_arr.append(g3)
                #g_arr.append(g3_neg)

                g_power = np.abs(Yhkm[good_freq_inds])**2
                sorted = np.argsort(g_power)[::-1]
                ind = 0
                while good_freq[sorted[ind]] in g_arr:
                    ind += 1
                g4 = good_freq[sorted[ind]]
                
                #g3 = freq[np.argmax(np.abs(Ypqe[1:half])**2)+1]
                #g3_neg = freq[np.argmax(np.abs(Ypqe[half:-2])**2)+half]
            
                g_arr.append(g4)

                g_power = np.abs(Yhkv[good_freq_inds])**2
                sorted = np.argsort(g_power)[::-1]
                ind = 0
                while good_freq[sorted[ind]] in g_arr:
                    ind += 1
                g2 = good_freq[sorted[ind]]
                
                #g3 = freq[np.argmax(np.abs(Ypqe[1:half])**2)+1]
                #g3_neg = freq[np.argmax(np.abs(Ypqe[half:-2])**2)+half]
            
                g_arr.append(g2)


                s_power = np.abs(Ypqe[good_freq_inds])**2
                sorted = np.argsort(s_power)[::-1]
                ind = 0
                while good_freq[sorted[ind]] in s_arr:
                    ind += 1
                s3 = good_freq[sorted[ind]]
                #s3 = freq[np.argmax(np.abs(Ypqe[1:half])**2)+1]
                #s3_neg = freq[np.argmax(np.abs(Ypqe[half:-2])**2)+half]
            
                s_arr.append(s3)

                s_power = np.abs(Ypqm[good_freq_inds])**2
                sorted = np.argsort(s_power)[::-1]
                ind = 0
                while good_freq[sorted[ind]] in s_arr:
                    ind += 1
                s4 = good_freq[sorted[ind]]
                #s3 = freq[np.argmax(np.abs(Ypqe[1:half])**2)+1]
                #s3_neg = freq[np.argmax(np.abs(Ypqe[half:-2])**2)+half]
            
                s_arr.append(s4)

                s_power = np.abs(Ypqv[good_freq_inds])**2
                sorted = np.argsort(s_power)[::-1]
                ind = 0
                while good_freq[sorted[ind]] in s_arr:
                    ind += 1
                s2 = good_freq[sorted[ind]]
                #s3 = freq[np.argmax(np.abs(Ypqe[1:half])**2)+1]
                #s3_neg = freq[np.argmax(np.abs(Ypqe[half:-2])**2)+half]
            
                s_arr.append(s2)
                #s_arr.append(s3_neg)
                
                #s_arr.append(s4_neg)
                #s_arr.append(s4)
                '''
                
            #print('g_arr',1/np.array(g_arr))
            #print('s_arr',1/np.array(s_arr))
            g_ex = np.array([g5-g6,g5-g7,g5-g8,g6-g7,g6-g8,g7-g8],dtype=np.float64)
            s_ex = np.array([s6-s7,s6-s8,s7-s8],dtype=np.float64)
            
            #for i in g_ex:
            #    g_arr.append(i)
            #for i in s_ex:
            #    s_arr.append(i)

            n_s = len(hk_arr_s)
            freq_s = np.fft.fftfreq(n_s,d=dt)
            
            freq_tol = fbin_og
            fbin = 1/len(hk_arr_s)/dt
            tol_bins = abs(round(fbin_og/fbin))
            tol_bins = 1
            i = 1

            Yhk_s = np.fft.fft(hk_arr_s)
            power = np.abs(Yhk_s)**2

            dist = round(tol_bins*2)
            #dist = int(tol_bins*5)
            
            j=0
            
            if small_planets_flag:
                num_p = 8
            else:
                num_p = 5
                
            for i in g_arr[:num_p]:
                ind = np.argmin(abs(freq_s-i))
                if j==0 or j==1:
                    mult = 100
                else:
                    mult = 20
                if small_planets_flag == False and j == 3:
                    mult = 100
                    
                power[ind-dist:ind+dist+1] = power[ind-dist:ind+dist+1]/mult
                j += 1

            short_period_g = (1/abs(np.array(g_arr[1]*4)))
            short_period_s = (1/abs(np.array(s_arr[0]*4)))
            short_period = 1/abs(2*g_arr[1]-2*s_arr[0])
            short_period = min(short_period,short_period_g)
            short_period = min(short_period,short_period_s)
            short_ind = np.where(1/abs(freq_s) < short_period/4)[0]
            power[short_ind] = power[short_ind]/5
            #sorted_indices = np.argsort(power)[::-1]
            #g_idx = sorted_indices[0]
            #g = freq[g_idx]
            #i = 1
            #best_i = i

            window_dex = 0.15  
            n_bins = len(freq_s)

            #windows = round(n_bins/100*window_dex)
            
            local_power = np.zeros(n_bins)
            
            window_half_dex = window_dex / 2

            #Ypq = np.fft.fft(pq_zp)
            #power = np.abs(Ypq_zp)**2
            n_bins = len(power)

            
            g_idx, g, local_power_all, protect_g_bins = find_local_max_windowed(freq_s, power, window_half_dex=0.05, window_protect_dex=0.15)
            #print('running Yhk dom search')

            #windows = round(n_bins/100*window_dex)

            if abs(1/g) > 5e5:
                protect_g_bins = protect_g_bins*5
            
           
            Ypq_s = np.fft.fft(pq_arr_s)
            power = np.abs(Ypq_s)**2
            power[abs(1/freq_s) > abs(t_init[-1])/2] = 0

            j=0
            for i in s_arr[:num_p-1]:
                if j==0 or j==1:
                    mult = 100
                else:
                    mult = 20
                if small_planets_flag == False and j == 2:
                    mult = 100
                ind = np.argmin(abs(freq_s-i))
                power[ind-dist:ind+dist+1] = power[ind-dist:ind+dist+1]/mult
                j += 1
            #sorted_indices = np.argsort(power)[::-1]
            #s_idx = sorted_indices[0]
            #s = freq[s_idx]
            #i = 1

            window_dex = 0.12  
            n_bins = len(freq_s)

            
            power[short_ind] = power[short_ind]/10
            
            s_idx, s, local_power_all, protect_s_bins = find_local_max_windowed(freq_s, power, window_half_dex=0.02, window_protect_dex=0.15)
            if abs(1/s) > 5e5:
                protect_s_bins = protect_s_bins*2
            
   
    
            z1 = -(-g6-s6); z2 = -(-g5-s7); z3 = -(-g5-s6); z4 = (2*g6-g5); z5 = (2*g6-g7); z6 = -(-s6-g5+g6); z7 = -(-3*g6+2*g5)
            z8 = -(2*(-g6)-s6); z9 = -(3*(-g6)-s6); z10 = -((-g6)-s6)
            z11 = -(-2*g7+g6); z12 = -(-2*g5); z13 = -(4*g7); z14 = -(-s6)

            z1 = -(g+s-g6-s6); z2 = -(g+s-g5-s7); z3 = -(g+s-g5-s6); z4 = -(2*g6-g5); z5 = -(2*g6-g7); z6 = -(s-s6-g5+g6); z7 = -(g-3*g6+2*g5)
            z8 = -(2*(g-g6)+s-s6); z9 = -(3*(g-g6)+s-s6); z10 = -((g-g6)+s-s6)
            z11 = -(g-2*g7+g6); z12 = -(2*g-2*g5); z13 = -(-4*g+4*g7); z14 = -(-2*s-s6)

            if small_planets_flag:
        #freq1 = [(g1),(g2),(g3),(g4),(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,z8,z9,g-z8,g-z9,g-z10,z11,z12,z13]
        #freq2 = [(s1),(s2),(s3),(s4),(s6),(s7),(s8),z1,z2,z3,z6,z8,z9,s-z8,s-z9,s-z10,z14]
                freq1 = [(g2),(g3),(g4),(g5),(g6),(g7),(g8),g-z8,g-z9,g-z10,z11,z12,z13,-g+2*s-g5]
                freq2 = [(s2),(s3),(s4),(s6),(s7),(s8),s-z8,s-z9,s-z10,z14,g-s+g5-s7,g+g5-2*s6,2*g-2*s6]
            else:
                freq1 = [(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,g-z8,g-z9,g-z10,z11,z12]
                freq2 = [(s6),(s7),(s8),z1,z2,z3,z6,z8,z9,s-z8,s-z9,s-z10,z14]

            g_freqs = [-s+g6+s6,-s+g5+s7,-s+g5+s6,2*g6-g5,2*g6-g7,3*g6-2*g5,(-s+s6+2*g6)/2,(-s+s6+2*g6)/3,-s+s6+g6,2*g7-g6,2*g6,g6/2]
            s_freqs = [-g+g6+s6,-g+g5+s7,-g+g5+s6,s6+g5-g6,(-2*g+s6+2*g6)/2,(-3*g+s6+2*g6)/3,-g+s6+g6,-s6/2,2*g6-2*s6]
            
            g_freqs = [2*g6-g5,2*g6-g7,3*g6-2*g5,2*g7-g6,2*g6,g6+g7-g5,g5+g6-g7]
            s_freqs = [s6+g5-g6,2*g6-2*s6,g5,g6,g7,g8]

            
            gs_freqs = [2*g6-g5+s6-s7,g5-s6+s7]

            #FROM ORBFIT INL VARIABLE IN SELRE9.90, added 2*g6-2*s6 as the shortest possible 4th order 
            gs_freqs = [2*g6-g5,2*g6-g7,3*g6-2*g5,-g5+g6+g7,g5+g6-g7,-g5+2*g6-s6-s7,g5+s7-s6,2*g5-g6,2*g6-2*s6]

            #for i in g_freqs:
            #    g_arr.append(i)
            #for i in s_freqs:
            #    s_arr.append(i)
            for i in gs_freqs:
                g_arr.append(i)
                s_arr.append(i)

            #protect_gs = np.array([g-s,2*g-s,g-2*s,3*g-s,g-3*s,2*g-2*s,3*g-3*s,2*g-3*s,3*g-2*s,
            #      g+s,2*g+s,g+2*s,3*g+s,g+3*s,2*g+2*s,3*g+3*s,2*g+3*s,3*g+2*s])
            
            #protect_gs = np.array([g-s,2*g-s,g-2*s,2*g-2*s,g+s,2*g+s,g+2*s,2*g+2*s])
            
            protect_gs = np.array([g-s,2*g-s,g-2*s,2*g-2*s,g+s],dtype=np.float64)
            
            protect_gs = np.array([g-s,2*g-s,g-2*s,g+s],dtype=np.float64)
            #protect_gs = np.array([g-s,2*g-s,g-2*s])
            
            #protect_g = np.append([g,2*g,3*g],protect_gs)
            #protect_s = np.append([s,2*s,3*s,g,2*g],protect_gs)
            protect_g = np.append([g,2*g],protect_gs)
            #protect_g = np.array([g,2*g])
            protect_s = np.append([s,2*s],protect_gs)

            #for i in protect_gs:
            #    protect_g.append(i)
            #    protect_s.append(i)

    
      
            #de = [g-g5,g-g6,g5-g6,s-s7,s-s6,s7-s6,g+s-s7-g5,g+s-s7-g6,g+s-s6-g5,g+s-s6-g6,2*g-2*s,g-2*g5+g6,g+g5-2*g6,2*g-g5-g6,-g+s+g5-s7,-g+s+g6-s7,-g+s+g5-s6,-g+s+g6-s6,g-g5+s7-s6, g-g5-s7+s6,g-g6+s7-s6,g-g6-s7+s6,2*g-s-s7,2*g-s-s6,-g+2*s-g5,-g+2*s-g6, 2*g-2*s7,2*g-2*s6, 2*g-s7-s6,g-s+g5-s7,g-s+g5-s6,g-s+g6-s7,g-s+g6-s6,g+g5-2*s7,g+g6-2*s7,g+g5-2*s6,g+g6-2*s6,g+g5-s7-s6,g+g6-s7-s6,s-2*s7+s6,s+s7-2*s6,2*s-s7-s6,s+g5-g6-s7,s-g5+g6-s7,s+g5-g6-s6, s-g5+g6-s6,2*s-2*g5, 2*s-2*g6, 2*s-g5-g6,s-2*g5+s7, s-2*g5+s6,s-2*g6+s7, s-2*g6+s6,s-g5-g6+s7, s-g5-g6+s6,2*g-2*g5,2*g-2*g6, 2*s-2*s7, 2*s-2*s6,g-2*g6+g7,g-3*g6+2*g5, 2*(g-g6)+(s-s6),g+g5-g6-g7,g-g5-g6+g7,g+g5-2*g6-s6+s7,3*(g-g6)+(s-s6)]

            #new_de_g = -np.array([-g5,-g6,g5-g6,-s7,-s6,s7-s6,-s7-g5,-s7-g6,-s6-g5,-s6-g6,-2*s,-2*g5+g6,+g5-2*g6,-g5-g6,g5-s7,g6-s7,g5-s6,g6-s6,-g5+s7-s6,-g5-s7+s6,-g6+s7-s6,-g6-s7+s6,-s7,-s6,-g5,-g6, -2*s7,-2*s6,-s7-s6,g5-s7,g5-s6,g6-s7,g6-s6,g5-2*s7,g6-2*s7,g5-2*s6,g6-2*s6,g5-s7-s6,g6-s7-s6,-2*s7+s6,s7-2*s6,-s7-s6,g5-g6-s7,-g5+g6-s7,g5-g6-s6, -g5+g6-s6,-2*g5, -2*g6, -g5-g6,-2*g5+s7,-2*g5+s6,-2*g6+s7, -2*g6+s6,-g5-g6+s7, -g5-g6+s6,-2*g5,-2*g6, -2*s7, -2*s6,-2*g6+g7,g-3*g6+2*g5, 2*(-g6)+(-s6),g5-g6-g7,-g5-g6+g7,+g5-2*g6-s6+s7,3*(-g6)+(-s6)])

            #new_de_s = -np.array([-g5,-g6,g5-g6,-2*g5+g6,g5-2*g6,(-g5-g6)/2])
            #new_de_g = 
            #new_de_gs = -np.array([-s7-g5,-s7-g6,-s6-g5,-s6-g6,2*g-2*s,g5-s7,g6-s7,g5-s6,g6-s6,-s7,-s6,-g5,-g6,2*(-g6)+(-s6),3*(-g6)+(-s6)])

                            


        
            
    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno

        error_message = "An error occurred in at line "+str(line_number)
    
        # Print the error message
        print(error_message)
        print(error)
        return [objname,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    
    #print(t_init)
    #Outputs proper a, proper e, proper sin(inc)
    #ds = int(len(t_init)/10)
    '''
    t_init = t_init[:ds*6]
    a_init = a_init[:ds*6]
    p_init = p_init[:ds*6]
    q_init = q_init[:ds*6]
    h_init = h_init[:ds*6]
    k_init = k_init[:ds*6]
    
    ds = int(len(t_init)/10)
    '''
    #pes = pe_vals(t_init[:ds],a_init[:ds],p_init[:ds],q_init[:ds],h_init[:ds],k_init[:ds],g_arr,s_arr,small_planets_flag,debug)
    #pes = pe_vals(t_init,a_init,p_init,q_init,h_init,k_init,g_arr,s_arr,small_planets_flag,debug)    
    

    protect_hk=0
    protect_pq=0
    dt = np.abs(t_init[1]-t_init[0]) 
    freqs_a = np.fft.rfftfreq(len(hk_arr),dt)
    #print('g_arr',1/np.array(g_arr))
    #print('s_arr',1/np.array(s_arr))
    dex_protect = 0.04
    freq_low_g = 10**(np.log10(abs(protect_g[0]))-dex_protect)
    print(1/freq_low_g,1/protect_g[0])
    ind_low_g = np.argmin(abs(freq-abs(freq_low_g)))
    ind = np.argmin(abs(freq-abs(protect_g[0])))
    kernel_g = max(2*round(abs(ind-ind_low_g)),round(len(freq)/2500))
    kernel_g = max(2*round(abs(ind-ind_low_g)),2)
    
    if kernel_g > round(len(freq)/100):
        kernel_g = round(len(freq)/100)
    
    freq_low_s = 10**(np.log10(abs(protect_s[0]))-dex_protect)
    ind_low_s = np.argmin(abs(freq-abs(freq_low_s)))
    ind = np.argmin(abs(freq-abs(protect_s[0])))

    
    kernel_s = max(2*round(abs(ind-ind_low_s)),round(len(freq)/2500))
    kernel_s = max(2*round(abs(ind-ind_low_s)),2)
    if kernel_s > round(len(freq)/100):
        kernel_s = round(len(freq)/100)
                          
    try:
        dt = np.abs(t_init[1]-t_init[0]) 
        #dt = np.abs(archive[1].t) 
    
        kernel=round(200*2**(2-np.log10(dt)))
        kernel=1000
        protect = round(10*2**(2-np.log10(dt)))
        tol = 1/dt/(len(t_init)-1)
        toln = 1/dt/(len(hk_arr)-1)
        tol_bins = round(tol/toln)
        kernel=tol_bins*40
        
        
        protect = tol_bins*6
        per_shave = 0.025
        print('dt',dt,'tol',tol, 'N', len(hk_arr),'protect_g',protect_g_bins,'protect_s',protect_s_bins,'kernel_g',kernel_g,'kernel_s',kernel_s)
    
        #hk_new, hk_freq, protect_hk, hk_signal = extract_proper_mode(hk_arr, t_init, g_arr, freq_tol=tol, kernel=kernel, proper_freq=g)
        #pq_new, pq_freq, protect_pq, pq_signal = extract_proper_mode(pq_arr, t_init, s_arr, freq_tol=tol, kernel=kernel, proper_freq=s, inc_filt = True)
        hk_new, hk_freq, protect_hk, hk_signal = extract_proper_mode(hk_arr, t_init, g_arr, freq_tol=tol, kernel=kernel_g, protect_bins = protect_g_bins, proper_freq=protect_g)
        pq_new, pq_freq, protect_pq, pq_signal = extract_proper_mode(pq_arr, t_init, s_arr, freq_tol=tol, kernel=kernel_s, proper_freq=protect_s, protect_bins = protect_s_bins, inc_filt = True)

        #print('pq_new', pq_new)
        N = N_og

        #pe_e = np.mean(np.array([np.mean(np.abs(hk_new)[int(0.05*N):int((1-0.05)*N)]),np.mean(np.abs(hk_new)[int(0.1*N):int((1-0.1)*N)]),np.mean(np.abs(hk_new)[int(0.2*N):int((1-0.2)*N)]),np.mean(np.abs(hk_new)[int(0.3*N):int((1-0.3)*N)])]))
        #pe_i = np.mean(np.array([np.mean(np.abs(pq_new)[int(0.05*N):int((1-0.05)*N)]),np.mean(np.abs(pq_new)[int(0.1*N):int((1-0.1)*N)]),np.mean(np.abs(pq_new)[int(0.2*N):int((1-0.2)*N)]),np.mean(np.abs(pq_new)[int(0.3*N):int((1-0.3)*N)])]))

        pe_e = np.nanmean(np.abs(hk_new[int(per_shave*N):int((1-per_shave)*N)]))
        pe_i = np.nanmean(np.abs(pq_new[int(per_shave*N):int((1-per_shave)*N)]))

        hk_news = hk_new[int(per_shave*N):int((1-per_shave)*N)]
        pq_news = pq_new[int(per_shave*N):int((1-per_shave)*N)]

        e_win = np.abs(hk_news)*np.kaiser(len(hk_news),6)
        inc_win = np.abs(pq_news)*np.kaiser(len(pq_news),6)

        pe_e = np.nansum(e_win)/np.sum(np.kaiser(len(hk_news),6))
        pe_i = np.nansum(inc_win)/np.sum(np.kaiser(len(pq_news),6))
        #pes = np.array([np.nanmean(a_init),np.nanmean(np.abs(hk_new)[int(per_shave*N):int((1-per_shave)*N)]),np.nanmean(np.abs(pq_new)[int(per_shave*N):int((1-per_shave)*N)])])

        #Ya = np.fft.rfft(a_init)
        #short_inds = np.where(abs(1/freqs_a) <= dt*3)[0]
        #Ya[short_inds] = 0
        #a_new = np.fft.irfft(Ya)
        
        #pes = np.array([np.nanmean(a_new[int(per_shave*N):int((1-per_shave)*N)]),pe_e,pe_i])
        pes = np.array([np.nanmean(a_init),pe_e,pe_i],dtype=np.float64)
    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno

        error_message = "An error occurred in at line "+str(line_number)
    
        # Print the error message
        print(error_message)
        print(error)
        pes = np.array([10,10,10],dtype=np.float64)
    
    error_list = np.zeros((windows,3))
    ds = int(len(t_init)/(windows+1))
    ds = int(len(t_init)/(windows+3))
    #=========================================================================================================================
    #Begin Error Calculation
    #=========================================================================================================================

    for j in range(windows):

        
        t_input = t_init[j*ds:(j+4)*ds]
        Nn = len(t_input)
        a_input = a_init[j*ds:(j+4)*ds]
        
        hk_in = hk_arr[j*ds:(j+4)*ds]
        pq_in = pq_arr[j*ds:(j+4)*ds]
        
        mult = round(len(t_init)/(4*ds))
        #p_input = p_init[j*ds:(j+2)*ds]
        #q_input = q_init[j*ds:(j+2)*ds]
        #h_input = h_init[j*ds:(j+2)*ds]
        #k_input = k_init[j*ds:(j+2)*ds]

        #freqs_a = np.fft.rfftfreq(len(t_input),dt)
        #Ya_in = np.fft.rfft(a_input)
        #short_inds_win = np.where(abs(1/freqs_a) <= dt*3)[0]
        #Ya_in[short_inds_win] = 0
        #a_win = np.fft.irfft(Ya_in)
        

        #N_window = len(hk_in)
        N_window = Nn
        
        #j = 10
        #while N_window < goal:
        #    goal = 2**j
        #    j += 1
        #try:    
        goal = len(hk_arr)
        #hk_in = np.append(hk_in,np.zeros(goal-N_window))
        #pq_in = np.append(pq_in,np.zeros(goal-N_window))
        #except:
        
        #    do = None

        
        protecte_hk = max(round(protect_hk/mult),3)
        protecte_pq = max(round(protect_pq/mult),3)
        #protecte_hk = round(protect_hk)
        #protecte_pq = round(protect_pq)
        #kernele_g = max(round(kernel_g/mult),3)
        #kernele_s = max(round(kernel_s/mult),3)
        kernele_g = kernel_g
        kernele_s = kernel_s
        tole = tol/mult
        #tole = tol
        try:        
            hk_newe, hk_freqe, protect_hk_waste, hk_waste = extract_proper_mode(hk_in, t_input, g_arr, freq_tol=tol, protect_bins = protecte_hk, kernel=kernele_g, proper_freq=hk_freq, win=True)
            pq_newe, pq_freqe, protect_pq_waste, hk_waste = extract_proper_mode(pq_in, t_input, s_arr, freq_tol=tol, protect_bins = protecte_pq, kernel=kernele_s, proper_freq=pq_freq,  inc_filt = True, win=True)  
        except Exception as e:
            print('line 975', e)
            error_list[j][0] = 10
            error_list[j][1] = 10
            error_list[j][2] = 10
            continue
        #pes_e = np.array([np.mean(a_init),np.mean(np.abs(hk_new)),np.mean(np.abs(pq_new))])
        
        #pes_e = pe_vals(t_init,a_init,p_init,q_init,h_init,k_init,g_arr,s_arr,small_planets_flag,debug)    
        #Nn = 2*ds

            
        #error_list[j][0] = np.nanmean(a_win[int(per_shave*Nn):int((1-per_shave)*Nn)])
        error_list[j][0] = np.nanmean(a_input)
        #error_list[j][1] = np.nanmean(np.abs(hk_newe)[int(per_shave*Nn):int((1-per_shave)*Nn)])
        #error_list[j][2] = np.nanmean(np.abs(pq_newe)[int(per_shave*Nn):int((1-per_shave)*Nn)])
        
        #pe_en = np.nanmean(np.array([np.nanmean(np.abs(hk_newe)[int(0.05*Nn):int((1-0.05)*Nn)]),np.nanmean(np.abs(hk_newe)[int(0.1*Nn):int((1-0.1)*Nn)]),np.nanmean(np.abs(hk_newe)[int(0.2*Nn):int((1-0.2)*Nn)]),np.nanmean(np.abs(hk_newe)[int(0.3*Nn):int((1-0.3)*Nn)])]))
        #pe_in = np.nanmean(np.array([np.nanmean(np.abs(pq_newe)[int(0.05*Nn):int((1-0.05)*Nn)]),np.nanmean(np.abs(pq_newe)[int(0.1*Nn):int((1-0.1)*Nn)]),np.nanmean(np.abs(pq_newe)[int(0.2*Nn):int((1-0.2)*Nn)]),np.nanmean(np.abs(pq_newe)[int(0.3*Nn):int((1-0.3)*Nn)])]))

        pe_en = np.nanmean(np.abs(hk_newe[int(per_shave*Nn):int((1-per_shave)*Nn)]))
        pe_in = np.nanmean(np.abs(pq_newe[int(per_shave*Nn):int((1-per_shave)*Nn)]))
        
        hk_newes = hk_newe[int(per_shave*Nn):int((1-per_shave)*Nn)]
        pq_newes = pq_newe[int(per_shave*Nn):int((1-per_shave)*Nn)]
            
        
        e_wine = np.abs(hk_newes)*np.kaiser(len(hk_newes),6)
        inc_wine = np.abs(pq_newes)*np.kaiser(len(pq_newes),6)

        pe_en = np.nansum(e_wine)/np.sum(np.kaiser(len(hk_newes),6))
        pe_in = np.nansum(inc_wine)/np.sum(np.kaiser(len(hk_newes),6))
        
        error_list[j][1] = pe_en
        error_list[j][2] = pe_in
        #error_list[j][1] = np.nanmean(np.abs(hk_newe))
        #error_list[j][2] = np.nanmean(np.abs(pq_newe))
        '''
        if j <= 10:
            #plt.plot(np.abs(hk_in[:Nn]))
            plt.plot(np.abs(hk_newe[int(Nn*0.1):int(Nn*0.15)]))
            plt.title('ecc:'+str(j))
            plt.show()
            fig,ax = plt.subplots(1,2,figsize=(10,3))
            freqe = np.fft.fftfreq(len(hk_newe),dt)
            #freq_ind = np.where(freq == freqe)[0]
            ax[0].scatter(1/np.flip(freqe),np.abs(np.fft.fft(hk_newe))**2,s=5,alpha=0.5)
            ax[0].scatter(1/np.flip(freq),np.abs(np.fft.fft(hk_new))**2,s=5,alpha=0.5)
            ax[0].set_xscale('log')
            ax[0].set_yscale('log')
            freqe = np.fft.fftfreq(len(hk_in),dt)
            ax[1].scatter(1/np.flip(freqe),np.abs(np.fft.fft(hk_in))**2,s=5)
            ax[1].set_xscale('log')
            ax[1].set_yscale('log')
            plt.show()
            fig,ax = plt.subplots(1,2,figsize=(10,3))
            ax[0].scatter(1/(freqe),np.abs(np.fft.fft(hk_newe))**2,s=5)
            ax[0].scatter(1/(freq),np.abs(np.fft.fft(hk_new))**2,s=5)
            ax[0].set_xscale('log')
            ax[0].set_yscale('log')
            freqe = np.fft.fftfreq(len(hk_in),dt)
            ax[1].scatter(1/(freqe),np.abs(np.fft.fft(hk_in))**2,s=5)
            ax[1].set_xscale('log')
            ax[1].set_yscale('log')
            plt.show()
        #'''
        '''
        if j <= 10:
            #plt.plot(np.abs(pq_in[:Nn]))
            plt.plot(np.abs(pq_newe[int(Nn*0.1):int(Nn*0.15)]))
            plt.title('inc:'+str(j))
            plt.show()
            fig,ax = plt.subplots(1,2,figsize=(10,3))
            freqe = np.fft.fftfreq(len(hk_newe),dt)
            #freq_ind = np.where(freq == freqe)[0]
            ax[0].scatter(1/np.flip(freqe),np.abs(np.fft.fft(pq_newe))**2,s=5)
            ax[0].scatter(1/np.flip(freq),np.abs(np.fft.fft(pq_new))**2,s=5)
            ax[0].set_xscale('log')
            ax[0].set_yscale('log')
            ax[0].set_ylim(1e-8,1e3)
            freqe = np.fft.fftfreq(len(pq_in),dt)
            ax[1].scatter(1/np.flip(freqe),np.abs(np.fft.fft(pq_in))**2,s=5)
            ax[1].set_xscale('log')
            ax[1].set_yscale('log')
            plt.show()
            fig,ax = plt.subplots(1,2,figsize=(10,3))
            ax[0].scatter(1/(freqe),np.abs(np.fft.fft(pq_newe))**2,s=5)
            ax[0].scatter(1/(freq),np.abs(np.fft.fft(pq_new))**2,s=5)
            ax[0].set_xscale('log')
            ax[0].set_yscale('log')
            freqe = np.fft.fftfreq(len(pq_in),dt)
            ax[1].scatter(1/(freqe),np.abs(np.fft.fft(pq_in))**2,s=5)
            ax[1].set_xscale('log')
            ax[1].set_yscale('log')
            plt.show()

        print('Sum Win hk',np.sum(np.abs(np.fft.fft(hk_newe))**2))
        print('Sum Win pq',np.sum(np.abs(np.fft.fft(pq_newe))**2))
        #'''
        #error_list[j][0] = pes_e[0]
        #error_list[j][1] = pes_e[1]
        #error_list[j][2] = pes_e[2]

    #print('Sum hk',np.sum(np.abs(np.fft.fft(hk_new))**2))
    #print('Sum pq',np.sum(np.abs(np.fft.fft(pq_new))**2))
    rms = np.sqrt(np.nanmean((np.array(error_list,dtype=np.float64)-np.array(pes,dtype=np.float64))**2,axis=0))
    rms = np.nanstd(error_list,axis=0)
    if debug == True:
        #rms = np.sqrt(np.nanmean((np.array(error_list)-np.array(pes))**2,axis=0))
        #rms = np.nanstd(error_list,axis=0)
        return pes,error_list,rms,g_arr,s_arr, hk_arr[:N_og], pq_arr[:N_og], hk_new[:N_og], pq_new[:N_og], hk_freq, pq_freq, hk_signal, pq_signal
    #print(pes,error_list,rms)
    
    maxvals = np.max(np.array(error_list,dtype=np.float64)-np.array(pes,dtype=np.float64),axis=0)
    return_data = [objname]
    return_data.append(np.mean(a_init))
    return_data.append(np.mean(e_init))
    return_data.append(np.mean(inc_init))
    for i in range(len(pes)):
        return_data.append(pes[i])
    for i in range(len(error_list)):
        for j in range(len(error_list[0])):
            return_data.append(error_list[i][j])
    return_data.append(rms[0])
    return_data.append(rms[1])
    return_data.append(rms[2])
    return_data.append(maxvals[0])
    return_data.append(maxvals[1])
    return_data.append(maxvals[2])
    return return_data

def prop_multi(filename):
    names_df = pd.read_csv('../data/data_files/'+filename+'.csv').iloc[165:167]
    data = []
    for i,objname in enumerate(names_df['Name']):
        if i%50==0:
            print(i)
        #archive = rebound.SimulationArchive(fullfile)
        windows=5
        data_line = prop_calc(objname,filename,windows=windows)
        #data_line = prop_calc(str(i),filename)
        #print(data_line)
        data.append(data_line)
    column_names = ['Objname','ObsSMA','ObsEcc','ObsSin(Inc)','PropSMA','PropEcc','PropSin(Inc)']
    for i in range(windows):
        numrange = str(i)+'_'+str(i+2)+'PE'
        column_names.append(numrange+'_a')
        column_names.append(numrange+'_e')
        column_names.append(numrange+'_sinI')
            #print(numrange)
    column_names.append('RMS_err_a')
    column_names.append('RMS_err_e')
    column_names.append('RMS_err_sinI')
    column_names.append('Delta_a')
    column_names.append('Delta_e')
    column_names.append('Delta_sinI')
    data_df = pd.DataFrame(data,columns=column_names)
    data_df.to_csv('../data/results/'+filename+'_prop_elem_helio_2.csv')
    return data

if __name__ == "__main__":
    filename = str(sys.argv[1])
    
    if filename != 'Single':
        data = prop_multi(filename)  
    else:
        column_names = ['Objname','ObsSMA','ObsEcc','ObsSin(Inc)','PropSMA','PropEcc','PropSin(Inc)']
        windows = 5
        for i in range(windows):
            numrange = str(i)+'_'+str(i+2)+'PE'
            column_names.append(numrange+'_a')
            column_names.append(numrange+'_e')
            column_names.append(numrange+'_sinI')
            #print(numrange)
        column_names.append('RMS_err_a')
        column_names.append('RMS_err_e')
        column_names.append('RMS_err_sinI')
        column_names.append('Delta_a')
        column_names.append('Delta_e')
        column_names.append('Delta_sinI')
        objname = str(sys.argv[2])
        fullfile = '../data/'+filename+'/'+objname+'/archive.bin'
        #fullfile = '~/../../../hdd/haumea-data/djspenc/SBDynT_sims/'+filename+'/'+objname+'/archive.bin'
        #archive = rebound.SimulationArchive(fullfile)
        data_line = [np.array(prop_calc(objname,filename,windows),dtype=np.float64)]
        #print(data_line,len(data_line),len(column_names))
        #data_df = pd.DataFrame(np.zeros((1,len(column_names))),columns = column_names)
        data_df = pd.DataFrame(data_line,columns = column_names)
        #print(data_df)
        #data_df.iloc[i] = data_line
        data_df.to_csv('../data/Single/'+objname+'/'+objname+'_prop_elem_helio.csv')
        print(data_df)
        
                       

        
