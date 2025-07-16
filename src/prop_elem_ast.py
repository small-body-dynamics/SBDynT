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
                minarr = np.array([(Yhk_copy[secresind1[i]-2*spreads[i]-1:secresind1[i]-spreads[i]-1]),(Yhk_copy[secresind1[i]+spreads[i]+1:secresind1[i]+2*spreads[i]+1])]).astype(float)
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
                minarr = np.array([(Ypq_copy[secresind2[i]-2*spreads[i+len(secresind1)]-1:secresind2[i]-spreads[i+len(secresind1)]-1]),(Ypq_copy[secresind2[i]+spreads[i+len(secresind1)]+1:secresind2[i]+2*spreads[i+len(secresind1)]+1])]).astype(float)
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


def smooth_fft_convolution(fft_signal, freqs, primary_freqs, protect_radius_bins=3, kernel_size=15, method="gaussian"):
    """
    Fast convolutional smoothing of FFT log-power spectrum, excluding primary peaks.
    """
    fft_signal = np.asarray(fft_signal)
    log_power = 10 * np.log10(np.abs(fft_signal)**2 + 1e-12)
    phase = np.angle(fft_signal)

    N = len(fft_signal)
    mask = np.ones(N, dtype=bool)
    mask_opp = np.ones(N, dtype=bool)
    mask_long = np.ones(N, dtype=bool)
    mask_half = np.ones(N, dtype=bool)
    mask_double = np.ones(N, dtype=bool)
    
    # Mask protected bins around each primary frequency
    for pf in primary_freqs:
        idx = np.argmin(np.abs(freqs - pf))
        start = max(idx - protect_radius_bins, 0)
        end = min(idx + protect_radius_bins + 1, N)
        mask[start:end] = False
        
        idx_opp = np.argmin(np.abs(- pf - freqs))
        start = max(idx_opp - round(protect_radius_bins*0.5), 0)
        end = min(idx_opp + round(protect_radius_bins*0.5) + 1, N)
        mask_opp[start:end] = False

        inds_long = np.where(1/np.abs(freqs) >= 1e7)[0]
        mask_long[inds_long] = False
        
        inds_half = np.argmin(np.abs(freqs - pf/2))
        mask_half[inds_half-1:inds_half+2] = False
        
        inds_double = np.argmin(np.abs(freqs - pf*2))
        mask_double[inds_double-1:inds_double+2] = False
        
    peak = log_power[idx]
    # Prepare data and mask for convolution
    masked_log_power = log_power.copy()
    masked_log_power[~mask] = 0  # zero out protected region before smoothing
    masked_log_power[~mask_opp] = 0  # zero out protected region before smoothing
    #masked_log_power[~mask_long] = 0  # zero out protected region before smoothing
    #masked_log_power[~mask_half] = 0  # zero out protected region before smoothing
    #masked_log_power[~mask_double] = 0  # zero out protected region before smoothing
    mask_float = mask.astype(float)

    # Convolve both signal and mask to normalize after convolution
    if method == "gaussian":
        smoothed_log_power = gaussian_filter1d(masked_log_power, sigma=kernel_size, mode="mirror")
        smoothed_mask = gaussian_filter1d(mask_float, sigma=kernel_size, mode="mirror")
    else:  # fallback to uniform boxcar
        kernel = np.ones(kernel_size) / kernel_size
        smoothed_log_power = convolve1d(masked_log_power, kernel, mode="nearest")
        smoothed_mask = convolve1d(mask_float, kernel, mode="nearest")

    # Normalize result: divide smoothed signal by smoothed mask
    #smoothed_mask[int(N/2)-3:int(N/2)+3] = 1
    #smoothed_mask[N-6:int(N)-1] = 0
    #smoothed_mask[0] = 0
    with np.errstate(invalid='ignore', divide='ignore'):
        
        normalized_log_power = np.where(smoothed_mask > 1e-6 , smoothed_log_power / smoothed_mask, log_power)

    # Restore original log_power in protected regions
    
    #normalized_log_power[normalized_log_power > 1e-5*np.max(log_power)] = 1e-3*np.max(log_power)
    normalized_log_power[~mask] = log_power[~mask]
    normalized_log_power[~mask_opp] = log_power[~mask_opp]
    normalized_log_power[~mask_long] = log_power[~mask_long]
    #normalized_log_power[~mask_half] = log_power[~mask_half]
    #normalized_log_power[~mask_double] = log_power[~mask_double]
    
    

    # Back to magnitude
    magnitude = 10 ** (0.5 * normalized_log_power / 10)

    # Normalize power to match original
    orig_power = np.sum(np.abs(fft_signal)**2)
    new_power = np.sum(magnitude**2)
    scale = np.sqrt(orig_power / new_power)

    smoothed_fft = magnitude * np.exp(1j * phase) * scale
    #smoothed_fft[idx] = np.median(smoothed_fft[idx-1:idx+2])
    #smoothed_fft[idx] = 0
    
    scaling_factor = np.sqrt(1/(np.max(np.abs(smoothed_fft)**2) / np.max(np.abs(fft_signal)**2)))
    scaling_factor = np.sqrt(1/((np.abs(smoothed_fft[idx])**2) / (np.abs(fft_signal[idx])**2)))
    #scaling_factor=1
    #smoothed_fft[0] = np.median(smoothed_fft[1:10])
    return smoothed_fft * scaling_factor

def extract_proper_mode(signal, time, known_planet_freqs, freq_tol=2e-7, protect_bins=None,kernel=60, proper_freq = None, inc_filt = False):
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
    N = len(signal)
    dt = np.abs(time[1] - time[0])
    freqs = fftfreq(N, d=dt)
    spectrum = np.fft.fft(signal)

    if inc_filt:
        spectrum[0] = 0
    # Find the frequency with the highest power that is NOT near any planetary frequency
    power = np.abs(spectrum)**2
    sorted_indices = np.argsort(power)[::-1]  # sort descending by power

    dt = time[1]-time[0]
    
    if proper_freq == None:
        for i in range(len(sorted_indices)):
            idx = sorted_indices[i]
            #print('469',idx)
            f = freqs[idx]
            #'''
            if inc_filt == False:
                if dt*f < 0:
                    idx_n = sorted_indices[i+1]
                    pow_f = np.mean(np.abs(spectrum[idx-1:idx+2])**2)
                    pow_next = np.mean(np.abs(spectrum[idx_n-1:idx_n+2])**2)
                    if pow_f < pow_next:
                        continue
            
            if inc_filt == True:
                if dt*f > 0:
                    idx_n = sorted_indices[i+1]
                    pow_f = np.mean(np.abs(spectrum[idx-1:idx+2])**2)
                    pow_next = np.mean(np.abs(spectrum[idx_n-1:idx_n+2])**2)
                    if pow_f < pow_next:
                        continue
            #'''
            #proper_freq = f
            #break
            if all(np.abs(f - pf) > freq_tol for pf in known_planet_freqs):
                idx_n = sorted_indices[i+1]
                pow_f = np.sum(np.abs(spectrum[idx-1:idx+2])**2)
                pow_next = np.sum(np.abs(spectrum[idx_n-1:idx_n+2])**2)

                if pow_f < pow_next:
                    continue
                
                proper_idx = np.argmax(np.abs(spectrum[idx-2:idx+3])**2)+idx-2
                proper_freq = freqs[proper_idx]
                
                break
            '''
            else:
                idx_n = sorted_indices[i+1]

                pow_f = np.sum(np.abs(spectrum[idx-1:idx+2])**2)
                pow_next = np.sum(np.abs(spectrum[idx_n-1:idx_n+2])**2)

                if pow_f < pow_next:
                    continue
                    
                idx_double = np.argmin(abs(freqs-f*2))
                idx_half = np.argmin(abs(freqs-f/2))

                idx_higher = 0
                pow_higher = 0
                if np.sum(np.abs(spectrum[idx_double-1:idx_double+2])**2) > np.sum(np.abs(spectrum[idx_half-1:idx_half+2])**2):
                    idx_higher = idx_double
                    #pow_higher = np.sum(np.abs(spectrum[idx_double-1:idx_double+2])**2)
                    pow_higher = np.max(np.abs(spectrum[idx_double-1:idx_double+2])**2)
                else:
                    idx_higher = idx_half
                    #pow_higher = np.sum(np.abs(spectrum[idx_half-1:idx_half+2])**2)
                    pow_higher = np.max(np.abs(spectrum[idx_half-1:idx_half+2])**2)

                #db_diff = np.sum(np.abs(spectrum[idx-1:idx+2])**2)/pow_higher
                db_diff = np.abs(spectrum[idx])**2/pow_higher
                x = 10/db_diff

                spectrum[idx] = x*spectrum[idx]

                proper_idx = np.argmax(np.abs(spectrum[idx-2:idx+3])**2)+idx-2
                proper_freq = freqs[proper_idx]
                #spectrum[idx] = spectrum[idx]*x
                
                break
            #'''
    else:
        proper_idx = np.argmin(abs(freqs - proper_freq))

    new_spec = spectrum.copy()

    #if inc_filt:
    #    known_planet_freqs.append(-2*proper_freq-known_planet_freqs[0])
    #    known_planet_freqs.append(proper_freq-known_planet_freqs[0])
    #    known_planet_freqs.append(proper_freq-known_planet_freqs[2])
    #    known_planet_freqs.append(proper_freq-known_planet_freqs[4])

    '''
    z1 = (g+s-g6-s6); z2 = (g+s-g5-s7); z3 = (g+s-g5-s6); z4 = (2*g6-g5); 
    z5 = (2*g6-g7); z6 = (s-s6-g5+g6); z7 = (g-3*g6+2*g5)    
    z8 = (2*(g-g6)+s-s6); z9 = (3*(g-g6)+s-s6); z10 = ((g-g6)+s-s6)
    z11 = g-2*g7+g6; z12 = 2*g-2*g5; z13 = -4*g+4*g7; z14 = -2*s-s6
    '''

    
    
    '''
    if inc_filt != None:
        for idx in sorted_indices[:10]:
            #print(idx,len(freqs))
            f = freqs[idx]
            if f == proper_freq:
                continue
            #if np.abs(spectrum[idx])**2 > np.abs(spectrum[proper_idx])**2:
            #    continue
            if abs(proper_idx-idx) <= 3:
                continue
            if abs(len(spectrum)-idx) < 4:
                continue
            if any(np.abs(f - pf) <= freq_tol for pf in known_planet_freqs):
                #print(idx,np.abs(spectrum[idx])**2)
                #print(len(new_spec))
                print(f,1/f)
                spectrum[idx-1:idx+2]=new_spec[idx-1:idx+2]*0.01
            #if any(np.abs(f - abs(pf)) <= freq_tol for pf in known_planet_freqs):
                #print(idx,np.abs(spectrum[idx])**2)
                #print(len(new_spec))
            #    print(f,1/f)
            #    spectrum[idx-6:idx+7]=np.mean(new_spec[idx-6:idx+7])
    #'''    
    #if inc_filt == False:
    #print('536')
    if inc_filt != None:
        for pf in known_planet_freqs:
            idx = np.argmin(abs(freqs - pf))
            
            #print('544',idx)
            #idx_opp = np.argmin(abs(- pf - freqs))
        
            if abs(proper_idx-idx) <= 5:
                #spectrum[idx] = new_spec[idx]/1.5
                continue
            if pf == proper_freq:
                continue
            
        #print(1/pf,idx)    
        #print(spectrum[idx-40:idx+41])
            #spectrum[idx]=new_spec[idx]*0.1
            #if abs(len(spectrum) - idx) > 5:
            #    spectrum[idx-4:idx+5] = np.mean(np.array([new_spec[idx-4],new_spec[idx+5]]))
            #if abs(len(spectrum) - idx) > 4:
            #    spectrum[idx-3:idx+4] = np.mean(np.array([new_spec[idx-3],new_spec[idx+4]]))
            #if abs(len(spectrum) - idx) > 3:
            #    spectrum[idx-2:idx+3] = np.mean(np.array([new_spec[idx-2],new_spec[idx+3]]))
            elif abs(len(spectrum) - idx) > 2:
                spectrum[idx-1:idx+2] = np.mean(np.array([new_spec[idx-1],new_spec[idx+2]]))
            elif abs(len(spectrum) - idx) > 1:
                spectrum[idx] = new_spec[idx]*0.5
            else:
                continue
            #if abs(len(spectrum) - idx_opp) > 2:
            #    spectrum[idx_opp-2:idx_opp+3] = np.mean(np.array([new_spec[idx_opp-2],new_spec[idx_opp+3]]))
            #else: 
            #    spectrum[idx_opp] = new_spec[idx_opp]*0.1
            
            #if abs(len(spectrum) - idx) > 2:
            #    spectrum[idx-1] = np.mean(np.array([new_spec[idx-2],new_spec[idx]])) 
            #    spectrum[idx+1] = np.mean(np.array([new_spec[idx],new_spec[idx+2]]))
        #print(spectrum[idx-40:idx+41])
            
        #print(1/pf,idx)    
        #print(spectrum[idx-40:idx+41])
        #spectrum[idx-15:idx+16]=np.median(new_spec[idx-15:idx+16])
        #print(spectrum[idx-40:idx+41])
    #'''
    if proper_freq is None:
        raise ValueError("No proper (free) frequency found distinct from planetary modes.")

    '''
    plt.scatter(1/freqs,power,s=10)
    plt.vlines(1/proper_freq,ymin=1e-2,ymax=1e6,colors='r')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    '''

    #leakage_profile = np.sinc(bins - delta)**2
    #leakage_db = 10 * np.log10(leakage_profile / np.max(leakage_profile))

    if protect_bins == None:
        freq_power = np.abs(spectrum[proper_idx])**2
        max_power = np.max(np.abs(spectrum[proper_idx-4:proper_idx+4])**2)
        

        
        j1 = 1
        j2 = 1
        down_power = np.abs(spectrum[proper_idx-j1])**2
        up_power = np.abs(spectrum[proper_idx+j2])**2
        #while 10*np.log10(next_power/freq_power) > -30:
        lim = -36
        if inc_filt:
            if abs(proper_freq) > 1e-6:
                lim = -50
        else:
            if abs(proper_freq) > 1e-6:
                lim = -45
        

        first_lim_hit = False

        idx_double = np.argmin(abs(freqs-proper_freq*2))
        idx_half = np.argmin(abs(freqs-proper_freq/2))
        idx_min = np.min(np.array([idx_double,idx_half]))
        idx_max = np.max(np.array([idx_double,idx_half]))
        while ((10*np.log10(down_power/max_power) > lim) and (10*np.log10(up_power/max_power) > lim)) or first_lim_hit == False:
            if proper_idx-j1 != 0:
                j1 += 1
            if proper_idx+j2 != len(spectrum)-1:
                j2 += 1
            down_power = np.abs(spectrum[proper_idx-j1])**2
            up_power = np.abs(spectrum[proper_idx+j2])**2
            if proper_idx - j1 < idx_min or proper_idx + j2 > idx_max:
                first_lim_hit = True

            if proper_idx-j1 == 0 and proper_idx+j2 != len(spectrum)-1:
                break
                

        protect_bins = np.max(np.array([j1+1,j2+1]))
        protect_second = round(abs(proper_freq*dt*N*0.15))

        if inc_filt:
            protect_second = round(protect_second*2)
        if abs(proper_freq) > 1e-6:
            protect_second = round(protect_second*1.5)

        if protect_second > protect_bins:
            #print('switch')
            protect_bins = protect_second
            #if dt*proper_freq < 0:            
            #    protect_bins = round(protect_bins*1.25)
        #if inc_filt:
        #    protect_bins = round(protect_bins/1.5**2)
        #if dt*f < 0:
        #    protect_bins = round(protect_bins/1.5)
        #protect_bins = 500
    if protect_bins < 1:
        protect_bins = 1
    #print('protect_bins:',protect_bins)
    filt_signal = smooth_fft_convolution(spectrum, freqs, [proper_freq], 
                                         protect_radius_bins=protect_bins, 
                                         kernel_size=kernel, method="gaussian")
    #filt_signal = smooth_fft_convolution(filt_signal, freqs, [proper_freq], 
    #                                     protect_radius_bins=int(protect_bins/3), 
    #                                     kernel_size=int(kernel*2), 
    #                                     method="gaussian")
    '''
    for idx in sorted_indices[:10]:
            #print(idx,len(freqs))
        f = freqs[idx]
        if f == proper_freq:
            continue
            #if np.abs(spectrum[idx])**2 > np.abs(spectrum[proper_idx])**2:
            #    continue
        if abs(proper_idx-idx) <= 2:
            continue
        if abs(len(spectrum)-idx) < 4:
            continue
        if any(np.abs(f - pf) <= freq_tol for pf in known_planet_freqs):
                #print(idx,np.abs(spectrum[idx])**2)
                #print(len(new_spec))
            spectrum[idx-3:idx+4]=np.mean(new_spec[idx-3:idx+4])
    '''
    '''plt.scatter(1/freqs,np.abs(filt_signal)**2,s=10)
    plt.vlines(1/proper_freq,ymin=1e-2,ymax=1e6,colors='r')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()'''
    # Build a band-pass filter centered on the proper frequency
    #mask = np.abs(freqs - proper_freq) < keep_width
    #filtered_spectrum = np.zeros_like(spectrum)
    #filtered_spectrum[mask] = spectrum[mask]

    # Inverse FFT to get the filtered signal
    proper_signal = np.fft.ifft(filt_signal)

    
    return proper_signal, proper_freq, protect_bins



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

        if abs(time_run)<=9e6:
            print('Integration not long enough for accurate proper element calculation')
            
            return [objname,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        
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

        Yhk = np.fft.fft(hk_arr)
        Ypq = np.fft.fft(pq_arr)
        
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
    
            Yhkj = np.fft.fft(kj+1j*hj); Yhks = np.fft.fft(ks+1j*hs); Yhku = np.fft.fft(ku+1j*hu); Yhkn = np.fft.fft(kn+1j*hn)
                    
            Ypqj = np.fft.fft(qj+1j*pj); Ypqs = np.fft.fft(qs+1j*ps); Ypqu = np.fft.fft(qu+1j*pu); Ypqn = np.fft.fft(qn+1j*pn)
            
    
            g_arr = []
            s_arr = []
            
            n = len(aj_init)
            dt = abs(archive[1].t)
            if n < 10001:
                print(n)
            freq = np.fft.fftfreq(n,d=dt)
            #freqn = np.fft.rfftfreq(len(aj_init),d=dt)

            half = int(len(freq)/2)
            
            g5 = freq[np.argmax(np.abs(Yhkj[1:])**2)+1]
            #g_arr.append(g5)
            g5 = freq[np.argmax(np.abs(Yhkj[1:half])**2)+1]
            g5_neg = freq[np.argmax(np.abs(Yhkj[half:-2])**2)+half]
            
            g_arr.append(g5)
            g_arr.append(g5_neg)
            
            while freq[np.argmax(np.abs(Yhks[1:])**2)+1] in g_arr:
                Yhks[np.argmax(np.abs(Yhks[1:])**2)+1] = 0
                
            g6 = freq[np.argmax(np.abs(Yhks[1:])**2)+1]
            
            g6 = freq[np.argmax(np.abs(Yhks[1:half])**2)+1]
            g6_neg = freq[np.argmax(np.abs(Yhks[half:-2])**2)+half]
            
            g_arr.append(g6)
            g_arr.append(g6_neg)
            
            while freq[np.argmax(np.abs(Yhku[1:])**2)+1] in g_arr:
                Yhku[np.argmax(np.abs(Yhku[1:])**2)+1] = 0
            g7 = freq[np.argmax(np.abs(Yhku[1:])**2)+1]

            g7 = freq[np.argmax(np.abs(Yhku[1:half])**2)+1]
            g7_neg = freq[np.argmax(np.abs(Yhku[half:-2])**2)+half]
            
            g_arr.append(g7)
            g_arr.append(g7_neg)
            
            while freq[np.argmax(np.abs(Yhkn[1:])**2)+1] in g_arr:
                Yhkn[np.argmax(np.abs(Yhkn[1:])**2)+1] = 0
            g8 = freq[np.argmax(np.abs(Yhkn[1:])**2)+1]
            #g_arr.append(g8)

            g8 = freq[np.argmax(np.abs(Yhkn[1:half])**2)+1]
            g8_neg = freq[np.argmax(np.abs(Yhkn[half:-2])**2)+half]
            
            g_arr.append(g8)
            g_arr.append(g8_neg)
            
            s6 = freq[np.argmax(np.abs(Ypqs[1:])**2)+1]
            s6 = freq[np.argmax(np.abs(Ypqs[1:half])**2)+1]
            s6_neg = freq[np.argmax(np.abs(Ypqs[half:-2])**2)+half]
            
            s_arr.append(s6)
            s_arr.append(s6_neg)

            
            while freq[np.argmax(np.abs(Ypqu[1:])**2)+1] in s_arr:
                Ypqu[np.argmax(np.abs(Ypqu[1:])**2)+1] = 0
            s7 = freq[np.argmax(np.abs(Ypqu[1:])**2)+1]

            s7 = freq[np.argmax(np.abs(Ypqu[1:half])**2)+1]
            s7_neg = freq[np.argmax(np.abs(Ypqu[half:-2])**2)+half]
            
            s_arr.append(s7)
            s_arr.append(s7_neg)
            #s_arr.append(s7)
            while freq[np.argmax(np.abs(Ypqn[1:])**2)+1] in s_arr:
                Ypqn[np.argmax(np.abs(Ypqn[1:])**2)+1] = 0
            s8 = freq[np.argmax(np.abs(Ypqn[1:])**2)+1]

            s8 = freq[np.argmax(np.abs(Ypqn[1:half])**2)+1]
            s8_neg = freq[np.argmax(np.abs(Ypqn[half:-2])**2)+half]
            
            s_arr.append(s8)
            s_arr.append(s8_neg)
            
            #s_arr.append(s8)
            
            if small_planets_flag:
                while freq[np.argmax(np.abs(Yhke[1:])**2)+1] in g_arr:
                    Yhke[np.argmax(np.abs(Yhke[1:])**2)+1] = 0
                g3 = freq[np.argmax(np.abs(Yhke[1:])**2)+1]
                
                g3 = freq[np.argmax(np.abs(Ypqe[1:half])**2)+1]
                g3_neg = freq[np.argmax(np.abs(Ypqe[half:-2])**2)+half]
            
                g_arr.append(g3)
                g_arr.append(g3_neg)
                
                while freq[np.argmax(np.abs(Yhkv[1:])**2)+1] in g_arr:
                    Yhkv[np.argmax(np.abs(Yhkv[1:])**2)+1] = 0
                g2 = freq[np.argmax(np.abs(Yhkv[1:])**2)+1]
                g2 = freq[np.argmax(np.abs(Ypqv[1:half])**2)+1]
                g2_neg = freq[np.argmax(np.abs(Ypqv[half:-2])**2)+half]
            
                g_arr.append(g2)
                g_arr.append(g2_neg)

                while freq[np.argmax(np.abs(Yhkm[1:])**2)+1] in g_arr:
                    Yhkm[np.argmax(np.abs(Yhkm[1:])**2)+1] = 0
                g4 = freq[np.argmax(np.abs(Yhkm[1:])**2)+1]
                g4 = freq[np.argmax(np.abs(Ypqm[1:half])**2)+1]
                g4_neg = freq[np.argmax(np.abs(Ypqm[half:-2])**2)+half]
            
                g_arr.append(g4)
                g_arr.append(g4_neg)
                
                while freq[np.argmax(np.abs(Ypqe[1:])**2)+1] in s_arr:
                    Ypqe[np.argmax(np.abs(Ypqe[1:])**2)+1] = 0
                
                s3 = freq[np.argmax(np.abs(Ypqe[1:])**2)+1]
                s3 = freq[np.argmax(np.abs(Ypqe[1:half])**2)+1]
                s3_neg = freq[np.argmax(np.abs(Ypqe[half:-2])**2)+half]
            
                s_arr.append(s3)
                s_arr.append(s3_neg)
                
                while freq[np.argmax(np.abs(Ypqv[1:])**2)+1] in s_arr:
                    Ypqv[np.argmax(np.abs(Ypqv[1:])**2)+1] = 0
                s2 = freq[np.argmax(np.abs(Ypqv[1:])**2)+1]
                s2 = freq[np.argmax(np.abs(Ypqv[1:half])**2)+1]
                s2_neg = freq[np.argmax(np.abs(Ypqv[half:-2])**2)+half]
            
                s_arr.append(s2)
                s_arr.append(s2_neg)
                while freq[np.argmax(np.abs(Ypqm[1:])**2)+1] in s_arr:
                    Ypqm[np.argmax(np.abs(Ypqm[1:])**2)+1] = 0
                s4 = freq[np.argmax(np.abs(Ypqm[1:])**2)+1]
                s4 = freq[np.argmax(np.abs(Ypqm[1:half])**2)+1]
                s4_neg = freq[np.argmax(np.abs(Ypqm[half:-2])**2)+half]
            
                s_arr.append(s4)
                s_arr.append(s4_neg)
                #s_arr.append(s4)

            g = freq[np.argmax(np.abs(Yhk[1:])**2)]
            s = freq[np.argmax(np.abs(Ypq[1:])**2)]
    
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

            g_ex = np.array([g5-g6,g5-g7,g5-g8,g6-g7,g6-g8,g7-g8])
            s_ex = np.array([s6-s7,s6-s8,s7-s8])
            
            for i in g_ex:
                g_arr.append(i)
            #for i in freq1:
            #    g_arr.append(i)
            for i in s_ex:
                s_arr.append(i)
            #for i in freq2:
            #    s_arr.append(i)
        
            
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
    #print('g_arr',1/np.array(g_arr))
    #print('s_arr',1/np.array(s_arr))
    try:
        dt = np.abs(t_init[1]-t_init[0]) 
        #dt = np.abs(archive[1].t) 
    
        kernel=round(250*2**(2-np.log10(dt)))
        protect = round(10*2**(2-np.log10(dt)))
        tol = 1/dt/(len(hk_arr)-1)*0.9
        per_shave = 0.05
        print('dt',dt,'tol',tol)
    
        hk_new, hk_freq, protect_hk = extract_proper_mode(hk_arr, t_init, g_arr, freq_tol=tol, kernel=kernel)
        pq_new, pq_freq, protect_pq = extract_proper_mode(pq_arr, t_init, s_arr, freq_tol=tol, kernel=kernel, inc_filt = True)
        N = len(hk_new)

        pe_e = np.mean(np.array([np.mean(np.abs(hk_new)[int(0.05*N):int((1-0.05)*N)]),np.mean(np.abs(hk_new)[int(0.1*N):int((1-0.1)*N)]),np.mean(np.abs(hk_new)[int(0.2*N):int((1-0.2)*N)]),np.mean(np.abs(hk_new)[int(0.3*N):int((1-0.3)*N)])]))
        pe_i = np.mean(np.array([np.mean(np.abs(pq_new)[int(0.05*N):int((1-0.05)*N)]),np.mean(np.abs(pq_new)[int(0.1*N):int((1-0.1)*N)]),np.mean(np.abs(pq_new)[int(0.2*N):int((1-0.2)*N)]),np.mean(np.abs(pq_new)[int(0.3*N):int((1-0.3)*N)])]))

        pe_e = np.mean(np.abs(hk_new[int(per_shave*N):int((1-per_shave)*N)]))
        pe_i = np.mean(np.abs(pq_new[int(per_shave*N):int((1-per_shave)*N)]))
        #pes = np.array([np.nanmean(a_init),np.nanmean(np.abs(hk_new)[int(per_shave*N):int((1-per_shave)*N)]),np.nanmean(np.abs(pq_new)[int(per_shave*N):int((1-per_shave)*N)])])
        pes = np.array([np.nanmean(a_init),pe_e,pe_i])
    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno

        error_message = "An error occurred in at line "+str(line_number)
    
        # Print the error message
        print(error_message)
        print(error)
        pes = np.array([10,10,10])
    
    error_list = np.zeros((windows,3))
    ds = int(len(t_init)/(windows+1))
    #=========================================================================================================================
    #Begin Error Calculation
    #=========================================================================================================================

    for j in range(windows):
        
        t_input = t_init[j*ds:(j+2)*ds]
        a_input = a_init[j*ds:(j+2)*ds]
        p_input = p_init[j*ds:(j+2)*ds]
        q_input = q_init[j*ds:(j+2)*ds]
        h_input = h_init[j*ds:(j+2)*ds]
        k_input = k_init[j*ds:(j+2)*ds]
        
        hk_in = hk_arr[j*ds:(j+2)*ds]
        pq_in = pq_arr[j*ds:(j+2)*ds]

        protecte_hk = round(protect_hk/(windows-1))
        protecte_pq = round(protect_pq/(windows-1))
        kernele = round(kernel/(windows-1))
        tole = tol/(windows-1)
        try:        
            hk_newe, hk_freqe, protect_hk_waste = extract_proper_mode(hk_in, t_input, g_arr, freq_tol=tol, protect_bins = protecte_hk, kernel=kernele, proper_freq=hk_freq)
            pq_newe, pq_freqe, protect_pq_waste = extract_proper_mode(pq_in, t_input, s_arr, freq_tol=tol, protect_bins = protecte_pq, kernel=kernele, proper_freq=pq_freq,  inc_filt = True)  
        except Exception as e:
            print('line 975', e)
            error_list[j][0] = 10
            error_list[j][1] = 10
            error_list[j][2] = 10
            continue
        #pes_e = np.array([np.mean(a_init),np.mean(np.abs(hk_new)),np.mean(np.abs(pq_new))])
        
        #pes_e = pe_vals(t_init,a_init,p_init,q_init,h_init,k_init,g_arr,s_arr,small_planets_flag,debug)    
        Nn = len(hk_newe)

            
        error_list[j][0] = np.nanmean(a_input)
        #error_list[j][1] = np.nanmean(np.abs(hk_newe)[int(per_shave*Nn):int((1-per_shave)*Nn)])
        #error_list[j][2] = np.nanmean(np.abs(pq_newe)[int(per_shave*Nn):int((1-per_shave)*Nn)])
        
        #pe_en = np.nanmean(np.array([np.nanmean(np.abs(hk_newe)[int(0.05*Nn):int((1-0.05)*Nn)]),np.nanmean(np.abs(hk_newe)[int(0.1*Nn):int((1-0.1)*Nn)]),np.nanmean(np.abs(hk_newe)[int(0.2*Nn):int((1-0.2)*Nn)]),np.nanmean(np.abs(hk_newe)[int(0.3*Nn):int((1-0.3)*Nn)])]))
        #pe_in = np.nanmean(np.array([np.nanmean(np.abs(pq_newe)[int(0.05*Nn):int((1-0.05)*Nn)]),np.nanmean(np.abs(pq_newe)[int(0.1*Nn):int((1-0.1)*Nn)]),np.nanmean(np.abs(pq_newe)[int(0.2*Nn):int((1-0.2)*Nn)]),np.nanmean(np.abs(pq_newe)[int(0.3*Nn):int((1-0.3)*Nn)])]))

        pe_en = np.nanmean(np.abs(hk_newe[int(per_shave*Nn):int((1-per_shave)*Nn)]))
        pe_in = np.nanmean(np.abs(pq_newe[int(per_shave*Nn):int((1-per_shave)*Nn)]))
        error_list[j][1] = pe_en
        error_list[j][2] = pe_in
        #error_list[j][1] = np.nanmean(np.abs(hk_newe))
        #error_list[j][2] = np.nanmean(np.abs(pq_newe))
        '''
        if j <= 10:
            plt.plot(np.abs(hk_in))
            plt.plot(np.abs(hk_newe))
            plt.title('ecc:'+str(j))
            plt.show()
            fig,ax = plt.subplots(1,2,figsize=(10,3))
            freqe = np.fft.fftfreq(len(hk_newe),dt)
            ax[0].scatter(1/np.flip(freqe),np.abs(np.fft.fft(hk_newe))**2,s=5)
            ax[0].set_xscale('log')
            ax[0].set_yscale('log')
            freqe = np.fft.fftfreq(len(hk_in),dt)
            ax[1].scatter(1/np.flip(freqe),np.abs(np.fft.fft(hk_in))**2,s=5)
            ax[1].set_xscale('log')
            ax[1].set_yscale('log')
            plt.show()
            fig,ax = plt.subplots(1,2,figsize=(10,3))
            ax[0].scatter(1/(freqe),np.abs(np.fft.fft(hk_newe))**2,s=5)
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
            plt.plot(np.abs(pq_in))
            plt.plot(np.abs(pq_newe))
            plt.title('ecc:'+str(j))
            plt.show()
            fig,ax = plt.subplots(1,2,figsize=(10,3))
            freqe = np.fft.fftfreq(len(pq_newe),dt)
            ax[0].scatter(1/np.flip(freqe),np.abs(np.fft.fft(pq_newe))**2,s=5)
            ax[0].set_xscale('log')
            ax[0].set_yscale('log')
            freqe = np.fft.fftfreq(len(pq_in),dt)
            ax[1].scatter(1/np.flip(freqe),np.abs(np.fft.fft(pq_in))**2,s=5)
            ax[1].set_xscale('log')
            ax[1].set_yscale('log')
            plt.show()
            fig,ax = plt.subplots(1,2,figsize=(10,3))
            ax[0].scatter(1/(freqe),np.abs(np.fft.fft(pq_newe))**2,s=5)
            ax[0].set_xscale('log')
            ax[0].set_yscale('log')
            freqe = np.fft.fftfreq(len(pq_in),dt)
            ax[1].scatter(1/(freqe),np.abs(np.fft.fft(pq_in))**2,s=5)
            ax[1].set_xscale('log')
            ax[1].set_yscale('log')
            plt.show()
        #'''
        #error_list[j][0] = pes_e[0]
        #error_list[j][1] = pes_e[1]
        #error_list[j][2] = pes_e[2]

    if debug == True:
        rms = np.sqrt(np.nanmean((np.array(error_list)-np.array(pes))**2,axis=0))
        rms = np.nanstd(error_list,axis=0)
        return pes,error_list,rms,g_arr,s_arr, hk_arr, pq_arr, hk_new, pq_new, hk_freq, pq_freq
    rms = np.sqrt(np.nanmean((np.array(error_list)-np.array(pes))**2,axis=0))
    rms = np.nanstd(error_list,axis=0)
    #print(pes,error_list,rms)
    
    maxvals = np.max(np.array(error_list)-np.array(pes),axis=0)
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
    names_df = pd.read_csv('../data/data_files/'+filename+'.csv')
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
    data_df.to_csv('../data/results/'+filename+'_prop_elem_helio.csv')
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
        data_line = [np.array(prop_calc(objname,filename,windows))]
        #print(data_line,len(data_line),len(column_names))
        #data_df = pd.DataFrame(np.zeros((1,len(column_names))),columns = column_names)
        data_df = pd.DataFrame(data_line,columns = column_names)
        #print(data_df)
        #data_df.iloc[i] = data_line
        data_df.to_csv('../data/Single/'+objname+'/'+objname+'_prop_elem_helio.csv')
        print(data_df)
        
                       

        
