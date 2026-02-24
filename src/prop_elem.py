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
#import tools
from tools import *
from scipy.stats import circmean
import plotting_scripts as ps



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

class proper_element_class:
    def __init__(self, des=''):
        
        self.des = des
        self.planets = {}
        self.planet_freqs = {}
        self.tmax = 0
        self.tout = 0
        self.osculating_elements = {}
        self.proper_elements = {}
        self.mean_elements = {}
        self.proper_errors = {}
        self.proper_extras = {}
        self.proper_indicators = {}
        self.proper_internal = {}
        self.prop_finish = False
        self.scattered = {'scattered': False, 'scat_time': np.inf, 'Max delta-E': 0}
        self.family_results = {'family_name': None, 'pairwise_dMet': np.inf}

        #Plotting Flags
        self.p_hkpq = True
        self.p_eI = False
        self.p_vO = False


    def plot_time_arrays(self):
        ps.plot_osc_and_prop(self)

    def plot_freq_space(self, plotflag='1', ifreqs={}):
        if '1' in plotflag:
            self.p_hkpq = True
        else:
            self.p_hkpq = False
        if '2' in plotflag:
            self.p_eI = True
        else:
            self.p_eI = False
        if '3' in plotflag:
            self.p_vO = True
        else:
            self.p_vO = False
            
        ps.plot_freq_space(self, ifreqs = ifreqs)

    def plot_hkpq(self):
        ps.plot_hkpq(self)

    def plot_angles(self, plot_cos=False, ifreqs={}):
        ps.plot_angles(self, plot_cos=plot_cos, ifreqs=ifreqs)

    #Function that calls the proper element computation
    def compute_proper(self, windows=5, time_run = 0, rms = True, debug = False):
        self.windows = windows
        self.time_run = time_run
        
        outputs = prop_calc(self.designation, filename=self.filename, windows=windows, direction = self.direction, time_run = time_run, rms = rms, debug=debug)

        #
            
        self.proper_elements['a'] = outputs[10]    
        self.proper_elements['e'] = outputs[11]    
        self.proper_elements['sinI'] = outputs[12] 
        self.proper_elements['omega'] = outputs[13]    
        self.proper_elements['Omega'] = outputs[14]
        
        self.osculating_elements['a'] = outputs[1]    
        self.osculating_elements['e'] = outputs[2]    
        self.osculating_elements['sinI'] = outputs[3] 
        self.osculating_elements['omega'] = outputs[4]    
        self.osculating_elements['Omega'] = outputs[5]    
        self.osculating_elements['M'] = outputs[6]

        self.mean_elements['a'] = outputs[7]    
        self.mean_elements['e'] = outputs[8]    
        self.mean_elements['sinI'] = outputs[9] 

        self.proper_errors['RMS_a'] = outputs[15 + self.windows*3]
        self.proper_errors['RMS_e'] = outputs[15 + self.windows*3 + 1]
        self.proper_errors['RMS_sinI'] = outputs[15 + self.windows*3 + 2]
        
        self.proper_elements['g'] = outputs[15 + self.windows*3 + 6]    
        self.proper_elements['s'] = outputs[15 + self.windows*3 + 7]

        self.proper_windows = outputs[15:15 + self.windows*3]

        self.proper_extras['res_e'] = outputs[15 + self.windows*3 + 8]
        self.proper_extras['res_I'] = outputs[15 + self.windows*3 + 9]
        
        self.proper_extras['sec_res_e'] = outputs[15 + self.windows*3 + 10]
        self.proper_extras['sec_res_I'] = outputs[15 + self.windows*3 + 11]
        
        self.proper_extras['e_osc_amp'] = outputs[15 + self.windows*3 + 12]
        self.proper_extras['I_osc_amp'] = outputs[15 + self.windows*3 + 13]
        
        self.proper_extras['e_filt_amp'] = outputs[15 + self.windows*3 + 14]
        self.proper_extras['I_filt_amp'] = outputs[15 + self.windows*3 + 15]
        
        self.proper_extras['angle_sec_res'] = outputs[15 + self.windows*3 + 16]
        self.proper_extras['librating_angle'] = outputs[15 + self.windows*3 + 17]
        self.proper_extras['phi_entropy'] = outputs[15 + self.windows*3 + 18]
        self.proper_extras['phi_frac'] = outputs[15 + self.windows*3 + 19]
        self.prop_finish = True

        for i in outputs[-1]:
            self.planet_freqs[i] = outputs[-1][i]*3600*360
        print('Proper Elements:',self.proper_elements)


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


def smooth_fft_convolution_old(fft_signal, freqs, primary_freqs, time, protect_radius_bins_init=3, kernel_size=15, method="gaussian", inc_filt = False, known_planet_freqs = [],freq_tol=2e-7, win=False, shortfilt = True):
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

    # Setting mask values to False causes them to be protected/preserved to original amplitudes
    mask = np.ones(N, dtype=bool)
    mask_opp = np.ones(N, dtype=bool)
    mask_long = np.ones(N, dtype=bool)
    mask_short = np.ones(N, dtype=bool)
    #mask_double = np.ones(N, dtype=bool)ll
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
    
    
    new_spec = log_power.copy()
    dt = abs(time[1]-time[0])
    tol_bins = abs(round(freq_tol*len(fft_signal)*dt))
    if tol_bins < 1:
        tol_bins = 1
    
    #if inc_filt == False:
    
    mask_filt = np.ones(N, dtype=bool)
    
    masked_log_power = log_power.copy()

    lowest_per_p = np.min(abs(1/np.array(known_planet_freqs)))
    lowest_per_gs = np.min(abs(1/np.array(primary_freqs)))
    lowest_period = min(lowest_per_p,lowest_per_gs)
        
    shortperiod = np.where(1/abs(freqs) < lowest_period/4)[0]
    #shortperiod = np.intersect1d(shortperiod, np.where(masked_log_power > np.median(masked_log_power))[0])

    if len(shortperiod) > 0 and shortfilt:    
        #print('filtering shortperiod', shortfilt)
        short_ref = new_spec[shortperiod]
        masked_log_power[shortperiod] = np.median(masked_log_power)
        
        
    
    if inc_filt != None:
        for pf in known_planet_freqs:

            dex_filt = dex_protect/2
            if win:
                dex_filt = dex_filt*4/3
            
            idx = np.nanargmin(abs(freqs - pf))
            if idx-tol_bins <= 0 or idx+tol_bins >= len(log_power):
                continue

            dist_freq = np.abs(primary_freqs[0] - pf)

            kernel_ind_low = max(0,f_pf_idx-kernel_size)
            kernel_ind_high = min(len(freqs)-1,f_pf_idx+kernel_size)
            dist_kernel = np.abs(freqs[kernel_ind_low] - freqs[kernel_ind_high])
            
            if dist_freq <= freq_tol*4:
                continue

            if dist_freq <= freq_tol*10:
                dex_filt=dex_filt/2
                #continue

            if pf < 0:
                min_logf = -10**(np.log10(abs(pf)) - dex_filt*extra)
                max_logf = -10**(np.log10(abs(pf)) + dex_filt)
                in_window = np.where((freqs >= max_logf) & (freqs <= min_logf))[0]
                if len(in_window) <= 2 and (len(log_power) - idx) > 1 and idx >1:
                    in_window = [idx-1,idx,idx+1]
                elif len(in_window) <= 3 and (len(log_power) - idx) > 2 and idx > 2:
                    in_window = [idx-2,idx-1,idx,idx+1,idx+2]
            
            else:
                min_logf = 10**(np.log10(abs(pf)) - dex_filt*extra)
                max_logf = 10**(np.log10(abs(pf)) + dex_filt)
                in_window = np.where((freqs >= min_logf) & (freqs <= max_logf))[0]
                if len(in_window) <= 2 and (len(log_power) - idx) > 1 and idx >1:
                    in_window = [idx-1,idx,idx+1]
                elif len(in_window) <= 3 and (len(log_power) - idx) > 2 and idx > 2:
                    in_window = [idx-2,idx-1,idx,idx+1,idx+2]

            
            
            try:
                if dist_freq <= freq_tol*8:
                    mask_filt[idx] = False
                    masked_log_power[idx] = log_power[idx]/2
                    continue
                elif dist_freq <= dist_kernel:
                    mask_filt[idx-1:idx+2] = False
                    masked_log_power[idx-1:idx+2] = log_power[idx-1:idx+2]/2
                    continue
                else:
                    mask_filt[in_window] = False
                    #masked_log_power[in_window] = log_power[in_window]
                    #masked_log_power[idx] = log_power[idx]/2
                    #masked_log_power[idx+1] = log_power[idx+1]/2
                    #masked_log_power[idx-1] = log_power[idx-1]/2
                    continue
            except Exception as e:
                #print('Line 521:',e)
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
    nep_num = 3
    if inc_filt:
        nep_num = 2
    #if abs(1/primary_freqs[0]) > abs(1/known_planet_freqs[nep_num]):
    #    long_protect = True
    #    long_inds = np.where(abs(1/freqs) > abs(1/primary_freqs[0]))[0]
    #    mask_long[long_inds] = False
        
    ind0 = np.where(freqs == 0)[0][0]
    power_temp = log_power.copy()
    
    power_temp[ind0] = power_temp[ind0]/10
    
    indmax = np.argmax(abs(power_temp))

    if abs(ind0 - indmax) <= 2:
        mask[ind0] = False
    else:
        mask[ind0] = True
    
    #og_val = log_power[ind0]

#    mean_val = np.nanmean(np.array([log_power[ind0-1],log_power[ind0+1],log_power[ind0]]))

#    if np.max(abs(log_power)) == abs(log_power[ind0]):
#        mask_long[ind0] = False
#    else:
#        mask_long[ind0] = True

    
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
    mask[normalized_log_power > log_power] = False
    mask_opp[normalized_log_power > log_power] = False
    #mask_filt[normalized_log_power > log_power] = False
    
    normalized_log_power[~mask] = log_power[~mask]
    normalized_log_power[~mask_opp] = log_power[~mask_opp]
    #normalized_log_power[~mask_short] = log_power[~mask_short]
    if long_protect:
        normalized_log_power[~mask_long] = log_power[~mask_long]

    #print(np.sum(mask_filt),len(norm_og))
    #print(len(norm_og[~mask_filt]))
    #print(len(normalized_log_power[~mask_filt]))
    #normalized_log_power = log_power
    normalized_log_power[~mask_filt] = norm_og[~mask_filt]
    if shortfilt == False:
        #print('setting back short period terms', shortfilt)
        normalized_log_power[shortperiod] = log_power[shortperiod]

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


def smooth_fft_convolution(fft_signal, freqs, primary_freqs, time, protect_radius_bins_init=3, kernel_size=15, method="gaussian", inc_filt = False, known_planet_freqs = [],freq_tol=2e-7, win=False, shortfilt = True):
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

    # Setting mask values to False causes them to be protected/preserved to original amplitudes
    mask = np.ones(N, dtype=bool)
    mask_opp = np.ones(N, dtype=bool)

    first = True
    protect_radius_bins = protect_radius_bins_init
    
    dt = time[1]-time[0]
    
    for pf in primary_freqs:
        
        dex_protect = 0.05
        #if win:
        #    dex_protect = 0.015

        idx = np.argmin(np.abs(freqs - pf))
        if first: 
            p_idx = idx
            #if inc_filt == False:
            #    dex_protect = 0.06
            #first = False
            f_pf_idx = np.argmin(abs(pf-freqs))
            dex_protect = dex_protect
        else:
            protect_radius_bins = round(protect_radius_bins_init/2)

        extra = 1

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

        mask[in_window] = False
        
        #idx_opp = np.argmin(np.abs(- pf - freqs))
        mask[in_window_opp] = False




 
    mask_float = mask.astype(float)
    mask_opp_float = mask_opp.astype(float)
    #mask_long_float = mask_long.astype(float)

    #rmasked_low_power = log_power.copy()
    # Convolve both signal and mask to normalize after convolution

    masked_log_power = log_power.copy()

    if method == "gaussian":
        smoothed_log_power = gaussian_filter1d(masked_log_power, sigma=kernel_size, mode="mirror")
        smoothed_mask = gaussian_filter1d(mask_float, sigma=kernel_size, mode="mirror")
        #smoothed_mask_filt = gaussian_filter1d(mask_filt_float, sigma=kernel_size, mode="mirror")

    else:  # fallback to uniform boxcar
        kernel = np.ones(kernel_size) / kernel_size
        smoothed_log_power = convolve1d(masked_log_power, kernel, mode="nearest")
        smoothed_mask = convolve1d(mask_float, kernel, mode="nearest")
        #smoothed_mask_filt = convolve1d(mask_filt_float, kernel, mode="nearest")

    with np.errstate(invalid='ignore', divide='ignore'):
        normalized_log_power = np.where(smoothed_mask > 1e-6 , smoothed_log_power, log_power)

    norm_og = normalized_log_power.copy()    
    
    normalized_log_power[~mask] = log_power[~mask]
    normalized_log_power[~mask_opp] = log_power[~mask_opp]
    if shortfilt == False:
        print('setting back short period terms', shortfilt)
        normalized_log_power[shortperiod] = log_power[shortperiod]

    magnitude = 10 ** (0.5 * normalized_log_power / 10) 

    normalized_log_power[normalized_log_power > log_power] = log_power[normalized_log_power > log_power]
    # Normalize power to match original
    
    orig_power = np.nansum(np.abs(fft_signal)**2)
    new_power = np.nansum(np.abs(magnitude)**2)
    scale = np.sqrt(orig_power / new_power)

    smoothed_fft = magnitude * np.exp(1j * phase) * scale
    scaling_factor = np.sqrt(1/((np.abs(smoothed_fft[f_pf_idx])**2) / (np.abs(fft_signal[f_pf_idx])**2)))

    return smoothed_fft * scaling_factor
    
def argmedian(x):
    """
    Returns the index of the median element for odd-length arrays,
    or one of the middle elements for even-length arrays.
    """
    return np.argpartition(x, len(x) // 2)[len(x) // 2]

def smooth_fft_time(fft_signal, freqs, primary_freqs, time, protect_radius_bins_init=3, kernel_size=15, method="gaussian", inc_filt = False, known_planet_freqs = [],freq_tol=2e-7, win=False, shortfilt = True):
    """
    Fast convolutional smoothing of FFT log-power spectrum, excluding primary peaks.
    """
    fft_signal = np.asarray(fft_signal)
    log_power = 10 * np.log10(np.abs(fft_signal)**2 + 1e-10)


    log_power[np.isnan(log_power)] = 1e-10

    phase = np.angle(fft_signal)

    N = len(fft_signal)

    # Setting mask values to False causes them to be protected/preserved to original amplitudes
    
    
    masked_log_power = log_power.copy()
    ind0 = np.argmin(abs(freqs))

    if masked_log_power[ind0] > np.median(masked_log_power):
        masked_log_power[ind0] = np.median(masked_log_power)
    

    for pf in primary_freqs:
        idx_p = np.argmin(abs(freqs-pf))
        masked_log_power[idx_p-2:idx_p+3] = masked_log_power[idx_p-2:idx_p+3]/5
        
    for pf in known_planet_freqs:
        idx_p = np.argmin(abs(freqs-pf))
        masked_log_power[idx_p-2:idx_p+3] = masked_log_power[idx_p-2:idx_p+3]/5
    
    if method == "gaussian":
        smoothed_log_power = gaussian_filter1d(masked_log_power, sigma=kernel_size, mode="mirror")

    else:  # fallback to uniform boxcar
        kernel = np.ones(kernel_size) / kernel_size
        smoothed_log_power = convolve1d(masked_log_power, kernel, mode="nearest")
    '''
    # Normalize result: divide smoothed signal by smoothed mask
    with np.errstate(invalid='ignore', divide='ignore'):
        #normalized_log_power = np.where(smoothed_mask_filt < 1e-6 , smoothed_log_power / smoothed_mask_filt, log_power)
        normalized_log_power = np.where(smoothed_mask > 1e-6 , smoothed_log_power, log_power)

    # Restore original log_power in protected regions
    #normalized_log_power = smoothed_log_power.copy()
    norm_og = normalized_log_power.copy()
    #'''    

    smoothed_log_power[smoothed_log_power > log_power] = log_power[smoothed_log_power > log_power]

    smoothed_log_power[ind0] = log_power[ind0]
    # Back to magnitude
    magnitude = 10 ** (0.5 * smoothed_log_power / 10) 

    # Normalize power to match original
    orig_power = np.nansum(np.abs(fft_signal)**2)
    new_power = np.nansum(np.abs(magnitude)**2)
    scale = np.sqrt(orig_power / new_power)

    smoothed_fft = magnitude * np.exp(1j * phase) 
    
    f_pf_idx = np.argmin(abs(freqs - primary_freqs[0]))
    #scaling_factor = np.sqrt(1/((np.abs(smoothed_fft[f_pf_idx])**2) / (np.abs(fft_signal[f_pf_idx])**2)))
    scaling_factor = 1

    '''
    plt.scatter(1/freqs, log_power, s=1)
    plt.scatter(1/freqs, smoothed_log_power, s=1)
    plt.xscale('symlog',linthresh=1e4, linscale=1e-2)
    plt.yscale('log')
    plt.show()
    '''
    
    return smoothed_fft * scaling_factor

def extract_proper_mode(signal, time, known_planet_freqs, freq_tol=2e-7, protect_bins=None,kernel=60, proper_freq = None, inc_filt = False, win=False, shortfilt = True, afilt=False, filt_time = False):
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
        #if inc_filt:
        #    spectrum[ind_0] = np.nanmean(np.array([ref_spec[ind_0-1],ref_spec[ind_0+1]]),dtype=np.float64)
        #    spectrum[ind_0] = 0

        lowest_per_p = np.min(abs(1/np.array(known_planet_freqs)))
        lowest_per_gs = np.min(abs(1/np.array(proper_freq)))
        #low_2g2s = 1/(2*g_arr[1]-2*s_arr[0])

        lowest_period = min(lowest_per_p,lowest_per_gs)
        #lowest_period = min(lowest_period,low_2g2s)
        
        shortperiod = np.where(1/abs(freqs) < lowest_period/4)[0]
        #if abs(1/proper_freq[0]) > abs(10*dt):
        if len(shortperiod) > 0 and shortfilt:    
            #print('filtering shortperiod line 834', shortfilt)
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
        #spectrum[nanvals] = np.nanmean(ref_spec)

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
        if filt_time:
            filt_signal = smooth_fft_time(spectrum, freqs, proper_freq, 
                                         time, protect_radius_bins_init=protect_bins, 
                                         kernel_size=kernel, method="gaussian", inc_filt=inc_filt,
                                         known_planet_freqs = known_planet_freqs, freq_tol=freq_tol,
                                         win=win, shortfilt = shortfilt)
        else:
            filt_signal = smooth_fft_convolution(spectrum, freqs, proper_freq, 
                                         time, protect_radius_bins_init=protect_bins, 
                                         kernel_size=kernel, method="gaussian", inc_filt=inc_filt,
                                         known_planet_freqs = known_planet_freqs, freq_tol=freq_tol,
                                         win=win, shortfilt = shortfilt)

        
        nan_inds = np.where(np.isnan(filt_signal))[0]
        #filt_signal[nan_inds] = np.nanmedian(filt_signal)
        
        #shortperiod = np.where(1/abs(freqs) < abs(3*dt))[0]
        #if abs(1/proper_freq[0]) > abs(10*dt):
            #filt_signal[shortperiod] = ref_spec[np.argmin(np.abs(ref_spec)**2)]
        #    filt_signal[shortperiod] = 0

        if len(shortperiod) > 0 and shortfilt == False:   
            #print('putting back shortperiod')
            filt_signal[shortperiod] = ref_spec[shortperiod]
            
        ind_0 = np.where(freqs == 0)[0][0]
        
        if afilt:
            filt_signal[ind_0] = spectrum[ind_0]
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

def handle_gs8_resonance(proper_signal, gs, gs8, freqs):
    
    obj_temp = proper_signal.copy()
    elem_temp = np.abs(obj_temp)
    ang_temp = np.angle(obj_temp)
    
    Ytemp = np.fft.rfft(elem_temp)
    prop_freq = np.argmin(abs(freqs - gs))

    indres = np.argmin(abs(freqs - (gs - gs8)))
        
    indmax = np.argmax(proper_signal[indres-1:indres+2]) + indres - 1
    if abs(indmax - prop_freq) < 2:
        print(indmax, prop_freq)
        return proper_signal
        
    scale = np.sqrt(abs(np.mean(Ytemp[indmax-1], Ytemp[indmax+1]) / Ytemp[indmax])**2)
    Ytemp[indmax] = Ytemp[indmax]*scale

    indres = np.argmin(abs( - freqs - (gs - gs8)))
        
    indmax = np.argmax(proper_signal[indres-1:indres+2]) + indres - 1
    if abs(indmax - prop_freq) < 2:
        
        print(indmax, prop_freq)
        return proper_signal
        
    scale = np.sqrt(abs(np.mean(Ytemp[indmax-1], Ytemp[indmax+1]) / Ytemp[indmax])**2)
    Ytemp[indmax] = Ytemp[indmax]*scale

    obj_temp = np.fft.irfft(Ytemp)
    proper_signal = obj_temp*np.exp(1j*ang_temp)
    print('modified sig for gs-gs8', indmax, scale, 1/freqs[indmax])
    return proper_signal

def handle_gs_gs8_resonance(proper_signal, prop, g, s, g8, s8, freqs):
    
    obj_temp = proper_signal.copy()
    elem_temp = np.abs(obj_temp)
    ang_temp = np.angle(obj_temp)
    
    Ytemp = np.fft.rfft(elem_temp)
    prop_freq = np.argmin(abs(freqs - prop))

    indres = np.argmin(abs(freqs - (g + s - g8 - s8)))
        
    indmax = np.argmax(proper_signal[indres-1:indres+2])  + indres - 1
    if(abs(indmax - prop_freq) < 2):
        print(indmax, prop_freq)
        return proper_signal
        
    scale = np.sqrt(abs(np.mean(Ytemp[indmax-1], Ytemp[indmax+1]) / Ytemp[indmax])**2)
    Ytemp[indmax] = Ytemp[indmax]*scale
        
    #proper_signal = obj_temp*np.exp(1j*ang_temp)
    
    indres = np.argmin(abs(- freqs - (g + s - g8 - s8)))
        
    indmax = np.argmax(proper_signal[indres-1:indres+2])  + indres - 1
    if(abs(indmax - prop_freq) < 2):
        print(indmax, prop_freq)
        return proper_signal
        
    scale = np.sqrt(abs(np.mean(Ytemp[indmax-1], Ytemp[indmax+1]) / Ytemp[indmax])**2)
    Ytemp[indmax] = Ytemp[indmax]*scale

    obj_temp = np.fft.irfft(Ytemp)
    proper_signal = obj_temp*np.exp(1j*ang_temp)
    print('modified sig for g+s-g8-s8', indmax, scale, 1/freqs[indmax])
    return proper_signal

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
    
    dominant_freq = np.sum(freqs[ind_low:ind_high]*(powers[ind_low:ind_high])**2)/np.sum((powers[ind_low:ind_high])**2)
    #print(dominant_freq,dominant_freq*2*np.pi*206265)
    max_idx = np.argmin(np.abs(freqs-dominant_freq))
    #if abs(len(freqs) - max_idx) <= 10:
    #    dominant_freq = np.sum(freqs[ind_low:ind_high]*(powers[ind_low:ind_high]))/np.sum((powers[ind_low:ind_high]))
    #if max_idx <= 10:
    #    dominant_freq = np.sum(freqs[ind_low:ind_high]*(powers[ind_low:ind_high]))/np.sum((powers[ind_low:ind_high]))

    #if powers[0] > powers[max_idx]:
    #    max_idx = 0
    #    dominant_freq = freqs[0]
    
    #min_logf = np.log10(abs(dominant_freq)) - window_protect_dex
    #max_logf = np.log10(abs(dominant_freq)) + window_protect_dex

    #if dominant_freq < 0:
    #    in_window = np.where((np.log10(-freqs) >= min_logf) & (np.log10(-freqs) <= max_logf))[0]
    #else:
    #    in_window = np.where((np.log10(freqs) >= min_logf) & (np.log10(freqs) <= max_logf))[0]

    '''
    plt.scatter(1/freqs, powers, s=1)
    plt.xscale('symlog', linthresh=1e3, linscale=1e-3)
    plt.yscale('log')
    plt.axvline(1/dominant_freq)
    plt.title('Dominant Freq='+ str(dominant_freq*3600*360) + ' "/yr')
    plt.show()
    '''
 
    protect_bins = 10

    return max_idx, dominant_freq, local_power_all, protect_bins


def get_planet_arrays(archive,small_planets_flag,fullfile):
    try:
        g1 = 0; g2 = 0; g3 = 0; g4 = 0; g5 = 0; g6 = 0; g7 = 0; g8 = 0
        s1 = 0; s2 = 0; s3 = 0; s4 = 0; s5 = 0; s6 = 0; s7 = 0; s8 = 0
        av_init = 0
        aj_init = 0

        #Ypq[0] = 0
        time_run = archive[-1].t
        goal_p = len(archive)
        N_og = len(archive)

        
        if small_planets_flag:
            
            if isinstance(av_init, int):
                if archive[-1].t > 0:
                    flag, av_init, ev_init, incv_init, lanv_init, aopv_init, Mv_init, t_init = read_sa_for_sbody(des = str('venus'), archivefile=fullfile,clones=0,tmax=(time_run),tmin=0,center='helio',s=archive)
                    flag, ae_init, ee_init, ince_init, lane_init, aope_init, Me_init, t_init = read_sa_for_sbody(des = str('earth'), archivefile=fullfile,clones=0,tmax=(time_run),tmin=0,center='helio',s=archive)
                    flag, am_init, em_init, incm_init, lanm_init, aopm_init, Mm_init, t_init = read_sa_for_sbody(des = str('mars'), archivefile=fullfile,clones=0,tmax=(time_run),tmin=0,center='helio',s=archive)
                else:
                    flag, av_init, ev_init, incv_init, lanv_init, aopv_init, Mv_init, t_init = read_sa_for_sbody(des = str('venus'), archivefile=fullfile,clones=0,tmin=(time_run),tmax=0,center='helio',s=archive)
                    flag, ae_init, ee_init, ince_init, lane_init, aope_init, Me_init, t_init = read_sa_for_sbody(des = str('earth'), archivefile=fullfile,clones=0,tmin=(time_run),tmax=0,center='helio',s=archive)
                    flag, am_init, em_init, incm_init, lanm_init, aopm_init, Mm_init, t_init = read_sa_for_sbody(des = str('mars'), archivefile=fullfile,clones=0,tmin=(time_run),tmax=0,center='helio',s=archive)
            
                hv = ev_init*np.sin(lanv_init+aopv_init); he = ee_init*np.sin(lane_init+aope_init); hm = em_init*np.sin(lanm_init+aopm_init)
                kv = ev_init*np.cos(lanv_init+aopv_init); ke = ee_init*np.cos(lane_init+aope_init); km = em_init*np.cos(lanm_init+aopm_init)                
                pv = np.sin(incv_init)*np.sin(lanv_init); pe = np.sin(ince_init)*np.sin(lane_init); pm = np.sin(incm_init)*np.sin(lanm_init)                
                qv = np.sin(incv_init)*np.cos(lanv_init); qe = np.sin(ince_init)*np.cos(lane_init); qm = np.sin(incm_init)*np.cos(lanm_init)

                hv = np.append(hv,np.zeros(goal_p-N_og)); he = np.append(he,np.zeros(goal_p-N_og)); hm = np.append(hm,np.zeros(goal_p-N_og))
                kv = np.append(kv,np.zeros(goal_p-N_og)); ke = np.append(ke,np.zeros(goal_p-N_og)); km = np.append(km,np.zeros(goal_p-N_og))
                pv = np.append(pv,np.zeros(goal_p-N_og)); pe = np.append(pe,np.zeros(goal_p-N_og)); pm = np.append(pm,np.zeros(goal_p-N_og))
                qv = np.append(qv,np.zeros(goal_p-N_og)); qe = np.append(qe,np.zeros(goal_p-N_og)); qm = np.append(qm,np.zeros(goal_p-N_og))
                            
                
        
        if isinstance(aj_init, int):
            if archive[-1].t > 0 :
                flag, aj_init, ej_init, incj_init, lanj_init, aopj_init, Mj_init, t_init = read_sa_for_sbody(des = str('jupiter'), archivefile=fullfile,clones=0,tmax=(time_run),tmin=0,center='helio',s=archive)
                flag, as_init, es_init, incs_init, lans_init, aops_init, Ms_init, t_init = read_sa_for_sbody(des = str('saturn'), archivefile=fullfile,clones=0,tmax=(time_run),tmin=0,center='helio',s=archive)
                flag, au_init, eu_init, incu_init, lanu_init, aopu_init, Mu_init, t_init = read_sa_for_sbody(des = str('uranus'), archivefile=fullfile,clones=0,tmax=(time_run),tmin=0,center='helio',s=archive)
                flag, an_init, en_init, incn_init, lann_init, aopn_init, Mn_init, t_init = read_sa_for_sbody(des = str('neptune'), archivefile=fullfile,clones=0,tmax=(time_run),tmin=0,center='helio',s=archive)
            else:
                flag, aj_init, ej_init, incj_init, lanj_init, aopj_init, Mj_init, t_init = read_sa_for_sbody(des = str('jupiter'), archivefile=fullfile,clones=0,tmin=(time_run),tmax=0,center='helio',s=archive)
                flag, as_init, es_init, incs_init, lans_init, aops_init, Ms_init, t_init = read_sa_for_sbody(des = str('saturn'), archivefile=fullfile,clones=0,tmin=(time_run),tmax=0,center='helio',s=archive)
                flag, au_init, eu_init, incu_init, lanu_init, aopu_init, Mu_init, t_init = read_sa_for_sbody(des = str('uranus'), archivefile=fullfile,clones=0,tmin=(time_run),tmax=0,center='helio',s=archive)
                flag, an_init, en_init, incn_init, lann_init, aopn_init, Mn_init, t_init = read_sa_for_sbody(des = str('neptune'), archivefile=fullfile,clones=0,tmin=(time_run),tmax=0,center='helio',s=archive)
            
            hj = ej_init*np.sin(lanj_init+aopj_init); hs = es_init*np.sin(lans_init+aops_init); hu = eu_init*np.sin(lanu_init+aopu_init); hn = en_init*np.sin(lann_init+aopn_init)
            kj = ej_init*np.cos(lanj_init+aopj_init); ks = es_init*np.cos(lans_init+aops_init); ku = eu_init*np.cos(lanu_init+aopu_init); kn = en_init*np.cos(lann_init+aopn_init)
            pj = np.sin(incj_init)*np.sin(lanj_init); ps = np.sin(incs_init)*np.sin(lans_init); pu = np.sin(incu_init)*np.sin(lanu_init); pn = np.sin(incn_init)*np.sin(lann_init)
            qj = np.sin(incj_init)*np.cos(lanj_init); qs = np.sin(incs_init)*np.cos(lans_init); qu = np.sin(incu_init)*np.cos(lanu_init); qn = np.sin(incn_init)*np.cos(lann_init)

            hj = np.append(hj,np.zeros(goal_p-N_og)); hs = np.append(hs,np.zeros(goal_p-N_og)); hu = np.append(hu,np.zeros(goal_p-N_og)); hn = np.append(hn,np.zeros(goal_p-N_og))
            kj = np.append(kj,np.zeros(goal_p-N_og)); ks = np.append(ks,np.zeros(goal_p-N_og)); ku = np.append(ku,np.zeros(goal_p-N_og)); kn = np.append(kn,np.zeros(goal_p-N_og))
            pj = np.append(pj,np.zeros(goal_p-N_og)); ps = np.append(ps,np.zeros(goal_p-N_og)); pu = np.append(pu,np.zeros(goal_p-N_og)); pn = np.append(pn,np.zeros(goal_p-N_og))
            qj = np.append(qj,np.zeros(goal_p-N_og)); qs = np.append(qs,np.zeros(goal_p-N_og)); qu = np.append(qu,np.zeros(goal_p-N_og)); qn = np.append(qn,np.zeros(goal_p-N_og))
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno

        error_message = "An error occurred in at line "+str(line_number)
        print(e)
        return 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

    if small_planets_flag:
        return hj,hs,hu,hn,kj,ks,ku,kn,pj,ps,pu,pn,qj,qs,qu,qn,hv,he,hm,kv,ke,km,pv,pe,pm,qv,qe,qm
    else:
        return hj,hs,hu,hn,kj,ks,ku,kn,pj,ps,pu,pn,qj,qs,qu,qn

def ind_filt(ind_list,powers,freq,dexl,div_num):
    for i in ind_list:
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
    return powers
    
def hkpq(p_elems):
    h = p_elems[1]*np.sin(p_elems[3]+p_elems[4])
    k = p_elems[1]*np.cos(p_elems[3]+p_elems[4])
    p = np.sin(p_elems[2])*np.sin(p_elems[4])
    q = np.sin(p_elems[2])*np.cos(p_elems[4])
    return h,k,p,q
    
def get_planet_freqs(t_init, planet_elems,small_planets_flag = False):
    
    try:     
        hj,kj,pj,qj = hkpq(planet_elems['jupiter'])
        hs,ks,ps,qs = hkpq(planet_elems['saturn'])
        hu,ku,pu,qu = hkpq(planet_elems['uranus'])
        hn,kn,pn,qn = hkpq(planet_elems['neptune'])
        
        if small_planets_flag:    
            hv,kv,pv,qv = hkpq(planet_elems['venus'])
            he,ke,pe,qe = hkpq(planet_elems['earth'])
            hm,km,pm,qm = hkpq(planet_elems['mars'])
            
            
        Yhkj = np.fft.fft(kj+1j*hj); Yhks = np.fft.fft(ks+1j*hs); Yhku = np.fft.fft(ku+1j*hu); Yhkn = np.fft.fft(kn+1j*hn)
        Ypqj = np.fft.fft(qj+1j*pj); Ypqs = np.fft.fft(qs+1j*ps); Ypqu = np.fft.fft(qu+1j*pu); Ypqn = np.fft.fft(qn+1j*pn)

        if small_planets_flag:
            Yhkv = np.fft.fft(kv+1j*hv); Yhke = np.fft.fft(ke+1j*he); Yhkm = np.fft.fft(km+1j*hm)
            Ypqv = np.fft.fft(qv+1j*pv); Ypqe = np.fft.fft(qe+1j*pe); Ypqm = np.fft.fft(qm+1j*pm)
            
    
        g_arr = []
        s_arr = []

        g_inds = []
        s_inds = []

        gs_dict = {}
            
        n = len(hj)
        dt = abs(t_init[1]-t_init[0])
            #if n < 10001:
                #print(n)
        freq = np.fft.fftfreq(n,d=dt)
            #freqn = np.fft.rfftfreq(len(aj_init),d=dt)

        half = int(len(freq)/2)

        good_freq_inds = np.where(abs(1/freq) <= np.max(abs(t_init)))[0]
        bad_freq_inds = np.where(abs(1/freq) > np.max(abs(t_init)))[0]

        #print(Yhkj.shape)
        #print(1/freq)
        #print(freq.shape)
        #print(t_init[-1])
        #print(len(good_freq_inds))
        g5 = freq[np.argmax(np.abs(Yhkj[good_freq_inds])**2)]

        dex = 0.02
        dexl = dex/2
        div_num = 500

        #Jupiter g
        powers = np.abs(Yhkj)**2
        powers[bad_freq_inds] = 0
        max_idx, g5, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
        g_arr.append(g5)
        g_inds.append(max_idx)
        gs_dict['g5'] = g5

        #Saturn g
        powers = np.abs(Yhks)**2
        powers[bad_freq_inds] = 0
        powers = ind_filt(g_inds,powers,freq,dexl,div_num)
        max_idx, g6, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
        g_arr.append(g6)
        g_inds.append(max_idx)
        gs_dict['g6'] = g6

        #Uranus g
        powers = np.abs(Yhku)**2
        powers[0] = 0
        powers = ind_filt(g_inds,powers,freq,dexl,div_num)
        max_idx, g7, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
        g_arr.append(g7)
        g_inds.append(max_idx)
        gs_dict['g7'] = g7

        #Neptune g
        powers = np.abs(Yhkn)**2
        #powers[bad_freq_inds] = 0
        powers[0] = 0
        powers = ind_filt(g_inds,powers,freq,dexl,div_num)
        max_idx, g8, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
        g_arr.append(g8)
        g_inds.append(max_idx)
        gs_dict['g8'] = g8

        #Saturn s
        powers = np.abs(Ypqs)**2
        powers[bad_freq_inds] = 0
        max_idx, s6, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
        s_arr.append(s6)
        s_inds.append(max_idx)
        gs_dict['s6'] = s6

        #Uranus s
        powers = np.abs(Ypqu)**2
        powers[bad_freq_inds] = 0
        powers = ind_filt(s_inds,powers,freq,dexl,div_num)
        max_idx, s7, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
        s_arr.append(s7)
        s_inds.append(max_idx)
        gs_dict['s7'] = s7

        #Neptune s
        powers = np.abs(Ypqn)**2
        powers[bad_freq_inds] = 0
        powers = ind_filt(s_inds,powers,freq,dexl,div_num)
        max_idx, s8, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
        s_arr.append(s8)
        s_inds.append(max_idx)
        gs_dict['s8'] = s8
            
        if small_planets_flag:
            #Venus g
            powers = np.abs(Yhkv)**2
            powers[bad_freq_inds] = 0
            powers = ind_filt(g_inds,powers,freq,dexl,div_num)
                
            max_idx, g2, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
            g_arr.append(g2)
            g_inds.append(max_idx)
            gs_dict['g2'] = g2
                
            #Earth g
            powers = np.abs(Yhke)**2
            powers[bad_freq_inds] = 0
            powers = ind_filt(g_inds,powers,freq,dexl,div_num) 
            max_idx, g3, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
            g_arr.append(g3)
            g_inds.append(max_idx)
            gs_dict['g3'] = g3


            #Mars g
            powers = np.abs(Yhkm)**2
            powers[bad_freq_inds] = 0
            powers = ind_filt(g_inds,powers,freq,dexl,div_num)    
            max_idx, g4, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
            g_arr.append(g4)
            g_inds.append(max_idx)
            gs_dict['g4'] = g4

                
            #Earth s
            powers = np.abs(Ypqe)**2
            powers[bad_freq_inds] = 0
            powers = ind_filt(s_inds,powers,freq,dexl,div_num)
            max_idx, s3, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
            s_arr.append(s3)
            s_inds.append(max_idx)
            gs_dict['s3'] = s3

            
            #Mars s
            powers = np.abs(Ypqm)**2
            powers[bad_freq_inds] = 0
            powers = ind_filt(s_inds,powers,freq,dexl,div_num)
            max_idx, s4, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
            s_arr.append(s4)
            s_inds.append(max_idx)
            gs_dict['s4'] = s4

            
            #Venus s
            powers = np.abs(Ypqv)**2
            powers[bad_freq_inds] = 0
            powers = ind_filt(s_inds,powers,freq,dexl,div_num)
            max_idx, s2, local_power_all, protect_bins = find_local_max_windowed(freq, powers, window_half_dex=dex, window_protect_dex=0.15)
            s_arr.append(s2)
            s_inds.append(max_idx)
            gs_dict['s2'] = s2

        g_ex = np.array([g5-g6,g5-g7,g5-g8,g6-g7,g6-g8,g7-g8],dtype=np.float64)
        s_ex = np.array([s6-s7,s6-s8,s7-s8],dtype=np.float64)

            
    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno

        error_message = "An error occurred in at line "+str(line_number)
        print(error_message)
        print(err)
        print('Gathering planetary frequencies failed')

        return 0,0,0,0,0
    return g_arr,g_inds,s_arr,s_inds, gs_dict

def power_filt(power, p_arr, freq_s, small_planets_flag = False, dist = 1, e_or_I = True):
    j=0
            
    if small_planets_flag:
        num_p = 8
    else:
        num_p = 5

    if e_or_I:
        for i in p_arr[:num_p]:
            ind = np.argmin(abs(freq_s-i))
            if j==0 or j==1:
                mult = 10
            else:
                mult = 2
            if small_planets_flag == False and j == 3:
                mult = 2
                    
            power[ind-dist:ind+dist+1] = power[ind-dist:ind+dist+1]/mult
            j += 1
    else:
        j=0
        for i in s_arr[:num_p-1]:
            if j==0 or j==1:
                mult = 100
            else:
                    #mult = 20
                mult = 20
            if small_planets_flag == False and j == 2:
                mult = 100
            ind = np.argmin(abs(freq_s-i))
            power[ind-dist:ind+dist+1] = power[ind-dist:ind+dist+1]/mult
            j += 1

    short_period = (1/abs(np.array(p_arr[1]*4)))
    short_ind = np.where(1/abs(freq_s) < short_period/4)[0]
    power[short_ind] = power[short_ind]/5

    return power


def compute_prop(a_init,e_init,inc_init,aop_init,lan_init,t_init,g_arr,s_arr,gs_dict,small_planets_flag,windows=5,debug=False,objname='', rms = True, shortfilt=True, output_arrays = False):
    try:
        p_init = np.sin(inc_init)*np.sin(lan_init)
        q_init = np.sin(inc_init)*np.cos(lan_init)
        h_init = (e_init)*np.sin(lan_init+aop_init)
        k_init = (e_init)*np.cos(lan_init+aop_init)

        if output_arrays:
            a_wins = []
            hk_wins = []
            pq_wins = []
            t_wins = []
            
        
        hk_arr = k_init+1j*h_init
        pq_arr = q_init+1j*p_init

        n = len(hk_arr)
        dt = abs(t_init[1] - t_init[0])
            #if n < 10001:
                #print(n)
        freq = np.fft.fftfreq(n,d=dt)
        
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
            
        Yhk = np.fft.fft(hk_arr_s)
        Ypq = np.fft.fft(pq_arr_s)
            #freqn = np.fft.rfftfreq(len(aj_init),d=dt)

        half = int(len(freq)/2)

        time_mag = abs(max(t_init) - min(t_init))
        good_freq_inds = np.where(abs(1/freq) <= time_mag)[0]
        bad_freq_inds = np.where(abs(1/freq) > time_mag)[0]

        try:
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

            window_dex = 0.15  
            n_bins = len(power)
            local_power = np.zeros(n_bins)
            
            window_half_dex = window_dex / 2
            n_bins = len(power)

            power = power_filt(power, g_arr, freq_s, small_planets_flag, dist)
            
            g_idx, g, local_power_all, protect_g_bins = find_local_max_windowed(freq_s, power, window_half_dex=0.05, window_protect_dex=0.15)
            #print('running Yhk dom search')

            #windows = round(n_bins/100*window_dex)

            if abs(1/g) > 5e5:
                protect_g_bins = protect_g_bins*5
            
           
            Ypq_s = np.fft.fft(pq_arr_s)
            power = np.abs(Ypq_s)**2
            power[abs(1/freq_s) > time_mag/2] = 0

            power = power_filt(power, s_arr, freq_s, small_planets_flag, dist)
            
            s_idx, s, local_power_all, protect_s_bins = find_local_max_windowed(freq_s, power, window_half_dex=0.02, window_protect_dex=0.15)
            if abs(1/s) > 5e5:
                protect_s_bins = protect_s_bins*2
            

            #FROM ORBFIT INL VARIABLE IN SELRE9.90, added 2*g6-2*s6 as the shortest possible 4th order 
            #gs_freqs = [2*g6-g5,2*g6-g7,3*g6-2*g5,-g5+g6+g7,g5+g6-g7,-g5+2*g6-s6-s7,g5+s7-s6,2*g5-g6,2*g6-2*s6]

            '''
            #Same as above, but with secular frequencies included in secular frequency map
            gs_freqs = [2*gs_dict['g6']-gs_dict['g5'],
                        2*gs_dict['g6']-gs_dict['g7'],
                        3*gs_dict['g6']-2*gs_dict['g5'],
                        -gs_dict['g5']+gs_dict['g6']+gs_dict['g7'],
                        gs_dict['g5']+gs_dict['g6']-gs_dict['g7'],
                        -gs_dict['g5']+2*gs_dict['g6']-gs_dict['s6']-gs_dict['s7'],
                        gs_dict['g5']+gs_dict['s7']-gs_dict['s6'],
                        2*gs_dict['g5']-gs_dict['g6'],
                        2*gs_dict['g6']-2*gs_dict['s6'],
                        gs_dict['s6']+gs_dict['g5']-gs_dict['g6'],
                        2*gs_dict['g6']+gs_dict['s6'],
                        3*gs_dict['g6']+gs_dict['s6'],
                        gs_dict['g5']+gs_dict['s6'],
                        gs_dict['g6']+gs_dict['s6']]

            for i in gs_freqs:
                g_arr.append(i)
                s_arr.append(i)

            g_arr.append(gs_dict['g8']+gs_dict['s8'])
            s_arr.append(gs_dict['g8']+gs_dict['s8'])
            '''
    
            protect_g = np.array([g])
            protect_s = np.array([s])
        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            line_number = exc_tb.tb_lineno

            error_message = "An error occurred in at line "+str(line_number)
    
        # Print the error message
            print(error_message)
            print(error)
            return [False] + list(np.zeros(20))

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
        return [False] + list(np.zeros(20))

    protect_hk=0
    protect_pq=0
    dt = np.abs(t_init[1]-t_init[0]) 
    freqs_a = np.fft.rfftfreq(len(hk_arr),dt)
    #print('g_arr',1/np.array(g_arr))
    #print('s_arr',1/np.array(s_arr))
    dex_protect = 0.04
    freq_low_g = 10**(np.log10(abs(protect_g[0]))-dex_protect)
    #print(1/freq_low_g,1/protect_g[0])
    ind_low_g = np.argmin(abs(freq-abs(freq_low_g)))
    ind = np.argmin(abs(freq-abs(protect_g[0])))
    kernel_g = max(2*round(abs(ind-ind_low_g)),round(len(freq)/2500))
    kernel_g = max(2*round(abs(ind-ind_low_g)),4)
    
    if kernel_g > round(len(freq)/25): # PReviously / 100
        kernel_g = round(len(freq)/25)
    
    freq_low_s = 10**(np.log10(abs(protect_s[0]))-dex_protect)
    ind_low_s = np.argmin(abs(freq-abs(freq_low_s)))
    ind = np.argmin(abs(freq-abs(protect_s[0])))

    
    kernel_s = max(2*round(abs(ind-ind_low_s)),round(len(freq)/2500))
    kernel_s = max(2*round(abs(ind-ind_low_s)),4)
    if kernel_s > round(len(freq)/25): #Previously /100
        kernel_s = round(len(freq)/25)

                          
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
        if abs(dt) < 2000:
            per_shave = 0.025
        else:
            per_shave = 0
        #print('dt',dt,'tol',tol, 'N', len(hk_arr),'protect_g',protect_g_bins,'protect_s',protect_s_bins,'kernel_g',kernel_g,'kernel_s',kernel_s)

        
        protect_g_bins = min(int(kernel_g/4), protect_g_bins)
        protect_s_bins = min(int(kernel_s/4), protect_s_bins)
        protect_g_bins = max(protect_g_bins, 2)
        protect_s_bins = max(protect_s_bins, 2)
    
        #hk_new, hk_freq, protect_hk, hk_signal = extract_proper_mode(hk_arr, t_init, g_arr, freq_tol=tol, kernel=kernel, proper_freq=g)
        #pq_new, pq_freq, protect_pq, pq_signal = extract_proper_mode(pq_arr, t_init, s_arr, freq_tol=tol, kernel=kernel, proper_freq=s, inc_filt = True)
        hk_old, hk_freq, protect_hk, hk_signal = extract_proper_mode(hk_arr, t_init, g_arr, freq_tol=tol, kernel=kernel_g, protect_bins = protect_g_bins, proper_freq=protect_g, shortfilt=shortfilt)
        pq_old, pq_freq, protect_pq, pq_signal = extract_proper_mode(pq_arr, t_init, s_arr, freq_tol=tol, kernel=kernel_s, proper_freq=protect_s, protect_bins = protect_s_bins, shortfilt=shortfilt)

        
        ee_old = np.abs(hk_old)
        II_old = np.abs(pq_old)

        ee0 = np.fft.fft(ee_old)[0]
        II0 = np.fft.fft(II_old)[0]
        
        ee_ang = np.angle(hk_old)
        II_ang = np.angle(pq_old)
        
        ee_new, ee_freq, protect_ee, ee_signal = extract_proper_mode(ee_old, t_init, g_arr, freq_tol=tol, kernel=kernel_g*2, protect_bins = 0, proper_freq=np.append(protect_g[0],[0]), shortfilt=False, filt_time=True)
        II_new, II_freq, protect_II, II_signal = extract_proper_mode(II_old, t_init, s_arr, freq_tol=tol, kernel=kernel_s*2, proper_freq=np.append(protect_s[0],[0]), protect_bins = 0, shortfilt=False, filt_time=True)
        #ee_new, ee_freq, protect_ee, ee_signal = extract_proper_mode(ee_old, t_init, g_arr[:4], freq_tol=tol, kernel=kernel_g*10, protect_bins = max(1,int(protect_g_bins/2)), proper_freq=[0,1/abs(np.max(t_init))], shortfilt=False)
        #II_new, II_freq, protect_II, II_signal = extract_proper_mode(II_old, t_init, s_arr[:3], freq_tol=tol, kernel=kernel_s*10, proper_freq=[0,1/abs(np.max(t_init))], protect_bins = max(1,int(protect_s_bins/2)), shortfilt=False)

        Yee = np.fft.fft(ee_new)
        YII = np.fft.fft(II_new)
        Yee[0] = ee0
        YII[0] = II0
        ee_new = np.fft.ifft(Yee)
        II_new = np.fft.ifft(YII)


        #hk_new = ee_new*np.exp(1j*ee_ang)
        #pq_new = II_new*np.exp(1j*II_ang)
        
        hk_new = ee_new*np.cos(ee_ang) + 1j*ee_new*np.sin(ee_ang)
        pq_new = II_new*np.cos(II_ang) + 1j*II_new*np.sin(II_ang)

        max_idx_g, g_fin, local_power_all, protect_bins = find_local_max_windowed(freq, np.abs(np.fft.fft(hk_new))**2, window_half_dex=0.02, window_protect_dex=0.15)
        max_idx_s, s_fin, local_power_all, protect_bins = find_local_max_windowed(freq, np.abs(np.fft.fft(pq_new))**2, window_half_dex=0.02, window_protect_dex=0.15)

        #g_arr[0] = g_fin
        #s_arr[0] = s_fin

        g = g_fin
        s = s_fin


        pomega_n = np.angle(hk_new)
        Omega_n = np.angle(pq_new)
        omega_n = pomega_n - Omega_n

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

        #Ya = np.fft.rfft(a_init)a_filt
        #short_inds = np.where(abs(1/freqs_a) <= dt*3)[0]
        #Ya[short_inds] = 0
        #a_new = np.fft.irfft(Ya)
        
        a_filt, a_freq, protect_a, a_signal = extract_proper_mode(a_init, t_init, g_arr, freq_tol=tol, kernel=40, protect_bins = 10, proper_freq=[np.max(abs(t_init))], shortfilt=shortfilt, afilt=True)
        
        #pes = np.array([np.nanmean(a_new[int(per_shave*N):int((1-per_shave)*N)]),pe_e,pe_i])
        pes = np.array([np.nanmean(a_filt),pe_e,pe_i, g, s],dtype=np.float64)
    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno

        error_message = "An error occurred in at line "+str(line_number)
    
        # Print the error message
        print(error_message)
        print(error)
        pes = np.array([10,10,10],dtype=np.float64)
        return [False] + list(np.zeros(20))

    error_list = np.zeros((windows,5))
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

        N_window = Nn
        
        goal = len(hk_arr)

        
        protecte_hk = max(round(protect_hk/mult),3)
        protecte_pq = max(round(protect_pq/mult),3)

        kernele_g = kernel_g
        kernele_s = kernel_s
        tole = tol/mult
        #tole = tol
        try:        
            freqe = np.fft.fftfreq(len(hk_in), dt)
            Yhk_e = np.fft.fft(hk_in)
            power_e = np.abs(Yhk_e)**2
            power_e[abs(1/freqe) > max(abs(t_init))/2] = 0
            
            Ypq_I = np.fft.fft(pq_in)
            power_I = np.abs(Ypq_I)**2
            power_I[abs(1/freqe) > max(abs(t_init))/2] = 0

            power_e = power_filt(power_e, g_arr, freqe, small_planets_flag, dist, e_or_I = True)
            power_I = power_filt(power_I, s_arr, freqe, small_planets_flag, dist, e_or_I = False)

            g_idx_e, g_e, local_power_all_e, protect_g_bins_e = find_local_max_windowed(freqe, power_e, window_half_dex=0.05, window_protect_dex=0.15)
            s_idx_e, s_e, local_power_all_e, protect_s_bins_e = find_local_max_windowed(freqe, power_I, window_half_dex=0.05, window_protect_dex=0.15)

            hk_freqe = np.array([g_e, 2*g_e, 3*g_e])
            pq_freqe = np.array([s_e, 2*s_e, 3*s_e])

            protecte_hk = min(int(kernele_g/4), protecte_hk)
            protecte_pq = min(int(kernele_s/4), protecte_pq)
            protecte_hk = max(protecte_hk, 1)
            protecte_pq = max(protecte_pq, 1)
            
            hk_newe, hk_freqe, protect_hk_waste, hk_waste = extract_proper_mode(hk_in, t_input, g_arr, freq_tol=tol, protect_bins = protecte_hk, kernel=kernele_g, proper_freq=hk_freqe, win=True)
            pq_newe, pq_freqe, protect_pq_waste, hk_waste = extract_proper_mode(pq_in, t_input, s_arr, freq_tol=tol, protect_bins = protecte_pq, kernel=kernele_s, proper_freq=pq_freqe,  win=True)  
            #pq_newe, pq_freqe, protect_pq_waste, hk_waste = extract_proper_mode(pq_in, t_input, s_arr, freq_tol=tol, protect_bins = protecte_pq, kernel=kernele_s, proper_freq=pq_freq,  inc_filt = True, win=True)  
        except Exception as e:
            error_list[j][0] = 10
            error_list[j][1] = 10
            error_list[j][2] = 10
            continue


        ee_olde = np.abs(hk_newe)
        II_olde = np.abs(pq_newe)

        ee0 = np.fft.fft(ee_olde)[0]
        II0 = np.fft.fft(II_olde)[0]
        
        ee_ange = np.angle(hk_newe)
        II_ange = np.angle(pq_newe)
        ee_newe, ee_freqe, protect_eee, ee_signale = extract_proper_mode(ee_olde, t_input, g_arr, freq_tol=tol, kernel=kernele_g*2, protect_bins = 0, shortfilt=shortfilt, proper_freq=np.append(hk_freq[0],[0]), filt_time=True)
        II_newe, II_freqe, protect_IIe, II_signale = extract_proper_mode(II_olde, t_input, s_arr, freq_tol=tol, kernel=kernele_s*2, protect_bins = 0, shortfilt=shortfilt, proper_freq=np.append(pq_freq[0],[0]), filt_time=True)
        #ee_newe, ee_freqe, protect_eee, ee_signale = extract_proper_mode(ee_olde, t_input, g_arr, freq_tol=tol, kernel=kernele_g*10, protect_bins = max(1,int(protecte_hk/2)), shortfilt=False, proper_freq=[0,1/abs(np.max(t_input))])
        #II_newe, II_freqe, protect_IIe, II_signale = extract_proper_mode(II_olde, t_input, s_arr, freq_tol=tol, kernel=kernele_s*10, protect_bins = max(1,int(protecte_pq/2)), shortfilt=False, proper_freq=[0,1/abs(np.max(t_input))])

        
        Yeee = np.fft.fft(ee_newe)
        YIIe = np.fft.fft(II_newe)
        Yeee[0] = ee0
        YIIe[0] = II0
        ee_newe = np.fft.ifft(Yeee)
        II_newe = np.fft.ifft(YIIe)

        #hk_newe = ee_newe*np.exp(1j*ee_ange)
        #pq_newe = II_newe*np.exp(1j*II_ange)

        hk_newe = ee_newe*np.cos(ee_ange) + 1j*ee_newe*np.sin(ee_ange)
        pq_newe = II_newe*np.cos(II_ange) + 1j*II_newe*np.sin(II_ange)

        a_filtn, a_freqn, protect_an, a_signaln = extract_proper_mode(a_input, t_input, g_arr, freq_tol=tol, kernel=40, protect_bins = 10, proper_freq=[np.max(abs(t_init))], shortfilt=shortfilt, afilt=True)
        
        #error_list[j][0] = np.nanmean(a_win[int(per_shave*Nn):int((1-per_shave)*Nn)])
        error_list[j][0] = np.nanmean(a_filtn)


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
        error_list[j][3] = g_e
        error_list[j][4] = s_e

        if output_arrays:
            a_wins.append(a_filtn)
            hk_wins.append(hk_newe)
            pq_wins.append(pq_newe)
            t_wins.append(t_input)

    if rms:
        rms_val = np.sqrt(np.nanmean((np.array(error_list,dtype=np.float64)-np.array(pes,dtype=np.float64))**2,axis=0))
    else:
        rms_val = np.nanstd(error_list,axis=0)

    #Identfy the longest period frequency that could appear in the simulation. This helps determine whether the hcm metric indicates 
    #long-term periodicity or instability. 
    longest = np.array([abs(g8+s8),abs(g8-s8),abs(g+s),abs(g-s),abs(g+s-g8-s8),abs(g-s-g8+s8),abs(s-g-s8+g8)])
    names = ['g8+s8','g8-s8','g+s','g-s','g+s-g8-s8','g-s-g8+s8','s-g-s8+g8']
    longest_ind = np.argmax(longest)

    sec_res = 'Resonant in: None'

    e_new = np.abs(hk_new)
    sinI_new = np.abs(pq_new)
    varpi_new = np.angle(hk_new) 
    O_new = np.angle(pq_new) 

    v_O = (varpi_new + O_new) % (2*np.pi)

    v_O_hist,edges = np.histogram(v_O, bins=np.linspace(0,360,61)/180*np.pi)
    varpi_hist,edges = np.histogram(varpi_new, bins=np.linspace(-180,180,61)/180*np.pi)
    O_hist,edges = np.histogram(O_new, bins=np.linspace(-180,180,61)/180*np.pi)

    v_O_ent = abs(-np.sum(v_O_hist/len(hk_new)*np.log(v_O_hist/len(hk_new)+1e-12))/np.log(60))
    varpi_ent = abs(-np.sum(varpi_hist/len(hk_new)*np.log(varpi_hist/len(hk_new)+1e-12))/np.log(60))
    O_ent = abs(-np.sum(O_hist/len(hk_new)*np.log(O_hist/len(hk_new)+1e-12))/np.log(60))
    

    eP = np.abs(np.fft.fft(e_new))**2
    sinIP = np.abs(np.fft.fft(sinI_new))**2

    sec_res_e = eP[0]/np.sum(eP)
    sec_res_I = sinIP[0]/np.sum(sinIP)

    sec_res_e = np.sqrt(np.sum(eP[1:]))/len(e_new)
    sec_res_I = np.sqrt(np.sum(sinIP[1:]))/len(e_new)
    e0 = np.sqrt(eP[0])/len(e_new)
    I0 = np.sqrt(sinIP[0])/len(e_new)


    rese = False
    resI = False

    if sec_res_e > 0.01 or sec_res_e/e0 > 0.1:
        rese = True
    if sec_res_I > 0.01  or sec_res_I/I0 > 0.1:
        resI = True

    angle_sec_res = 'No angle'
    librate_angle = -10
    angle_ent = 1
    if v_O_ent < 0.95:
        angle_sec_res = 'Phi'
        #librate_angle = np.median(v_O)
        librate_angle = circmean(v_O, low=0, high = 2*np.pi)
        angle_ent = v_O_ent
    elif varpi_ent < 0.95:
        angle_sec_res = 'Varpi'
        #librate_angle = np.median(varpi_new)
        librate_angle = circmean(varpi_new, low=-np.pi, high = np.pi)
        angle_ent = varpi_ent
    elif O_ent < 0.95:
        angle_sec_res = 'Omega'
        #librate_angle = np.median(O_new)
        librate_angle = circmean(O_new, low=-np.pi, high = np.pi)
        angle_ent = O_ent
    
    e_sort = np.sort(e_new[int(per_shave*N):int((1-per_shave)*N)])
    I_sort = np.sort(sinI_new[int(per_shave*N):int((1-per_shave)*N)])

    e_max = np.mean(e_sort[:5])
    e_min = np.mean(e_sort[-5:])
    I_max = np.mean(I_sort[:5])
    I_min = np.mean(I_sort[-5:])

    e_amp = abs(e_max-e_min)/2
    I_amp = abs(I_max-I_min)/2
    
    e_osc_amp = abs(np.max(np.abs(hk_arr))-np.min(np.abs(hk_arr)))/2
    I_osc_amp = abs(np.max(np.abs(pq_arr))-np.min(np.abs(pq_arr)))/2
    
    maxvals = np.max(np.array(error_list,dtype=np.float64)-np.array(pes,dtype=np.float64),axis=0)
    if debug == True:
        #rms = np.sqrt(np.nanmean((np.array(error_list)-np.array(pes))**2,axis=0))
        #rms = np.nanstd(error_list,axis=0)
        return pes,error_list,rms_val,g_arr,s_arr, hk_arr[:N_og], pq_arr[:N_og], hk_new[:N_og], pq_new[:N_og], hk_freq, pq_freq, hk_signal, pq_signal, rese, resI, sec_res_e, sec_res_I, e_osc_amp, I_osc_amp, e_amp, I_amp, angle_sec_res, librate_angle

    phifrac = (g+s)/(gs_dict['g8'] + gs_dict['s8'])

    if output_arrays:
        return True, pes, rms_val, error_list, omega_n, Omega_n, maxvals, g, s, rese, resI, sec_res_e, sec_res_I, e_osc_amp, I_osc_amp, e_amp, I_amp, angle_sec_res, librate_angle, angle_ent, phifrac, hk_new, pq_new, a_filt, hk_wins, pq_wins, a_wins, t_wins, hk_arr, pq_arr
    
    return True, pes, rms_val, error_list, omega_n, Omega_n, maxvals, g, s, rese, resI, sec_res_e, sec_res_I, e_osc_amp, I_osc_amp, e_amp, I_amp, angle_sec_res, librate_angle, angle_ent, phifrac


def read_archive_for_pe(des, clones=3, datadir=None,archivefile=None, logfile=None, object_type= None):
    if(archivefile==None):
        file = archive_file_name(des=des)
    else:
        file = archivefile
    if(datadir):
        file = datadir + '/' + file

    try:
        sim = rebound.Simulationarchive(file)
    except:
        print(file)
        print('failed to read archivefile')
    if object_type == None:
        try:
            testval = sim[0].particles['venus']
            testval = sim[0].particles['earth']
            testval = sim[0].particles['mars']
            vem = True
            planets = ['venus','earth','mars','jupiter','saturn','uranus','neptune']
        except:
            vem = False
            planets = ['jupiter','saturn','uranus','neptune']
    else:
        if object_type == 'tno':
            vem = False
            planets = ['jupiter','saturn','uranus','neptune']
        elif object_type == 'asteroid':
            vem = True
            planets = ['venus','earth','mars','jupiter','saturn','uranus','neptune']
        else:
            print('small_body object does not have a valid object_type. Checking planets contained in simulation by hash.')
            try:
                testval = sim[0].particles['venus']
                testval = sim[0].particles['earth']
                testval = sim[0].particles['mars']
                print('simulation includes venus, earth, and mars. Will filter out these inner planets')
                vem = True
                planets = ['venus','earth','mars','jupiter','saturn','uranus','neptune']
            except:
                print('simulation does not include one or all of venus, earth, and mars. Will only consider the outer planets in proper element filtering.')
                vem = False
                planets = ['jupiter','saturn','uranus','neptune']


    j_elems = np.zeros((5, len(sim)))
    s_elems = np.zeros((5, len(sim)))
    u_elems = np.zeros((5, len(sim)))
    n_elems = np.zeros((5, len(sim)))

    if vem:
        v_elems = np.zeros((5, len(sim)))
        e_elems = np.zeros((5, len(sim)))
        m_elems = np.zeros((5, len(sim)))
    
    sb_elems = np.zeros((6, len(sim)))

    clone_elems = np.zeros((clones, 5, len(sim)))

    #for i in range(0,len(sim),10):


    for i in range(len(sim)):
        #if i%20000 == 0:
        #    print('Reading snapshot:', i, sim[i].t)

        #if int(sim[i].t) % 20000 != 0:
        #    continue
        s = sim[i]
        s.move_to_com()
        particles = s.particles
        
        com = s.com()
        #orbs = s.orbits()     

        try:
            sb_idx = particles[des].index
        except:
            continue
        #o = orbs[sb_idx]
        o = particles[des].orbit(com)

        sb_elems[0,i] = s.t
        sb_elems[1,i] = o.a
        sb_elems[2,i] = o.e
        sb_elems[3,i] = o.inc
        sb_elems[4,i] = o.omega
        sb_elems[5,i] = o.Omega
        
        for j in range(clones):
            try:
                c_idx = particles[des+'_'+str(j+1)].index
            except:
                continue
            #clone = orbs[c_idx]
            c_name = des+'_'+str(j+1)
            clone = sim[i].particles[c_name].orbit(com)
            clone_elems[j,0,i] = clone.a
            clone_elems[j,1,i] = clone.e
            clone_elems[j,2,i] = clone.inc
            clone_elems[j,3,i] = clone.omega
            clone_elems[j,4,i] = clone.Omega
            
            
        
        '''
        for j in range(clones):

            c_name = str(des)+'_'+str(j+1)

            #clone = orbs[j+c_idx1]
            clone = sim[i].particles[c_name].orbit(primary=sim[i].particles['sun'])
            clone_elems[j,0,i] = clone.a
            clone_elems[j,1,i] = clone.e
            clone_elems[j,2,i] = clone.inc
            clone_elems[j,3,i] = clone.omega
            clone_elems[j,4,i] = clone.Omega
        '''

        #'''
        jorb = particles['jupiter'].orbit(com)
        sorb = particles['saturn'].orbit(com)
        uorb = particles['uranus'].orbit(com)
        norb = particles['neptune'].orbit(com)
        
        j_elems[0,i] = jorb.a
        j_elems[1,i] = jorb.e
        j_elems[2,i] = jorb.inc
        j_elems[3,i] = jorb.omega
        j_elems[4,i] = jorb.Omega
        
        s_elems[0,i] = sorb.a
        s_elems[1,i] = sorb.e
        s_elems[2,i] = sorb.inc
        s_elems[3,i] = sorb.omega
        s_elems[4,i] = sorb.Omega
        
        u_elems[0,i] = uorb.a
        u_elems[1,i] = uorb.e
        u_elems[2,i] = uorb.inc
        u_elems[3,i] = uorb.omega
        u_elems[4,i] = uorb.Omega
        
        n_elems[0,i] = norb.a
        n_elems[1,i] = norb.e
        n_elems[2,i] = norb.inc
        n_elems[3,i] = norb.omega
        n_elems[4,i] = norb.Omega


        if vem:
            vorb = particles['venus'].orbit(com)
            eorb = particles['earth'].orbit(com)
            morb = particles['mars'].orbit(com)
            
            v_elems[0,i] = vorb.a
            v_elems[1,i] = vorb.e
            v_elems[2,i] = vorb.inc
            v_elems[3,i] = vorb.omega
            v_elems[4,i] = vorb.Omega
            
            e_elems[0,i] = eorb.a
            e_elems[1,i] = eorb.e
            e_elems[2,i] = eorb.inc
            e_elems[3,i] = eorb.omega
            e_elems[4,i] = eorb.Omega
            
            m_elems[0,i] = morb.a
            m_elems[1,i] = morb.e
            m_elems[2,i] = morb.inc
            m_elems[3,i] = morb.omega
            m_elems[4,i] = morb.Omega

        #'''
        '''

        planet_idx = []
        for j in planets:
            planet_idx.append(particles[j].index-1)
            
        if vem:
            j_idx = 3
        else:
            j_idx = 0
            
        j_elems[0,i] = orbs[planet_idx[j_idx]].a
        j_elems[1,i] = orbs[planet_idx[j_idx]].e
        j_elems[2,i] = orbs[planet_idx[j_idx]].inc
        j_elems[3,i] = orbs[planet_idx[j_idx]].omega
        j_elems[4,i] = orbs[planet_idx[j_idx]].Omega
        
        s_elems[0,i] = orbs[planet_idx[j_idx+1]].a
        s_elems[1,i] = orbs[planet_idx[j_idx+1]].e
        s_elems[2,i] = orbs[planet_idx[j_idx+1]].inc
        s_elems[3,i] = orbs[planet_idx[j_idx+1]].omega
        s_elems[4,i] = orbs[planet_idx[j_idx+1]].Omega
        
        u_elems[0,i] = orbs[planet_idx[j_idx+2]].a
        u_elems[1,i] = orbs[planet_idx[j_idx+2]].e
        u_elems[2,i] = orbs[planet_idx[j_idx+2]].inc
        u_elems[3,i] = orbs[planet_idx[j_idx+2]].omega
        u_elems[4,i] = orbs[planet_idx[j_idx+2]].Omega
        
        n_elems[0,i] = orbs[planet_idx[j_idx+3]].a
        n_elems[1,i] = orbs[planet_idx[j_idx+3]].e
        n_elems[2,i] = orbs[planet_idx[j_idx+3]].inc
        n_elems[3,i] = orbs[planet_idx[j_idx+3]].omega
        n_elems[4,i] = orbs[planet_idx[j_idx+3]].Omega

        if vem:
            v_elems[0,i] = orbs[planet_idx[0]].a
            v_elems[1,i] = orbs[planet_idx[0]].e
            v_elems[2,i] = orbs[planet_idx[0]].inc
            v_elems[3,i] = orbs[planet_idx[0]].omega
            v_elems[4,i] = orbs[planet_idx[0]].Omega
            
            e_elems[0,i] = orbs[planet_idx[1]].a
            e_elems[1,i] = orbs[planet_idx[1]].e
            e_elems[2,i] = orbs[planet_idx[1]].inc
            e_elems[3,i] = orbs[planet_idx[1]].omega
            e_elems[4,i] = orbs[planet_idx[1]].Omega
        
            m_elems[0,i] = orbs[planet_idx[2]].a
            m_elems[1,i] = orbs[planet_idx[2]].e
            m_elems[2,i] = orbs[planet_idx[2]].inc
            m_elems[3,i] = orbs[planet_idx[2]].omega
            m_elems[4,i] = orbs[planet_idx[2]].Omega
    #'''

    #First sort and remove values from the time array that aren't consistent with the higher resolution data. 
    t_arr = sb_elems[0].copy()
    sortt = np.sort(t_arr)

    dt = round(abs(sortt[-1] - sortt[-2]))


    test_arr = t_arr.copy()
    #print(dt, test_arr)
    #print(test_arr % dt)
    #skip_short_res = np.where(abs(test_arr % dt) <= 1)[0]
    skip_short_res = np.where(test_arr.astype(int) % dt == 0)[0]
    
    #t_arr = np.sort(t_arr[skip_short_res])
    
    # archive may have backwards integration as well, sort and remove the second 0-time point where the array restarts    
    filt_t_arr = t_arr[skip_short_res]
    sort_t, sort_inds = np.unique(filt_t_arr, return_index=True)

    sb_elems = sb_elems[:,skip_short_res]
    clone_elems = clone_elems[:,:,skip_short_res]
    
    sb_elems = sb_elems[:,sort_inds]
    clone_elems = clone_elems[:,:, sort_inds]
    
    j_elems = j_elems[:,skip_short_res]
    s_elems = s_elems[:,skip_short_res]
    u_elems = u_elems[:,skip_short_res]
    n_elems = n_elems[:,skip_short_res]
    
    j_elems = j_elems[:,sort_inds]
    s_elems = s_elems[:,sort_inds]
    u_elems = u_elems[:,sort_inds]
    n_elems = n_elems[:,sort_inds]

    if vem:
        v_elems = v_elems[:,skip_short_res]
        e_elems = e_elems[:,skip_short_res]
        m_elems = m_elems[:,skip_short_res]
        
        v_elems = v_elems[:,sort_inds]
        e_elems = e_elems[:,sort_inds]
        m_elems = m_elems[:,sort_inds]

    if vem:
        planet_elems = {'venus': v_elems, 'earth': e_elems, 'mars': m_elems, 'jupiter': j_elems, 'saturn': s_elems, 'uranus': u_elems, 'neptune': n_elems}
    else:
        planet_elems = {'jupiter': j_elems, 'saturn': s_elems, 'uranus': u_elems, 'neptune': n_elems}

    '''
    if small_planets_flag:
            hj,hs,hu,hn,kj,ks,ku,kn,pj,ps,pu,pn,qj,qs,qu,qn,hv,he,hm,kv,ke,km,pv,pe,pm,qv,qe,qm = equinoct_arrays
        else:
            hj,hs,hu,hn,kj,ks,ku,kn,pj,ps,pu,pn,qj,qs,qu,qn = equinoct_arrays

    p_elems = [a,e,I,o,O]
    '''
    
    if vem:
        p_equinoct_arr = [*hkpq(j_elems),*hkpq(s_elems),*hkpq(u_elems),*hkpq(n_elems),*hkpq(v_elems),*hkpq(e_elems),*hkpq(m_elems)]    
    else:
        p_equinoct_arr = [*hkpq(j_elems),*hkpq(s_elems),*hkpq(u_elems),*hkpq(n_elems)]    
    '''
    print('dt')
    
    print('sorted t_arr')
    plt.plot(sb_elems[0])
    plt.show()
    print('sort_t')
    plt.plot(sort_t)
    plt.show()
    '''
    return 1, sb_elems[0], sb_elems[1:], planet_elems, clone_elems, vem

def hcm_calc(a,rmsa,rmse,rmsi):    
    G = 6.673e-11
    M = 1.989e30
    n = np.sqrt(G*M/(a*1.496e11)**3)
    return n*a*1.498e11*np.sqrt(5/4*(rmsa/a)**2+2*rmse**2+2*rmsi**2)   

def calc_proper_elements(des='', times= [], sb_elems = [], planet_elems = [], small_planets_flag = False, output_arrays = False, gs_dict = None):

    #t_init, sb_elems, planet_elems, clone_elems, p_equinoct_arr, small_planets_flag = read_archive_for_pe(des=des,datadir=datadir,archivefile=archivefile,
    #                                    clones=clones,return_timeseries=return_timeseries,logfile=logfile, tno_results = tno_results)
    proper_object = proper_element_class(des=des)
    #if len(int_data) < 1:
    #    return 0, proper_object, []
        
    #times, sb_elems, planet_elems, clone_elems, small_planets_flag = int_data


    osc_elem = {}
    mean_elem = {}
    prop_elem = {}
    ind0 = np.where(times == 0.0)[0]

    dt = abs(times[-1]-times[-2])

    a_init = sb_elems[0]
    e_init = sb_elems[1]
    I_init = sb_elems[2]
    o_init = sb_elems[3]
    O_init = sb_elems[4]
        
    osc_elem['a'] = a_init[ind0]
    osc_elem['e'] = e_init[ind0]
    osc_elem['I'] = I_init[ind0]
    osc_elem['o'] = o_init[ind0]
    osc_elem['O'] = O_init[ind0]

    mean_elem['a'] = np.mean(a_init)
    mean_elem['e'] = np.mean(e_init)
    mean_elem['sinI'] = np.sin(np.mean(I_init))
        
    diffg = np.gradient((o_init+O_init)%(2*np.pi))
    diffs = np.gradient((O_init)%(2*np.pi))
    
    mean_elem['g(rev/yr)'] = np.median(diffg)/dt/2/np.pi
    mean_elem['s(rev/yr)'] = np.median(diffs)/dt/2/np.pi
    
    mean_elem['g("/yr)'] = np.median(diffg)/dt*3600*360/2/np.pi
    mean_elem['s("/yr)'] = np.median(diffs)/dt*3600*360/2/np.pi

    if len(planet_elems) == 0:
        g_arr = []
        s_arr = []
        if gs_dict == None:
            if small_planets_flag:
                gs_dict = {'g5': 3.299e-6 ,'g6': 2.197e-5, 'g7': 2.398e-6, 'g8': 5.022e-7, 's6': -2.032e-5, 's7': -2.309e-6, 's8': -5.3395e-7, 'g2': 7.34474/1296000, 'g3': 17.32832/1296000, 'g4': 18.00233/1296000, 's2': -6.57080/1296000, 's3': -18.74359/1296000, 's4': -17.63331/1296000}
            else:
                gs_dict = {'g5': 3.299e-6 ,'g6': 2.197e-5, 'g7': 2.398e-6, 'g8': 5.022e-7, 's6': -2.032e-5, 's7': -2.309e-6, 's8': -5.3395e-7}
        
        for key, value in gs_dict.items():
            if 'g' in key:
                g_arr.append(value)
            elif 's' in key:
                s_arr.append(value)
    else:        
        g_arr,g_inds,s_arr,s_inds, gs_dict = get_planet_freqs(times, planet_elems, small_planets_flag = small_planets_flag)

    '''
    #OrbFit SELRE9.f90 frequencies
    gs_freqs = [2*gs_dict['g6']-gs_dict['g5'],
                        2*gs_dict['g6']-gs_dict['g7'],
                        3*gs_dict['g6']-2*gs_dict['g5'],
                        -gs_dict['g5']+gs_dict['g6']+gs_dict['g7'],
                        gs_dict['g5']+gs_dict['g6']-gs_dict['g7'],
                        -gs_dict['g5']+2*gs_dict['g6']-gs_dict['s6']-gs_dict['s7'],
                        gs_dict['g5']+gs_dict['s7']-gs_dict['s6'],
                        2*gs_dict['g5']-gs_dict['g6'],
                        2*gs_dict['g6']-2*gs_dict['s6'],
                        gs_dict['s6']+gs_dict['g5']-gs_dict['g6'],
                        2*gs_dict['g6']+gs_dict['s6'],
                        3*gs_dict['g6']+gs_dict['s6'],
                        gs_dict['g5']+gs_dict['s6'],
                        gs_dict['g6']+gs_dict['s6']]
    '''

    g_freqs = {
        '2g6-g5': 2*gs_dict['g6'] - gs_dict['g5'],
        '2g5-g6': 2*gs_dict['g5'] - gs_dict['g6'],
        '2g7-g6': 2*gs_dict['g7'] - gs_dict['g6'],
        '2g6-g7': 2*gs_dict['g6'] - gs_dict['g7'],
        '3g6-g5': 3*gs_dict['g6'] - 2*gs_dict['g5'],
        'g6-g5+g7': gs_dict['g6'] - gs_dict['g5'] + gs_dict['g7'],
        'g6+g5-g7': gs_dict['g6'] + gs_dict['g5'] - gs_dict['g7'],
        '2g6-g5+s6-s7': 2*gs_dict['g6'] - gs_dict['g5'] + gs_dict['s6'] - gs_dict['s7'],
        'g5+s7-s6': gs_dict['g5'] + gs_dict['s7'] - gs_dict['s6']
    }
    
    s_freqs = {
        's6+s7': gs_dict['s6'] + gs_dict['s7'], 
        's6+s8': gs_dict['s6'] + gs_dict['s8'], 
        's7+s8': gs_dict['s7'] + gs_dict['s8'], 
        '2s6-s7': 2*gs_dict['s6'] - gs_dict['s7'],     
        '2s6-s8': 2*gs_dict['s6'] - gs_dict['s8'],     
        '2s7-s6': 2*gs_dict['s7'] - gs_dict['s6'],     
        '2s7-s8': 2*gs_dict['s7'] - gs_dict['s8'],     
        '2s8-s7': 2*gs_dict['s8'] - gs_dict['s7'],        
        '2g5-s6': 2*gs_dict['g5'] - gs_dict['s6'],       
        '2g5-s7': 2*gs_dict['g5'] - gs_dict['s7'],   
        '2g6-s6': 2*gs_dict['g6'] - gs_dict['s6'],       
        '2g6-s7': 2*gs_dict['g6'] - gs_dict['s7'],       
        'g5+g6-s6': gs_dict['g5'] + gs_dict['g6'] - gs_dict['s6'],      
        'g5+g6-s7': gs_dict['g5'] + gs_dict['g6'] - gs_dict['s7'],      
        'g5-g6+s6': gs_dict['g5'] - gs_dict['g6'] + gs_dict['s6'],                   
    }

    gs_freqs = {
        'g6+s6': gs_dict['g6'] + gs_dict['s6'],
        '2g6+s6': 2*gs_dict['g6'] + gs_dict['s6'],
        '3*g6+s6': 3*gs_dict['g6'] + gs_dict['s6'],
        'g7+s7': gs_dict['g7'] + gs_dict['s7'],
        'g8+s8': gs_dict['g8'] + gs_dict['s8'],
        'g5+s6': gs_dict['g5'] + gs_dict['s6'],
        'g5+s7': gs_dict['g5'] + gs_dict['s7'],
        'g6+s7': gs_dict['g6'] + gs_dict['s7'],
        'g7+s8': gs_dict['g7'] + gs_dict['s8'],
        'g6-s6': gs_dict['g6'] - gs_dict['s6'],
        'g7-s7': gs_dict['g7'] - gs_dict['s7'],
        'g8-s8': gs_dict['g8'] - gs_dict['s8'],
        
    }

    for key, val in g_freqs.items():
        g_arr.append(val)
    for key, val in s_freqs.items():
        s_arr.append(val)
    for key, val in gs_freqs.items():
        g_arr.append(val)
        s_arr.append(val)

    
    if output_arrays:
        flag, pes, rms_val, error_list, omega_n, Omega_n, maxvals, g, s, rese, resI, sec_res_e, sec_res_I, e_osc_amp, I_osc_amp, e_amp, I_amp, angle_sec_res, librate_angle, angle_ent, phifrac, hk_new, pq_new, a_filt, hk_wins, pq_wins, a_wins, t_wins, hk_arr, pq_arr = compute_prop(a_init,e_init,I_init,o_init,O_init,times,g_arr,s_arr,gs_dict,small_planets_flag,windows=5,debug=False,objname=des, rms = True, shortfilt=True, output_arrays = output_arrays)
    else:
        flag, pes, rms_val, error_list, omega_n, Omega_n, maxvals, g, s, rese, resI, sec_res_e, sec_res_I, e_osc_amp, I_osc_amp, e_amp, I_amp, angle_sec_res, librate_angle, angle_ent, phifrac = compute_prop(a_init,e_init,I_init,o_init,O_init,times,g_arr,s_arr,gs_dict,small_planets_flag,windows=5,debug=False,objname=des, rms = True, shortfilt=True)

    prop_elem = {}
    prop_elem['a'] = pes[0]
    prop_elem['e'] = pes[1]
    prop_elem['sinI'] = pes[2]
    prop_elem['g(rev/yr)'] = g
    prop_elem['s(rev/yr)'] = s
    prop_elem['g("/yr)'] = g*3600*360
    prop_elem['s("/yr)'] = s*3600*360
    
    prop_errs = {}
    prop_errs['RMS_a'] = rms_val[0]
    prop_errs['RMS_e'] = rms_val[1]
    prop_errs['RMS_sinI'] = rms_val[2]
    prop_errs['RMS_g(rev/yr)'] = rms_val[3]
    prop_errs['RMS_s(rev/yr)'] = rms_val[4]
    prop_errs['RMS_g("/yr)'] = rms_val[3]*3600*360
    prop_errs['RMS_s("/yr)'] = rms_val[4]*3600*360

    prop_win_list = {}
    prop_win_list['a_win'] = error_list[:,0]
    prop_win_list['e_win'] = error_list[:,1]
    prop_win_list['sinI_win'] = error_list[:,2]
    prop_win_list['g_win'] = error_list[:,3]
    prop_win_list['s_win'] = error_list[:,4]

    if small_planets_flag:
        proper_object.planets = ['venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']
    else:
        proper_object.planets = ['jupiter', 'saturn', 'uranus', 'neptune']

    proper_object.planet_freqs = gs_dict
    proper_object.osculating_elements = osc_elem
    proper_object.mean_elements = mean_elem
    proper_object.proper_elements = prop_elem
    proper_object.proper_windows = prop_win_list

    proper_object.proper_errors = prop_errs

    #rese, resI, sec_res_e, sec_res_I, e_osc_amp, I_osc_amp, e_amp, I_amp, angle_sec_res, librate_angle, angle_ent, phifrac
    proper_object.proper_indicators['Ecc Mean Flag'] = rese
    proper_object.proper_indicators['sinI Mean Flag'] = resI
    proper_object.proper_indicators['Ecc Mean Indicator'] = sec_res_e
    proper_object.proper_indicators['sinI Mean Indicator'] = sec_res_I
    
    proper_object.proper_indicators['Ecc Osculating Amplitude'] = e_osc_amp
    proper_object.proper_indicators['sinI Osculating Amplitude'] = I_osc_amp
    proper_object.proper_indicators['Ecc Filtered Amplitude'] = e_amp
    proper_object.proper_indicators['sinI Filtered Amplitude'] = I_amp

    proper_object.proper_indicators['Distance Metric'] = hcm_calc(prop_elem['a'], prop_errs['RMS_a'], prop_errs['RMS_e'], prop_errs['RMS_sinI'])

    proper_object.proper_internal['Secular Resonant Angle'] = angle_sec_res
    proper_object.proper_internal['Librating Angle'] = librate_angle
    proper_object.proper_internal['Angle Entropy'] = angle_ent
    proper_object.proper_internal['Phi Entropy'] = phifrac

    #if clones > 0:
    #    sb_elems = np.concatenate((np.array([sb_elems[1:]]),clone_elems), axis = 0)

    g5 = gs_dict['g5']; g6 = gs_dict['g6']; g7 = gs_dict['g7']; g8 = gs_dict['g8']
    s6 = gs_dict['s6']; s7 = gs_dict['s7']; s8 = gs_dict['s8']
    
    if output_arrays:
        proper_object.hk_original = hk_arr
        proper_object.pq_original = pq_arr
        
        proper_object.hk_filtered = hk_new
        proper_object.pq_filtered = pq_new

        proper_object.hk_windows = hk_wins
        proper_object.pq_windows = pq_wins
        proper_object.time_windows = t_wins

        proper_object.a_original = a_init
        proper_object.a_filtered = a_filt
        proper_object.a_windows = a_wins

        proper_object.time = times

        

    proper_object.secular_frequencies = {
        'g-g5': g-g5, 'g-g6': g-g6, 'g-g7': g-g7, 'g-g8': g-g8,
        's-s6': s-s6, 's-s7': s-s7, 's-s8': s-s8,
        #g-frequencies
        'g-2g6+g5': g-2*g6+g5, 
        'g-2g5+g6': g-2*g5+g6, 
        'g-2g7+g6': g-2*g7+g6, 
        'g-3g6+2g5': g-3*g6+2*g5, 
        'g-g6+g5-g7': g-g6+g5-g7, 
        'g-g6-g5=g7': g-g6-g5+g7, 
        'g-2g6+g5-s6+s7': g-2*g6+g5-s6+s7, 
        'g-g5-s7+s6': g-g5-s7+s6, 
        # s-frequencies
        '2s-s6-s7': 2*s-s6-s7,
        '2s-s6-s8': 2*s-s6-s8,
        '2s-s7-s8': 2*s-s7-s8,
        's-2s6+s7': s-2*s6+s7,
        's-2s6+s8': s-2*s6+s8,
        's-2s7+s6': s-2*s7+s6,
        's-2s7+s8': s-2*s7+s8,
        's-2s8+s7': s-2*s8+s6,
        's-2g5+s6': s-2*g5+s6,
        's-2g5+s7': s-2*g5+s7,
        's-2g6+s6': s-2*g6+s6,
        's-2g6+s7': s-2*g6+s7,
        's-g5-g6+s6': s-g5-g6+s6,
        's-g5-g6+s7': s-g5-g6+s7,
        's-g5+g6-s6': s-g5+g6-s6,
        #g+s-frequencies
        'g+s-g6-s6': g+s-g6-s6,
        '2g+s-2g6-s6': 2*g+s-2*g6-s6,
        '3g+s-3g6-s6': 3*g+s-3*g6-s6,
        'g+s-g7-s7': g+s-g7-s7,
        'g+s-g8-s8': g+s-g8-s8,
        '2(g+s)-g7-s7-g8-s8': 2*(g+s)-g7-s7-g8-s8,
        'g+s-g5-s6': g+s-g5-s6,
        'g+s-g5-s7': g+s-g5-s7,
        'g+s-g6-s7': g+s-g6-s7,
        'g+s-g7-s8': g+s-g7-s8,
        'g-s-g6+s6': g-s-g6+s6,
        'g-s-g7+s7': g-s-g7+s7,
        'g-s-g8+s8': g-s-g8+s8,
    }

    proper_object.secfreq_flags = {}

    for key,val in proper_object.secular_frequencies.items():
        proper_object.secular_frequencies[key] = val*3600*360

        proper_object.secfreq_flags[key] = (abs(val*3600*360) < 0.01, abs(val*3600*360) < 0.05, abs(val*3600*360) < 0.1, abs(val*3600*360) < 0.2)

    fam_results = check_family_candidates(proper_object)

    proper_object.family_results = fam_results


    scat_results = check_scatter(times,a_init)
    proper_object.scattered = scat_results
        
    return 1, proper_object
    

def check_scatter(t,a):
    da = np.gradient(a)
    de = np.abs(da/a)[1:-2]

    scat_result = {'scattered': False, 'scat_time': np.inf, 'Max delta-E': np.max(de)}

    if np.nanmean(a) < 20:
        thresh = 1e-2
    else:
        thresh = 1e-3
    
    if np.max(de) > thresh:
        scat_ind = np.argmax(de)
        
        scat_result['scattered'] = True
        scat_result['scat_time'] = t[scat_ind]
        scat_result['Max delta-E'] = de[scat_ind]
    return scat_result
        
def hcm_pair(a1, a2, e1, e2, sini1, sini2):
    G = 6.673e-11
    M = 1.989e30
    am = np.mean(np.array([a1,a2]))
    n = np.sqrt(G*M/(am*1.496e11)**3)
    return n*am*1.498e11*np.sqrt(5/4*((a2-a1)/am)**2+2*(e2-e1)**2+2*(sini2-sini1)**2)  
    
def check_family_candidates(proper_object):

    family_occupancy = {'family_name': None, 'pairwise_dMet': np.inf}
    pe = proper_object.proper_elements
    
    if pe['a'] > 20:
        fam_df = pd.read_csv('../data/sbdynt_files/tno_family_centers.txt', index_col=0)
    #elif pe['a'] < 5:
    #    fam_df = pd.read_csv('../data/sbdynt_files/ast_family_centers.txt', index_col=0)
    else:
        return family_occupancy
    
    for i in range(len(fam_df)):
        fam_obj = fam_df.iloc[i]
        hcm_cen = hcm_pair(fam_obj['cen_a'], pe['a'], 
                           fam_obj['cen_e'], pe['e'], 
                           np.sin(fam_obj['cen_I']/180*np.pi), pe['sinI'])

        if (hcm_cen < fam_obj['hcm_cut']) and (pe['a'] > fam_obj['low_a']) and (pe['a'] < fam_obj['high_a']) and (pe['e'] > fam_obj['low_e']) and (pe['e'] < fam_obj['high_e']) and (pe['sinI'] > np.sin(np.pi/180*fam_obj['low_I'])) and (pe['sinI'] < np.sin(np.pi/180*fam_obj['high_I'])):
            family_occupancy['family_name'] = fam_obj['objname']
            family_occupancy['pairwise_dMet'] = hcm_cen

            break

    return family_occupancy
        
    


def prop_calc(objname, filename='Single',windows=5, direction = 'both', time_run = 0, rms = True, shortfilt=True, debug=False):
    
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
        
        The Running Block calculations are used to calculate a mean error for the proper elements. The default run will be a 150 Myr integration, with running blocks from 0-50,25-75,...100-150 Myr, producing 5 running blocks total. So your default run will produce an outputs list consisting of 45 values total.  
        
        If an error occurs in the code, then outputs is instead returned as [objname,0,0,0,0,0,0,0].
        
    """ 
#    print(objname)
    try:       
        if direction == 'both':
            fullfile = '../data/'+filename+'/'+str(objname)+'/archive_forward.bin'
        else:
            fullfile = '../data/'+filename+'/'+str(objname)+'/archive.bin'

        home = str(os.path.expanduser("~"))

        if debug:
            print(os.listdir('../data/'+filename+'/'+str(objname)))
        
        archive = rebound.Simulationarchive(fullfile)
        
        try:
            earth = archive[0].particles['earth']
            small_planets_flag = True
        except:
            small_planets_flag = False

        nump = len(archive[0].particles)
        if abs(time_run) > abs(archive[-1].t) or time_run == 0:
            time_run = archive[-1].t

        if direction == 'both':
            fullfile_b = '../data/'+filename+'/'+str(objname)+'/archive_back.bin'
            fullfile_f = '../data/'+filename+'/'+str(objname)+'/archive_forward.bin'

            archive_b = rebound.Simulationarchive(fullfile_b)
            archive_f = rebound.Simulationarchive(fullfile_f)

            flagb, a_initb, e_initb, inc_initb, lan_initb, aop_initb, M_initb, t_initb = read_sa_for_sbody(des = str(objname), archivefile=fullfile_b,clones=0,tmax=0.,tmin=-abs(time_run),center='helio',s=archive_b)
            flagf, a_initf, e_initf, inc_initf, lan_initf, aop_initf, M_initf, t_initf = read_sa_for_sbody(des = str(objname), archivefile=fullfile_f,clones=0,tmin=0.,tmax=abs(time_run),center='helio',s=archive_f)

            a_init = np.concatenate((a_initb[::-1], a_initf[1:]))
            e_init = np.concatenate((e_initb[::-1], e_initf[1:]))
            inc_init = np.concatenate((inc_initb[::-1], inc_initf[1:]))
            lan_init = np.concatenate((lan_initb[::-1], lan_initf[1:]))
            aop_init = np.concatenate((aop_initb[::-1], aop_initf[1:]))
            t_init = np.concatenate((t_initb[::-1], t_initf[1:]))
            M_init = np.concatenate((M_initb[::-1], M_initf[1:]))
            
        else:
            if archive[-1].t > 0:
                flag, a_init, e_init, inc_init, lan_init, aop_init, M_init, t_init = read_sa_for_sbody(des = str(objname), archivefile=fullfile,clones=0,tmin=0.,tmax=time_run,center='helio',s=archive)
            else:
                flag, a_init, e_init, inc_init, lan_init, aop_init, M_init, t_init = read_sa_for_sbody(des = str(objname), archivefile=fullfile,clones=0,tmax=0.,tmin=time_run,center='helio',s=archive)
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno

        error_message = "An error occurred in at line "+str(line_number)

        print(error_message)
        print(e)
        return list(np.zeros(51))
        

    dt = abs(archive[1].t-archive[0].t)
    
    try:
        equinoct_arrays = get_planet_arrays(archive,small_planets_flag,fullfile)
        g_arr,g_inds,s_arr,s_inds, gs_dict = get_planet_freqs(equinoct_arrays,small_planets_flag,t_init)
    except:
        outs = np.zeros(51)
        return list(outs)
    if debug == True:
        return compute_prop(a_init,e_init,inc_init,aop_init,lan_init,t_init,g_arr,s_arr,gs_dict,small_planets_flag, windows=windows,debug = debug,objname=objname, rms = rms, shortfilt=shortfilt)  
    else:
        flag, pes, rms_val, error_list, omega_n, Omega_n, maxvals, g, s, rese, resI, sec_res_e, sec_res_I, e_osc_amp, I_osc_amp, e_amp, I_amp, angle_sec_res, librate_angle, angle_ent, phifrac = compute_prop(a_init,e_init,inc_init,aop_init,lan_init,t_init,g_arr,s_arr,gs_dict,small_planets_flag, windows=windows,debug = debug,objname=objname, rms = rms)

    ind0 = np.where(t_init == 0)[0][0]
    if flag == False:
        return list(np.zeros(51))

    return_data = [objname]
    return_data.append(a_init[ind0])
    return_data.append(e_init[ind0])
    
    return_data.append(inc_init[ind0])
    return_data.append(aop_init[ind0])
    return_data.append(lan_init[ind0])
    return_data.append(M_init[ind0])
    
    return_data.append(np.mean(a_init))
    return_data.append(np.mean(e_init))
    return_data.append(np.mean(inc_init))
    for i in range(len(pes)):
        return_data.append(pes[i])
    
    return_data.append(omega_n[ind0])
    return_data.append(Omega_n[ind0])
    
    for i in range(len(error_list)):
        for j in range(len(error_list[0])):
            return_data.append(error_list[i][j])
    return_data.append(rms_val[0])
    return_data.append(rms_val[1])
    return_data.append(rms_val[2])
    return_data.append(maxvals[0])
    return_data.append(maxvals[1])
    return_data.append(maxvals[2])
    return_data.append(g*2*np.pi*206265)
    return_data.append(s*2*np.pi*206265)
   
    return_data.append(rese)    
    return_data.append(resI)    
    return_data.append(sec_res_e)    
    return_data.append(sec_res_I)
    return_data.append(e_osc_amp)    
    return_data.append(I_osc_amp)
    return_data.append(e_amp)    
    return_data.append(I_amp)
    return_data.append(angle_sec_res)    
    return_data.append(librate_angle)
    return_data.append(angle_ent)
    return_data.append(phifrac)
    return_data.append(gs_dict)
    return return_data

def prop_multi(filename):
    names_df = pd.read_csv('../data/data_files/'+filename+'.csv').iloc[0:4]
    data = []
    for i,objname in enumerate(names_df['Name']):
        if i%50==0:
            print(i)
        windows=5        
        data_line = prop_calc(objname,filename,windows=windows)
        data.append(data_line)
        
    column_names = ['Objname','ObsSMA','ObsEcc','ObsSin(Inc)','Obs_omega','Obs_Omega','Obs_M','MeanSMA','MeanEcc','MeanSin(Inc)','PropSMA','PropEcc','PropSin(Inc)','Prop_omega','Prop_Omega']
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
    column_names.append('g("/yr)')
    column_names.append('s("/yr)')
    column_names.append('Res_E')
    column_names.append('Res_I')
    column_names.append('Ecc_0_bin_%')
    column_names.append('Inc_0_bin_%')
    column_names.append('Ecc_osc_amplitude')
    column_names.append('Sin(Inc)_osc_amplitude')
    column_names.append('Ecc_filtered_amplitude')
    column_names.append('Sin(Inc)_filtered_amplitude')
    column_names.append('Secular Resonant Angle')
    column_names.append('Median Librating Angle (rad)')
    column_names.append('Angle Entropy')
    column_names.append('Phi Fraction')
    data_df = pd.DataFrame(data,columns=column_names)
    data_df.to_csv('../data/results/'+filename+'_prop_elem_helio_2.csv')
    return data

if __name__ == "__main__":
    filename = str(sys.argv[1])
    
    if filename != 'Single':
        data = prop_multi(filename)  
    else:
        column_names = ['Objname','ObsSMA','ObsEcc','ObsSin(Inc)','Obs_omega','Obs_Omega','Obs_M','MeanSMA','MeanEcc','MeanSin(Inc)','PropSMA','PropEcc','PropSin(Inc)','Prop_omega','Prop_Omega']
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
        column_names.append('g("/yr)')
        column_names.append('s("/yr)')
        column_names.append('Res_e')
        column_names.append('Res_I')
        column_names.append('Ecc_0_bin_%')
        column_names.append('Inc_0_bin_%')
        column_names.append('Ecc_filtered_amplitude')
        column_names.append('Sin(Inc)_filtered_amplitude')
        column_names.append('Secular Resonant Angle')
        column_names.append('Median Librating Angle (rad)')
        column_names.append('Angle Entropy')
        column_names.append('Phi Fraction')

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
        
                       

        
