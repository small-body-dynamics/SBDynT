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

av_init = 0
ev_init = 0
incv_init = 0
lanv_init = 0
aopv_init = 0
Mv_init = 0

ae_init = 0
ee_init = 0
ince_init = 0
lane_init = 0
aope_init = 0
Me_init = 0

am_init = 0
e_init = 0
incm_init = 0
lanm_init = 0
aopm_init = 0
Mm_init = 0

aj_init = 0
ej_init = 0
incj_init = 0
lanj_init = 0
aopj_init = 0
Mj_init = 0

as_init = 0
es_init = 0
incs_init = 0
lans_init = 0
aops_init = 0
Ms_init = 0

au_init = 0
eu_init = 0
incu_init = 0
lanu_init = 0
aopu_init = 0
Mu_init = 0

an_init = 0
en_init = 0
incn_init = 0
lann_init = 0
aopn_init = 0
Mn_init = 0


g2 = 0
g3 = 0
g4 = 0
g5 = 0
g6 = 0
g7 = 0
g8 = 0

s2 = 0
s3 = 0
s4 = 0
s6 = 0
s7 = 0
s8 = 0
#small_planets_flag=False

def prop_calc(objname, filename='Single',windows=9,objdes=None,debug=False):
    
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
        #print(objname)
        if archive[-1].t > 0:
            flag, a_init, e_init, inc_init, lan_init, aop_init, M_init, t_init = tools.read_sa_for_sbody(sbody = str(objname), archivefile=fullfile,nclones=0,tmin=0.,tmax=archive[-1].t)
        else:
            flag, a_init, e_init, inc_init, lan_init, aop_init, M_init, t_init = tools.read_sa_for_sbody(sbody = str(objname), archivefile=fullfile,nclones=0,tmax=0.,tmin=archive[-1].t)
        
        global av_init
        global ev_init
        global incv_init
        global lanv_init
        global aopv_init
        global Mv_init

        global ae_init
        global ee_init
        global ince_init
        global lane_init
        global aope_init
        global Me_init

        global am_init
        global em_init 
        global incm_init
        global lanm_init
        global aopm_init
        global Mm_init

        global aj_init
        global ej_init
        global incj_init
        global lanj_init
        global aopj_init
        global Mj_init

        global as_init
        global es_init
        global incs_init
        global lans_init
        global aops_init
        global Ms_init

        global au_init
        global eu_init
        global incu_init
        global lanu_init
        global aopu_init
        global Mu_init
        
        global an_init
        global en_init
        global incn_init
        global lann_init
        global aopn_init
        global Mn_init
        
        global g1
        global g2
        global g3
        global g4
        global g5
        global g6
        global g7
        global g8
        
        global s1
        global s2
        global s3
        global s4
        global s6
        global s7
        global s8

        
        if small_planets_flag:
            '''
            flag, av_init, ev_init, incv_init, lanv_init, aopv_init, Mv_init, t_init = tools.read_sa_for_sbody(sbody = str('venus'), archivefile=fullfile,nclones=0,tmin=(sim[-1].t),tmax=0)
            flag, ae_init, ee_init, ince_init, lane_init, aope_init, Me_init, t_init = tools.read_sa_for_sbody(sbody = str('earth'), archivefile=fullfile,nclones=0,tmin=(sim[-1].t),tmax=0)
            flag, am_init, em_init, incm_init, lanm_init, aopm_init, Mm_init, t_init = tools.read_sa_for_sbody(sbody = str('mars'), archivefile=fullfile,nclones=0,tmin=(sim[-1].t),tmax=0)
        flag, aj_init, ej_init, incj_init, lanj_init, aopj_init, Mj_init, t_init = tools.read_sa_for_sbody(sbody = str('jupiter'), archivefile=fullfile,nclones=0,tmin=(sim[-1].t),tmax=0)
        flag, as_init, es_init, incs_init, lans_init, aops_init, Ms_init, t_init = tools.read_sa_for_sbody(sbody = str('saturn'), archivefile=fullfile,nclones=0,tmin=(sim[-1].t),tmax=0)
        flag, au_init, eu_init, incu_init, lanu_init, aopu_init, Mu_init, t_init = tools.read_sa_for_sbody(sbody = str('uranus'), archivefile=fullfile,nclones=0,tmin=(sim[-1].t),tmax=0)
        flag, an_init, en_init, incn_init, lann_init, aopn_init, Mn_init, t_init = tools.read_sa_for_sbody(sbody = str('neptune'), archivefile=fullfile,nclones=0,tmin=(sim[-1].t),tmax=0)
        '''
            
            if isinstance(av_init, int):
                if archive[-1].t > 0:
                    flag, av_init, ev_init, incv_init, lanv_init, aopv_init, Mv_init, t_init = tools.read_sa_for_sbody(sbody = str('venus'), archivefile=fullfile,nclones=0,tmax=(archive[-1].t),tmin=0)
                    flag, ae_init, ee_init, ince_init, lane_init, aope_init, Me_init, t_init = tools.read_sa_for_sbody(sbody = str('earth'), archivefile=fullfile,nclones=0,tmax=(archive[-1].t),tmin=0)
                    flag, am_init, em_init, incm_init, lanm_init, aopm_init, Mm_init, t_init = tools.read_sa_for_sbody(sbody = str('mars'), archivefile=fullfile,nclones=0,tmax=(archive[-1].t),tmin=0)
                else:
                    flag, av_init, ev_init, incv_init, lanv_init, aopv_init, Mv_init, t_init = tools.read_sa_for_sbody(sbody = str('venus'), archivefile=fullfile,nclones=0,tmin=(archive[-1].t),tmax=0)
                    flag, ae_init, ee_init, ince_init, lane_init, aope_init, Me_init, t_init = tools.read_sa_for_sbody(sbody = str('earth'), archivefile=fullfile,nclones=0,tmin=(archive[-1].t),tmax=0)
                    flag, am_init, em_init, incm_init, lanm_init, aopm_init, Mm_init, t_init = tools.read_sa_for_sbody(sbody = str('mars'), archivefile=fullfile,nclones=0,tmin=(archive[-1].t),tmax=0)
            
                hv = ev_init*np.sin(lanv_init+aopv_init)
                he = ee_init*np.sin(lane_init+aope_init)
                hm = em_init*np.sin(lanm_init+aopm_init)
                kv = ev_init*np.cos(lanv_init+aopv_init)
                ke = ee_init*np.cos(lane_init+aope_init)
                km = em_init*np.cos(lanm_init+aopm_init)
                pv = np.sin(incv_init)*np.sin(lanv_init)
                pe = np.sin(ince_init)*np.sin(lane_init)
                pm = np.sin(incm_init)*np.sin(lanm_init)
                qv = np.sin(incv_init)*np.cos(lanv_init)
                qe = np.sin(ince_init)*np.cos(lane_init)
                qm = np.sin(incm_init)*np.cos(lanm_init)

                                
                Yhkv = np.fft.fft(kv+1j*hv)
                Yhke = np.fft.fft(ke+1j*he)
                Yhkm = np.fft.fft(km+1j*hm)
                    
                Ypqv = np.fft.fft(qv+1j*pv)
                Ypqe = np.fft.fft(qe+1j*pe)
                Ypqm = np.fft.fft(qm+1j*pm)
        
        if isinstance(aj_init, int):
            if archive[-1].t > 0 :
                flag, aj_init, ej_init, incj_init, lanj_init, aopj_init, Mj_init, t_init = tools.read_sa_for_sbody(sbody = str('jupiter'), archivefile=fullfile,nclones=0,tmax=(archive[-1].t),tmin=0)
                flag, as_init, es_init, incs_init, lans_init, aops_init, Ms_init, t_init = tools.read_sa_for_sbody(sbody = str('saturn'), archivefile=fullfile,nclones=0,tmax=(archive[-1].t),tmin=0)
                flag, au_init, eu_init, incu_init, lanu_init, aopu_init, Mu_init, t_init = tools.read_sa_for_sbody(sbody = str('uranus'), archivefile=fullfile,nclones=0,tmax=(archive[-1].t),tmin=0)
                flag, an_init, en_init, incn_init, lann_init, aopn_init, Mn_init, t_init = tools.read_sa_for_sbody(sbody = str('neptune'), archivefile=fullfile,nclones=0,tmax=(archive[-1].t),tmin=0)
            else:
                flag, aj_init, ej_init, incj_init, lanj_init, aopj_init, Mj_init, t_init = tools.read_sa_for_sbody(sbody = str('jupiter'), archivefile=fullfile,nclones=0,tmin=(archive[-1].t),tmax=0)
                flag, as_init, es_init, incs_init, lans_init, aops_init, Ms_init, t_init = tools.read_sa_for_sbody(sbody = str('saturn'), archivefile=fullfile,nclones=0,tmin=(archive[-1].t),tmax=0)
                flag, au_init, eu_init, incu_init, lanu_init, aopu_init, Mu_init, t_init = tools.read_sa_for_sbody(sbody = str('uranus'), archivefile=fullfile,nclones=0,tmin=(archive[-1].t),tmax=0)
                flag, an_init, en_init, incn_init, lann_init, aopn_init, Mn_init, t_init = tools.read_sa_for_sbody(sbody = str('neptune'), archivefile=fullfile,nclones=0,tmin=(archive[-1].t),tmax=0)
            
            hj = ej_init*np.sin(lanj_init+aopj_init)
            hs = es_init*np.sin(lans_init+aops_init)
            hu = eu_init*np.sin(lanu_init+aopu_init)
            hn = en_init*np.sin(lann_init+aopn_init)
            kj = ej_init*np.cos(lanj_init+aopj_init)
            ks = es_init*np.cos(lans_init+aops_init)
            ku = eu_init*np.cos(lanu_init+aopu_init)
            kn = en_init*np.cos(lann_init+aopn_init)
            pj = np.sin(incj_init)*np.sin(lanj_init)
            ps = np.sin(incs_init)*np.sin(lans_init)
            pu = np.sin(incu_init)*np.sin(lanu_init)
            pn = np.sin(incn_init)*np.sin(lann_init)
            qj = np.sin(incj_init)*np.cos(lanj_init)
            qs = np.sin(incs_init)*np.cos(lans_init)
            qu = np.sin(incu_init)*np.cos(lanu_init)
            qn = np.sin(incn_init)*np.cos(lann_init)
    
            Yhkj = np.fft.fft(kj+1j*hj)
            Yhks = np.fft.fft(ks+1j*hs)
            Yhku = np.fft.fft(ku+1j*hu)
            Yhkn = np.fft.fft(kn+1j*hn)
                    
            Ypqj = np.fft.fft(qj+1j*pj)
            Ypqs = np.fft.fft(qs+1j*ps)
            Ypqu = np.fft.fft(qu+1j*pu)
            Ypqn = np.fft.fft(qn+1j*pn)
            
    
            g_arr = []
            s_arr = []
            
            n = len(aj_init)
            dt = abs(t_init[1])
            if n < 10001:
                print(n)
            freq = np.fft.fftfreq(n,d=dt)
            #freqn = np.fft.rfftfreq(len(aj_init),d=dt)
            
            g5 = freq[np.argmax(np.abs(Yhkj[1:])**2)+1]
            g_arr.append(g5)
            while freq[np.argmax(np.abs(Yhks[1:])**2)+1] in g_arr:
                Yhks[np.argmax(np.abs(Yhks[1:])**2)+1] = 0
            g6 = freq[np.argmax(np.abs(Yhks[1:])**2)+1]
            g_arr.append(g6)
            while freq[np.argmax(np.abs(Yhku[1:])**2)+1] in g_arr:
                Yhku[np.argmax(np.abs(Yhku[1:])**2)+1] = 0
            g7 = freq[np.argmax(np.abs(Yhku[1:])**2)+1]
            g_arr.append(g7)
            while freq[np.argmax(np.abs(Yhkn[1:])**2)+1] in g_arr:
                Yhkn[np.argmax(np.abs(Yhkn[1:])**2)+1] = 0
            g8 = freq[np.argmax(np.abs(Yhkn[1:])**2)+1]
            g_arr.append(g8)
            
            s6 = freq[np.argmax(np.abs(Ypqs[1:])**2)+1]
            s_arr.append(g6)
            while freq[np.argmax(np.abs(Ypqu[1:])**2)+1] in s_arr:
                Ypqu[np.argmax(np.abs(Ypqu[1:])**2)+1] = 0
            s7 = freq[np.argmax(np.abs(Ypqu[1:])**2)+1]
            s_arr.append(s7)
            while freq[np.argmax(np.abs(Ypqn[1:])**2)+1] in s_arr:
                Ypqn[np.argmax(np.abs(Ypqn[1:])**2)+1] = 0
            s8 = freq[np.argmax(np.abs(Ypqn[1:])**2)+1]
            s_arr.append(s8)
            
            if small_planets_flag:
                while freq[np.argmax(np.abs(Yhke[1:])**2)+1] in g_arr:
                    Yhke[np.argmax(np.abs(Yhke[1:])**2)+1] = 0
                g3 = freq[np.argmax(np.abs(Yhke[1:])**2)+1]
                g_arr.append(g3)
                while freq[np.argmax(np.abs(Yhkv[1:])**2)+1] in g_arr:
                    Yhkv[np.argmax(np.abs(Yhkv[1:])**2)+1] = 0
                g2 = freq[np.argmax(np.abs(Yhkv[1:])**2)+1]
                g_arr.append(g2)
                while freq[np.argmax(np.abs(Yhkm[1:])**2)+1] in g_arr:
                    Yhkm[np.argmax(np.abs(Yhkm[1:])**2)+1] = 0
                g4 = freq[np.argmax(np.abs(Yhkm[1:])**2)+1]
                g_arr.append(g4)
                
                while freq[np.argmax(np.abs(Ypqe[1:])**2)+1] in s_arr:
                    Ypqe[np.argmax(np.abs(Ypqe[1:])**2)+1] = 0
                s3 = freq[np.argmax(np.abs(Ypqe[1:])**2)+1]
                s_arr.append(s3)
                while freq[np.argmax(np.abs(Ypqv[1:])**2)+1] in s_arr:
                    Ypqv[np.argmax(np.abs(Ypqv[1:])**2)+1] = 0
                s2 = freq[np.argmax(np.abs(Ypqv[1:])**2)+1]
                s_arr.append(s2)
                while freq[np.argmax(np.abs(Ypqm[1:])**2)+1] in s_arr:
                    Ypqm[np.argmax(np.abs(Ypqm[1:])**2)+1] = 0
                s4 = freq[np.argmax(np.abs(Ypqm[1:])**2)+1]
                s_arr.append(s4)
        
            
    except Exception as error:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno

        error_message = "An error occurred in at line "+str(line_number)
    
        # Print the error message
        print(error_message)
        print(error)
        return [objname,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    p_init = np.sin(inc_init)*np.sin(lan_init)
    q_init = np.sin(inc_init)*np.cos(lan_init)
    h_init = (e_init)*np.sin(lan_init+aop_init)
    k_init = (e_init)*np.cos(lan_init+aop_init)
    
    #print(t_init)
    
    try:
        dt = abs(t_init[1])
    
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
        window = signal.windows.hamming(n)
        #window = signal.windows.general_hamming(n,0.1)
        Yhk_win = Yhk*window
        Ypq_win = Ypq*window
        Ypq[0]=0
        #Yhk[0]=0
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
      
        #gind = np.argmax(np.abs(Yhk[1:])**2)+1    
        #sind = np.argmax(np.abs(Ypq[1:])**2)+1
        gind = np.argmax(np.abs(Yhk_win[1:])**2)+1    
        sind = np.argmax(np.abs(Ypq_win[1:])**2)+1
        g = freq[gind]  
        s = freq[sind]
        
    except:
        return [objname,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

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

    if small_planets_flag:
        freq1 = [(g2),(g3),(g4),(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,z8,z9]
        freq2 = [(s2),(s3),(s4),(s6),(s7),(s8),z1,z2,z3,z6,z8,z9]
    else:
        freq1 = [(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,z8,z9]
        freq2 = [(s6),(s7),(s8),z1,z2,z3,z6,z8,z9]
        
    if small_planets_flag:
        freq1 = [(g2),(g3),(g4),(g5),(g6),(g7),(g8)]
        freq2 = [(s2),(s3),(s4),(s6),(s7),(s8)]
    else:
        freq1 = [(g5),(g6),(g7),(g8)]
        freq2 = [(s6),(s7),(s8)]
               
    #if small_planets_flag:
    #    freq1 = [(g1),(g2),(g3),(g4),(g5),(g6),(g7),(g8),g-z8,g-z9,g-z10,z11,z12,z13]
    #    freq2 = [(s1),(s2),(s3),(s4),(s6),(s7),(s8),s-z8,s-z9,s-z10,z14]
    #else:
    #    freq1 = [(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,g-z8,g-z9,g-z10,z11,z12,z13]
    #    freq2 = [(s6),(s7),(s8),z1,z2,z3,z6,z8,z9,s-z8,s-z9,s-z10,z14]
    
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

    if small_planets_flag:
        #freq1 = [(g1),(g2),(g3),(g4),(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,z8,z9,g-z8,g-z9,g-z10,z11,z12,z13]
        #freq2 = [(s1),(s2),(s3),(s4),(s6),(s7),(s8),z1,z2,z3,z6,z8,z9,s-z8,s-z9,s-z10,z14]
        freq1 = [(g2),(g3),(g4),(g5),(g6),(g7),(g8),g-z8,g-z9,g-z10,z11,z12,z13,-g+2*s-g5]
        freq2 = [(s2),(s3),(s4),(s6),(s7),(s8),s-z8,s-z9,s-z10,z14,g-s+g5-s7,g+g5-2*s6,2*g-2*s6]
    else:
        freq1 = [(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,g-z8,g-z9,g-z10,z11,z12]
        freq2 = [(s6),(s7),(s8),z1,z2,z3,z6,z8,z9,s-z8,s-z9,s-z10,z14]
    
    
    de = [
        g-g5,
        g-g6,
        g5-g6, 
        s-s7, 
        s-s6, 
        s7-s6, 
        g+s-s7-g5, 
        g+s-s7-g6, 
        g+s-s6-g5, 
        g+s-s6-g6, 
        2*g-2*s, 
        g-2*g5+g6, 
        g+g5-2*g6, 
        2*g-g5-g6, 
        -g+s+g5-s7, 
        -g+s+g6-s7, 
        -g+s+g5-s6, 
        -g+s+g6-s6, 
        g-g5+s7-s6, 
        g-g5-s7+s6, 
        g-g6+s7-s6, 
        g-g6-s7+s6, 
        2*g-s-s7, 
        2*g-s-s6, 
        -g+2*s-g5, 
        -g+2*s-g6, 
        2*g-2*s7, 
        2*g-2*s6, 
        2*g-s7-s6, 
        g-s+g5-s7, 
        g-s+g5-s6, 
        g-s+g6-s7, 
        g-s+g6-s6, 
        g+g5-2*s7, 
        g+g6-2*s7, 
        g+g5-2*s6, 
        g+g6-2*s6, 
        g+g5-s7-s6, 
        g+g6-s7-s6, 
        s-2*s7+s6, 
        s+s7-2*s6, 
        2*s-s7-s6, 
        s+g5-g6-s7, 
        s-g5+g6-s7, 
        s+g5-g6-s6, 
        s-g5+g6-s6, 
        2*s-2*g5, 
        2*s-2*g6, 
        2*s-g5-g6, 
        s-2*g5+s7, 
        s-2*g5+s6, 
        s-2*g6+s7, 
        s-2*g6+s6, 
        s-g5-g6+s7, 
        s-g5-g6+s6, 
        2*g-2*g5, 
        2*g-2*g6, 
        2*s-2*s7, 
        2*s-2*s6, 
        g-2*g6+g7, 
        g-3*g6+2*g5, 
        2*(g-g6)+(s-s6), 
        g+g5-g6-g7, 
        g-g5-g6+g7, 
        g+g5-2*g6-s6+s7, 
        3*(g-g6)+(s-s6)
    ]
    
    freq1.extend(de)
    freq2.extend(de)
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
    
    spread_dist1 = 1.05
    spread_dist2 = 1.05
    freq_dist1 = 1.1
    freq_dist2 = 1.1
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
    '''
    ##########################################################################################
    #Using different spreads for different values
    for i in range(len(secresind1)):
        if secresind1[i] == gind:
            Yhk_f[secresind1[i]] = 0
            continue
        if abs(secresind1[i] - gind) < freq_dist_lims[i]:
            continue
    
        if spreads[i] > 0:
            Yhk_f[secresind1[i]-spreads[i]:secresind1[i]+spreads[i]] = 0
        else:
            Yhk_f[secresind1[i]] = 0
                
    for i in range(len(secresind2)):
        if secresind2[i] == sind:
            #Ypq_f[secresind2[i]] = 0
            continue
        if abs(secresind2[i] - sind) < freq_dist_lims[i+len(secresind1)]:
            continue
    
        if spreads[i+len(secresind1)] > 0:
            Ypq_f[secresind2[i]-spreads[len(secresind1)+i]:secresind2[i]+spreads[len(secresind1)+i]] = 0
        else:
            Ypq_f[secresind2[i]] = 0
    ##########################################################################################
'''    
    
##########################################################################################################################
    Yhk_copy = Yhk_f.copy()    
    Ypq_copy = Ypq_f.copy()    
    
    for i in range(len(secresind1)):
        if secresind1[i] == gind:
            '''
            if small_planets_flag:
                if i<=3 or i>=8:
                    Yhk_f[secresind1[i]] = Yhk_copy[secresind1[i]]*0.995
                else:
                    Yhk_f[secresind1[i]] = Yhk_copy[secresind1[i]]*0.1
            else:
                if i>=4:
                    Yhk_f[secresind1[i]] = Yhk_copy[secresind1[i]]*0.995
                else:
                    Yhk_f[secresind1[i]] = Yhk_copy[secresind1[i]]*0.1
            #Yhk_f[secresind1[i]-1] = 0
            #Yhk_f[secresind1[i]-1] = 0
            '''
            continue
            
        if abs(secresind1[i] - gind) < freq_dist_lims[i]:
            '''
            if small_planets_flag:
                if i<=3 or i>=8:
                    Yhk_f[gind] = Yhk_copy[gind]*0.995
                else:
                    Yhk_f[gind] = Yhk_copy[gind]*0.1
            else:
                if i>=4:
                    Yhk_f[gind] = Yhk_copy[gind]*0.995
                else:
                    Yhk_f[gind] = Yhk_copy[gind]*0.1
            '''
            continue
            
        if spreads[i] > 0:
            if secresind1[i] <= 3:
                continue
            if len(Yhk_copy) - secresind1[i] <= 3:
                continue
                
            if secresind1[i]-2*spreads[i]-1 < 0:
                minarr = Yhk_copy[secresind1[i]+spreads[i]+1:secresind1[i]+2*spreads[i]+1]
            elif secresind1[i]+2*spreads[i]+1 > len(freq):
                minarr = Yhk_copy[secresind1[i]-2*spreads[i]-1:secresind1[i]-spreads[i]-1]
            else:
                minarr = np.array([(Yhk_copy[secresind1[i]-2*spreads[i]-1:secresind1[i]-spreads[i]-1]),(Yhk_copy[secresind1[i]+spreads[i]+1:secresind1[i]+2*spreads[i]+1])]).astype(float)
            #print(len(Yhk_copy),minarr,spreads[i],secresind1[i],i,z13,freq[secresind1[i]],g,g7)
            #minind = np.unravel_index(np.argmin(np.abs(minarr), axis=None), minarr.shape)
            #print(minind,minarr)
            #print(spreads[i],minarr[minind],minarr,minind,secresind1[i],len(freq))
            #Yhk_f[secresind1[i]-spreads[i]:secresind1[i]+spreads[i]] = minarr[minind]
            Yhk_f[secresind1[i]-spreads[i]:secresind1[i]+spreads[i]] = 10**(np.nanmean(np.log10(abs(minarr))))
            #print(minarr,Yhk_f[secresind1[i]-spreads[i]:secresind1[i]+spreads[i]])
            #Yhk_f[secresind1[i]-spreads[i]:secresind1[i]+spreads[i]] = np.mean(minarr)
        else:
        #Yh_f[secresind1[i]] = 0
            #Yk_f[secresind1[i]] = 0
            Yhk_f[secresind1[i]] = 0.5+0.5j
                
    for i in range(len(secresind2)):
        if secresind2[i] == sind:
            #Ypq_f[sind-2:sind] = 0
            #Ypq_f[sind+1:sind+3] = 0
            #Yhk_f[secresind1[i]] = 0.5+0.5j
            continue
        if abs(secresind2[i] - sind) < freq_dist_lims[i+len(secresind1)]:
            #Ypq_f[sind-2:sind] = 0
            #Ypq_f[sind+1:sind+3] = 0
            continue
    
        if spreads[i] > 0:
            if secresind2[i] <= 3:
                continue
            if len(Ypq_copy) - secresind2[i] <= 3:
                continue
            
            if secresind2[i]-2*spreads[i+len(secresind1)]-1 < 0:
                minarr = Ypq_copy[secresind2[i]+spreads[i+len(secresind1)]+1:secresind2[i]+2*spreads[i+len(secresind1)]+1]
            elif secresind2[i]+2*spreads[i+len(secresind1)]+1 > len(freq):
                minarr = Ypq_copy[secresind2[i]-2*spreads[i+len(secresind1)]-1:secresind2[i]-spreads[i+len(secresind1)]-1]
            else:
                minarr = np.array([(Ypq_copy[secresind2[i]-2*spreads[i+len(secresind1)]-1:secresind2[i]-spreads[i+len(secresind1)]-1]),(Ypq_copy[secresind2[i]+spreads[i+len(secresind1)]+1:secresind2[i]+2*spreads[i+len(secresind1)]+1])]).astype(float)
            
            #minind = np.unravel_index(np.argmin(np.abs(minarr), axis=None), minarr.shape)
            
            #Ypq_f[secresind2[i]-spreads[i+len(secresind1)]:secresind2[i]+spreads[i+len(secresind1)]] = minarr[minind]
            Ypq_f[secresind2[i]-spreads[i+len(secresind1)]:secresind2[i]+spreads[i+len(secresind1)]] = 10**(np.nanmean(np.log10(abs(minarr))))
        #Ypq_f[secresind1[i]-spreads[i]:secresind1[i]+spreads[i]] = np.mean(minarr)
        else:
            Ypq_f[secresind2[i]] = 0.5+0.5j
######################################################################################################################                
                    
    limit_ind = np.where(freq >= freqlim)[0]
    limit_indr = np.where(freqn >= freqlim)[0]

    
    
    Ypq_f[limit_ind] = 0
    Yhk_f[limit_ind] = 0
    Ya_f[limit_indr] = 0
    
    pq_f = np.fft.ifft(Ypq_f,len(p_init))
    hk_f = np.fft.ifft(Yhk_f,len(h_init))
    a_f = np.fft.irfft(Ya_f,len(a_init))
    #print('hk_f',hk_f)
    sini_f = np.abs(pq_f)
    ecc_f = np.abs(hk_f) 
    #print('ecc',ecc_f)

    h_mom = np.sqrt(a_f*(1-ecc_f**2))*np.cos(np.arcsin(sini_f))+1/2/a_f
    
    #print(secresind1,secresind2,1/np.array(freq1),1/np.array(freq2))
    pc5 = int(n*0.05)
    
    outputs =  np.array([np.nanmean(a_init),np.nanmean(e_init),np.nanmean(np.sin(inc_init)),np.nanmean(np.sqrt(a_init*(1-e_init**2))),np.nanmean(a_init),np.mean(ecc_f[pc5:-pc5]),np.nanmean(sini_f[pc5:-pc5]),np.nanmean(h_mom)])
    outputs =  np.array([np.nanmedian(a_init),np.nanmedian(e_init),np.nanmedian(np.sin(inc_init)),np.nanmedian(np.sqrt(a_init*(1-e_init**2))),np.nanmedian(a_init),np.median(ecc_f[pc5:-pc5]),np.nanmedian(sini_f[pc5:-pc5]),np.nanmedian(h_mom)])
    if debug == True:
        return outputs,Yhk_f,Ypq_f,g,s,freq1,freq2,spreads,freq_dist_lims,freq,n,dt
    #outputs =  np.array([np.nanmedian(a_init),np.nanmedian(e_init),np.nanmedian(np.sin(inc_init)),np.nanmedian(np.sqrt(a_init*(1-e_init**2))),np.nanmedian(a_f),np.median(ecc_f),np.nanmedian(sini_f),np.nanmedian(h_mom)])
    #windows=5
    error_list = np.zeros((windows,4))
    ds = int(len(t_init)/(windows+1))
    for j in range(windows):
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
    
        if small_planets_flag:
            freq1 = [(g2),(g3),(g4),(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,z8,z9]
            freq2 = [(s2),(s3),(s4),(s6),(s7),(s8),z1,z2,z3,z6,z8,z9]
        else:
            freq1 = [(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,z8,z9]
            freq2 = [(s6),(s7),(s8),z1,z2,z3,z6,z8,z9]
            
        if small_planets_flag:
            freq1 = [(g2),(g3),(g4),(g5),(g6),(g7),(g8)]
            freq2 = [(s2),(s3),(s4),(s6),(s7),(s8)]
        else:
            freq1 = [(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,z8,z9]
            freq2 = [(s6),(s7),(s8),z1,z2,z3,z6,z8,z9]
                   
        if small_planets_flag:
            freq1 = [(g2),(g3),(g4),(g5),(g6),(g7),(g8),g-z8,g-z9,g-z10,z11,z12,z13]
            freq2 = [(s2),(s3),(s4),(s6),(s7),(s8),s-z8,s-z9,s-z10,z14]
        else:
            freq1 = [(g5),(g6),(g7),(g8),z1,z2,z3,z4,z5,z7,g-z8,g-z9,g-z10,z11,z12,z13]
            freq2 = [(s6),(s7),(s8),z1,z2,z3,z6,z8,z9,s-z8,s-z9,s-z10,z14]
    
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
                while ((freq[secresind1[i]+spreads[i]]/freq[secresind1[i]-spreads[i]])) < spread_dist1:
                    spreads[i] = spreads[i]+1   
                    if secresind1[i]+spreads[i] >= len(freq):
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
                while ((freq[secresind1[i]-freq_dist_lims[i]]/freq[secresind1[i]+freq_dist_lims[i]])) < freq_dist1:
                    freq_dist_lims[i] = freq_dist_lims[i]+1
                    if secresind1[i]+freq_dist_lims[i] >= len(freq):
                         break
                    
        for i in range(len(secresind2)):
            if freq[secresind2[i]] > 0:
                while ((freq[secresind2[i]+spreads[i+len(secresind1)]]/freq[secresind2[i]-spreads[i+len(secresind1)]])) < spread_dist2:
                #print(i,((freqn[secresind2[i]-spreads[i+len(secresind1)]])/(freqn[secresind2[i]+spreads[i+len(secresind1)]])))
                    spreads[i+len(secresind1)] = spreads[i+len(secresind1)]+1
                    if secresind2[i]+spreads[i+len(secresind1)] >= len(freq):
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
                while ((freq[secresind2[i]-freq_dist_lims[i+len(secresind1)]]/freq[secresind2[i]+freq_dist_lims[i+len(secresind1)]])) < freq_dist2:
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
        if i%50==0:
            print(i)
        #archive = rebound.SimulationArchive(fullfile)
        windows=5
        data_line = prop_calc(objname,filename,windows=windows)
        #data_line = prop_calc(str(i),filename)
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
        windows = 9
        for i in range(windows):
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
        data_line = prop_calc(objname,fullfile,windows)
        data_df = pd.DataFrame(data_line,columns = column_names)
        data_df.to_csv('../data/Single/'+objname+'/'+objname+'_prop_elem.csv')
        
                       

        
