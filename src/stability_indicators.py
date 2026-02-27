import numpy as np
import pandas as pd
import rebound
#import integrate_multi as im
import sys
import os
import functools
#import schwimmbad
import run_reb
import json
from prop_elem import *

from sbdynt import *


class stability_indicators:

    def __init__(self):

        self.ACFI = None
        self.Entropy = None
        self.Power = None
        self.Distance_metric = None
        self.Clone_RMS_a = None
        self.Clone_RMS_e = None
        self.Clone_RMS_sinI = None
        self.scattered = {'scattered': False, 'scat_time': np.inf, 'Max delta-E': 0, 'qlim': 0, 'Qlim': np.inf, 'pcrossing_flag': False, 'qmin': 0, 'Qmax': 0}


    
def compute_stability(times = [], sb_elem = [], clones=0, pe_obj = None, clone_elems = []):
    ci = stability_indicators()
    try:
        if len(sb_elem) > 0:
            a_arr = sb_elem[0,:]
            e_arr = sb_elem[1,:]
            I_arr = sb_elem[2,:]
            o_arr = sb_elem[3,:]
            O_arr = sb_elem[4,:]

            scat_results = check_scatter(times,a_arr,e_arr)
            ci.scattered = scat_results
                 
            ci.ACFI = ACFI_calc(a_arr)
            ci.Entropy = entropy_calc(a_arr, e_arr, I_arr)
            ci.Power = power_calc(e_arr, I_arr, o_arr, O_arr, size=5)

            if clones > 0:
                if len(clone_elems) == 0:
                    print('clones > 0, but clone_elems is not provided. Please supply the clone_elems to copute the RMS indicators')

                else:
                    for i in range(clones):
                        #print(a_arr, clone_elems[i,0])
                        diff_a = ((a_arr - clone_elems[i,0])/np.mean(a_arr))**2
                        diff_e = (e_arr - clone_elems[i,1])**2
                        diff_I = (np.sin(I_arr) - np.sin(clone_elems[i,2]))**2
                
                    ci.Clone_RMS_a = np.sqrt(np.nanmean(diff_a))
                    ci.Clone_RMS_e = np.sqrt(np.nanmean(diff_e))
                    ci.Clone_RMS_sinI = np.sqrt(np.nanmean(diff_I))

        if pe_obj != None:
            ci.Distance_metric = hcm_calc(pe_obj.proper_elements['a'], pe_obj.proper_errors['RMS_a'], 
                                          pe_obj.proper_errors['RMS_e'], pe_obj.proper_errors['RMS_sinI'])

            
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno

        error_message = "An error occurred in at line "+str(line_number)

        print(error_message)
        print(e)
        
    return ci
        

#def rms_diff_calc(a_arr, e_arr, I_arr, clones=0):
    

def autocorr(x):
    result = numpy.correlate(x, x, mode='full')
    return result[result.size//2:]    

def get_Kchao1(a,e,bins=100):
    try:
        hist, xedges, yedges = np.histogram2d(a, e, bins=[bins, bins])
    except:
        return 0.001
    K = len(np.where(hist > 0)[0])
    f1 = len(np.where(hist == 1)[0])
    f2 = len(np.where(hist == 2)[0])
    #print(K, f1, f2)
    
    if f2 == 0:
        return K + f1*(f1-1)/2
    else:
        return K + f1**2/2/f2

def eff_sample(x):
    x = x - np.mean(x)
    rho1 = np.corrcoef(x[:-1],x[1:])[0,1]
    Neff = len(x)*(1-rho1)/(1+rho1) if rho1<1 else 1
    return max(1,Neff)

def entropy_baseline(a,e,bins=100):
    hist, xedges, yedges = np.histogram2d(a, e, bins=[bins, bins])
    occupied = hist > 0

    M = np.prod(hist.shape)
    #M = occupied.sum() + dilation margin
    Neff = int(0.5*(eff_sample(a) + eff_sample(e)))
    Kexp = M*(1-np.exp(-Neff/M))
    H_stable = np.log10(Kexp)

    return H_stable

def integrate_chaos(objname, tmax1=5e4, tout1=1e3, tmax2=1e6, tout2=2e4, objtype='Single'):
    """
    Integrate the given archive.bin file which has been prepared.

    Parameters:
        objname (str or int): Index of the celestial body in the names file.
        tmax (float): The number of years the integration will run for. Default set for 10 Myr.
        tmin (float): The interval of years at which to save. Default set to save every 1000 years.  
        objtype (str): Name of the file containing the list of names, and the directory containing the archive.bin files. 

    Returns:
        None
        
    """   
    try:
        #print(objname)
        # Construct the file path
        file = '../data/' + objtype + '/' + str(objname)
        
        # Load the simulation from the archive
        #print(file)
        if os.path.isfile(file+'/archive_init.bin') == False:
            print('No init file') 
            return 0
        sim2 = rebound.Simulation(file + "/archive_init.bin")
        #print('read init')
        sim2.integrator = 'whfast'
        #print(sim2.integrator)
        sim2.ri_whfast.safe_mode = 0

        try:
            earth = sim2[0].particles['earth']
            small_planets_flag = True
        except:
            small_planets_flag = False
        
        clones_flag = False
        #print(small_planets_flag,len(sim2.particles))
        if small_planets_flag and len(sim2.particles) > 10:
            clones_flag = True
            numclones = len(sim2.particles)-10
        elif not small_planets_flag and len(sim2.particles) > 6:
            #print('yes')
            clones_flag = True
            numclones = len(sim2.particles)-6
        else:
            #print('no')
            numclones = 0
        numparts = len(sim2.particles)
        #if numparts <= 6:
        #    print(objname)
        for i in range(numclones):
            sim2.remove(numparts-i-1)
            
        sim2.init_megno()
        #print('init megno')
        #print(sim2.integrator
        
        sim1 = run_reb.run_simulation(sim2, tmax=tmax1, tout=tout1, filename=file + "/archive_chaos_short.bin", deletefile=False,integrator=sim2.integrator)
        print('short made')
        
        
        sim4 = rebound.Simulation(file + "/archive_init.bin")

        #print(clones_flag)
        if clones_flag:
            print(objname)
            sim4.integrator = 'whfast'
            #sim4.init_megno()
            sim3 = run_reb.run_simulation(sim4, tmax=tmax2, tout=tout2, filename=file + "/archive_chaos_long.bin", deletefile=False,integrator=sim2.integrator)
        # Uncomment if you need to print simulation information
        # print(sim2, sim2.particles)
        
    except Exception as error:
        # Raise a specific exception with an informative error message
        #raise ValueError(f"Failed to integrate {objtype} {objname}. Error: {error}")
        print(error)
        return 0

    # Rest of the integration code

    return 1

mn = 1.024e26
M = 1.989e30
G = 6.674e-11
def D_diff(q,an):
    return 8/5/np.pi*mn/M*np.sqrt(G*M*an)*np.exp(-((q/an)**2)/2)

from scipy import stats

from scipy.special import betainc

# import cupy as xp
# from cupyx.scipy.special import betainc

def pearsonr2(x, y):
    # Assumes inputs are DataFrames and computation is to be performed
    # pairwise between columns. We convert to arrays and reshape so calculation
    # is performed according to normal broadcasting rules along the last axis.
    #x = x.T[:, np.newaxis, :]
    #y = y.T
    n = len(x)

    # Compute Pearson correlation coefficient. We can't use `cov` or `corrcoef`
    # because they want to compute everything pairwise between rows of a
    # stacked x and y.
    xm = np.mean(x)
    ym = np.mean(y)
    cov = np.sum((x - xm) * (y - ym))/(n-1)
    sx = np.std(x)
    sy = np.std(y)
    rho = cov/(sx * sy)

    # Compute the two-sided p-values. See documentation of scipy.stats.pearsonr.
    ab = n/2 - 1
    x = (abs(rho) + 1)/2
    p = 2*(1-betainc(ab, ab, x))
    return rho, p

def ACFI_calc(a):
    atenth = int(len(a)/10)
    num = atenth
    if num > 250:
        num = 250
    Rs = np.zeros(num)
    for i in range(num):
        Rs[i] = pearsonr2(a[0:num], a[atenth*7+i:atenth*7+num+i])[0]
    ACFI = np.sum(abs(Rs)>0.05)/num
    return ACFI

def power_calc(e,inc,omega,Omega, size=3):
    from scipy import stats

    
    hk = e*np.cos(Omega+omega)+1j*e*np.sin(Omega+omega)
    pq = np.sin(inc)*np.cos(Omega)+1j*np.sin(inc)*np.sin(Omega)

    Fhk = np.abs(np.fft.fft(hk))**2
    Fpq = np.abs(np.fft.fft(pq))**2

    Fhk_sorted = np.sort(Fhk[1:])[::-1]
    Fpq_sorted = np.sort(Fpq[1:])[::-1]
    top3_hk = np.sum(Fhk_sorted[:size])
    top3_pq = np.sum(Fpq_sorted[:size])

    '''

    maxindhk = np.argmax(Fhk[1:]) + 1
    maxindpq = np.argmax(Fpq[1:]) + 1

    if size == 1:
        top3_hk = np.max(Fhk[1:])
        top3_pq = np.max(Fpq[1:])
    else:
        if maxindhk < size//2 + 1:
            indlow = 1
            indhigh = 1 + size
        elif len(Fhk) - maxindhk <= size//2:
            indlow = len(Fhk) - size - 1
            indhigh = len(Fhk) - 1
        else:
            indlow = maxindhk - size//2
            indhigh = maxindhk + size//2 + 1
    
        top3_hk = np.sum(Fhk[indlow:indhigh])

        if maxindpq < size//2 + 1:
            indlow = 1
            indhigh = 1 + size
        elif len(Fpq) - maxindpq <= size//2:
            indlow = len(Fhk) - size - 1
            indhigh = len(Fhk) - 1
        else:
            indlow = maxindpq - size//2
            indhigh = maxindpq + size//2 + 1
        top3_pq = np.sum(Fpq[indlow:indhigh])
    '''
    power_percentage = np.mean((top3_hk/np.sum(Fhk_sorted),top3_pq/np.sum(Fpq_sorted)))
    return power_percentage

def power_prop_calc(num, y, z, e,inc,omega,Omega, g_arr, s_arr, g_inds, s_inds, small_planets_flag, t, size, debug = False):
    if num%5 == 0:
        if y%20 == 0 and z == 0:
            print('a:',num, '/', 81, 'inc:',y, '/', 41)

    
    hk = e*np.cos(omega+Omega) + 1j*e*np.sin(omega+Omega)
    pq = np.sin(inc)*np.cos(Omega) + 1j*np.sin(inc)*np.sin(Omega)

    dt = abs(t[1]-t[0])
    freq_s = np.fft.fftfreq(len(e),d=dt)

    tol_bins = 1
    dist = round(tol_bins*1)

    Yhk_s = np.fft.fft(hk)
    power_hk = np.abs(Yhk_s)**2

    if small_planets_flag:
        num_p = 8
    else:
        num_p = 5

    j=0
    
    for i in g_arr[:num_p]:
        ind = np.argmin(abs(freq_s-i))
        if j==0 or j==1:
            mult = 100
        else:
            mult = 20
        if small_planets_flag == False and j == 3:
            mult = 20
                    
        power_hk[ind-dist:ind+dist+1] = power_hk[ind-dist:ind+dist+1]/mult
        j += 1

    short_period_g = (1/abs(np.array(g_arr[1]*4)))
    short_period_s = (1/abs(np.array(s_arr[0]*4)))
    short_period = 1/abs(2*g_arr[1]-2*s_arr[0])
    short_period = min(short_period,short_period_g)
    short_period = min(short_period,short_period_s)
    short_ind = np.where(1/abs(freq_s) < short_period/4)[0]
    power_hk[short_ind] = power_hk[short_ind]/5

    window_dex = 0.15  
    n_bins = len(freq_s)

    local_power = np.zeros(n_bins)
            
    window_half_dex = window_dex / 2

    n_bins = len(power_hk)

            
    g_idx, g, local_power_all, protect_g_bins = prop_elem.find_local_max_windowed(freq_s, power_hk, window_half_dex=0.02, window_protect_dex=0.15)

    Ypq_s = np.fft.fft(pq)
    power_pq = np.abs(Ypq_s)**2
    #power_pq[abs(1/freq_s) > abs(t[-1])/2] = 0

    j=0
    for i in s_arr[:num_p-1]:
        if j==0 or j==1:
            mult = 100
        else:
            mult = 20
        if small_planets_flag == False and j == 2:
            mult = 100
        ind = np.argmin(abs(freq_s-i))
        power_pq[ind-dist:ind+dist+1] = power_pq[ind-dist:ind+dist+1]/mult
        j += 1

    window_dex = 0.12  
    n_bins = len(freq_s)
            
    power_pq[short_ind] = power_pq[short_ind]/10
            
    s_idx, s, local_power_all, protect_s_bins = prop_elem.find_local_max_windowed(freq_s, power_pq, window_half_dex=0.02, window_protect_dex=0.15)
    
    if size == 1:
        top3_hk = power_hk[g_idx]
        top3_pq = power_pq[s_idx]
        if g_idx == 0:
            hk_percent = top3_hk/np.sum(power_hk)
        else:
            hk_percent = top3_hk/np.sum(power_hk[1:])
        if s_idx == 0:
            pq_percent = top3_pq/np.sum(power_pq)
        else:
            pq_percent = top3_pq/np.sum(power_pq[1:])
    else:
        if g_idx < size//2 + 1:
            if g_idx == 0:
                indlow1 = 0
                indhigh1 = 0 + size
            else:
                indlow1 = 1
                indhigh1 = 1 + size
        elif len(power_hk) - g_idx <= size//2:
            indlow1 = len(power_hk) - size 
            indhigh1 = len(power_hk) 
        else:
            indlow1 = g_idx - size//2
            indhigh1 = g_idx + size//2 + 1
    
        top3_hk = np.sum(power_hk[indlow1:indhigh1])
        if g_idx == 0:
            hk_percent = top3_hk/np.sum(power_hk)
        else:
            hk_percent = top3_hk/np.sum(power_hk[1:])

        if s_idx < size//2 + 1:
            if s_idx == 0:
                indlow2 = 0
                indhigh2 = 0 + size
            else:
                indlow2 = 1
                indhigh2 = 1 + size
        elif len(power_pq) - s_idx <= size//2:
            indlow2 = len(power_pq) - size 
            indhigh2 = len(power_pq) 
        else:
            indlow2 = s_idx - size//2
            indhigh2 = s_idx + size//2 + 1
        top3_pq = np.sum(power_pq[indlow2:indhigh2])
        if s_idx == 0:
            pq_percent = top3_pq/np.sum(power_pq)
        else:
            pq_percent = top3_pq/np.sum(power_pq[1:])

    #if y > 30:
    #    print('inc ind:',y,'g:',g, 1/g,'s:',s,1/s)
    
    power_percent = np.mean((hk_percent, pq_percent))
    if debug:
        return power_percent, g, s, indlow1, indhigh1, indlow2, indhigh2
    return power_percent
    

def entropy_calc(a,e,inc):
    #num_bins_a = 50
    #num_bins_e = 50
            #num_bins_i = 100
            #hist, edges = np.histogramdd((a, e, inc), bins=(num_bins_a, num_bins_e, num_bins_i))
            #hist, a_edges, e_edges = np.histogram2d(a, e, bins=[num_bins_a, num_bins_e])
            #print(a_edges)
            #hist, a_edges = np.histogram(a_clone, bins=num_bins_a)
    
            # Normalize to get probabilities
            #print(hist,hist.sum())
            #hist = hist / hist.sum()

            # Calculate the Shannon entropy
            
            #entropy = -np.sum(hist * np.log10(hist+1e-12))  # adding a small value to avoid log(0)

    #try:
    #    H_star = np.log10(min(get_Kchao1(a,e),num_bins_a*num_bins_e))
    #    H_base = entropy_baseline(a,e)
    #except:
    #    return 0
            
    #entropy = H_star - H_base

    try:
        hs = np.sqrt(a*(1-e**2))
        #Lz = np.cos(inc)*hs
        #Lz_hat = np.cos(inc)*np.sqrt(1-e**2)
        #hs[np.isnan(hs)] = 0
        #inc[np.isnan(inc)] = np.pi/2
        #Lz[np.isnan(Lz)] = 0
        #'''
        bins = int(len(a)/10)
        hs_nonan = hs[~np.isnan(hs)]
        hist,edges = np.histogram(hs_nonan,bins=bins)
        hist = hist / hist.sum()
        baseline = np.log10(bins)
        entropy = -np.sum(hist * np.log10(hist+1e-12))
        #'''
        '''
        bins = int(np.sqrt(len(hs)/10))
        hist,xedges,yedges = np.histogram2d(hs,np.cos(inc),bins=(bins,bins))
        hist = hist / hist.sum()
        baseline = np.log10(bins**2)
        entropy = -np.sum(hist * np.log10(hist+1e-12))
        #'''
    
        '''
        a_s = (a-np.mean(a))/np.std(a)
        Lz_s = (Lz_hat-np.mean(Lz_hat))/np.std(Lz_hat)
        bins = int(np.sqrt(len(hs)/10))
        hist_j,xedges,yedges = np.histogram2d(a_s,Lz_s,bins=(bins,bins))
        hist_j = hist_j / hist_j.sum()
        baseline = np.log10(bins**2)
        entropy_j = -np.sum(hist_j * np.log10(hist_j+1e-12))
        
        hist_1,xedges= np.histogram(a_s,bins=bins**2)
        hist_1 = hist_1/ hist_1.sum()
        baseline = np.log10(bins**2)
        entropy_1 = -np.sum(hist_1 * np.log10(hist_1+1e-12))
        
        hist_2,xedges = np.histogram(Lz_s,bins=bins**2)
        hist_2 = hist_2 / hist_2.sum()
        baseline = np.log10(bins**2)
        entropy_2 = -np.sum(hist_2 * np.log10(hist_2+1e-12))
        entropy = entropy_1 + entropy_2 - entropy_j
        #'''
    except Exception as error:
        print(error)
        #print(hs[:10])
        #print(Lz[:10])
        #print(np.min(hs),np.max(hs))
        #print(np.min(Lz),np.max(Lz))
        return 1

    
    return entropy/baseline

def hcm_calc(a,rmsa,rmse,rmsi):
    G = 6.673e-11
    M = 1.989e30
    n = np.sqrt(G*M/(a*1.496e11)**3)
    return n*a*1.498e11*np.sqrt(5/4*(rmsa/a)**2+2*rmse**2+2*rmsi**2)
    
            
    

def calc_stability(objname,objtype,prop_vals=None):
    #print(objname)
    file = '../data/' + objtype + '/' + str(objname)
    try:
        short_sim = rebound.Simulationarchive(file + "/archive_chaos_short.bin") 
        MEGNO = short_sim[-1].calculate_megno()
    except:
        MEGNO = 0

         
    CloneDist = np.zeros(4)
    GreatestDist = np.zeros(3)

    long_sim = None
    if os.path.exists("../data/"+objtype+"/"+objname+"/archive.bin"):
        sim = rebound.Simulationarchive("../data/"+objtype+"/"+objname+"/archive.bin")
        a = np.zeros(len(sim))
        e = np.zeros(len(sim))
        inc = np.zeros(len(sim))
        try:
            for i in range(len(sim)):
                a[i] = sim[i].particles[-1].a
                e[i] = sim[i].particles[-1].e
                inc[i] = sim[i].particles[-1].inc        

            CloneDist[3] = entropy_calc(entropy)

            ACFI = ACFI_calc(a)
            #autocorr = autocorr(a)
            #half = int(len(autocorr)/2)
            #ACFI = (np.abs(autocorr[half:]) > 0.05).sum()/(len(autocorr)-half)
            #ACFI_flag = (ACFI < 0.5)        
            
        except:
            CloneDist[3] = 0
            ACFI = 0
        try:
            earth = sim[0].particles['earth']
            small_planets_flag = True
            nump = 7
        except:
            small_planets_flag = False
            nump = 4

        if len(sim[0].particles) - nump >=5:
            long_sim = sim
    
    if os.path.exists(file + "/archive_chaos_long.bin") or long_sim != None:
        if long_sim == None:
            long_sim = rebound.Simulationarchive(file + "/archive_chaos_long.bin")

        
        clones_flag = False
        if small_planets_flag and len(long_sim[-1].particles) > 10:
            clones_flag = True
            numclones = len(sim2.particles)-10
            objnum = 9
            
        elif not small_planets_flag and len(long_sim[-1].particles) > 6:
            clones_flag = True
            numclones = len(long_sim[-1].particles)-6
            #print('nuclones',numclones)
            objnum = 5
        else:
            numclones = 0
        diff = np.zeros((numclones,3))   
        sb = long_sim[-1].particles[objnum]
        #print(sb)
            
        a_diffusion = np.zeros(numclones+1)  

        a_p = np.zeros(len(long_sim))
        e_p = np.zeros(len(long_sim))
        i_p = np.zeros(len(long_sim))

        a_n = np.zeros(len(long_sim))
        
        for i in range(len(long_sim)):
            a_p[i] = long_sim[i].particles[objnum].a
            e_p[i] = long_sim[i].particles[objnum].e
            i_p[i] = long_sim[i].particles[objnum].inc
            
            a_n[i] = long_sim[i].particles[objnum-1].a
        

        if a_p[0] >= 30:
            q = (1-e_p)*a_p
            a_diffusion[0] = np.max(D_diff(q*1.496e11,a_n*1.496e11))/1.496e11**2*24*60*60*365
        
        for i in range(numclones):          
            a_clone = np.zeros(len(long_sim))
            e_clone = np.zeros(len(long_sim))
            i_clone = np.zeros(len(long_sim))
            
            for j in range(len(long_sim)):
                a_clone[j] = long_sim[j].particles[objnum+i+1].a
                e_clone[j] = long_sim[j].particles[objnum+i+1].e
                i_clone[j] = long_sim[j].particles[objnum+i+1].inc
            
            #print(abs(sb.a - long_sim[-1].particles[objnum+i].a) > GreatestDist[0])
            if np.max(abs(a_p - a_clone)) > GreatestDist[0]:
                GreatestDist[0] = np.max(abs(np.mean(a_p - a_clone)))
            if np.max(abs(e_p - e_clone)) > GreatestDist[1]:
                GreatestDist[1] = np.max(abs(e_p - e_clone))
            if np.max(abs(i_p - i_clone)) > GreatestDist[2]:
                GreatestDist[2] = np.max(abs(np.sin(i_p) - np.sin(i_clone)))

            diff[i][0] = np.mean(((sb.a - long_sim[-1].particles[objnum+i].a)/sb.a)**2)
            diff[i][1] = np.mean((sb.e - long_sim[-1].particles[objnum+i].e)**2)
            diff[i][2] = np.mean((np.sin(sb.inc) - np.sin(long_sim[-1].particles[objnum+i].inc))**2)

            if a_p[0] >= 30:
                q = (1-e_clone)*a_clone
                a_diffusion[i+1] = np.max(D_diff(q*1.496e11,a_n*1.496e11))/1.496e11**2*24*60*60*365

        
            
        D_metric = np.nanmean(a_diffusion)
        D_std = np.std(a_diffusion)
        #print(sb.a,sb.e,sb.inc)
        #print(long_sim[-1].particles[objnum+i].a,long_sim[-1].particles[objnum+i].e,long_sim[-1].particles[objnum+i].inc)
        CloneDist[0:3] = np.sqrt(np.nanmean(diff,axis=0))
        if np.isnan(CloneDist[0]):
            print(diff)
    else:
        CloneDist = np.array([0,0,0,0])
        GreatestDist = np.array([0,0,0])
        
    prop_err = np.array([0,0,0])
    prop_delta = np.array([0,0,0])
    if os.path.exists("../data/results/"+objtype+"_prop_elem_multi.csv"):
        prop_elem = pd.read_csv("../data/results/"+objtype+"_prop_elem_multi.csv",index_col=1)
        prop_err = [prop_elem.loc[objname]['RMS_err_a']/prop_elem.loc[objname]['PropSMA'],prop_elem.loc[objname]['RMS_err_e'],prop_elem.loc[objname]['RMS_err_sinI']]
        prop_delta = [prop_elem.loc[objname]['Delta_a']/prop_elem.loc[objname]['PropSMA'],prop_elem.loc[objname]['Delta_e'],prop_elem.loc[objname]['Delta_sinI']]
          
    #print(prop_vals)
    if prop_vals != None:
        #print('making prop_err and delta')
        prop_err = np.abs(np.array([prop_vals[-8]/prop_vals[5],prop_vals[-7],prop_vals[-6]]))
        prop_delta = np.abs(np.array([prop_vals[-4]/prop_vals[5],prop_vals[-3],prop_vals[-2]]))    


    M_flag = (MEGNO > 2.5)
    P_flag = (prop_err[0] > 0.01)
    D_flag = (CloneDist[0] > 0.01)
    E_flag = (CloneDist[3] < -0.1)        
    ACFI_flag = (ACFI < 0.5)

    if (D_metric < 0.5*1e-3 and D_std < D_metric):
        Diff_Flag = 0
    elif (D_metric < 5e-3 and D_std < D_metric):
        Diff_flag = 1
    elif D_metric == 0:
        Diff_flag = np.nan
    else:
        Diff_flag = 2
        
    return [str(objname),MEGNO,CloneDist[0],CloneDist[1],CloneDist[2],CloneDist[3],GreatestDist[0],GreatestDist[1],GreatestDist[2],prop_err[0],prop_err[1],prop_err[2],prop_delta[0],prop_delta[1],prop_delta[2],ACFI,D_metric,D_std,a_diffusion,M_flag,P_flag,D_flag,E_flag,ACFI_flag,Diff_flag]
    
if __name__ == "__main__":
    
    from schwimmbad import MPIPool
    #print('schwimmbad in')
    with MPIPool() as pool:
        #print('mpipool pooled')    
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    # Check if the required command-line arguments are provided
        if len(sys.argv) < 2:
            print("Usage: python integrate.py <Filename>")
            sys.exit(1)

        objtype = str(sys.argv[1])
        # Load data file for the given objtype
        names_df = pd.read_csv('../data/data_files/' + objtype + '.csv')
        
        run1 = functools.partial(integrate_chaos, objtype=objtype)

        des = np.array(names_df['Name'])
        #des = np.array(names_df['Name'][:980])
        
        pool.map(run1, des)
        
        run2 = functools.partial(calc_stability, objtype=objtype)
        data = pool.map(run2, des)
        
        chaos_df = pd.DataFrame(data,columns=['Name','MEGNO','Div_RMS_a','Div_RMS_e','Div_RMS_sinI','Entropy_DelH','Delta_a','Delta_e','Delta_sinI','Prop_RMS_a','Prop_RMS_e','Prop_RMS_sinI','Prop_Delta_a','Prop_Delta_e','Prop_Delta_sinI','ACFI','Diffusion Coefficient','Diffusion STD','Diffusion_Clones','MEGNO_flag','Proper_SMA_flag','Clone_SMA_flag','Entropy_flag','AFCI_flag','Diffusion Flag'])
        
        #entropy_flag = np.where(chaos_df['Info Entropy'] < 0.9*np.median(chaos_df['Info Entropy']))[0]
        PSMA_flag = np.where(chaos_df['Prop_RMS_a'] > 0.01)[0]
        PSMA_flag = np.where(chaos_df['Div_RMS_a'] > 0.01)[0]
        MEGNO_flag = np.where(chaos_df['MEGNO'] > 2.5)[0]
        AFCI_flag = np.where(chaos_df['AFCI'] < 0.5)[0]
        
        chaos_df['MEGNO_flag'] = np.zeros(len(data))
        chaos_df['Proper_SMA_flag'] = np.zeros(len(data))
        chaos_df['Entropy_flag'] = np.zeros(len(data))
        chaos_df['Clone_SMA_flag'] = np.zeros(len(data))
        chaos_df['AFCI_flag'] = np.zeros(len(data))
        
        chaos_df['MEGNO_flag'].iloc[MEGNO_flag] = 1
        chaos_df['Proper_SMA_flag'].iloc[PSMA_flag] = 1
        chaos_df['Clone_SMA_flag'].iloc[PSMA_flag] = 1

        entropy_f = np.zeros(len(data))
        earr = np.array(chaos_df['Entropy_DelH'])
        entropy_f[(earr < -0.1) and (earr > -0.2)] = 1
        entropy_f[earr < -0.2] = 2
        
        chaos_df['Entropy_flag'] = entropy_f        
        
        chaos_df['AFCI_flag'].iloc[entropy_flag] = 1
        
        
        chaos_df.to_csv('../data/results/'+objtype+'_chaos.csv')
        

            
