import numpy as np
import pandas as pd
import rebound
import sys
import os
import functools
import json

#internal modules
import tools
import prop_elem
import plotting_scripts



class stability_indicators:
    """
    The class object built for containing the stability indicator results. 

    Parameters:
        # Metadata Parameters
        - des (str): The name of the small body being analyzed. 
        - flag_limits (dict): The limits for each indicator at which the 
          indicator transitions from stable to unstable. Used by the print_results() function.

        # Result parameters
        
        - ACFI (float): The ACFI indicator value for the small body. Should be a float between 
          0 and 1. 1 represents perfect correlation and stability and 0 represents complete 
          non-correlation and instability. 
        - Entropy (float): The Entropy indicator value for this small body. Should be a float 
          between 0 and 1, with values closer to 1 representing stability.
        - Power (float): The Power indicator value for this small body. A float between 0 and 1. 
          Values close to 1 indicate stability, and values closer to 0 indicate instability. 
        - Distance_metric (float): The Distance_metirc indicator value for this small body. Is 
          only computed if a pe_obj is submitted to compute_stability. Values are reported in m/s. 
          A value < 10 m/s indciates stability, 10 < D.M. < 100 indicates metastability, and 
          D.M. > 100 indicates instability. 
        - Clone_RMS_a (float): The Root mean square of the semi-major axis for the clones compared 
          to the best-fit orbit. Requires the user to include clone_elems for comparison to the 
          best-fit orbit. Reported in units of AU. 
        - Clone_RMS_e (float): The Root mean square of the eccentricity for the clones compared to 
          the best-fit orbit. Requires the user to include clone_elems for comparison to the 
          best-fit orbit.
        - Clone_RMS_sinI (float): The Root mean square of the sine of the inclination for the 
          clones compared to the best-fit orbit. Requires the user to include clone_elems for 
          comparison to the best-fit orbit.

        # Output arrays: If output_arrays = True, than these parameters are filled. Required for 
          the plotting functions.
        
        - t (1D np.array): The array of times corresponding to the orbital elements. 
        - sb_elems (2D np.array): The orbital elements for the best-fit orbit of the small body. 
          Contains [a, e, I, omega, Omega].
        - clone_elems (3D np.array): The orbital elements for the clone orbits of the small body. 
          Has shape (#Clones, 5, len(simulation)).


        # Functions:

        - print_results(): Prints out the results of the stability_indicator analysis in an 
          easy-to-read format. 
        - plot_ACFI(): Calls the plot_ACFI function from the plotting_scripts.py file. Requires 
          output_arrays to have been called previously.
        - plot_entropy(): Calls the plot_entropy function from the plotting_scripts.py file. 
          Requires output_arrays to have been called previously.
        - plot_power(): Calls the plot_power function from the plotting_scripts.py file. Requires 
          output_arrays to have been called previously.
        - plot_clones(): Calls the plot_clones function from the plotting_scripts.py file. Requires 
          output_arrays to have been called previously.
        
    """
    
    def __init__(self, des=''):

        self.des = des
        self.flag_limits = {'ACFI' : 0.75, 'Entropy' : 0.95, 'Power': 0.9, 'Distance_metric': (10,100), 
                            'Clone_RMS_a': 0.01, 'Clone_RMS_e': 0.01, 'Clone_RMS_sinI': 0.01}

        
        self.ACFI = None
        self.Entropy = None
        self.Power = None
        self.Distance_metric = None
        self.Clone_RMS_a = None
        self.Clone_RMS_e = None
        self.Clone_RMS_sinI = None
        self.scattered = {'scattered': False, 'scat_time': np.inf, 'Max delta-E': 0, 'qlim': 0, 
                          'Qlim': np.inf, 'pcrossing_flag': False, 'qmin': 0, 'Qmax': 0}

        self.t = []
        self.sb_elems = []
        self.clone_elems = []

    def print_results(self):
        print('Small Body:' + str(self.des) + ', Stability Indicator Results')
        if self.ACFI != None:
            sign = ' < ' if self.ACFI < self.flag_limits['ACFI'] else ' > '
            print('ACFI: (' , self.ACFI < self.flag_limits['ACFI'], '), ', self.ACFI, sign ,self.flag_limits['ACFI'])
        else:
            print('ACFI: Undefined')
                
        if self.Entropy != None:
            sign = ' < ' if self.Entropy < self.flag_limits['Entropy'] else ' > '
            print('Entropy: (' , self.Entropy < self.flag_limits['Entropy'], '), ', self.Entropy, sign ,
                  self.flag_limits['Entropy'])
        else:
            print('Entropy: Undefined')
                
        if self.Power != None:
            sign = ' < ' if self.Power < self.flag_limits['Power'] else ' > '
            print('Power: (' , self.Power < self.flag_limits['Power'], '), ', self.Power, sign ,
                  self.flag_limits['Power'])
        else:
            print('Power: Undefined')
                
        if self.Distance_metric != None:
            if self.Distance_metric < self.flag_limits['Distance_metric'][0]:
                print('Distance Metric: Stable,', self.Distance_metric, ' < ', 
                      self.flag_limits['Distance_metric'][0], 'm/s')
            elif self.Distance_metric < self.flag_limits['Distance_metric'][1]:
                print('Distance Metric: Metastable,', self.flag_limits['Distance_metric'][0], ' < ', 
                      self.Distance_metric, ' < ', 
                      self.flag_limits['Distance_metric'][1], 'm/s')
            elif self.Distance_metric > self.flag_limits['Distance_metric'][1]:
                print('Distance Metric: Unstable,', self.Distance_metric, ' > ', 
                      self.flag_limits['Distance_metric'][1], 'm/s')
            else:
                print('Distance Metric: Undefined')
        else:
            print('Distance Metric: Undefined')

        if self.Clone_RMS_a != None:
            sign = ' < ' if self.Clone_RMS_a < self.flag_limits['Clone_RMS_a'] else ' > '
            print('Clone_RMS_a: (' , self.Clone_RMS_a > self.flag_limits['Clone_RMS_a'], '), ', 
                  self.Clone_RMS_a, sign ,self.flag_limits['Clone_RMS_a'])
          
        if self.Clone_RMS_e != None:  
            sign = ' < ' if self.Clone_RMS_e < self.flag_limits['Clone_RMS_a'] else ' > '
            print('Clone_RMS_e: (' , self.Clone_RMS_a > self.flag_limits['Clone_RMS_e'], '), ', 
                  self.Clone_RMS_e, sign ,self.flag_limits['Clone_RMS_e'])
            
        if self.Clone_RMS_sinI != None:
            sign = ' < ' if self.Clone_RMS_sinI < self.flag_limits['Clone_RMS_sinI'] else ' > '
            print('Clone_RMS_sinI: (' , self.Clone_RMS_sinI > self.flag_limits['Clone_RMS_sinI'], '), ',
                  self.Clone_RMS_sinI, sign ,self.flag_limits['Clone_RMS_sinI'])



    def plot_ACFI(self):
        plotting_scripts.plot_ACFI(self)

    def plot_entropy(self):
        plotting_scripts.plot_entropy(self)
            
    def plot_power(self, pe_obj = None):
        plotting_scripts.plot_power(self, pe_obj = pe_obj)
            
    def plot_clones(self):
        plotting_scripts.plot_clone_osc(self)


        
            


    
def compute_stability(des=None,times=[], sb_elems=[], clones=0, pe_obj=None, 
                      clone_elems=[], output_arrays=False,logfile=False):
    """
    Compute stability indicators for the given orbital elements.

    Parameters:
        times (str): Name/designation of the celestial body as contained in the simulation archive.
        sb_elems (2D numpy array): 
        clones (int): Number of clones contained in clone_elems to include in clone stability analysis
        pe_obj (proper_element_class object): A proper_element_class object which contains numerical 
            uncertainties for the proper element computation. This is used to compute the Distance 
            Metric stability indicator.
        clone_elems (3D numpy array): 
        output_arrays (boolean): If True, saves sb_elems and clone_elems to variables in the 
            stability_indicators class object. Essential for visualization options.
        logfile (boolean or string, optional): if False, only critical error messages are printed
          to screen; if a string is passed, log messages are printed to screen (if set to 'screen') or
          to a file with the path given by the string.

    Returns:
        flag (integer): 0 for failure, 1 for success
        stability_indicators: An stability_indicators.stability_indicators class object, with parameters 
            filled according to the available data which have been provided. Providing no proper elements 
            object or clone elements will cause those results to be left blank/empty. See the 
            stability_indicators class above to see what variables are included in the file. 
    """ 

    if(des==None):
        print("must provide a designation to stability_indicators.compute_stability")
        return 0, None
    
    ci = stability_indicators(des)

    if len(sb_elems) == 0:
        logmessage='The sb_elems variable is empty. Call the function again with the sb_elems variable defined\n'
        logmessage += "failed at stability_indicators.compute_stability"
        tools.writelog(logfile,logmessage)
        if(logfile != 'screen'):
            print(logmessage)            
        return 0, ci
        
    if output_arrays:
        ci.t = times
        ci.sb_elems = sb_elems
        ci.clone_elems = clone_elems

    try:
        if(len(sb_elems) > 0):
            a_arr = sb_elems[0]
            e_arr = sb_elems[1]
            I_arr = sb_elems[2]
            o_arr = sb_elems[3]
            O_arr = sb_elems[4]

            if(len(times) != len(a_arr)):
                logmessage = 'The times variable is not equal in length to the sb_elems variable.\n'
                logmessage += "Skipping computing the scattering flags\n"
                tools.writelog(logfile,logmessage)
            else:
                scat_results = prop_elem.check_scatter(times,a_arr,e_arr)
                ci.scattered = scat_results
                 
            ci.ACFI = ACFI_calc(a_arr)
            ci.Entropy = entropy_calc(a_arr, e_arr, I_arr)
            ci.Power = power_prop_calc(times, e_arr, I_arr, o_arr, O_arr, size = 5, pe_obj = pe_obj)[0]

            
            if(clones > 0):
                if(len(clone_elems) == 0):
                    logmessage = 'clones > 0, but clone_elems is not provided.\n'
                    logmessage += 'Please supply the clone_elems variable to compute the RMS indicators\n'
                    tools.writelog(logfile,logmessage)
                else:
                    diff_a = []; diff_e = []; diff_I = []
                    for i in range(clones):
                        if(i >= len(clone_elems)):
                            logmessage = str(i) + 'is larger than the number of clones in clone_elems\n'
                            logmessage += 'which is ' + str(len(clone_elems)) + 'Stopping iterating over clone_elems\n'
                            tools.writelog(logfile,logmessage)
                            break
                        diff_a.append(((a_arr - clone_elems[i,0])/np.mean(a_arr))**2)
                        diff_e.append((e_arr - clone_elems[i,1])**2)
                        diff_I.append((np.sin(I_arr) - np.sin(clone_elems[i,2]))**2)

                    diff_a = np.array(diff_a); diff_e = np.array(diff_e); diff_I = np.array(diff_I)
                    
                    ci.Clone_RMS_a = np.sqrt(np.nanmean(diff_a))
                    ci.Clone_RMS_e = np.sqrt(np.nanmean(diff_e))
                    ci.Clone_RMS_sinI = np.sqrt(np.nanmean(diff_I))

        if(pe_obj != None):
            try:
                ci.Distance_metric = prop_elem.hcm_calc(pe_obj.proper_elements['a'][0], pe_obj.proper_errors['RMS_a'][0], 
                                          pe_obj.proper_errors['RMS_e'][0], pe_obj.proper_errors['RMS_sinI'][0])
            except Exception as err:
                logmessage = str(err)
                logmessage += '-> pe_obj.proper_elements and/or pe_obj.proper_errors does not contain valid\n'
                logmessage += 'inputs to compute the Distance metric.\n'
                tools.writelog(logfile,logmessage)
            
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno

        error_message = "An error occurred in at line "+str(line_number)

        print(error_message)
        print(e)
        
    return 1, ci
        




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


def pearsonr2(x, y):
    # Assumes inputs are DataFrames and computation is to be performed
    # pairwise between columns. We convert to arrays and reshape so calculation
    # is performed according to normal broadcasting rules along the last axis.
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




def entropy_calc(a,e,inc):
    try:
        hs = np.sqrt(a*(1-e**2))
        bins = int(len(a)/10)
        hs_nonan = hs[~np.isnan(hs)]
        hist,edges = np.histogram(hs_nonan,bins=bins)
        hist = hist / hist.sum()
        baseline = np.log10(bins)
        entropy = -np.sum(hist * np.log10(hist+1e-12))
    except Exception as error:
        print(error)
        return 1
    
    return entropy/baseline




def power_prop_calc(t, e, I, omega, Omega, size = 3, pe_obj = None):

    if(pe_obj != None and len(pe_obj.planet_freqs) > 0):
        gs_dict = pe_obj.planet_freqs
    else:
        gs_dict = {'g5': 3.299e-6 ,'g6': 2.197e-5, 'g7': 2.398e-6, 'g8': 5.022e-7, 
                   's6': -2.032e-5, 's7': -2.309e-6, 's8': -5.3395e-7, 'g2': 7.34474/1296000, 
                   'g3': 17.32832/1296000, 'g4': 18.00233/1296000, 's2': -6.57080/1296000, 
                   's3': -18.74359/1296000, 's4': -17.63331/1296000}

    g_arr = []; s_arr = []

    for key, val in gs_dict.items():
        if 'g' in key:
            g_arr.append(val)
        if 's' in key:
            s_arr.append(val)
    
    hk = e*np.cos(omega+Omega) + 1j*e*np.sin(omega+Omega)
    pq = np.sin(I)*np.cos(Omega) + 1j*np.sin(I)*np.sin(Omega)

    dt = abs(t[1]-t[0])
    freq_s = np.fft.fftfreq(len(e),d=dt)

    Yhk_s = np.fft.fft(hk)
    power_hk = np.abs(Yhk_s)**2

    power_hk0 = power_hk.copy()

    j = 0
    dist = 2
    for i in g_arr:
        ind = np.argmin(abs(freq_s-i))
        if j==0 or j==1:
            mult = 100
        else:
            mult = 20
                    
        power_hk[ind-dist:ind+dist+1] = power_hk[ind-dist:ind+dist+1]/mult
        j += 1

    short_period_g = (1/abs(np.array(gs_dict['g6']*4)))
    short_period_s = (1/abs(np.array(gs_dict['s6']*4)))
    short_period = 1/abs(2*gs_dict['g6']-2*gs_dict['s6'])
    short_period = min(short_period,short_period_g)
    short_period = min(short_period,short_period_s)
    short_ind = np.where(1/abs(freq_s) < short_period/4)[0]
    power_hk[short_ind] = power_hk[short_ind]/5

    window_dex = 0.15  
    n_bins = len(freq_s)

    local_power = np.zeros(n_bins)
            
    window_half_dex = window_dex / 2

    n_bins = len(power_hk)

    g_idx, g, local_power_all, protect_g_bins = prop_elem.find_local_max_windowed(freq_s, power_hk, 
                                                    window_half_dex=0.02, window_protect_dex=0.15)

    Ypq_s = np.fft.fft(pq)
    power_pq = np.abs(Ypq_s)**2
    
    power_pq0 = power_pq.copy()

    j=0
    for i in s_arr:
        if j==0 or j==1:
            mult = 100
        else:
            mult = 20
        ind = np.argmin(abs(freq_s-i))
        power_pq[ind-dist:ind+dist+1] = power_pq[ind-dist:ind+dist+1]/mult
        j += 1

    window_dex = 0.12  
    n_bins = len(freq_s)
            
    power_pq[short_ind] = power_pq[short_ind]/10
            
    s_idx, s, local_power_all, protect_s_bins = prop_elem.find_local_max_windowed(freq_s, power_pq, 
                                                        window_half_dex=0.02, window_protect_dex=0.15)
    
    if(size == 1):
        top3_hk = power_hk0[g_idx]
        top3_pq = power_pq0[s_idx]
        if g_idx == 0:
            hk_percent = top3_hk/np.sum(power_hk0)
        else:
            hk_percent = top3_hk/np.sum(power_hk0[1:])
        if s_idx == 0:
            pq_percent = top3_pq/np.sum(power_pq0)
        else:
            pq_percent = top3_pq/np.sum(power_pq0[1:])
    else:
        if g_idx < size//2 + 1:
            if g_idx == 0:
                indlow1 = 0
                indhigh1 = 0 + size
            else:
                indlow1 = 1
                indhigh1 = 1 + size
        elif len(power_hk) - g_idx <= size//2:
            indlow1 = len(power_hk0) - size 
            indhigh1 = len(power_hk0) 
        else:
            indlow1 = g_idx - size//2
            indhigh1 = g_idx + size//2 + 1
    
        top3_hk = np.sum(power_hk[indlow1:indhigh1])
        if g_idx == 0:
            hk_percent = top3_hk/np.sum(power_hk0)
        else:
            hk_percent = top3_hk/np.sum(power_hk0[1:])

        if s_idx < size//2 + 1:
            if s_idx == 0:
                indlow2 = 0
                indhigh2 = 0 + size
            else:
                indlow2 = 1
                indhigh2 = 1 + size
        elif len(power_pq0) - s_idx <= size//2:
            indlow2 = len(power_pq0) - size 
            indhigh2 = len(power_pq0) 
        else:
            indlow2 = s_idx - size//2
            indhigh2 = s_idx + size//2 + 1
        top3_pq = np.sum(power_pq0[indlow2:indhigh2])
        if s_idx == 0:
            pq_percent = top3_pq/np.sum(power_pq0)
        else:
            pq_percent = top3_pq/np.sum(power_pq0[1:])

    power_percent = np.mean((hk_percent, pq_percent))
    
    return power_percent, g, s, gs_dict

