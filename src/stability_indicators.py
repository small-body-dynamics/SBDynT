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
            print('Clone_RMS_a: (' , self.Clone_RMS_a < self.flag_limits['Clone_RMS_a'], '), ', 
                  self.Clone_RMS_a, sign ,self.flag_limits['Clone_RMS_a'])
          
        if self.Clone_RMS_e != None:  
            sign = ' < ' if self.Clone_RMS_e < self.flag_limits['Clone_RMS_a'] else ' > '
            print('Clone_RMS_e: (' , self.Clone_RMS_a < self.flag_limits['Clone_RMS_e'], '), ', 
                  self.Clone_RMS_e, sign ,self.flag_limits['Clone_RMS_e'])
            
        if self.Clone_RMS_sinI != None:
            sign = ' < ' if self.Clone_RMS_sinI < self.flag_limits['Clone_RMS_sinI'] else ' > '
            print('Clone_RMS_sinI: (' , self.Clone_RMS_sinI < self.flag_limits['Clone_RMS_sinI'], '), ',
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
                ci.Distance_metric = hcm_calc(pe_obj.proper_elements['a'], pe_obj.proper_errors['RMS_a'], 
                                          pe_obj.proper_errors['RMS_e'], pe_obj.proper_errors['RMS_sinI'])
            except Exception as err:
                logmessage = err
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

def power_prop_calc(t, e, I, omega, Omega, size = 3, pe_obj = None):

    #e = ci.sb_elems[1]; I = ci.sb_elems[2]
    #omega = ci.sb_elems[3]; Omega = ci.sb_elems[4]

    if pe_obj != None and len(pe_obj.planet_freqs) > 0:
        gs_dict = pe_obj.planet_freqs
        
    else:
        gs_dict = {'g5': 3.299e-6 ,'g6': 2.197e-5, 'g7': 2.398e-6, 'g8': 5.022e-7, 's6': -2.032e-5, 's7': -2.309e-6, 's8': -5.3395e-7, 'g2': 7.34474/1296000, 'g3': 17.32832/1296000, 'g4': 18.00233/1296000, 's2': -6.57080/1296000, 's3': -18.74359/1296000, 's4': -17.63331/1296000}

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

    g_idx, g, local_power_all, protect_g_bins = prop_elem.find_local_max_windowed(freq_s, power_hk, window_half_dex=0.02, window_protect_dex=0.15)

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
            
    s_idx, s, local_power_all, protect_s_bins = prop_elem.find_local_max_windowed(freq_s, power_pq, window_half_dex=0.02, window_protect_dex=0.15)
    
    if size == 1:
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
        

            
