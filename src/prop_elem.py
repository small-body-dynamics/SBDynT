# external modules 
import numpy as np
import pandas as pd
import rebound
import warnings
warnings.filterwarnings("ignore")
import scipy.signal as signal
from scipy.stats import circmean
from scipy.fft import fft, ifft, fftfreq
from scipy.ndimage import convolve1d, gaussian_filter1d
import random
import string
from os import remove
from importlib import resources as impresources


# internal modules
import plotting_scripts
import tools
import PEdata
import run_reb


g2 = 0;g3 = 0;g4 = 0;g5 = 0;g6 = 0;g7 = 0;g8 = 0
s2 = 0;s3 = 0;s4 = 0;s6 = 0;s7 = 0;s8 = 0
g_arr = []
s_arr=[]


class proper_element_class:
    """
    The class object built for containing the synthetic proper element results. 

    Parameters:
        # Metadata Parameters
        - des (str): The name of the small body being analyzed. 
        - planets (list): A list of which planets are included in the analysis.
        - planet_freqs (dict): A dictionary containing the proper frequencies used 
          in the analysis. These typically result from the analysis itself, though users 
          may define their own planetary frequencies if they wish.
        - tmax (float): The total length of time of the simulation
        - tout (float): The output interval of the input orbital data.

        # Result parameters
        - osculating_elements (dict): The osculating elements at time t=0. 
          Contains {a, e, I, omega, Omega, M}.
        - mean_elements (dict): The average of the orbital elements over the full integration. 
          Contains {a, e, sinI, g("/yr), s("/yr), g(rev/yr), s(rev/yr)}. The average precession 
          rates are computed from the median gradient of the varpi and Omega angles. 
        - proper_elements (dict): The proper orbital elements computed by this code. 
          Contains {a, e, sinI, omega, Omega, g("/yr), s("/yr), g(rev/yr), s(rev/yr)}. 
          The proper omega and Omega angles are defined with respect to the time t=0 angles in the 
          filtered time arrays. 
        - proper_errors (dict): The numerical uncertainties associated with the computed proper 
          elements from the windowed error computation. 
          Contains {RMS_a, RMS_e, RMS_sinI, RMS_g("/yr), RMS_s("/yr)}.

        # Flags and Indicators
        - proper_indicators (dict): A dictionary of several helpful values that may be relevant to 
          users hoping to better understand the amplitude or impact of secular dynamics on the small 
          body. Contains {Ecc Mean Flag, sinI Mean Flag, Ecc Mean Indicator, sinI Mean Indicator, Ecc 
          Osculating Amplitude, sinI Osculating Amplitude, Ecc Filtered Amplitude, sinI Filtered 
          Amplitude, Distance Metric}. Descriptions of these indicators and how they can be used are 
          found in Spencer et al. 2026. 
        - scattered (dict): A dictionary with flags and indicators related to identifying whether an 
          object experiences any potential planet crossing or scattering events. 
          Contains {scattered, scat_time, scat_ind, Max delta-E, qlim, Qlim, pcrossing_flag, qmin, Qmax}.
        - family_results (dict, work in progress, not yet tested): An in progress dictionary related to 
          a feature for automatic collisional family candidate identification. Using the Nesvorny 
          familylist.tab file to identify family boundaries, the code runs a brief check to see if the 
          object exists near the collisional center of any collisional families for an asteroid, or the 
          Haumea family collisional center for a TNO. Note that this does not take H-mag into account, 
          and so this is simply a 1st order check to see if the object is within the dynamical region 
          for family membership. Follow-up analysis should be done using the V-shape criterion and other 
          methods to prove family membership.

        
        # Arrays: If output_arrays = True in original function call, time arrays will be saved to these 
        variables within the proper_element_class object.

        - time (1D np.array): The time array provided by the user. 
        - hk_original (1D np.array): Equivalent to k + 1j*h, where h and k are computed from the initial 
          osculating orbital elements provided by the user.
        - pq_original (1D np.array): Equivalent to q + 1j*p, where p and q are computed from the initial 
          osculating orbital elements provided by the user.
        - hk_filtered (1D np.array): Equivalent to kfilt + 1j*hfilt, where hfilt and kfilt are computed 
          from the filtered orbital elements produced by the analysis.
        - pq_filtered (1D np.array): Equivalent to qfilt + 1j*pfilt, where pfilt and qfilt are computed 
          from the filtered orbital elements produced by the analysis.

        - a_original (1D np.array): The initial osculating semi-major axis time array provided by the user.
        - a_filtered (1D np.array): The filtered semi-major axis time array produced by the analysis.

        - t_windows (list of 1D np.arrays): A list of the times used for each window when computing the 
          numerical uncertainty. 
        - hk_windows (list of 1D np.arrays): A list of the filtered k + 1j*h arrays produced for each window 
          when computing the numerical uncertainty. 
        - pq_windows (list of 1D np.arrays): A list of the filtered q + 1j*p arrays produced for each window 
          when computing the numerical uncertainty.
        - a_windows (list of 1D np.arrays): A list of the filtered semi-major axis arrays produced for each 
          window when computing the numerical uncertainty.



        # Miscellaneous
        - proper_internal (dict): A dictionary with some very niche indicators used primarily for debugging 
          and analysis. 

        # Functions:

        - print_results(): Prints out the results of the stability_indicator analysis in an easy-to-read format. 
        - plot_time_arrays(): Calls the plot_osc_and_prop function from the plotting_scripts.py file. Requires 
          output_arrays to have been called previously.
        
        - plot_freq_space(plotflag='123', ifreqs={}): Calls the plot_freq_space function from the 
          plotting_scripts.py file. Requires output_arrays to have been called previously. 
            - plotflag (str): The plotflag variable can be used to select what plots are made by defining a str 
              of any combination of 1, 2, and 3. 1 = hk and pq, 2 = e and I, and 3 = varpi and Omega plots.
            - ifreq (dict): The ifreq variable can be used to include a vertical line at a given frequency in 
              the plot associated with the key. Ex: ifreq = {'2': ['50,000 years',2e-5]} would produce a vertical 
              line at the frequency 2e-5 rev/yr in the e and I plots, with a label in the legend = '50,000'.
            
        - plot_hkpq(): Calls the plot_hkpq function from the plotting_scripts.py file. Requires output_arrays 
          to have been called previously.
        
        - plot_angles(plot_cos=False, ifreqs={}): Calls the plot_angles function from the plotting_scripts.py file. 
          Requires output_arrays to have been called previously.
            - plot_cos (boolean): If True, these plots will show the cosine of the angles rather than the 
              circulating angles between(0,2*pi). 
            - ifreqs (dict): ifreq can be used to plot a custom precession frequency. This could be a useful tool 
              for someone testing secular frequencies. Ex: ifreq = {'3': ['0-freq', 0], '4': ['0-freq', 0]} would 
              plot a single line corresponding to the 0 frequency in both the omega and phi plots, with the label 
              '0-freq' in the legend.
    
    """
    
    def __init__(self, des=''):
        
        self.des = des
        self.planets = []
        self.planet_freqs = {}
        self.tmax = 0
        self.tout = 0
        ########################
        # Dallin, I think we need to make these arrays and not a dictionary so that we can store all the information for
        # the clones and not just the best-fit orbit
        # so something like:
        # self.osculating_elements.a = np.zeros(clones), etc, like in the TNO ML outputs
        # I don't see an elegant way otherwise to do this
        # we can add a helper function that prints out the units of the variables instead
        # of storing them in the variable neames
        #######################
        self.osculating_elements = {'a': [], 'e': [], 'I': [], 'omega': [], 'Omega': []}
        self.proper_elements = {'a': [], 'e': [], 'sinI': [], 'g("/yr)': [], 's("/yr)': [], 
                                'g(rev/yr)': [], 's(rev/yr)': [], 'omega': [], 'Omega': []}
        self.mean_elements = {'a': [], 'e': [], 'sinI': [], 'g("/yr)': [], 's("/yr)': [], 
                              'g(rev/yr)': [], 's(rev/yr)': []}
        self.proper_errors = {'RMS_a': [], 'RMS_e': [], 'RMS_sinI': [], 'RMS_g("/yr)': [], 
                              'RMS_s("/yr)': []}
        self.proper_indicators = {}
        self.proper_internal = {}
        self.scattered = {'scattered': False, 'scat_time': np.inf, 'scat_ind': np.inf, 'Max delta-E': 0, 
                          'qlim': 0, 'Qlim': np.inf, 'pcrossing_flag': False, 'qmin': 0, 'Qmax': np.inf}
        self.family_results = {'family_name': None, 'pairwise_dMet': np.inf}

        #Plotting Flags
        self.p_hkpq = True
        self.p_eI = False
        self.p_vO = False

        #Output arrays values
        self.hk_original = np.array([])
        self.pq_original = np.array([])
        
        self.hk_filtered = np.array([])
        self.pq_filtered = np.array([])

        self.hk_windows = np.array([])
        self.pq_windows = np.array([])
        self.time_windows = np.array([])

        self.a_original = np.array([])
        self.a_filtered = np.array([])
        self.a_windows = np.array([])

        self.time = np.array([])

    def print_results(self):

        print(str(self.des), " Proper Element Results from a ", str(round(self.tmax/1e6)), " Myr integration with outputs every", 
              str(round(self.tout)), "years")

        print("# \t\t\t SMA(AU) \t Ecc    \t Inc(deg) \t g(\"/yr) \t s(\"/yr)")
        print("#Osculating Elements: \t", round(self.osculating_elements['a'][0], 5) ,"\t" , 
              round(self.osculating_elements['e'][0], 5) , "\t" , round(self.osculating_elements['I'][0]*180/np.pi, 5) , 
              "\t N/A    \t N/A" )
        print("#Mean Elements: \t" , round(self.mean_elements['a'][0], 5) , "\t" , round(self.mean_elements['e'][0], 5) , "\t" , 
              round(np.arcsin(self.mean_elements['sinI'][0])*180/np.pi, 5) , "\t" , round(self.mean_elements['g("/yr)'][0], 5) , 
              "\t" , round(self.mean_elements['s("/yr)'][0], 5))
        print("#Proper Elements: \t" , round(self.proper_elements['a'][0], 5) , "\t" , round(self.proper_elements['e'][0], 5) , 
              "\t" , round(np.arcsin(self.proper_elements['sinI'][0])*180/np.pi, 5) , " \t" , round(self.proper_elements['g("/yr)'][0], 5) , 
              "\t" , round(self.proper_elements['s("/yr)'][0], 5))
        
        propa_err = self.proper_errors['RMS_a'][0]
        prope_err = self.proper_errors['RMS_e'][0]
        propI_err = np.arcsin(self.proper_errors['RMS_sinI'][0])*180/np.pi
        propg_err = self.proper_errors['RMS_g("/yr)'][0]
        props_err = self.proper_errors['RMS_s("/yr)'][0]
        print("#Proper Errors: \t", f"{propa_err:.3e}", "\t", f"{prope_err:.3e}",  "\t",   f"{propI_err:.3e}",  
              "\t",  f"{propg_err:.3e}", "\t", f"{props_err:.3e}")
        print()
        if self.scattered['scattered']:
            print(str(self.des) + ' may have been scattered during the simulation at t=' +str(self.scattered['scat_time']) + 
                  ' years, near snapshot #' + str(self.scattered['scat_ind']))

        if self.family_results['family_name'] != None:
            print(str(self.des) + ' maintains a proper orbit which corresponds dynamically to the ' + 
                  self.family_results['family_name'] + ' family.')
            print('The relative velocity of this object with respect to the family collisional center = ' + 
                  str(round(self.family_results['pairwise_dMet']), 2) + 'm/s')


    def plot_time_arrays(self):
        plotting_scripts.plot_osc_and_prop(self)

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
            
        plotting_scripts.plot_freq_space(self, ifreqs = ifreqs)

    def plot_hkpq(self):
        plotting_scripts.plot_hkpq(self)

    def plot_angles(self, plot_cos=False, ifreqs={}):
        plotting_scripts.plot_angles(self, plot_cos=plot_cos, ifreqs=ifreqs)




def integrate_for_pe(sim, des=None, archivefile=None,datadir='',icfile=False,
                     logfile=False,tmax=10e6,tout=500., direction='bf', 
                     deletefile=False, integrator='mercurius'):

    """
    Integrate a Rebound Simulation in the direction prescribed by the user,
    specifically built for synthetic proper element computation. 

    inputs:
        sim (Rebound Simulation): The Rebound Simulation to be used as the initial 
            starting point for the integration.
        des (str): Name/designation of the celestial body as contained in the 
            simulation archive.
        datadir (optional): string, path for saving any files produced in this 
            function; defaults to the current directory
        icfile (optional): boolean or string, default False, if True or a string,
            the path to the initial conditions of the rebound simulation is the 
            string or a default filename defined by sbdynt
            if False, there isn't a set initial conditions simulation archive
        archivefile (str; optional): name for the simulation
            archive file that rebound will generate. The default filename is 
            <des>-simarchive.bin if this variable is not defined.
        logfile (optional): boolean or string; if True:  will save some messages 
            to a default log file name or to a file with the name equal to the 
            string passed or to the screen if 'screen' is passed; (default) if 
            False nothing is saved (critical errors still printed to screen)
        deletefile (bool; optional): if set to True and an archivefile with
            the name/path of archivefile exists, it will be deleted before
            the new simulation archive file is created. The default
            is True for this function. If False, snapshots will be appended 
        tmax (float): The total integration time in years for the small body. The default 
            is 10e6 years, which is the default for an asteroid run.
        tout (float): The time interval for the integration outputs to be saved in the 
            Simulationarchive. The default is set to 500 years, the default for an asteroid run. 
            The combined tmax + tout default inputs results in a Simulationarchive with 
            20,000 individual snapshots saved to the Simulationarchive binary file.
        direction (str = 'bf','forwards', 'backwards'): The direction the integration should be 
            performed in, with options to do a forwards integration, a backwards integration, or 
            a combined backwards + forwards integration, which is the default setting. The 
            combined "bf" option integrates in each direction for half of the tmax setting. e.g. 
            tmax = 10 Myr years would result in a forwards integration of 5 Myr and a backwards 
            integration of 5 Myr. 
        integrator (optional): can be used to specify a choice other than the default of
            mercurius. NOTE: if you set this to whfast, close encounters won't be correctly 
            resolved!

    outputs:
        flag (0,1): 0 indicates a failure, while 1 indicates a successful run.
        sim (Rebound Simulation instance): The final Simulation from the full integration. 
        Equivalent to the last snapshot in the saved Simulationarchive. 
    """ 

    ic_file = icfile
    if(icfile == True):
        ic_file = tools.ic_file_name(des=des)
    elif(icfile):
        ic_file = icfile

    
    if(datadir):
        tools.check_datadir(datadir)
    if(datadir):
        if(icfile):
            ic_file = datadir + '/' + ic_file
    if(logfile==True):
        logf = tools.log_file_name(des=des[0])
    else:
        logf = logfile
    if(datadir and logf and logf!='screen'):
        logf = datadir + '/' +logf
    
    if(archivefile==None):
        archivefile = tools.archive_file_name(des)
    if(datadir):
        archivefile = datadir + '/' +archivefile
        
    if(direction == 'bf'):
        #check if there's a saved initial conditions file because we will need to be able
        #to re-start the simulation later to switch integration directions
        if(icfile == False):
            #save the current state to a randomly named file to make sure we don't over-write 
            #anyhing else in this directory
            random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
            ic_file = random_string+'.bin'
            if(datadir):
                ic_file = datadir + '/' + ic_file
            logmessage = "creating a temporary initial conditions file at " + ic_file
            tools.writelog(logf,logmessage)   
            sim.save_to_file(ic_file)
        
        #run the integration forward first
        rflag, sim = run_reb.run_simulation(sim,des=des,archivefile=archivefile,
                                            logfile=logf,tmax=(tmax/2),tout=tout, 
                                            deletefile=deletefile,integrator=integrator)
        
        if(rflag < 1):
            logmessage = "The forward simulation failed in prop_elem.integrate_for_pe\n"
            logmessage += "at run_reb.run_simulation\n";
            tools.writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)
            return 0, sim

        #read in the saved initial state to re-start for the backwards portion of the integration
        iflag, snew, clones = run_reb.initialize_simulation_from_simarchive(des=des,
                                                                            archivefile=ic_file,
                                                                            logfile=logf)
        if(iflag < 1):
            logmessage = "Failed to read back in the initial state in prop_elem.integrate_for_pe\n"
            logmessage += "from the file: " + ic_file
            logmessage += "at run_reb.initialize_simulation_from_simarchive to restart for the\n"
            logmessage += "backwards part of the integration\n"
            tools.writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)
            return 0, snew

        if(icfile==False):
            logmessage = "removing the temporary initial conditions file"
            tools.writelog(logf,logmessage)   
            remove(ic_file)

        rflag, snew = run_reb.run_simulation(snew,des=des,archivefile=archivefile,
                                             logfile=logf,tmax=-(tmax/2),tout=tout,
                                             deletefile=False, integrator=integrator)

        if(rflag < 1):
            logmessage = "The backward simulation failed in prop_elem.integrate_for_pe\n"
            logmessage += "at run_reb.run_simulation\n";
            tools.writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)
            return 0, snew

        return 1, sim


    elif(direction == 'forwards'):
        rflag, sim = run_reb.run_simulation(sim,des=des,archivefile=archivefile,
                                             logfile=logf,tmax=tmax,tout=tout, 
                                             deletefile=deletefile, integrator=integrator)
        if(rflag < 1):
            logmessage = "The forward simulation failed in prop_elem.integrate_for_pe\n"
            logmessage += "at run_reb.run_simulation\n";
            tools.writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)
            return 0, sim
        else:
            return 1, sim

    elif(direction == 'backwards'):
        rflag, sim = run_reb.run_simulation(sim,des=des,archivefile=archivefile,
                                            logfile=logf,tmax=-tmax,tout=tout, 
                                            deletefile=deletefile,integrator=integrator)
        if(rflag < 1):
            logmessage = "The backward simulation failed in prop_elem.integrate_for_pe\n"
            logmessage += "at run_reb.run_simulation\n";
            tools.writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)
            return 0, sim
        else:
            return 1, sim

    else:
        logmessage = "failed in  prop_elem.integrate_for_pe\n"
        logmessage += 'direction given not backwards, forwards, or bf.\n'
        logmessage += 'Call this function again with a valid direction\n'
        tools.writelog(logf,logmessage) 
        if(logf != 'screen'):
            print(logmessage)
        return 0,sim




    
def smooth_fft_convolution(fft_signal, freqs, primary_freqs, time, protect_radius_bins_init=3, kernel_size=15, 
                           method="gaussian", inc_filt = False, known_planet_freqs = [],freq_tol=2e-7, win=False, 
                           shortfilt = True):
    """
    Fast convolutional smoothing of FFT log-power spectrum, excluding primary peaks.

    This function takes an FFT signal, the proper frequencies to be protected, the planetary frequencies to be 
    filtered out, and some other related data, and performs a gaussian_1dfilter on the log10 signal. The 
    function returns the smoothed FFT signal, which corresponds to the proper orbital motion. 
    """
    fft_signal = np.asarray(fft_signal)
    log_power = 10 * np.log10(np.abs(fft_signal)**2 + 1e-10)
    
    log_power[np.isnan(log_power)] = 1e-10

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

        idx = np.argmin(np.abs(freqs - pf))
        if first: 
            p_idx = idx
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
        mask[in_window_opp] = False

 
    mask_float = mask.astype(float)
    mask_opp_float = mask_opp.astype(float)

    # Convolve both signal and mask to normalize after convolution

    masked_log_power = log_power.copy()

    if method == "gaussian":
        smoothed_log_power = gaussian_filter1d(masked_log_power, sigma=kernel_size, mode="mirror")
        smoothed_mask = gaussian_filter1d(mask_float, sigma=kernel_size, mode="mirror")
    else:  # fallback to uniform boxcar
        kernel = np.ones(kernel_size) / kernel_size
        smoothed_log_power = convolve1d(masked_log_power, kernel, mode="nearest")
        smoothed_mask = convolve1d(mask_float, kernel, mode="nearest")

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

def smooth_fft_time(fft_signal, freqs, primary_freqs, time, protect_radius_bins_init=3, kernel_size=15, 
                    method="gaussian", inc_filt = False, known_planet_freqs = [],freq_tol=2e-7, win=False, 
                    shortfilt = True):
    """
    This function takes in the FFT of a signal, and smooths the entire FFT. This is used to perform the 
    additional smoothing on the eccentricity and inclination arrays directly, which should retain now 
    long-period terms. 
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
    scaling_factor = 1

    return smoothed_fft * scaling_factor

def extract_proper_mode(signal, time, known_planet_freqs, freq_tol=2e-7, protect_bins=None,kernel=60, 
                        proper_freq = None, inc_filt = False, win=False, shortfilt = True, afilt=False, 
                        filt_time = False):
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

        tol_bins = round(freq_tol*N*dt)

        ref_spec = spectrum.copy()

        ind_0 = np.where(freqs == 0)[0][0]

        lowest_per_p = np.min(abs(1/np.array(known_planet_freqs)))
        lowest_per_gs = np.min(abs(1/np.array(proper_freq)))

        lowest_period = min(lowest_per_p,lowest_per_gs)
        
        shortperiod = np.where(1/abs(freqs) < lowest_period/4)[0]
        if len(shortperiod) > 0 and shortfilt:    
            short_ref = ref_spec[shortperiod]
            spectrum[shortperiod] = short_ref[tools.argmedian(np.abs(short_ref)**2)]
        #Find the frequency with the highest power that is NOT near any planetary frequency
        power = np.abs(spectrum)**2
        sorted_indices = np.argsort(power)[::-1]  # sort descending by power

        dt = time[1]-time[0]

        proper_idx = np.nanargmin(abs(freqs - proper_freq[0]))
        new_spec = spectrum.copy()

        if proper_freq is None:
            raise ValueError("No proper (free) frequency found distinct from planetary modes.")

        if protect_bins == None:
            protect_bins = 6*tol_bins

        if protect_bins < 1:
            protect_bins = 1

        nanvals = np.where(np.isnan(spectrum))[0]

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
        if len(shortperiod) > 0 and shortfilt == False:   
            filt_signal[shortperiod] = ref_spec[shortperiod]
            
        ind_0 = np.where(freqs == 0)[0][0]
        
        if afilt:
            filt_signal[ind_0] = spectrum[ind_0]
        filt_signal = np.fft.ifftshift(filt_signal)
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

    min_logf = np.log10(abs(max_freq)) - window_half_dex
    max_logf = np.log10(abs(max_freq)) + window_half_dex

    
    ind_low = np.argmin(abs(freqs-10**min_logf))
    ind_high = np.argmin(abs(freqs-10**max_logf))

    if max_freq < 0:
        ind_high = np.argmin(abs(freqs+10**min_logf))
        ind_low = np.argmin(abs(freqs+10**max_logf))

    if ind_high - ind_low <= 1:
        ind_low = max_idx - 1
        ind_high = max_idx + 1
    
    dominant_freq = np.sum(freqs[ind_low:ind_high]*(powers[ind_low:ind_high])**2)/np.sum((powers[ind_low:ind_high])**2)
    max_idx = np.argmin(np.abs(freqs-dominant_freq))
    protect_bins = 10

    return max_idx, dominant_freq, local_power_all, protect_bins



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
    """
    This function finds the proper planetary secular frequencies associated with the planetary 
    orbital elements arrays contained in the planet_elems variable, and the returns the result as a dictionary.
    """
    try:     
        hj,kj,pj,qj = hkpq(planet_elems['jupiter'])
        hs,ks,ps,qs = hkpq(planet_elems['saturn'])
        hu,ku,pu,qu = hkpq(planet_elems['uranus'])
        hn,kn,pn,qn = hkpq(planet_elems['neptune'])
        
        if small_planets_flag:    
            hv,kv,pv,qv = hkpq(planet_elems['venus'])
            he,ke,pe,qe = hkpq(planet_elems['earth'])
            hm,km,pm,qm = hkpq(planet_elems['mars'])
            
            
        Yhkj = np.fft.fft(kj+1j*hj); Yhks = np.fft.fft(ks+1j*hs); 
        Yhku = np.fft.fft(ku+1j*hu); Yhkn = np.fft.fft(kn+1j*hn)
        Ypqj = np.fft.fft(qj+1j*pj); Ypqs = np.fft.fft(qs+1j*ps); 
        Ypqu = np.fft.fft(qu+1j*pu); Ypqn = np.fft.fft(qn+1j*pn)

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
        freq = np.fft.fftfreq(n,d=dt)

        half = int(len(freq)/2)

        good_freq_inds = np.where(abs(1/freq) <= np.max(abs(t_init)))[0]
        bad_freq_inds = np.where(abs(1/freq) > np.max(abs(t_init)))[0]

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
    """
    This function takes a power spectrum for some signal, and the planetary frequencies to 
    filter out of the spectrum.
    It then performs a simple "filter" by dividing the planetary frequencies not associated 
    with the proper frequency of the planet by some scalar.
    The result is a "filtered" power spectrum. 
    This function is not the robust, comprehensive filter used by SBDynT to compute the 
    proper motion, but is instead used to prepare a signal to find the proper frequency 
    within the signal by reducing the planetary influence first.
    """
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


def compute_prop(a_init,e_init,inc_init,aop_init,lan_init,t_init,g_arr,s_arr,gs_dict,small_planets_flag,
                 windows=5,debug=False,objname='', rms = True, shortfilt=True, output_arrays = False):
    """
    Compute the proper motion for a Solar System small body, given the oscualting orbital elements and 
    the planetary secular frequencies to be filtered out.
    Parameters:
        a_init (1D numpy.array): The osculating semi-major axis of the small body.
        e_init (1D numpy.array): The osculating eccentricity of the small body.
        inc_init (1D numpy.array, radians): The osculating inclination of the small body.
        aop_init (1D numpy.array, radians): The osculating argument of periapse of the small body.
        lan_init (1D numpy.array, radians): The osculating longitude of the ascending node of the small body.
        t_init (1D numpy.array, years): The array of times/epochs axis of the small body arrays.

        g_arr (list, units=rev/yr): The list of eccentricity planetary secular frequencies to be 
            filtered out. i.e. [g5, g6, g7, g8]
        s_arr (list, units=rev/yr): The list of inclination planetary secular frequencies to be 
            filtered out. i.e. [s6, s7, s8]
        gs_dict (dict): The dictionary of all linear and non-linear planetary secular frequencies 
            to be filtered out. Values should be reported in rev/yr. 
            e.g. gs_dict = {'g8': g8, 's8': s8, 'g8+s8': g8+s8}
        small_planets_flag (boolean): If True, filteres out the planetary frequencies associated with 
            Venus, Earth, and Mars as well as the giant planets. 
            Otherwise, this functino only filters out the giant planets. 
        windows (int): The number of windows to use in estimating the numerical uncertainties. 
        objname (str, optional): The name of the object as contained in the Simulationarchive.
        shortfilt (boolean): A parameter which turns the short-period filter for objects on or off. It is 
            recommended that this variable remain True in all cases, though no change should be detected 
            in 99% of cases. 
        output_arrays (boolean): If True, the resulting outputs include the osculating and filtered time 
            arrays of the orbital elements.
        
    Returns:
        DALLIN: add return variables list
        
    """ 
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
        freq = np.fft.fftfreq(n,d=dt)
        
        goal = 2**16
        goal = len(hk_arr)
        j=10
        goal_p = 0
        while goal_p < len(hk_arr):
            goal_p = 2**j
            j += 1
        goal_p = len(hk_arr)
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
            if abs(1/g) > 5e5:
                protect_g_bins = protect_g_bins*5
            
           
            Ypq_s = np.fft.fft(pq_arr_s)
            power = np.abs(Ypq_s)**2
            power[abs(1/freq_s) > time_mag/2] = 0

            power = power_filt(power, s_arr, freq_s, small_planets_flag, dist)
            
            s_idx, s, local_power_all, protect_s_bins = find_local_max_windowed(freq_s, power, window_half_dex=0.02, window_protect_dex=0.15)
            if abs(1/s) > 5e5:
                protect_s_bins = protect_s_bins*2 
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
    dex_protect = 0.04
    freq_low_g = 10**(np.log10(abs(protect_g[0]))-dex_protect)
    ind_low_g = np.argmin(abs(freq-abs(freq_low_g)))
    ind = np.argmin(abs(freq-abs(protect_g[0])))
    kernel_g = max(2*round(abs(ind-ind_low_g)),round(len(freq)/2500))
    kernel_g = max(2*round(abs(ind-ind_low_g)),4)
    
    if kernel_g > round(len(freq)/25): 
        kernel_g = round(len(freq)/25)
    
    freq_low_s = 10**(np.log10(abs(protect_s[0]))-dex_protect)
    ind_low_s = np.argmin(abs(freq-abs(freq_low_s)))
    ind = np.argmin(abs(freq-abs(protect_s[0])))

    
    kernel_s = max(2*round(abs(ind-ind_low_s)),round(len(freq)/2500))
    kernel_s = max(2*round(abs(ind-ind_low_s)),4)
    if kernel_s > round(len(freq)/25): 
        kernel_s = round(len(freq)/25)

                          
    try:
        dt = np.abs(t_init[1]-t_init[0]) 
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
        protect_g_bins = min(int(kernel_g/4), protect_g_bins)
        protect_s_bins = min(int(kernel_s/4), protect_s_bins)
        protect_g_bins = max(protect_g_bins, 2)
        protect_s_bins = max(protect_s_bins, 2)
    
        hk_old, hk_freq, protect_hk, hk_signal = extract_proper_mode(hk_arr, t_init, g_arr, freq_tol=tol, 
                                                                     kernel=kernel_g, protect_bins = protect_g_bins, 
                                                                     proper_freq=protect_g, shortfilt=shortfilt)
        pq_old, pq_freq, protect_pq, pq_signal = extract_proper_mode(pq_arr, t_init, s_arr, freq_tol=tol, 
                                                                     kernel=kernel_s, proper_freq=protect_s, 
                                                                     protect_bins = protect_s_bins, shortfilt=shortfilt)
        
        ee_old = np.abs(hk_old)
        II_old = np.abs(pq_old)

        ee0 = np.fft.fft(ee_old)[0]
        II0 = np.fft.fft(II_old)[0]
        
        ee_ang = np.angle(hk_old)
        II_ang = np.angle(pq_old)
        
        ee_new, ee_freq, protect_ee, ee_signal = extract_proper_mode(ee_old, t_init, g_arr, freq_tol=tol, kernel=kernel_g*2, 
                                                                     protect_bins = 0, proper_freq=np.append(protect_g[0],[0]), 
                                                                     shortfilt=False, filt_time=True)
        II_new, II_freq, protect_II, II_signal = extract_proper_mode(II_old, t_init, s_arr, freq_tol=tol, kernel=kernel_s*2, 
                                                                     proper_freq=np.append(protect_s[0],[0]), protect_bins = 0, 
                                                                     shortfilt=False, filt_time=True)
        Yee = np.fft.fft(ee_new)
        YII = np.fft.fft(II_new)
        Yee[0] = ee0
        YII[0] = II0
        ee_new = np.fft.ifft(Yee)
        II_new = np.fft.ifft(YII)

        hk_new = ee_new*np.cos(ee_ang) + 1j*ee_new*np.sin(ee_ang)
        pq_new = II_new*np.cos(II_ang) + 1j*II_new*np.sin(II_ang)

        max_idx_g, g_fin, local_power_all, protect_bins = find_local_max_windowed(freq, np.abs(np.fft.fft(hk_new))**2, 
                                                                                  window_half_dex=0.02, window_protect_dex=0.15)
        max_idx_s, s_fin, local_power_all, protect_bins = find_local_max_windowed(freq, np.abs(np.fft.fft(pq_new))**2, 
                                                                                  window_half_dex=0.02, window_protect_dex=0.15)

        g = g_fin
        s = s_fin

        pomega_n = np.angle(hk_new)
        Omega_n = np.angle(pq_new)
        omega_n = pomega_n - Omega_n

        N = N_og

        pe_e = np.nanmean(np.abs(hk_new[int(per_shave*N):int((1-per_shave)*N)]))
        pe_i = np.nanmean(np.abs(pq_new[int(per_shave*N):int((1-per_shave)*N)]))

        hk_news = hk_new[int(per_shave*N):int((1-per_shave)*N)]
        pq_news = pq_new[int(per_shave*N):int((1-per_shave)*N)]

        e_win = np.abs(hk_news)*np.kaiser(len(hk_news),6)
        inc_win = np.abs(pq_news)*np.kaiser(len(pq_news),6)

        pe_e = np.nansum(e_win)/np.sum(np.kaiser(len(hk_news),6))
        pe_i = np.nansum(inc_win)/np.sum(np.kaiser(len(pq_news),6))
        
        a_filt, a_freq, protect_a, a_signal = extract_proper_mode(a_init, t_init, g_arr, freq_tol=tol, kernel=40, 
                                                                  protect_bins = 10, proper_freq=[np.max(abs(t_init))], 
                                                                  shortfilt=shortfilt, afilt=True)
        
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

    #=======================================================================================
    #Begin Error Calculation
    #=======================================================================================

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

            g_idx_e, g_e, local_power_all_e, protect_g_bins_e = find_local_max_windowed(freqe, power_e, 
                                                                                        window_half_dex=0.05, 
                                                                                        window_protect_dex=0.15)
            s_idx_e, s_e, local_power_all_e, protect_s_bins_e = find_local_max_windowed(freqe, power_I, 
                                                                                        window_half_dex=0.05, 
                                                                                        window_protect_dex=0.15)

            hk_freqe = np.array([g_e, 2*g_e, 3*g_e])
            pq_freqe = np.array([s_e, 2*s_e, 3*s_e])

            protecte_hk = min(int(kernele_g/4), protecte_hk)
            protecte_pq = min(int(kernele_s/4), protecte_pq)
            protecte_hk = max(protecte_hk, 1)
            protecte_pq = max(protecte_pq, 1)
            
            hk_newe, hk_freqe, protect_hk_waste, hk_waste = extract_proper_mode(hk_in, t_input, g_arr, freq_tol=tol, 
                                                                                protect_bins = protecte_hk, 
                                                                                kernel=kernele_g, proper_freq=hk_freqe, 
                                                                                win=True)
            pq_newe, pq_freqe, protect_pq_waste, hk_waste = extract_proper_mode(pq_in, t_input, s_arr, freq_tol=tol, 
                                                                                protect_bins = protecte_pq, 
                                                                                kernel=kernele_s, proper_freq=pq_freqe,  
                                                                                win=True)  
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
        ee_newe, ee_freqe, protect_eee, ee_signale = extract_proper_mode(ee_olde, t_input, g_arr, freq_tol=tol, 
                                                                         kernel=kernele_g*2, protect_bins = 0, 
                                                                         shortfilt=shortfilt, 
                                                                         proper_freq=np.append(hk_freq[0],[0]), 
                                                                         filt_time=True)
        II_newe, II_freqe, protect_IIe, II_signale = extract_proper_mode(II_olde, t_input, s_arr, freq_tol=tol,
                                                                         kernel=kernele_s*2, protect_bins = 0, 
                                                                         shortfilt=shortfilt, 
                                                                         proper_freq=np.append(pq_freq[0],[0]), 
                                                                         filt_time=True)
        
        Yeee = np.fft.fft(ee_newe)
        YIIe = np.fft.fft(II_newe)
        Yeee[0] = ee0
        YIIe[0] = II0
        ee_newe = np.fft.ifft(Yeee)
        II_newe = np.fft.ifft(YIIe)

        hk_newe = ee_newe*np.cos(ee_ange) + 1j*ee_newe*np.sin(ee_ange)
        pq_newe = II_newe*np.cos(II_ange) + 1j*II_newe*np.sin(II_ange)

        a_filtn, a_freqn, protect_an, a_signaln = extract_proper_mode(a_input, t_input, g_arr, freq_tol=tol, 
                                                                      kernel=40, protect_bins = 10, 
                                                                      proper_freq=[np.max(abs(t_init))], 
                                                                      shortfilt=shortfilt, afilt=True)
        
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
        rms_val = np.sqrt(np.nanmean((np.array(error_list,dtype=np.float64)-
                                      np.array(pes,dtype=np.float64))**2,axis=0))
    else:
        rms_val = np.nanstd(error_list,axis=0)

    #Identfy the longest period frequency that could appear in the simulation. This helps determine 
    #whether the hcm metric indicates long-term periodicity or instability. 
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
        librate_angle = circmean(v_O, low=0, high = 2*np.pi)
        angle_ent = v_O_ent
    elif varpi_ent < 0.95:
        angle_sec_res = 'Varpi'
        librate_angle = circmean(varpi_new, low=-np.pi, high = np.pi)
        angle_ent = varpi_ent
    elif O_ent < 0.95:
        angle_sec_res = 'Omega'
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
        return pes,error_list,rms_val,g_arr,s_arr, hk_arr[:N_og], pq_arr[:N_og], hk_new[:N_og], \
                pq_new[:N_og], hk_freq, pq_freq, hk_signal, pq_signal, rese, resI, sec_res_e, \
                sec_res_I, e_osc_amp, I_osc_amp, e_amp, I_amp, angle_sec_res, librate_angle

    phifrac = (g+s)/(gs_dict['g8'] + gs_dict['s8'])

    if output_arrays:
        return True, pes, rms_val, error_list, omega_n, Omega_n, maxvals, g, s, rese, resI, \
                sec_res_e, sec_res_I, e_osc_amp, I_osc_amp, e_amp, I_amp, angle_sec_res, \
                librate_angle, angle_ent, phifrac, hk_new, pq_new, a_filt, hk_wins, pq_wins, \
                a_wins, t_wins, hk_arr, pq_arr
    
    return True, pes, rms_val, error_list, omega_n, Omega_n, maxvals, g, s, rese, resI, sec_res_e, \
            sec_res_I, e_osc_amp, I_osc_amp, e_amp, I_amp, angle_sec_res, librate_angle, angle_ent, \
            phifrac


def read_archive_for_pe(des, clones=0, datadir='',archivefile=None, logfile=None, object_type= None):

    """
    Read a Rebound Simulationarchive binary file, and format the orbital array outputs to work with 
    the proper elements code.
    This function automatically handles issues that may arise from Simulationarchive's that may have 
    varying time resolution, or integrate in both the forwards and backwards direction, which results 
    naturally from the default TNO SBDynT analysis. 

    Parameters:
        des (str): Name/designation of the celestial body as contained in the simulation archive.
        clones (int): Number of clones in the Simulationarchive for the TNO to be read in and returned 
           by the function. This number should not exceed the number of clones contained in the actual 
           Simulationarchive itself. 
        datadir (str): Directory where the Simulationarchive is located. Is defined with respect to the 
           user's working directory.
        archivefile (str): Name of the Simulationarchive binary file to be read in. The default filename 
           is <des>-simarchive.bin.
        logfile (boolean): If True, saves a log file of the run for debugging or benchmarking purposes. 
        objectype (str = 'tno', 'asteroid'): Determines whether this function returns orbital element 
           time arrays for only the 4 giant planets ('tno' = JSUN) or the the 4 giant planets + the 3 major 
           rocky planets ('asteroid' = JSUN + VEM). 
        If the object_type is not defined, SBDynT will check to see if Venus, Earth, and Mars are contained in the 
           Simulationarchive. If these objects are present in the Simulation, it will return their orbital 
           elements, and set vem = True. 

    Returns:
        flag (0,1):  0 indicates a failure occurred. 1 indicates a successful function run.
        time (1D numpy.array, shape= len(Simulationarchive)) = The corresponding times for the outputs contained in 
           the Simulationarchive.  
        sb_elems (2D numpy.array, shape= (5, len(Simulationarchive))): The orbital element time arrays for the 
           small body particle contained in the Simulationarchive. The array contains the orbital elements in 
           this order: [sma, ecc, inc, omega, Omega]. 
        planet_elems (dict): A dictionary containing the orbital element time arrays for the planets to be returned. 
           {'jupiter': jupiter_orb, 'saturn': saturn_orb .... }. The individual "planet_orb" variables have the 
           same shape and format as the sb_elems variable.
        clone_elems (3D numpy.array, shape = (clones, 5, len(Simulationarchive))): The orbital element time arrays 
           for the clones of the small body particle. 
        vem (boolean): A boolean corresponding to whether Venus, Earth, and Mars are contained in the planet_elems 
           dictionary, and if they should be included in the proper_elements computation. 
        
    """ 
    
    if(archivefile==None):
        file = tools.archive_file_name(des=des)
    else:
        file = archivefile
    if(datadir):
        file = datadir + '/' + file


    if(logfile==True):
        logf = tools.log_file_name(des=des)
    else:
        logf = logfile
    if(datadir and logf and logf!='screen'):
        logf = datadir + '/' + logf

    try:
        sim = rebound.Simulationarchive(file)
    except:
        logmessage = "failed to read archive file: "+file + "\n"
        logmessage += "in prop_elem.read_archive_for_pe\n"
        if(logf != 'screen'):
            print(logmessage)  
        if(logf):
            tools.writelog(logf,logmessage)
        return 0, None, None, None, None, None

    if(object_type == None):
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
        if(object_type == 'tno'):
            vem = False
            planets = ['jupiter','saturn','uranus','neptune']
        elif(object_type == 'asteroid'):
            vem = True
            planets = ['venus','earth','mars','jupiter','saturn','uranus','neptune']
        else:
            logmessage = 'small_body object does not have a valid object_type.\n'
            logmessage+= 'Checking planets contained in simulation by hash.'
            if(logf != 'screen'):
                print(logmessage)  
            if(logf):
                tools.writelog(logf,logmessage)
            try:
                testval = sim[0].particles['venus']
                testval = sim[0].particles['earth']
                testval = sim[0].particles['mars']
                logmessage = 'simulation includes venus, earth, and mars.\n'
                logmessage += 'SBDynt will filter out these inner planets for proper elements\n'
                if(logf != 'screen'):
                    print(logmessage)  
                if(logf):
                    tools.writelog(logf,logmessage)
                vem = True
                planets = ['venus','earth','mars','jupiter','saturn','uranus','neptune']
            except:
                logmessage = 'simulation does not includes venus, earth, and/or mars.\n'
                logmessage += 'SBDynt will filter out only the outer planets for proper elements\n'
                if(logf != 'screen'):
                    print(logmessage)  
                if(logf):
                    tools.writelog(logf,logmessage)
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

    for i in range(len(sim)):
        s = sim[i]
        s.move_to_com()
        particles = s.particles
        com = s.com()
        try:
            sb_idx = particles[des].index
        except:
            continue

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
            c_name = des+'_'+str(j+1)
            clone = sim[i].particles[c_name].orbit(com)
            clone_elems[j,0,i] = clone.a
            clone_elems[j,1,i] = clone.e
            clone_elems[j,2,i] = clone.inc
            clone_elems[j,3,i] = clone.omega
            clone_elems[j,4,i] = clone.Omega
            
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

    #First sort and remove values from the time array that aren't consistent with the higher resolution data. 
    t_arr = sb_elems[0].copy()
    sortt = np.sort(t_arr)

    dt = round(abs(sortt[-1] - sortt[-2]))


    test_arr = t_arr.copy()
    skip_short_res = np.where(test_arr.astype(int) % dt == 0)[0]
    
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
        planet_elems = {'venus': v_elems, 'earth': e_elems, 'mars': m_elems, 'jupiter': j_elems, 
                        'saturn': s_elems, 'uranus': u_elems, 'neptune': n_elems}
    else:
        planet_elems = {'jupiter': j_elems, 'saturn': s_elems, 'uranus': u_elems, 'neptune': n_elems}

    return 1, sb_elems[0], sb_elems[1:], planet_elems, clone_elems, vem



def hcm_calc(a,rmsa,rmse,rmsi):    
    G = 6.673e-11
    M = 1.989e30
    n = np.sqrt(G*M/(a*149597870700)**3)
    return n*a*149597870700*np.sqrt(5/4*(rmsa/a)**2+2*rmse**2+2*rmsi**2)   


def hcm_pair(a1, a2, e1, e2, sini1, sini2):
    G = 6.673e-11
    M = 1.989e30
    am = np.mean(np.array([a1,a2]))
    n = np.sqrt(G*M/(am*1.496e11)**3)
    return n*am*1.498e11*np.sqrt(5/4*((a2-a1)/am)**2+2*(e2-e1)**2+2*(sini2-sini1)**2)  



def calc_proper_elements(des=None, times= [], sb_elems = [], clones = 0, clone_elems = [], planet_elems = [], small_planets_flag = False, 
                         output_arrays = False, gs_dict = None,logfile=False):

    """
    Compute the synthetic proper elements from the given orbital element time arrays for a small body, 
    and the associated orbital elements for the planets included in the Simulation

    Parameters:
        des (str): Name/designation of the celestial body as contained in the simulation archive.
        times (1D numpy array): The integration times associated with the output orbital elements.
        sb_elems (2D numpy array): The orbital element time arrays for the small body. Should be 
          formatted as [sma, ecc, inc, omega, Omega].
        planet_elems (dict, optional but recommended): A dictionary of the planet orbital element 
          time arrays associated with the small body, to be used to compute the planetary secular 
          frequencies. Example = {'jupiter': jupiter_elems, 'saturn': saturn_elems ....}, 
          where *planet*_elems has the same shape and format as sb_elems. If this variable is left 
          empty, SBDynT will see if the gs_dict variable is provided. If not, hard coded planetary 
          secular frequencies as reported by Knezevic and Milani 2000 and Brouwer and van Woerkom 
          1950 are used instead.
        small_planets_flag (boolean): If True, the function will use the orital elements in 
          planet_elems corresponding to 'venus', 'earth', and 'mars', and will filter out their 
          secular frequencies in the small body's orbit. 
        output_arrays (boolean): If True, will save the orbital element time arrays and the filtered 
          orbital element time arrays to variables in the proper_element_class object for quick access. 
          This function must be set to true to use the visualization tools after the analysis.
        gs_dict (dict, optional): A dictionary of the secular planetary frequencies (in rev/yr) to be 
          filtered out. Only used if planet_elems is not defined. If both variables are left empty, 
          hard coded secular frequencies as reported by Knezevic and Milani 2000 and Brouwer and van 
          Woerkom 1950 are used instead. 
          Example: {'g5': 3.299e-6 ,'g6': 2.197e-5, 'g7': 2.398e-6, 'g8': 5.022e-7, 's6': -2.032e-5, 
          's7': -2.309e-6, 's8': -5.3395e-7}.
        logfile (boolean or string, optional): if False, only critical error messages are printed
          to screen; if a string is passed, log messages are printed to screen (if set to 'screen') or
          to a file with the path given by the string.

    Returns:
        flag (0,1):  0 indicates a failure occurred. 1 indicates a successful function run.
        proper_object: A proper_element_class object with results from the analysis. See the 
          proper_element_class for information of the variables which are included in the outputs. 
        
    """ 
    
    if(des==None):
        logmessage = 'You must provide an object designation\n'
        logmessage+='Failed at prop_elem.calc_proper_elements'
        tools.writelog(logfile,logmessage)
        if(logfile != 'screen'):
            print(logmessage)             
        return 0, None

    proper_object = proper_element_class(des=des)

    if len(times) == 0:
        logmessage = 'An array of the times associated with the sb_elems was not provided.\n'
        logmessage += 'Please include the times vairable in your function call\n'
        logmessage += 'Failed at prop_elem.calc_proper_elements\n'
        tools.writelog(logfile,logmessage)
        if(logfile != 'screen'):
            print(logmessage)                
        return 0, proper_object

    if len(sb_elems) == 0:
        logmessage ='No sb_elems were provided, or the array of sb_elems is empty.\n'
        logmessage += 'Failed at prop_elem.calc_proper_elements\n'
        tools.writelog(logfile,logmessage)
        if(logfile != 'screen'):
            print(logmessage)              
        return 0, proper_object

    try:
        if small_planets_flag:
            test = planet_elems['venus']
    except:
        logmessage = 'User set small_planets_flag = True, but planet_elems does not contain a\n'
        logmessage += '"venus" array. Setting small_planets_flag = False for this analysis.\n'
        tools.writelog(logfile,logmessage)        
        small_planets_flag = False
    try:
        if small_planets_flag:
            test = planet_elems['earth']
    except:
        logmessage = 'User set small_planets_flag = True, but planet_elems does not contain a \n'
        logmessage += '"earth" array. Setting small_planets_flag = False for this analysis.\n'
        tools.writelog(logfile,logmessage)        
        small_planets_flag = False
    try:
        if small_planets_flag:
            test = planet_elems['mars']
    except:
        logmessage += 'User set small_planets_flag = True, but planet_elems does not contain a \n'
        logmessage += '"mars" array. Setting small_planets_flag = False for this analysis.\n'
        tools.writelog(logfile,logmessage)        
        small_planets_flag = False
            
    osc_elem = {}
    mean_elem = {}
    prop_elem = {}
    ind0 = np.where(times == 0.0)[0][0]

    dt = abs(times[-1]-times[-2])

    proper_object.tmax = round(np.max(times) - np.min(times))
    proper_object.tout = round(dt)

    a_init = sb_elems[0]
    e_init = sb_elems[1]
    I_init = sb_elems[2]
    o_init = sb_elems[3]
    O_init = sb_elems[4]
        
    osc_elem['a'] = [a_init[ind0]]
    osc_elem['e'] = [e_init[ind0]]
    osc_elem['I'] = [I_init[ind0]]
    osc_elem['omega'] = [o_init[ind0]]
    osc_elem['Omega'] = [O_init[ind0]]

    mean_elem['a'] = [np.mean(a_init)]
    mean_elem['e'] = [np.mean(e_init)]
    mean_elem['sinI'] = [np.mean(I_init)]
        
    diffg = np.gradient((o_init+O_init)%(2*np.pi))
    diffs = np.gradient((O_init)%(2*np.pi))
    
    mean_elem['g(rev/yr)'] = [np.median(diffg)/dt/2/np.pi]
    mean_elem['s(rev/yr)'] = [np.median(diffs)/dt/2/np.pi]
    
    mean_elem['g("/yr)'] = [np.median(diffg)/dt*3600*360/2/np.pi]
    mean_elem['s("/yr)'] = [np.median(diffs)/dt*3600*360/2/np.pi]

    if clones > 0:
        for i in range(clones):
            osc_elem['a'].append(clone_elems[i,0,ind0])
            osc_elem['e'].append(clone_elems[i,1,ind0])
            osc_elem['I'].append(clone_elems[i,2,ind0])
            osc_elem['omega'].append(clone_elems[i,3,ind0])
            osc_elem['Omega'].append(clone_elems[i,4,ind0])

            mean_elem['a'].append(np.mean(clone_elems[i,0]))
            mean_elem['e'].append(np.mean(clone_elems[i,1]))
            mean_elem['sinI'].append(np.sin(np.mean(clone_elems[i,2])))
        
            diffg = np.gradient((clone_elems[i,3]+clone_elems[i,4])%(2*np.pi))
            diffs = np.gradient((clone_elems[i,4])%(2*np.pi))
    
            mean_elem['g(rev/yr)'].append(np.median(diffg)/dt/2/np.pi)
            mean_elem['s(rev/yr)'].append(np.median(diffs)/dt/2/np.pi)
    
            mean_elem['g("/yr)'].append(np.median(diffg)/dt*3600*360/2/np.pi)
            mean_elem['s("/yr)'].append(np.median(diffs)/dt*3600*360/2/np.pi)

    

    if len(planet_elems) == 0:
        g_arr = []
        s_arr = []
        if gs_dict == None:
            logmessage ='Neither orbital element arrays for the planets nor a gs_dict were provided by the user.\n'
            logmessage +='Proper elements will be computed using hard coded secular frequencies taken from the literature.\n'
            tools.writelog(logfile,logmessage)
            if small_planets_flag:
                gs_dict = {'g5': 3.299e-6 ,'g6': 2.197e-5, 'g7': 2.398e-6, 'g8': 5.022e-7, 's6': -2.032e-5, 
                           's7': -2.309e-6, 's8': -5.3395e-7, 'g2': 7.34474/1296000, 'g3': 17.32832/1296000, 
                           'g4': 18.00233/1296000, 's2': -6.57080/1296000, 's3': -18.74359/1296000, 's4': -17.63331/1296000}
            else:
                gs_dict = {'g5': 3.299e-6 ,'g6': 2.197e-5, 'g7': 2.398e-6, 'g8': 5.022e-7, 's6': -2.032e-5, 
                           's7': -2.309e-6, 's8': -5.3395e-7}
        
        for key, value in gs_dict.items():
            if 'g' in key:
                g_arr.append(value)
            elif 's' in key:
                s_arr.append(value)
    else:        
        g_arr,g_inds,s_arr,s_inds, gs_dict = get_planet_freqs(times, planet_elems, small_planets_flag = small_planets_flag)

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
        flag, pes, rms_val, error_list, omega_n, Omega_n, maxvals, g, s, rese, resI, sec_res_e, sec_res_I, \
                e_osc_amp, I_osc_amp, e_amp, I_amp, angle_sec_res, librate_angle, angle_ent, phifrac, hk_new, \
                pq_new, a_filt, hk_wins, pq_wins, a_wins, t_wins, hk_arr, pq_arr = \
                compute_prop(a_init,e_init,I_init,o_init,O_init,times,g_arr,s_arr,gs_dict,small_planets_flag,
                             windows=5,debug=False,objname=des, rms = True, shortfilt=True, output_arrays = output_arrays)
    else:
        flag, pes, rms_val, error_list, omega_n, Omega_n, maxvals, g, s, rese, resI, sec_res_e, sec_res_I, \
                e_osc_amp, I_osc_amp, e_amp, I_amp, angle_sec_res, librate_angle, angle_ent, phifrac = \
                compute_prop(a_init,e_init,I_init,o_init,O_init,times,g_arr,s_arr,gs_dict,small_planets_flag,
                             windows=5,debug=False,objname=des, rms = True, shortfilt=True)

    c_results = []
    if clones > 0:
        for i in range(clones):
            c_results.append(compute_prop(clone_elems[i,0],clone_elems[i,1],clone_elems[i,2],clone_elems[i,3],clone_elems[i,4],times,g_arr,s_arr,gs_dict,
                              small_planets_flag, windows=5,debug=False,objname=des, rms = True, shortfilt=True))
            

    if(flag <1):
        logmessage = 'prop_elem.compute_prop() did not succeed\n'
        logmessage += 'failed at prop_elem.calc_proper_elements\n'
        tools.writelog(logfile,logmessage)
        if(logfile != 'screen'):
            print(logmessage)            
        return flag, proper_object

    prop_elem = {}
    prop_elem['a'] = [pes[0]]
    prop_elem['e'] = [pes[1]]
    prop_elem['sinI'] = [pes[2]]
    prop_elem['g(rev/yr)'] = [g]
    prop_elem['s(rev/yr)'] = [s]
    prop_elem['g("/yr)'] = [g*3600*360]
    prop_elem['s("/yr)'] = [s*3600*360]
    prop_elem['omega'] = [omega_n[ind0]]
    prop_elem['Omega'] = [Omega_n[ind0]]
    
    prop_errs = {}
    prop_errs['RMS_a'] = [rms_val[0]]
    prop_errs['RMS_e'] = [rms_val[1]]
    prop_errs['RMS_sinI'] = [rms_val[2]]
    prop_errs['RMS_g(rev/yr)'] = [rms_val[3]]
    prop_errs['RMS_s(rev/yr)'] = [rms_val[4]]
    prop_errs['RMS_g("/yr)'] = [rms_val[3]*3600*360]
    prop_errs['RMS_s("/yr)'] = [rms_val[4]*3600*360]

    if clones > 0:
        for i in range(clones):
            prop_elem['a'].append(c_results[i][1][0])
            prop_elem['e'].append(c_results[i][1][1])
            prop_elem['sinI'].append(c_results[i][1][2])
            prop_elem['g(rev/yr)'].append(c_results[i][7])
            prop_elem['s(rev/yr)'].append(c_results[i][8])
            prop_elem['g("/yr)'].append(c_results[i][7]*3600*360)
            prop_elem['s("/yr)'].append(c_results[i][8]*3600*360)

            prop_errs['a'].append(c_results[i][2][0])
            prop_errs['e'].append(c_results[i][2][1])
            prop_errs['sinI'].append(c_results[i][2][2])
            prop_errs['g(rev/yr)'].append(c_results[i][2][3])
            prop_errs['s(rev/yr)'].append(c_results[i][2][4])
            prop_errs['g("/yr)'].append(c_results[i][2][3]*3600*360)
            prop_errs['s("/yr)'].append(c_results[i][2][4]*3600*360)

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

    proper_object.proper_indicators['Ecc Mean Flag'] = rese
    proper_object.proper_indicators['sinI Mean Flag'] = resI
    proper_object.proper_indicators['Ecc Mean Indicator'] = sec_res_e
    proper_object.proper_indicators['sinI Mean Indicator'] = sec_res_I
    
    proper_object.proper_indicators['Ecc Osculating Amplitude'] = e_osc_amp
    proper_object.proper_indicators['sinI Osculating Amplitude'] = I_osc_amp
    proper_object.proper_indicators['Ecc Filtered Amplitude'] = e_amp
    proper_object.proper_indicators['sinI Filtered Amplitude'] = I_amp

    proper_object.proper_indicators['Distance Metric'] = hcm_calc(prop_elem['a'][0], prop_errs['RMS_a'][0],
                                                                  prop_errs['RMS_e'][0], prop_errs['RMS_sinI'][0])

    proper_object.proper_internal['Secular Resonant Angle'] = angle_sec_res
    proper_object.proper_internal['Librating Angle'] = librate_angle
    proper_object.proper_internal['Angle Entropy'] = angle_ent
    proper_object.proper_internal['Phi Entropy'] = phifrac

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

        proper_object.secfreq_flags[key] = (abs(val*3600*360) < 0.01, abs(val*3600*360) < 0.05, 
                                            abs(val*3600*360) < 0.1, abs(val*3600*360) < 0.2)

    fam_results = check_family_candidates(proper_object)

    proper_object.family_results = fam_results


    scat_results = check_scatter(times, a_init, e_init)
    proper_object.scattered = scat_results
        
    return 1, proper_object
    

def check_scatter(t,a,e):
    """
    A function that checks whether a small body's orbit experiences a potential planet-corssing 
    orbit, or is scattered, causing a large change in the orbital energy.
    """
    da = np.gradient(a)
    de = np.abs(da/a)[1:-2]

    q = a*(1-e)
    Q = a*(1+e)

    scat_result = {'scattered': False, 'scat_time': np.inf, 'Max delta-E': np.max(de), 'qlim': 0, 'Qlim': np.inf, 
                   'pcrossing_flag': False, 'qmin': np.min(q), 'Qmax': np.max(Q)}

    if np.nanmean(a) < 20:
        thresh = 1e-2
        scat_result['qlim'] = 1.7
        scat_result['Qlim'] = 4.1
    else:
        thresh = 1e-3
        scat_result['qlim'] = 34
        scat_result['Qlim'] = 1e5
    
    if np.max(de) > thresh:
        scat_ind = np.argmax(de)
        
        scat_result['scattered'] = True
        scat_result['scat_time'] = t[scat_ind]
        scat_result['scat_ind'] = scat_ind
        scat_result['Max delta-E'] = de[scat_ind]

    if np.min(q) < scat_result['qlim']:
        scat_result['pcrossing_flag'] = True
    if np.max(Q) > scat_result['Qlim']:
        scat_result['pcrossing_flag'] = True
    
    return scat_result
        

    
def check_family_candidates(proper_object):
    '''
    Stub function still in development
    Currently only looks at TNOs 
    '''
    family_occupancy = {'family_name': None, 'pairwise_dMet': np.inf}
    prope = proper_object.proper_elements
    
    if prope['a'][0] > 20:
        default_family_file = 'tno_family_centers.txt'
        family_file =  impresources.files(PEdata) / default_family_file 
        fam_df = pd.read_csv(family_file, index_col=0)
    else:
        return family_occupancy
    
    for i in range(len(fam_df)):
        fam_obj = fam_df.iloc[i]
        hcm_cen = hcm_pair(fam_obj['cen_a'], prope['a'][0], 
                           fam_obj['cen_e'], prope['e'][0], 
                           np.sin(fam_obj['cen_I']/180*np.pi), prope['sinI'][0])

        if( (hcm_cen < fam_obj['hcm_cut']) and (prope['a'][0] > fam_obj['low_a']) and (prope['a'][0] < fam_obj['high_a']) 
                and (prope['e'][0] > fam_obj['low_e']) and (prope['e'][0] < fam_obj['high_e']) and 
                (prope['sinI'][0] > np.sin(np.pi/180*fam_obj['low_I'])) and 
                (prope['sinI'][0] < np.sin(np.pi/180*fam_obj['high_I']))):
            family_occupancy['family_name'] = fam_obj['objname']
            family_occupancy['pairwise_dMet'] = hcm_cen
            break

    return family_occupancy
        
   
