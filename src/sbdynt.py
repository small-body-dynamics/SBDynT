from horizons_api import *
from tools import *
from resonances import *
from run_reb import *
from plotting_scripts import *
from hard_coded_constants import *
from add_orbits import *

from horizons_api import *
from tools import *
from resonances import *
from run_reb import *
from plotting_scripts import *
from hard_coded_constants import *
from add_orbits import *
from machine_learning import *
from tno_classifier import *
#from proper_elements import *
from tno import *
from prop_elem import *
from chaos_indicators import *

from datetime import datetime
import os

import rebound

# Globally disable two rebound-specific warnings that come up a lot regarding the 
# use of simulation archive functions
import warnings
warnings.filterwarnings("ignore", message="You have to reset function pointers after creating a reb_simulation struct with a binary file.")
warnings.filterwarnings("ignore", message="File in use for Simulationarchive already exists. Snapshots will be appended.")

# Globally disable a syntaxwarning that has to do with generating latex-friendly plot labele
#warnings.filterwarnings("ignore", category=SyntaxWarning)


class small_body:
    # class that is returned by the main sbdynt function
    # it contains all of the calculated dynamical parameters and 
    # other pertinent information about the simulations
    def __init__(self, designation, object_type=None, clones=0, savefile=False, filetype='Single'):
        self.designation = designation
        self.object_type = object_type
        self.clones = clones
        self.cloning_method = None
        self.clone_weights = np.ones(clones+1)
        self.planets = None
        self.archivefile = None
        self.icfile = None
        self.logfile = None
        if(object_type == 'tno'):
            self.tno_ml_outputs = None
        self.proper_elements = None

        #DS added variables for proper_element branch
        self.savefile = savefile
        self.filetype = filetype
        self.init_file = ''
        self.out_file1 = ''
        self.out_file2 = ''
        
        self.a_arr = []
        self.e_arr = []
        self.I_arr = []
        self.o_arr = []
        self.O_arr = []
        self.t_arr = []
        
        self.sim_init = None
        self.planets = {}
        self.planet_freqs = {}
        self.tmax = 0
        self.tout = 0
        self.osculating_elements = {}
        self.proper_elements = {}
        self.mean_elements = {}
        self.proper_errors = {}
        self.proper_windows = []
        self.proper_extras = {}
        self.prop_finish = False
        
        self.chaos_indicators = {'ACFI': None, 'Entropy': None, 'Power': None, 'Distance Metric': None, 'Clone_RMS_a': None, 'Clone_RMS_e': None, 'Clone_RMS_sinI': None}
        self.chaos_clones = {'ACFI': None, 'Entropy': None, 'Power': None, 'Distance Metric': None, 'Clone_RMS_a': None, 'Clone_RMS_e': None, 'Clone_RMS_sinI': None}
        

    def init_pe(self, planets = None, filename=None):
        if filename != None:
            self.filetype = filename
        
        if planets == None:
            if self.object_type == 'asteroid':
                planet_id = {2: 'venus', 3: 'earth', 4: 'mars', 5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune'}
            elif self.object_type == 'tno':
                planet_id = {5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune'}
            else:
                print('This small_body has not been assigned an ``asteroid`` or ``tno`` designation.')
                print('Please either give this particle a valid ``object_type``, or define the ``planets`` variable in the ``init_pe`` function call to be one of the 3 presets [1,2,3], or some self-defined dictionary of planets, e.g. {8: " neptune"}')
                return
                
                #planet_id = {5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune'}
                
        elif isinstance(planets, int):
            if planets == 1:
                planet_id = {2: 'venus', 3: 'earth', 4: 'mars', 5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune'}
            elif planets == 2:
                planet_id = {5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune'}
            elif planets == 3:
                planet_id = {1:'merucry', 2: 'venus', 3: 'earth', 4: 'mars', 5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune'}
        elif isinstance(planets, dict):
            planet_id = planets
        else:
            print('Error: variable ``planets`` is neither None, an integer of type [1,2,3], or a dictionary of planets. Please correct the variable.')
            return 

        print('Adding: ',planet_id)
        if self.savefile:
            obj_directory = '../data/'+self.filetype+'/'+str(self.designation)
            os.makedirs(obj_directory, exist_ok=True)
        
        flag, epoch, sim = run_reb.initialize_simulation(planets=list(planet_id.values()), des=str(self.designation), clones=self.clones)
        
        if self.savefile:
            self.init_file = '../data/'+str(self.filetype)+'/'+str(self.designation) + '/archive_init.bin'
            sim.save_to_file(self.init_file)

        self.sim_init = sim

    def integrate(self, tmax=None, tout=None, direction='both', deletefile=True): 
        begin = datetime.now()
        if tmax == None:
            if self.object_type == 'asteroid':
                self.tmax = 1e7
            elif self.object_type == 'tno':
                self.tmax = 1.5e8
        else:
            self.tmax = tmax
            
        if tout == None:
            if self.object_type == 'asteroid':
                self.tout = 5e2
            elif self.object_type == 'tno':
                self.tout = 5e3
        else:
            self.tout = tout               
        
        if self.savefile == True:
            sim2 = rebound.Simulation(self.init_file)
        else:
            sim2 = self.sim_init

        obj_directory = '../data/'+str(self.filetype)+'/'+str(self.designation)
        os.makedirs(obj_directory, exist_ok=True)

        self.direction = direction
        try:
            if direction == 'backwards':
                self.tmax = -abs(self.tmax)
                self.out_file1 = 'data/'+str(self.filetype)+'/'+str(self.designation) + '/archive.bin'
                sim = run_reb.run_simulation(sim2, tmax=self.tmax, tout=self.tout, filename=self.out_file1, deletefile=deletefile)
                
            elif direction == 'both':
                tmax1 = -abs(self.tmax)/2
                tmax2 = abs(self.tmax)/2
                
                self.out_file1 = '../data/'+str(self.filetype)+'/'+str(self.designation) + '/archive_back.bin'
                self.out_file2 = '../data/'+str(self.filetype)+'/'+str(self.designation) + '/archive_forward.bin'
                
                simb = run_reb.run_simulation(sim2, tmax=tmax1, tout=self.tout, filename=self.out_file1, deletefile=deletefile)
                simf = run_reb.run_simulation(sim2, tmax=tmax2, tout=self.tout, filename=self.out_file2, deletefile=deletefile)
            else:
                sim = run_reb.run_simulation(sim2, tmax=abs(self.tmax), tout=self.tout, filename=self.out_file1, deletefile=deletefile)
        except:
            print('The particle was likely ejected, or some other break point occurred within the Rebound integration. Simulation ended.')
        print('Integration Completed; run took ', datetime.now() - begin, ' seconds.')
        
        self.read_orbel_to_small_body()
        #print('Now reading in orbital element time arrays to small_body object')
        

    def read_orbel_to_small_body(self):
        fullfile = self.out_file1        
        archive = rebound.Simulationarchive(fullfile)

        if self.direction == 'both':
            fullfile_b = self.out_file1
            fullfile_f = self.out_file2

            archive_b = rebound.Simulationarchive(fullfile_b)
            archive_f = rebound.Simulationarchive(fullfile_f)

            flagb, a_initb, e_initb, inc_initb, lan_initb, aop_initb, M_initb, t_initb = read_sa_for_sbody(des = str(self.designation), archivefile=fullfile_b,clones=self.clones,tmax=0,tmin=-abs(self.tmax/2),center='helio',s=archive_b)
            flagf, a_initf, e_initf, inc_initf, lan_initf, aop_initf, M_initf, t_initf = read_sa_for_sbody(des = str(self.designation), archivefile=fullfile_f,clones=self.clones,tmin=0.,tmax=abs(self.tmax/2),center='helio',s=archive_f)

            if self.clones > 0:
                self.a_arr = np.zeros((self.clones + 1, int(len(a_initb[0])*2-1)))
                self.e_arr = np.zeros((self.clones + 1, int(len(a_initb[0])*2-1)))
                self.I_arr = np.zeros((self.clones + 1, int(len(a_initb[0])*2-1)))
                self.o_arr = np.zeros((self.clones + 1, int(len(a_initb[0])*2-1)))
                self.O_arr = np.zeros((self.clones + 1, int(len(a_initb[0])*2-1)))
                self.M_arr = np.zeros((self.clones + 1, int(len(a_initb[0])*2-1)))
                self.t_arr = np.zeros(int(len(a_initb[0])*2-1))
                
                self.a_arr[:,:len(a_initb[0])] = a_initb[:,::-1]
                self.a_arr[:,len(a_initb[0]):] =  a_initf[:,1:]
                
                self.e_arr[:,:len(a_initb[0])] = e_initb[:,::-1]
                self.e_arr[:,len(a_initb[0]):] =  e_initf[:,1:]
                
                self.I_arr[:,:len(a_initb[0])] = inc_initb[:,::-1]
                self.I_arr[:,len(a_initb[0]):] =  inc_initf[:,1:]
                
                self.o_arr[:,:len(a_initb[0])] = aop_initb[:,::-1]
                self.o_arr[:,len(a_initb[0]):] =  aop_initf[:,1:]
                
                self.O_arr[:,:len(a_initb[0])] = lan_initb[:,::-1]
                self.O_arr[:,len(a_initb[0]):] =  lan_initf[:,1:]
                
                self.M_arr[:,:len(a_initb[0])] = M_initb[:,::-1]
                self.M_arr[:,len(a_initb[0]):] =  M_initf[:,1:]
                
                self.t_arr[:len(a_initb[0])] = t_initb[::-1]
                self.t_arr[len(a_initb[0]):] =  t_initf[1:]
                
            else:
                self.a_arr = np.concatenate((a_initb[::-1], a_initf[1:]))
                self.e_arr = np.concatenate((e_initb[::-1], e_initf[1:]))
                self.I_arr = np.concatenate((inc_initb[::-1], inc_initf[1:]))
                self.O_arr = np.concatenate((lan_initb[::-1], lan_initf[1:]))
                self.o_arr = np.concatenate((aop_initb[::-1], aop_initf[1:]))
                self.t_arr = np.concatenate((t_initb[::-1], t_initf[1:]))
                self.M_arr = np.concatenate((M_initb[::-1], M_initf[1:]))
                
            
        elif self.direction == 'forwards':
            flag, a_init, e_init, inc_init, lan_init, aop_init, M_init, t_init = tools.read_sa_for_sbody(des = str(self.designation), archivefile=fullfile,clones=self.clones,tmin=0.,tmax=abs(self.tmax),center='helio',s=archive)

            self.a_arr = a_init
            self.e_arr = e_init
            self.I_arr = I_init
            self.o_arr = aop_init
            self.O_arr = lan_init
            self.M_arr = M_init
            self.t_arr = t_init
            
        elif self.direction == 'backwards':
            flag, a_init, e_init, inc_init, lan_init, aop_init, M_init, t_init = tools.read_sa_for_sbody(des = str(objname), archivefile=fullfile,clones=self.clones,tmax=0.,tmin=-abs(self.tmax),center='helio',s=archive)

            self.a_arr = a_init
            self.e_arr = e_init
            self.I_arr = I_init
            self.o_arr = aop_init
            self.O_arr = lan_init
            self.M_arr = M_init
            self.t_arr = t_init
        
    

    def compute_proper(self, windows=5, time_run = 0, rms = True, debug = False):
        self.windows = windows
        self.time_run = time_run
        
        outputs = prop_calc(self.designation, filename=self.filetype, windows=windows, direction = self.direction, time_run = time_run, rms = rms, debug=debug)

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
        print('Proper Elements:',self.proper_elements)

    def compute_chaos(self):

        if len(self.a_arr) > 0:
            if self.clones == 0:
                self.chaos_indicators['ACFI'] = ACFI_calc(self.a_arr)
                self.chaos_indicators['Entropy'] = entropy_calc(self.a_arr, self.e_arr, self.I_arr)
                self.chaos_indicators['Power'] = power_calc(self.e_arr, self.I_arr, self.o_arr, self.O_arr, size=5)
            else:
                self.chaos_indicators['ACFI'] = ACFI_calc(self.a_arr[0])
                self.chaos_indicators['Entropy'] = entropy_calc(self.a_arr[0], self.e_arr[0], self.I_arr[0])
                self.chaos_indicators['Power'] = power_calc(self.e_arr[0], self.I_arr[0], self.o_arr[0], self.O_arr[0], size=5)

        if self.prop_finish:
            self.chaos_indicators['Distance Metric'] = hcm_calc(self.proper_elements['a'],
                                                                self.proper_errors['RMS_a'],self.proper_errors['RMS_e'],self.proper_errors['RMS_sinI'])

        if len(self.a_arr) > 0 and self.clones > 0:
            for i in range(1,self.clones+1):
                diff_a = ((self.a_arr[0] - self.a_arr[i])/np.mean(self.a_arr[0]))**2
                diff_e = (self.e_arr[0] - self.e_arr[i])**2
                diff_I = (np.sin(self.I_arr[0]) - np.sin(self.I_arr[i]))**2
                
            self.chaos_indicators['Clone_RMS_a'] = np.sqrt(np.nanmean(diff_a))
            self.chaos_indicators['Clone_RMS_e'] = np.sqrt(np.nanmean(diff_e))
            self.chaos_indicators['Clone_RMS_sinI'] = np.sqrt(np.nanmean(diff_I))

        
        

        

        
        

#function to do a standard TNO analysis run using all default choices
def run_tno(des=None, clones=None, datadir='',archivefile=None,
            logfile=False,deletefile=False):
    '''
    documentation here...
    '''
    if(des == None):
        print("The designation of a TNO must be provided")
        return None

    if(logfile==True):
        logf = log_file_name(des=des)
    else:
        logf = logfile
    if(datadir and logf and logf!='screen'):        
        logf = datadir + '/' +logf

    if(logf):
        logmessage = "Initializing a TNO simulation instance by querying JPL"
        writelog(logf,logmessage)  

    iflag, sim, epoch, clones, cloning_method, weights = \
                setup_default_tno_integration(des=des, clones=clones, datadir=datadir,
                                              save_sbdb=False,saveic=True,
                                              archivefile=archivefile,
                                              logfile=logfile)
    

    if(iflag < 1):
        print("Failed at initialization stage")
        return None

    object_type = 'tno'
    #initialize the results class
    tno_results = small_body(des,object_type,clones)
    tno_results.planets = ['outer']
    tno_results.cloning_method = cloning_method
    tno_results.clone_weights = weights

    icfile = ic_file_name(des=des)
    if(datadir):
        icfile = datadir + '/' + icfile

    if(archivefile==None):
        file = archive_file_name(des=des)
    if(datadir):
        file = datadir + '/' + file
    tno_results.archivefile = file
    tno_results.icfile = icfile
    tno_results.logfile = logf

    cflag, tno_class, sim = run_and_MLclassify_TNO(sim=sim,des=des,clones=clones,
                                    archivefile=archivefile,datadir=datadir,
                                    deletefile=deletefile,logfile=logfile)

    if(cflag < 1):
        print("Failed at the machine learning stage")
        return tno_results

    if(logf=='screen'):
        tno_class.print_results()

    tno_results.tno_ml_outputs = tno_class

    if(tno_class.most_common_class == 'scattering'):
        if(logf):
            logmessage = "This is most likely a scattering TNO.\n"
            logmessage += "Proper a,e,sini will be 10 Myr averages.\n"
            logmessage += "No further integration needed."
            writelog(logf,logmessage)  
        tno_results.proper_elements = proper_elements(clones)
        tno_results.proper_elements.a = tno_class.features.a_mean
        tno_results.proper_elements.e = tno_class.features.e_mean
        tno_results.proper_elements.sini = np.sin(tno_class.features.a_mean)
        return tno_results
    else:
        if(logf):
            logmessage = "Running additional forward integrations for the synthetic\n"
            logmessage+= "proper elements calculation\n"
            writelog(logf,logmessage)  

        rflag, sim = run_simulation(sim,des=des,archivefile=archivefile,datadir=datadir,
                                       logfile=logfile,tmax=50e6,tout=1000.)
        if(rflag < 1):
            print("Failed at additional forward integration stage")
            return tno_results
        
        if(logf):
            logmessage = "Running additional backward integrations for the synthetic\n"
            logmessage+= "proper elements calculation\n"
            writelog(logf,logmessage)  

        try:        
            sa = rebound.Simulationarchive(icfile)
            sim2 = sa[0]
        except:
            print("failed to read in the saved initial conditions file to restart from t=0")
            return tno_results

        rflag, sim2 = run_simulation(sim2,des=des,archivefile=archivefile,datadir=datadir,
                                         logfile=logfile,tmax=-50e6,tout=1000.)
        if(rflag < 1):
            print("Failed at backward integration stage")
            return tno_results

        if(logf):
            logmessage = "Running the synthetic proper elements calculation\n"
            writelog(logf,logmessage)  
            
        pflag, pe = calc_proper_elements(des=des,datadir=datadir,archivefile=archivefile,
                                        clones=clones,return_timeseries=True,logfile=logfile)

        if(pflag < 1):
            print("Failed at proper elements calculation stage")
            return tno_results

        tno_results.proper_elements = pe


    return tno_results