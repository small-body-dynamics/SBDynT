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
from asteroid import *
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
    def __init__(self, designation, object_type=None, clones=0):
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

        self.tmax = 0
        self.tout = 0
        self.int_direction = ''
        self.run_properties = self.analysis_vars()
            
        self.proper_elements = proper_element_class(designation)
        self.chaos_indicators = chaos_indicators()

        #DS added variables for proper_element branch
        self.a_arr = []
        self.e_arr = []
        self.I_arr = []
        self.o_arr = []
        self.O_arr = []
        self.t_arr = []


    def analysis_vars(self, tmax = None, tout = None, int_direction = None, planets=None):
        
        allowed_d = ['backwards','forwards','bf']
        if(tmax == None):
            if(self.object_type == 'asteroid'):
                self.tmax = 10e6
            elif(self.object_type == 'tno'):
                self.tmax = 150e6
            else:
                print('object_type is neither asteroid or tno. Setting default tmax = 50 Myr.')
                self.tmax = 50e6
                
        if(tout == None):
            if(self.object_type == 'asteroid'):
                self.tout = 500
            elif(self.object_type == 'tno'):
                self.tout = 5000
            else:
                print('object_type is neither asteroid or tno. Setting default tout = 2000 yr.')
                self.tout = 2000
                
        if(int_direction == None):
            self.int_direction = 'bf'
        elif(int_direction not in allowed_d):
            print('The given direction variable=', int_direction, ' is not one of the 4 allowed inputs ("backwards","forwards","b+f", None). Setting direction to "bf". Call this function again with a valid direction is you do not want a forwards + backwards integration.')  
            self.int_direction = 'bf'

        if(planets==None):
            if(self.object_type == 'asteroid'):
                self.planets = ['inner']
            elif(self.object_type == 'tno'):
                self.planets = ['outer']
            else:
                print('object_type is neither asteroid or tno. Setting default planets = ["outer"]')
                self.planets = ['outer']
            
        
def setup_sb_integration(des=None, sb_results=None, clones=None, datadir='',save_sbdb=False,
                                  saveic=False,archivefile=None,logfile=False):
    '''

    '''
    flag = 0
    if(des == None):
        print("The designation of a Small Body must be provided")
        print("failed at sbdynt.setup_default_sb_integration()")
        return flag, None, None, None, None, None
    
    if(logfile==True):
        logf = tools.log_file_name(des=des)
    else:
        logf = logfile
    if(datadir and logf and logf!='screen'):        
        logf = datadir + '/' +logf

    if(clones==None):
        #default to the Gladman approach of best fit + 3-sigma clones
        clones = 2
        cloning_method = 'find_3_sigma'
        if(logf):
            logmessage = "Clones were not specified, so the default behavior is to return\n"
            logmessage += "a best-fit and 3-sigma minimum and maximum semimajor axis clones\n"
            tools.writelog(logf,logmessage)  
        iflag, epoch, sim, weights = run_reb.initialize_simulation(planets=sb_results.planets,
                          des=des, clones=clones, cloning_method= cloning_method,datadir=datadir,
                          logfile=logfile, save_sbdb=save_sbdb, saveic=saveic)
    else:
        cloning_method = 'Gaussian'
        iflag, epoch, sim = run_reb.initialize_simulation(planets=sb_results.planets,
                          des=des, clones=clones, cloning_method= cloning_method,datadir=datadir,
                          logfile=logfile, save_sbdb=save_sbdb, saveic=saveic)
        weights = np.ones(clones+1)


    if(iflag < 1):
        print("failed at tno.setup_default_tno_integration()")
        return flag, None, None, None, None, None


    flag = 1
    return flag, sim, epoch, clones, cloning_method, weights            
    
#function to do a standard Asteroid analysis run using all default choices
def run_ast(des=None, clones=None, datadir='',archivefile=None,
            logfile=False,deletefile=False, run_proper = False, run_chaos = False):
    '''
    documentation here...
    '''
    if(des == None):
        print("The designation of an asteroid must be provided")
        return None

    if(logfile==True):
        logf = log_file_name(des=des)
    else:
        logf = logfile

    if(datadir):
        tools.check_datadir(datadir)

    if(datadir and logf and logf!='screen'):        
        logf = datadir + '/' +logf

    if(logf):
        logmessage = "Initializing an Asteroid simulation instance by querying JPL"
        writelog(logf,logmessage)  

    iflag, sim, epoch, clones, cloning_method, weights = \
                setup_default_ast_integration(des=des, clones=clones, datadir=datadir,
                                              save_sbdb=False,saveic=True,
                                              archivefile=archivefile,
                                              logfile=logfile)
    

    if(iflag < 1):
        print("Failed at initialization stage")
        return None

    object_type = 'asteroid'
    #initialize the results class
    ast_results = small_body(des,object_type,clones)
    ast_results.planets = ['inner']
    ast_results.cloning_method = cloning_method
    ast_results.clone_weights = weights

    icfile = ic_file_name(des=des)
    if(datadir):
        icfile = datadir + '/' + icfile

    
    if(archivefile==None):
        file = archive_file_name(des=des)
    else:
        file = archivefile
    if(datadir):
        file = datadir + '/' + file
    ast_results.archivefile = file
    ast_results.icfile = icfile
    ast_results.logfile = logf


    if(logf):
        logmessage = "Running additional forward integrations for the synthetic\n"
        logmessage+= "proper elements calculation\n"
        writelog(logf,logmessage)  

    print('Running Asteroid integration for Proper Elements')
    rflag, sim = run_simulation(sim,des=des,archivefile=archivefile,datadir=datadir,
                                       logfile=logfile,tmax=5e6,tout=500.)
    if(rflag < 1):
        print("Failed at forward integration stage")
        return ast_results
        
    if(logf):
        logmessage = "Running additional backward integrations for the synthetic\n"
        logmessage+= "proper elements calculation\n"
        writelog(logf,logmessage)  

    try:        
        sa = rebound.Simulationarchive(icfile)
        sim2 = sa[0]
    except:
        print("failed to read in the saved initial conditions file to restart from t=0")
        return ast_results

    rflag, sim2 = run_simulation(sim2,des=des,archivefile=archivefile,datadir=datadir,
                                         logfile=logfile,tmax=-5e6,tout=500.)
    if(rflag < 1):
        print("Failed at backward integration stage")
        return ast_results

    if(logf):
        logmessage = "Running the synthetic proper elements calculation\n"
        writelog(logf,logmessage)  

    print('Reading Asteroid integration')
    reflag, times, sb_elems, planet_elems, clone_elems, small_planets_flag = read_archive_for_pe(des=des,datadir=datadir,archivefile=archivefile,
                                        clones=clones,logfile=logfile, object_type = ast_results.object_type)

    if(reflag < 1):
        print("Failed when reading in integrated simulation for proper elements")
        return tno_results

    if run_proper:
        print('Running Asteroid Proper Elements')
        pflag, pe = calc_proper_elements(des=des, times = times, sb_elems = sb_elems, 
                                             planet_elems = planet_elems, small_planets_flag = small_planets_flag)

        if(pflag < 1):
            print("Failed at proper elements calculation stage")
            return ast_results
            
        ast_results.proper_elements = pe

    if run_chaos:
        print('Running Asteroid Chaos Indicators')
        ast_results.chaos_indicators = compute_chaos(sb_elem = sb_elems, clones=clones, pe_obj = pe, clone_elems = clone_elems)

    return ast_results
        
        

def integrate_for_pe(sim, des=None, archivefile=None,datadir=None,
                                       logfile=None,tmax=10e6,tout=500., direction='bf', deletefile = True):

    icfile = ic_file_name(des=des)
    if(datadir):
        icfile = datadir + '/' + icfile
        
    if direction == 'bf':
        
        rflag, sim = run_simulation(sim,des=des,archivefile=archivefile,datadir=datadir,
                                       logfile=logfile,tmax=(tmax/2),tout=tout, deletefile = deletefile)
        try:        
            sa = rebound.Simulationarchive(icfile)
            snew = sa[0]
        except:
            print("failed to read in the saved initial conditions file to restart from t=0")
            return 0, sim

        return run_simulation(snew,des=des,archivefile=archivefile,datadir=datadir,
                                         logfile=logfile,tmax=-(tmax/2),tout=tout, deletefile = False)


    elif direction == 'forwards':
        
        return run_simulation(sim,des=des,archivefile=archivefile,datadir=datadir,
                                       logfile=logfile,tmax=tmax,tout=tout, deletefile = deletefile)

    elif direction == 'backwards':

        return  run_simulation(sim,des=des,archivefile=archivefile,datadir=datadir,
                                       logfile=logfile,tmax=-tmax,tout=tout, deletefile = deletefile)

    else:
        print('direction given not backwards, forwards, or bf. Call this function again with a valid direction')
        return 0,sim

    

        
    
#function to do a standard TNO analysis run using all default choices
def run_tno(des=None, clones=None, datadir='',archivefile=None,
            logfile=False,deletefile=False, run_ML = True, run_proper = False, run_chaos = False):
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

    if(datadir):
        tools.check_datadir(datadir)

    if(datadir and logf and logf!='screen'):        
        logf = datadir + '/' +logf

    if(logf):
        logmessage = "Initializing a TNO simulation instance by querying JPL"
        writelog(logf,logmessage)  

    print('Initializing TNO')
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
    else:
        file = archivefile
    if(datadir):
        file = datadir + '/' + file
    tno_results.archivefile = file
    tno_results.icfile = icfile
    tno_results.logfile = logf

    #'''
    sim = None
    
    tno_class = TNO_ML_outputs()
    if run_ML:
        print('Running TNO ML')
        cflag, tno_class, sim = run_and_MLclassify_TNO(sim=sim,des=des,clones=clones,
                                    archivefile=archivefile,datadir=datadir,
                                    deletefile=deletefile,logfile=logfile)

        if(cflag < 1):
            print("Failed at the machine learning stage")
            return tno_results

        if(logf=='screen'):
            tno_class.print_results()

        tno_results.tno_ml_outputs = tno_class
    #'''

    if run_proper:
        if(tno_class.most_common_class == 'scattering'):
            if(logf):
                logmessage = "This is most likely a scattering TNO.\n"
                logmessage += "Proper a,e,sini will be 10 Myr averages.\n"
                logmessage += "No further integration needed."
                writelog(logf,logmessage)  
            
            tno_results.proper_elements.proper_elements.a = tno_class.features.a_mean
            tno_results.proper_elements.proper_elements.e = tno_class.features.e_mean
            tno_results.proper_elements.proper_elements.sinI = np.sin(tno_class.features.i_mean)
        
            return tno_results
        else:
            if(logf):
                logmessage = "Running additional forward integrations for the synthetic\n"
                logmessage+= "proper elements calculation\n"
                writelog(logf,logmessage)  

            #for i in sim.particles:
            #    print(i)
            print('Running TNO integration for PE')
            rflag, sim = run_simulation(sim,des=des,archivefile=archivefile,datadir=datadir,
                                       logfile=logfile,tmax=75e6,tout=5000.)
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
                                         logfile=logfile,tmax=-75e6,tout=5000.)
            if(rflag < 1):
                print("Failed at backward integration stage")
                return tno_results

            if(logf):
                logmessage = "Running the synthetic proper elements calculation\n"
                writelog(logf,logmessage)  

            print('Reading TNO integration')
            reflag, times, sb_elems, planet_elems, clone_elems, small_planets_flag = read_archive_for_pe(des=des,datadir=datadir,archivefile=archivefile,
                                        clones=clones,logfile=logfile, object_type = tno_results.object_type)

            if(reflag < 1):
                print("Failed when reading in integrated simulation for proper elements")
                return tno_results
                
            print('Running TNO PE')
            pflag, pe = calc_proper_elements(des=des, times = times, sb_elems = sb_elems, 
                                             planet_elems = planet_elems, small_planets_flag = small_planets_flag)
            #pflag, pe, sb_elems = calc_proper_elements(tno_results.proper_elements, des=des,datadir=datadir,archivefile=archivefile,
            #                            clones=clones, logfile=logfile, int_data=int_data)

            if(pflag < 1):
                print("Failed at proper elements calculation stage")
                return tno_results

            tno_results.proper_elements = pe
            
    elif run_chaos:
        reflag, times, sb_elems, planet_elems, clone_elems, small_planets_flag = read_archive_for_pe(des=des,datadir=datadir,archivefile=archivefile,
                                        clones=clones,logfile=logfile, object_type = tno_results.object_type)
    if run_chaos:
        tno_results.chaos_indicators = compute_chaos(sb_elem = sb_elems, clones=clones, pe_obj = tno_results.proper_elements, clone_elems = clone_elems)
        

    return tno_results


#function to do a standard small body analysis run using given or default choices
def run_sb(des=None, object_type=None, clones=None, datadir='',archivefile=None,
            logfile=False,deletefile=False):
    '''
    documentation here...
    '''
    if(des == None):
        print("The designation of a solar system small body must be provided")
        return None
    if(object_type == None):
        print("The object_type of a solar system small body must be provided ('asteroid' or 'tno'")
        return None

    if(logfile==True):
        logf = log_file_name(des=des)
    else:
        logf = logfile

    if(datadir):
        tools.check_datadir(datadir)

    if(datadir and logf and logf!='screen'):        
        logf = datadir + '/' +logf

    if(logf):
        logmessage = "Initializing a small body simulation instance by querying JPL"
        writelog(logf,logmessage)  

    print('Initializing Small Body')
    
    sb_results = small_body(des,object_type,clones)
    iflag, sim, epoch, clones, cloning_method, weights = \
                setup_default_sb_integration(des=des, sb_results=sb_results, clones=clones, datadir=datadir,
                                              save_sbdb=False,saveic=True,
                                              archivefile=archivefile,
                                              logfile=logfile)
    

    if(iflag < 1):
        print("Failed at initialization stage")
        return None

    
    #initialize the results class
    sb_results.cloning_method = cloning_method
    sb_results.clone_weights = weights

    icfile = ic_file_name(des=des)
    if(datadir):
        icfile = datadir + '/' + icfile

    
    if(archivefile==None):
        file = archive_file_name(des=des)
    else:
        file = archivefile
    if(datadir):
        file = datadir + '/' + file
    sb_results.archivefile = file
    sb_results.icfile = icfile
    sb_results.logfile = logf


    if(sb_results.object_type == 'tno'):
        print('Running SB ML')
        cflag, tno_class, sim = run_and_MLclassify_TNO(sim=sim,des=des,clones=clones,
                                    archivefile=archivefile,datadir=datadir,
                                    deletefile=deletefile,logfile=logfile)

        if(cflag < 1):
            print("Failed at the machine learning stage")
            return sb_results

        if(logf=='screen'):
            sb_class.print_results()

        sb_results.tno_ml_outputs = tno_class

        if(tno_class.most_common_class == 'scattering'):
            if(logf):
                logmessage = "This is most likely a scattering TNO.\n"
                logmessage += "Proper a,e,sini will be 10 Myr averages.\n"
                logmessage += "No further integration needed.\n"
                logmessage += "Chaos indicators will not be computed."
                writelog(logf,logmessage)  
            
            sb_results.proper_elements.proper_elements.a = tno_class.features.a_mean
            sb_results.proper_elements.proper_elements.e = tno_class.features.e_mean
            sb_results.proper_elements.proper_elements.sinI = np.sin(tno_class.features.i_mean)
        
            return sb_results

    if(logf):
        logmessage = "Running additional forward integrations for the synthetic\n"
        logmessage+= "proper elements calculation\n"
        writelog(logf,logmessage)  

        
    print('Running Small body integration for PE')

    
    if(sb_results.int_direction == 'forward'):
        rflag, sim = run_simulation(sim,des=des,archivefile=archivefile,datadir=datadir,
                                       logfile=logfile,tmax=sb_results.tmax,tout=sb_results.tout)
        if(rflag < 1):
            print("Failed at additional forward integration stage")
            return sb_results
    elif(sb_results.int_direction == 'backward'):
        try:        
            sa = rebound.Simulationarchive(icfile)
            sim2 = sa[0]
        except:
            print("failed to read in the saved initial conditions file to restart from t=0")
            return sb_results
        rflag, sim2 = run_simulation(sim2,des=des,archivefile=archivefile,datadir=datadir,
                                       logfile=logfile,tmax=-sb_results.tmax,tout=sb_results.tout)
        if(rflag < 1):
            print("Failed at backward integration stage")
            return sb_results
    elif(sb_results.int_direction == 'bf'):
        rflag, sim = run_simulation(sim,des=des,archivefile=archivefile,datadir=datadir,
                                       logfile=logfile,tmax=int(sb_results.tmax/2),tout=sb_results.tout)
        if(rflag < 1):
            print("Failed at additional forward integration stage")
            return sb_results
        
        if(logf):
            logmessage = "Running additional backward integrations for the synthetic\n"
            logmessage+= "proper elements calculation\n"
            writelog(logf,logmessage)  

        try:        
            sa = rebound.Simulationarchive(icfile)
            sim2 = sa[0]
        except:
            print("failed to read in the saved initial conditions file to restart from t=0")
            return sb_results

        rflag, sim2 = run_simulation(sim2,des=des,archivefile=archivefile,datadir=datadir,
                                         logfile=logfile,tmax=-int(sb_results.tmax/2),tout=sb_results.tout)
        if(rflag < 1):
            print("Failed at backward integration stage")
            return sb_results

    if(logf):
        logmessage = "Running the synthetic proper elements calculation\n"
        writelog(logf,logmessage)  

    print('Running TNO PE')
    pflag, pe, sb_elems = calc_proper_elements(des=des,datadir=datadir,archivefile=archivefile,
                                        clones=clones,return_timeseries=True,logfile=logfile)

    if(pflag < 1):
        print("Failed at proper elements calculation stage")
        return sb_results

    sb_results.proper_elements = pe

    sb_results.chaos_indicators = compute_chaos(sb_results.chaos_indicators, a_arr = sb_elems[1], e_arr = sb_elems[2], I_arr = sb_elems[3], o_arr = sb_elems[4], O_arr = sb_elems[5], clones=clones, pe_obj = pe)


    return sb_results


def run_existing_tno(des=None, clones=None, datadir='',archivefile=None,
            logfile=False,deletefile=False, run_ML = False, run_proper=False, run_chaos=False, output_arrays = True):
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

    if(datadir):
        tools.check_datadir(datadir)

    if(datadir and logf and logf!='screen'):        
        logf = datadir + '/' +logf
    

    iflag, sim, epoch, clones, cloning_method, weights = \
                setup_default_tno_integration(des=des, clones=clones, datadir=datadir,
                                              save_sbdb=False,saveic=True,
                                              archivefile=archivefile,
                                              logfile=logfile)
    object_type = 'tno'
    #initialize the results class
    tno_results = small_body(des,object_type,clones)
    tno_results.planets = ['outer']
    tno_results.cloning_method = 'Covariance Matrix'
    tno_results.clone_weights = None

    icfile = ic_file_name(des=des)
    if(datadir):
        icfile = datadir + '/' + icfile

    
    if(archivefile==None):
        file = archive_file_name(des=des)
    else:
        file = archivefile
    if(datadir):
        file = datadir + '/' + file
    tno_results.archivefile = file
    tno_results.icfile = icfile
    tno_results.logfile = logf

    tno_class = TNO_ML_outputs()
    if run_ML:
        #'''
        print('Running TNO ML')
        cflag, tno_class, sim = run_and_MLclassify_TNO(sim=sim,des=des,clones=clones,
                                    archivefile=archivefile,datadir=datadir,
                                    deletefile=deletefile,logfile=logfile, classify_only=True)

        if(cflag < 1):
            print("Failed at the machine learning stage")
            return tno_results

        if(logf=='screen'):
            tno_class.print_results()

        tno_results.tno_ml_outputs = tno_class
        #'''

    if run_proper or run_chaos:
        print('Reading TNO integration for Proper Elements and/or Chaos')
        reflag, times, sb_elems, planet_elems, clone_elems, small_planets_flag = read_archive_for_pe(des=des,datadir=datadir,archivefile=archivefile,
                                   clones=clones,logfile=logfile, object_type = tno_results.object_type)

        if(reflag < 1):
            print("Failed when reading in integrated simulation for proper elements")
            return tno_results
    
    if run_proper:
        print('Running TNO Proper Elements')
        
        if(tno_class.most_common_class == 'scattering'):
            if(logf):
                logmessage = "This is most likely a scattering TNO.\n"
                logmessage += "Proper a,e,sini will be 10 Myr averages.\n"
                logmessage += "No further integration needed."
                writelog(logf,logmessage)  
            
            tno_results.proper_elements.proper_elements.a = tno_class.features.a_mean
            tno_results.proper_elements.proper_elements.e = tno_class.features.e_mean
            tno_results.proper_elements.proper_elements.sinI = np.sin(tno_class.features.i_mean)
        
            return tno_results
        else:
                
            #print('Running TNO PE')
            pflag, pe = calc_proper_elements(des=des, times = times, sb_elems = sb_elems, 
                                             planet_elems = planet_elems, small_planets_flag = small_planets_flag, output_arrays = output_arrays)

            if(pflag < 1):
                print("Failed at proper elements calculation stage")
                return tno_results

            tno_results.proper_elements = pe
            
    if run_chaos:
        print('Running TNO Chaos Indicators')
        #print('clone elemes:', clone_elems.shape, clone_elems)
        tno_results.chaos_indicators = compute_chaos(sb_elem = sb_elems, clones=clones, pe_obj = pe, clone_elems = clone_elems)
        

    return tno_results


def run_existing_sb(des=None, clones=None, datadir='',archivefile=None,
            logfile=False,deletefile=False, run_proper=False, run_chaos=False, object_type=None, output_arrays = True):
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

    if(datadir):
        tools.check_datadir(datadir)

    if(datadir and logf and logf!='screen'):        
        logf = datadir + '/' +logf
    

    #object_type = obj
    #initialize the results class
    tno_results = small_body(des,object_type,clones)
    #tno_results.planets = ['outer']
    tno_results.cloning_method = 'Covariance Matrix'
    tno_results.clone_weights = None

    icfile = ic_file_name(des=des)
    if(datadir):
        icfile = datadir + '/' + icfile

    
    if(archivefile==None):
        file = archive_file_name(des=des)
    else:
        file = archivefile
    if(datadir):
        file = datadir + '/' + file
    tno_results.archivefile = file
    tno_results.icfile = icfile
    tno_results.logfile = logf

    tno_class = TNO_ML_outputs()
    
    if run_proper:
        if(tno_class.most_common_class == 'scattering'):
            if(logf):
                logmessage = "This is most likely a scattering TNO.\n"
                logmessage += "Proper a,e,sini will be 10 Myr averages.\n"
                logmessage += "No further integration needed."
                writelog(logf,logmessage)  
            
            tno_results.proper_elements.proper_elements.a = tno_class.features.a_mean
            tno_results.proper_elements.proper_elements.e = tno_class.features.e_mean
            tno_results.proper_elements.proper_elements.sinI = np.sin(tno_class.features.i_mean)
        
            return tno_results
        else:
            print('Reading Small Body integration')
            reflag, times, sb_elems, planet_elems, clone_elems, small_planets_flag = read_archive_for_pe(des=des,datadir=datadir,archivefile=archivefile,
                                        clones=clones,logfile=logfile, object_type = tno_results.object_type)

            if(reflag < 1):
                print("Failed when reading in integrated simulation for proper elements")
                return tno_results
                
            print('Running TNO PE')

            pflag, pe = calc_proper_elements(des=des, times = times, sb_elems = sb_elems, 
                                             planet_elems = planet_elems, small_planets_flag = small_planets_flag, output_arrays = output_arrays)

            if(pflag < 1):
                print("Failed at proper elements calculation stage")
                return tno_results

            tno_results.proper_elements = pe
            
    elif run_chaos:
        reflag, times, sb_elems, planet_elems, clone_elems, small_planets_flag = read_archive_for_pe(des=des,datadir=datadir,archivefile=archivefile,
                                        clones=clones,logfile=logfile, object_type = tno_results.object_type)
    if run_chaos:
        #print('clone elemes:', clone_elems.shape, clone_elems)
        tno_results.chaos_indicators = compute_chaos(sb_elem = sb_elems, clones=clones, pe_obj = pe, clone_elems = clone_elems)
        
   
        
    return tno_results