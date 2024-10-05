from horizons_api import *
from tools import *
from resonances import *
from run_reb import *
from plotting_scripts import *
from hard_coded_constants import *
from add_orbits import *
from machine_learning import *
from proper_elements import *
from tno import *

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
        self.proper_elements = None

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
