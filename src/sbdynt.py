from horizons_api import *
from tools import *
from resonances import *
from run_reb import *
from plotting_scripts import *
from hard_coded_constants import *
from add_orbits import *
from machine_learning import *
from tno_classifier import *

from tno import *
from asteroid import *
from stability_indicators import *
from prop_elem import *


from datetime import datetime
import os
import rebound
import random
import string
from os import remove

# Globally disable two rebound-specific warnings that come up a lot regarding the 
# use of simulation archive functions
import warnings
warnings.filterwarnings("ignore", message="You have to reset function pointers after creating a reb_simulation struct with a binary file.")
warnings.filterwarnings("ignore", message="File in use for Simulationarchive already exists. Snapshots will be appended.")

class small_body:
    # class that is returned by the main sbdynt function
    # it contains all of the calculated dynamical parameters and 
    # other pertinent information about the simulations
    def __init__(self, designation, object_type=None):
        self.designation = designation
        self.object_type = object_type
        self.epoch = None
        self.clones = None
        self.cloning_method = None
        self.clone_weights = None
        self.planets = None
        self.archivefile = None
        self.icfile = None
        self.sbdb_file = None
        self.logfile = None
        if(object_type == 'tno'):
            self.tno_class = None

        self.tmax = None
        self.tout = None
        self.int_direction = ''
        self.run_properties = self.analysis_vars()

        self.proper_elements = proper_element_class(des=designation)
        self.stability_indicators = stability_indicators(des=designation)

        #DS added variables for proper_element branch
        self.a_arr = []
        self.e_arr = []
        self.I_arr = []
        self.o_arr = []
        self.O_arr = []
        self.t_arr = []


    def analysis_vars(self, tmax=None, tout=None, int_direction=None, planets=None):
        
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
            print('The given direction variable=', int_direction, ' is not one of the 4 allowed inputs ')
            print('("backwards","forwards","bf", None). Setting direction to "bf". Call this function ')
            print('again with a valid direction is you do not want a forwards + backwards integration.')  
            self.int_direction = 'bf'

        if(planets==None):
            if(self.object_type == 'asteroid'):
                self.planets = ['inner+outer']
            elif(self.object_type == 'tno'):
                self.planets = ['outer']
            else:
                print('object_type is neither asteroid or tno. Setting default planets = ["all"]')
                self.planets = ['all']
                 
        

    
def run_ast(des=None, clones=None, datadir='',archivefile=None, saveic=True, save_sbdb=True,
            logfile=False, deletefile=False, run_proper=True, run_stability=True, 
            output_arrays=False,integrator='mercurius'):
    """
    Function to do a standard Asteroid analysis run using all default choices
    Initialize, integrate, and perform several different analyses for an asteroid small body contained 
    in JPL's small body database browser.

    inputs:
        des: string, the designation for the object in the SBDB
        clones (optional): integer, number of clones. Defaults to None, which will
            result in adding two, 3-sigma outlying clones in semimajor axis.
            If set to 0, only the best fit orbit is run
            If set to any integer, that many clones will be sampled in a Guassian
            manner from the orbit-fit covariance matrix
        datadir (optional): string, path for saving any files produced in this 
            function; defaults to the current directory
        saveic (optional): boolean or string; 
            (default) if True:  will save a rebound file with the simulation 
            state that can be used to restart later either to a default 
            file name or to a file with the name equal to the string passed
            if False nothing is saved
        logfile (optional): boolean or string; 
            if True:  will save some messages to a default log file name
            or to a file with the name equal to the string passed or
            to the screen if 'screen' is passed 
            (default) if False nothing is saved
        save_sbdb (optional): boolean or string; 
            (default) if True:  will save a pickle file with the results of the 
            JPL SBDB query either to a default file name or to a file
            with the name equal to the string passed
            if False nothing is saved
        archivefile (str; optional): name for the simulation
            archive file that rebound will generate. The default filename is 
            <des>-simarchive.bin if this variable is not defined.
        deletefile (boolean): Default False; if True, deletes any previous 
            Simulationarchive at archivefile
        run_proper (boolean): Default True, run and computes synthetic proper elements 
            from the Simulationarchive outputs. 
            if False, nothing currently happens in this function
        run_stability (boolean): Default True, computes stability indicators from the 
            Simulationarchive outputs of the proper element run. 
        output_arrays (boolean): Default False; if True, saves the osculating orbital 
            elements and the filtered orbital elements to the proper_elements class object, 
            so the user can have quick access to the time arrays for further analysis or 
            visualization. This speeds up the visualization options for the proper_elements 
            code significantly. 
        integrator (optional): if not set, the default integrator mercurius is used. But
            you can set this to 'whfast' to speed up the long proper elements integration.
            NOTE: The safest thing is to leave this parameter un-set unless you're sure
            you don't need to worry about close encounters.

    outputs:
        flag (0,1): 0 indicates a failure, while 1 indicates a successful run.
        ast_results: An sbdynt.small_body class object, with parameters filled according to the 
            analyses selected to be performed by 
            the user. The most significant variables for users will likely be:
                - ast_results.proper_elements (See prop_elem.proper_elements class for more information)
                - ast_results.stability_indicators (See stability_indicators.stability_indicators 
                  class for more information)
        sim (Rebound Simulation instance): The final Simulation from the full integration. Equivalent 
            to the last snapshot in the saved Simulationarchive. 
    """ 


    if(des == None):
        print("The designation of an asteroid must be provided")
        return 0, None, None


    object_type = 'asteroid'
    #initialize the results class
    ast_results = small_body(des,object_type)

    if(datadir):
        check_datadir(datadir)

    #define names/paths to all the files to be saved
    if(logfile==True):
        logf = log_file_name(des=des)
    else:
        logf = logfile
    if(datadir and logf and logf!='screen'):        
        logf = datadir + '/' +logf
    if(archivefile==None):
        archivefile = archive_file_name(des)
    if(datadir):
        archivefile = datadir + '/' +archivefile
    ic_file = saveic
    if(saveic):
        if(saveic == True):
            ic_file = ic_file_name(des=des)
        if(datadir):
            ic_file = datadir + '/'  +ic_file
    sbdb_file = save_sbdb
    if(save_sbdb):
        if(save_sbdb == True):
            sbdb_file = orbit_solution_file(des)
        if(datadir):
            sbdb_file = datadir + '/' + sbdb_file

    #define some of the run parameters into the results class
    ast_results.planets = ['inner+outer']
    ast_results.archivefile = archivefile
    ast_results.sbdb_file = sbdb_file
    ast_results.icfile = ic_file
    ast_results.logfile = logf


    if(logf):
        logmessage = "Initializing an Asteroid simulation instance by querying JPL for designation: "
        logmessage += des
        writelog(logf,logmessage)  

    #set up default asteroid run as defined in asteroid.py
    iflag, sim, epoch, clones, cloning_method, weights = \
        setup_default_ast_integration(des=des, clones=clones, save_sbdb=ast_results.sbdb_file,
                                      planets=ast_results.planets,saveic=ast_results.icfile, 
                                      archivefile=ast_results.archivefile, 
                                      logfile=ast_results.logfile)

    if(iflag < 1):
        logmessage = "Failed at initialization stage of sbdynt.run_ast\n"
        logmessage += "at asteroid.setup_default_ast_integration\n"
        writelog(logf,logmessage)  
        if(logf != 'screen'):
            print(logmessage)
        return 0, ast_results, sim

    ast_results.clones = clones
    ast_results.cloning_method = cloning_method
    ast_results.clone_weights = weights

    if(run_proper):
        if(logf):
            logmessage = "Running Asteroid integration for the synthetic\n"
            logmessage+= "proper elements calculation\n"
            writelog(logf,logmessage)  

        rflag, sim = integrate_for_pe(sim, des=des, archivefile=ast_results.archivefile,
                                      icfile=ast_results.icfile,logfile=ast_results.logfile, 
                                      tmax=10e6, tout=500., direction='bf', 
                                      deletefile=deletefile, integrator=integrator)

        if(rflag < 1):
            logmessage = "Failed during the integration in sbdynt.run_ast\n"
            logmessage += "at prop_elem.integrate_for_pe\n"
            writelog(logf,logmessage)  
            if(logf != 'screen'):
                print(logmessage)        
            return 0, ast_results, sim

        if(logf):
            logmessage = "Reading in the results of the Asteroid integration\n"
            writelog(logf,logmessage)  

        reflag, times, sb_elems, planet_elems, clone_elems, small_planets_flag = \
                read_archive_for_pe(des=des,archivefile=ast_results.archivefile,
                                clones=ast_results.clones,logfile=ast_results.logfile, 
                                object_type = ast_results.object_type)

        if(reflag < 1):
            logmessage = "Failed in sbdynt.run_ast when reading in integrated simulation for proper elements\n"
            logmessage += "at tools.read_archive_for_pe\n"
            writelog(logf,logmessage)  
            if(logf != 'screen'):
                print(logmessage)        
            return 0, ast_results, sim
        
        if(logf):
            logmessage = "calculating asteroid proper elements\n"
            writelog(logf,logmessage)  

        
        pflag, prope = calc_proper_elements(des=des, times=times, sb_elems=sb_elems, 
                                            planet_elems=planet_elems, clones = ast_results.clones, clone_elems = clone_elems,
                                            small_planets_flag=small_planets_flag, 
                                            output_arrays=output_arrays,
                                            logfile=logf)
        if(pflag < 1):
            logmessage = "Failed at proper elements calculation stage in sbdynt.run_ast\n"
            logmessage+= "at prop_elem.calc_proper_elements\n"
            writelog(logf,logmessage)  
            if(logf != 'screen'):
                print(logmessage)        
            return 0, ast_results, sim
            
        ast_results.proper_elements = prope

    else:
        logmessage = "run_proper wasn't selected in sbdynt.run_ast \n"
        logmessage += "right now that means nothing happens, so returning\n"
        logmessage += "an initialized, but not run, rebound simulation instance"
        writelog(logf,logmessage)  
        if(logf != 'screen'):
            print(logmessage)        
        return 0, ast_results, sim


    if run_stability:
        if(logf):
            logmessage = "running asteroid stability indicators\n"
            writelog(logf,logmessage)  

        st_flag, ast_results.stability_indicators = compute_stability(des=des, times=times, sb_elems=sb_elems, 
                                                             clones=ast_results.clones, 
                                                             pe_obj=ast_results.proper_elements, 
                                                             clone_elems = clone_elems, 
                                                             output_arrays = output_arrays,
                                                             logfile=ast_results.logfile)
        if(st_flag < 1):
            logmessage = "failure in sbdynt.run_ast when calculating stability indicators\n"
            logmessage += "at stability_indicators.compute_stability\n"
            writelog(logf,logmessage)  
            if(logf != 'screen'):
                print(logmessage)                  
            return 0, ast_results, sim

    return 1, ast_results, sim

def analyze_ast_run(des=None, clones=None, datadir='', archivefile=None,
            logfile=False, run_proper=True, run_stability=True, output_arrays=False):
    """
    Perform several default analyses on an already existing Simulationarchive for 
    a default asteroid small body simulation.
    inputs:
        des: string, the designation for the object in the archivefile
        clones (optional): integer, number of clones. Defaults to None, which
            will let the code figure out how many clones are in the simulation
        datadir (optional): string, path for finding or saving any files needed or
            produced in this function; defaults to the current directory
        logfile (optional): boolean or string; 
            if True:  will save some messages to a default log file name
            or to a file with the name equal to the string passed or
            to the screen if 'screen' is passed 
            (default) if False nothing is saved
        archivefile (str; optional): name for the simulation archive file that 
            contains the rebound simulation that was already run. The default 
            filename is <des>-simarchive.bin if this variable is not defined.
        run_proper (boolean): Default True, run and computes synthetic proper elements 
            from the Simulationarchive outputs. 
            if False, nothing currently happens in this function
        run_stability (boolean): Default True, computes stability indicators from the 
            Simulationarchive outputs of the proper element run. 
        output_arrays (boolean): Default False; if True, saves the osculating orbital 
            elements and the filtered orbital elements to the proper_elements class object, 
            so the user can have quick access to the time arrays for further analysis or 
            visualization. This speeds up the visualization options for the proper_elements 
            code significantly. 

    outputs:
        flag (integer): 0 if something went wrong, 1 if everything succeeded
        ast_results: An sbdynt.small_body class object, with parameters filled according to 
            the analyses selected to be performed by the user. The most significant variables for 
            users will likely be...
                - ast_results.proper_elements (See prop_elem.proper_elements class for more information)
                - ast_results.stability_indicators (See stability_indicators.stability_indicators 
                  class for more information)
    """ 

    if(des == None):
        print("The designation of the small body must be provided")
        return 0, None

    if(logfile==True):
        logf = log_file_name(des=des)
    else:
        logf = logfile
    if(datadir and logf and logf!='screen'):        
        logf = datadir + '/' +logf
    
    object_type = 'asteroid'

    #initialize the results class
    ast_results = small_body(des,object_type)

    if(archivefile==None):
        file = archive_file_name(des=des)
    else:
        file = archivefile
    if(datadir):
        file = datadir + '/' + file
    
    ast_results.archivefile = file
    ast_results.logfile = logf

    if(clones == None):
        #read in the simulation archive to determine the number of clones
        iflag, snew, clones = initialize_simulation_from_simarchive(des=des,archivefile=ast_results.archivefile,
                                                                    logfile=logf)
        if(iflag < 1):
            logmessage = "failed in sbdynt.analyze_ast_run when trying to read in the archivefile\n"
            logmessage += " to determine how many clones there are using \n"
            logmessage += "run_reb.initialize_simulation_from_simarchive"
            writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)
            return 0, ast_results

    ast_results.clones = clones

    if(run_proper):
        if(logf):
            logmessage = 'Reading Asteroid integration\n'
            writelog(logf,logmessage)  
        reflag, times, sb_elems, planet_elems, clone_elems, small_planets_flag = read_archive_for_pe(des=des,
                                                                        archivefile=ast_results.archivefile,
                                                                        clones=ast_results.clones,logfile=logf, 
                                                                        object_type=ast_results.object_type)

        if(reflag < 1):
            logmessage = "Failed in sbdynt.analyze_ast_run when reading in an integrated simulation for proper elements\n"
            logmessage += "at prop_elem.read_archive_for_pe\n"
            writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)
            return 0, ast_results
                
        if(logf):
            logmessage = 'calculating asteroid proper elements\n'
            writelog(logf,logmessage)  

        pflag, prope = calc_proper_elements(des=des, times=times, sb_elems=sb_elems, planet_elems=planet_elems, 
                                            clones = ast_results.clones, clone_elems = clone_elems,
                                            small_planets_flag=small_planets_flag, output_arrays=output_arrays)
        if(pflag < 1):
            logmessage = "Failed at proper elements calculation stage in sbdynt.analyze_ast_run\n"
            logmessage += "at prop_elem.calc_proper_elements\n"
            tools.writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)
            return 0, ast_results

        ast_results.proper_elements = prope

        if(logf=='screen'):
            prope.print_results()
            
    elif(run_stability):
        reflag, times, sb_elems, planet_elems, clone_elems, small_planets_flag = read_archive_for_pe(des=des,
                                                                            archivefile=ast_results.archivefile,
                                                                            clones=ast_results.clones, logfile=logf, 
                                                                            object_type = ast_results.object_type)
        if(reflag < 1):
            logmessage = "Failed in sbdynt.analyze_ast_run when reading in an integrated simulation for stability calc\n"
            logmessage += "at prop_elem.read_archive_for_pe\n"
            writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)
            return 0, ast_results
        
    if(run_stability):

        if(logf):
            logmessage = 'calculating asteroid stability indicators\n'
            writelog(logf,logmessage)         
        stflag, ast_results.stability_indicators = compute_stability(des=des, times=times, sb_elems=sb_elems, clones=clones, 
                                                                   pe_obj=ast_results.proper_elements, clone_elems=clone_elems, 
                                                                   output_arrays=output_arrays)
        if(stflag < 1):
            logmessage = "Failed in sbdynt.analyze_ast_run when calculating stability indicators \n"
            logmessage += "at stability_indicators.compute_stability\n"
            tools.writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)
            return 0, ast_results
        
        if(logf=='screen'):
            ast_results.stability_indicators.print_results()
       
    return 1, ast_results


        
    
def run_tno(des=None, clones=None, datadir='', archivefile=None,
            saveic=True, save_sbdb=True,logfile=False, deletefile=False,
            run_ML=True, run_proper=False, run_stability=False, 
            output_arrays=False,integrator='mercurius'):
    """
    Function to do a standard TNO analysis run using all default choices
    Initialize, Integrate, and perform several different analyses for a TNO small body contained in
    JPL's Small Body Database.

    inputs:
        des: string, the designation for the object in the SBDB
        clones (optional): integer, number of clones. Defaults to None, which will
            result in adding two, 3-sigma outlying clones in semimajor axis.
            If set to 0, only the best fit orbit is run
            If set to any integer, that many clones will be sampled in a Guassian
            manner from the orbit-fit covariance matrix
        datadir (optional): string, path for saving any files produced in this 
            function; defaults to the current directory
        saveic (optional): boolean or string; 
            (default) if True:  will save a rebound file with the simulation 
            state that can be used to restart later either to a default 
            file name or to a file with the name equal to the string passed
            if False nothing is saved
        logfile (optional): boolean or string; 
            if True:  will save some messages to a default log file name
            or to a file with the name equal to the string passed or
            to the screen if 'screen' is passed 
            (default) if False nothing is saved
        save_sbdb (optional): boolean or string; 
            (default) if True:  will save a pickle file with the results of the 
            JPL SBDB query either to a default file name or to a file
            with the name equal to the string passed
            if False nothing is saved
        archivefile (str; optional): name for the simulation
            archive file that rebound will generate. The default filename is 
            <des>-simarchive.bin if this variable is not defined.
        deletefile (boolean): Default False; if True, deletes any previous 
            Simulationarchive at archivefile
        run_ML (boolean): Default True, performs a classification analysis on 
            the TNO Simulationarchive outputs by way of the tno_classifier algorithm. 
        run_proper (boolean): Default True, run and computes synthetic proper elements 
            from the Simulationarchive outputs. 
            if False, nothing currently happens in this function
        run_stability (boolean): Default True, computes stability indicators from the 
            Simulationarchive outputs of the proper element run. 
        output_arrays (boolean): Default False; if True, saves the osculating orbital 
            elements and the filtered orbital elements to the proper_elements class object, 
            so the user can have quick access to the time arrays for further analysis or 
            visualization. This speeds up the visualization options for the proper_elements 
            code significantly.
        integrator (optional): if not set, the default integrator mercurius is used. But
            you can set this to 'whfast' to speed up the long proper elements integration.
            NOTE: mercurius is always used for the ML classifier run since that's short
            and you really want to resolve those close encounters correctly for scattering 
            objects. The safest thing is to leave this parameter un-set unless you're sure
            you don't need to worry about close encounters.

    outputs:
        flag (0,1): 0 indicates a failure, while 1 indicates a successful run.
        tno_results: An sbdynt.small_body class object, with parameters filled according to the 
            analyses selected to be performed by the user, including:
            - tno_results.tno_ml_outputs (See tno_classifier.TNO_ML_Outputs class for more information)
            - tno_results.proper_elements (See prop_elem.proper_elements class for more information)
            - tno_results.stability_indicators (See stability_indicators.stability_indicators class 
              for more information)
        sim (Rebound Simulation instance): The final Simulation from the full integration. Equivalent 
            to the last snapshot in the saved Simulationarchive.         
    """ 

    if(des == None):
        print("The designation of a TNO must be provided")
        return 0, None, None

    object_type = 'tno'
    #initialize the results class
    tno_results = small_body(des,object_type)

    if(datadir):
        check_datadir(datadir)

    #define names/paths to all the files to be saved
    if(logfile==True):
        logf = log_file_name(des=des)
    else:
        logf = logfile
    if(datadir and logf and logf!='screen'):        
        logf = datadir + '/' +logf
    if(archivefile==None):
        archivefile = archive_file_name(des)
    if(datadir):
        archivefile = datadir + '/' +archivefile
    ic_file = saveic
    if(saveic):
        if(saveic == True):
            ic_file = ic_file_name(des=des)
        if(datadir):
            ic_file = datadir + '/'  +ic_file
    sbdb_file = save_sbdb
    if(save_sbdb):
        if(save_sbdb == True):
            sbdb_file = orbit_solution_file(des)
        if(datadir):
            sbdb_file = datadir + '/' + sbdb_file

    #define some of the run parameters into the results class
    tno_results.planets = ['outer']
    tno_results.archivefile = archivefile
    tno_results.sbdb_file = sbdb_file
    tno_results.icfile = ic_file
    tno_results.logfile = logf


    if(logf):
        logmessage = "Initializing a TNO simulation instance by querying JPL"
        writelog(logf,logmessage)  

    iflag, sim, epoch, clones, cloning_method, weights = \
                setup_default_tno_integration(des=des, clones=clones,
                                              planets= tno_results.planets,
                                              archivefile=tno_results.archivefile,
                                              save_sbdb=tno_results.sbdb_file,
                                              saveic=tno_results.icfile,
                                              logfile=tno_results.logfile)
    tno_results.clones = clones
    tno_results.epoch = epoch
    
    if(iflag < 1):
        logmessage = "sbdynt.run_tno Failed at initialization stage\n"
        logmessage += " at tno.setup_default_tno_integration"
        writelog(logf,logmessage) 
        if(logf != 'screen'):
            print(logmessage)
        return 0, tno_results, sim

    tno_results.cloning_method = cloning_method
    tno_results.clone_weights = weights

    if(run_ML and (run_proper or run_stability) and tno_results.icfile == False):
        #we will need a saved copy of the initial conditions later
        #save the current state to a randomly named file to make sure we don't over-write 
        #anything else in this directory
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
        ic_file = random_string+'.bin'
        if(datadir):
            ic_file = datadir + '/' + ic_file
        logmessage = "creating a temporary initial conditions file at " + ic_file
        tools.writelog(logf,logmessage)
        sim.save_to_file(ic_file)

    
    tno_class = TNO_ML_outputs()
    if(run_ML):
        if(logf):
            logmessage= 'Running TNO ML\n'
            writelog(logf,logmessage)
        cflag, tno_class, sim = run_and_MLclassify_TNO(sim=sim,des=des,
                                    clones=tno_results.clones,
                                    archivefile=tno_results.archivefile,
                                    deletefile=deletefile,
                                    logfile=tno_results.logfile)
        tno_results.tno_ml_outputs = tno_class 
        if(cflag < 1):
            logmessage = "Failed at the machine learning stage of sbdynt.run_tno\n"
            logmessage += "at tno_classifier.run_and_MLclassify_TNO\n"
            writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)
            return 0, tno_results, sim

        if(logf=='screen'):
            tno_class.print_results()


    if(run_proper or run_stability):
        if(tno_class.most_common_class == 'scattering'):
            logmessage = str(des) + " is most likely a scattering TNO based on the ML classifier.\n"
            logmessage += "Proper a,e,sini will be given as 10 Myr averages.\n"
            logmessage += "No further integration needed.\n"
            logmessage += "If you *really* want to do the proper element integration, re-run this\n"
            logmessage += "object with run_ML=False\n"
            writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)
            tno_results.proper_elements.proper_elements.a = tno_class.features.a_mean
            tno_results.proper_elements.proper_elements.e = tno_class.features.e_mean
            tno_results.proper_elements.proper_elements.sinI = np.sin(tno_class.features.i_mean)
        
            return 2, tno_results, sim
        
        if(logf):
            logmessage = "Running additional forward integrations for the synthetic\n"
            logmessage+= "proper elements and/or stability indicators calculation\n"
            writelog(logf,logmessage)  

        #make sure we don't delete the initial ML integration if it was used
        if(run_ML==True and deletefile==True):
            pe_deletefile = False
        else:
            pe_deletefile = deletefile
        rflag, sim = integrate_for_pe(sim,des=des,archivefile=tno_results.archivefile, icfile=tno_results.icfile,
                                      logfile=tno_results.logfile,tmax=150e6,tout=5000., 
                                      direction='bf', deletefile=pe_deletefile, integrator=integrator)
        if(rflag < 1):
            logmessage = "failed in sbdynt.run_tno at the integration for proper elements and\n"
            logmessage += "stability indicators at prop_elem.integrate_for_pe\n"
            writelog(logf,logmessage)  
            if(logf != 'screen'):
                print(logmessage)
            return 0, tno_results, sim

        if(tno_results.icfile == False):
            logmessage = "removing the temporary initial conditions file"
            tools.writelog(logf,logmessage)   
            remove(ic_file)

        if(logf):
            logmessage = "Reading in the TNO integration\n"
            logmessage+= "proper elements and/or stability indicators calculation\n"
            writelog(logf,logmessage)  
        
        reflag, times, sb_elems, planet_elems, clone_elems, small_planets_flag = read_archive_for_pe(
                    des=des,archivefile=tno_results.archivefile,clones=tno_results.clones,
                    logfile=tno_results.logfile,object_type=tno_results.object_type)
        if(reflag < 1):
            logmessage = "Failed when reading in the integrated simulation for proper elements in sbdynt.run_tno\n"
            logmessage += "at prop_elem.read_archive_for_pe\n"
            writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)
            return 0, tno_results, sim

        if(run_proper):
            if(logf):
                logmessage = 'Running the TNO proper element calculation'
                writelog(logf,logmessage) 

            pflag, prope = calc_proper_elements(des=des, times=times, sb_elems=sb_elems, planet_elems=planet_elems, 
                                                clones=tno_results.clones, clone_elems = clone_elems,
                                                small_planets_flag=small_planets_flag, output_arrays=output_arrays)

            tno_results.proper_elements = prope
            if(pflag < 1):
                logmessage = "Failed at proper elements calculation stage in sbdynt.run_tno\n"
                logmessage += "at prop_elem.calc_proper_elements\n"
                writelog(logf,logmessage) 
                if(logf != 'screen'):
                    print(logmessage)                
                return 0, tno_results, sim

            if(logf=='screen'):
                prope.print_results()
            
        if(run_stability):
            stflag, tno_results.stability_indicators = compute_stability(des=des, times=times, sb_elems=sb_elems, 
                                                                 clones=tno_results.clones, 
                                                                 pe_obj=tno_results.proper_elements, 
                                                                 clone_elems=clone_elems, 
                                                                 output_arrays=output_arrays)
            if(stflag < 1):
                logmessage = "Failed at stability indicators calculation stage in sbdynt.run_tno\n"
                logmessage += "at stability_indicators.compute_stability\n"
                writelog(logf,logmessage) 
                if(logf != 'screen'):
                    print(logmessage)                
                return 0, tno_results, sim
            if(logf=='screen'):
                tno_results.stability_indicators.print_results()

    return 1, tno_results, sim


def analyze_tno_run(des=None, clones=None, datadir='',archivefile=None,
                    logfile=False, run_ML=True, run_proper=False, run_stability=False, 
                    output_arrays = True):
    """
    Perform several default analyses on an already existing Simulationarchive for 
    a default tno small body simulation.
    inputs:
        des: string, the designation for the object in the archivefile
        clones (optional): integer, number of clones. Defaults to None, which
            will let the code figure out how many clones are in the simulation
        datadir (optional): string, path for finding or saving any files needed or
            produced in this function; defaults to the current directory
        logfile (optional): boolean or string; 
            if True:  will save some messages to a default log file name
            or to a file with the name equal to the string passed or
            to the screen if 'screen' is passed 
            (default) if False nothing is saved
        archivefile (str; optional): name for the simulation archive file that 
            contains the rebound simulation that was already run. The default 
            filename is <des>-simarchive.bin if this variable is not defined.
        run_ML (boolean): Default True, performs a classification analysis on 
            the TNO Simulationarchive outputs by way of the tno_classifier algorithm.             
        run_proper (boolean): Default False, computes synthetic proper elements 
            from the Simulationarchive outputs. 
            if False, nothing currently happens in this function
        run_stability (boolean): Default False, computes stability indicators from the 
            Simulationarchive outputs of the proper element run. 
        output_arrays (boolean): Default False; if True, saves the osculating orbital 
            elements and the filtered orbital elements to the proper_elements class object, 
            so the user can have quick access to the time arrays for further analysis or 
            visualization. This speeds up the visualization options for the proper_elements 
            code significantly. 

    outputs:
        flag (0,1): 0 indicates a failure, while 1 indicates a successful run.
        tno_results: An sbdynt.small_body class object, with parameters filled according to the 
            analyses selected to be performed by the user, including:
            - tno_results.tno_ml_outputs (See tno_classifier.TNO_ML_Outputs class for more information)
            - tno_results.proper_elements (See prop_elem.proper_elements class for more information)
            - tno_results.stability_indicators (See stability_indicators.stability_indicators class 
              for more information)        
    """ 

    if(logfile==True):
        logf = log_file_name(des=des)
    else:
        logf = logfile
    if(datadir and logf and logf!='screen'):        
        logf = datadir + '/' +logf
    
    object_type = 'tno'

    #initialize the results class
    tno_results = small_body(des,object_type)

    if(archivefile==None):
        file = archive_file_name(des=des)
    else:
        file = archivefile
    if(datadir):
        file = datadir + '/' + file
    
    tno_results.archivefile = file
    tno_results.logfile = logf

    if(clones == None):
        #read in the simulation archive to determine the number of clones
        if(logf):
            logmessage = "reading in the simulation archive to determine how many clones there are\n"
            writelog(logf,logmessage)
        iflag, snew, clones = initialize_simulation_from_simarchive(des=des,archivefile=tno_results.archivefile,
                                                                    logfile=logf)
        if(iflag < 1):
            logmessage = "failed in sbdynt.analyze_tno_run when trying to read in the archivefile\n"
            logmessage += " to determine how many clones there are using \n"
            logmessage += "run_reb.initialize_simulation_from_simarchive"
            writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)
            return 0, tno_results

    tno_results.clones = clones
    tno_class = TNO_ML_outputs(clones)
    if(run_ML):
        if(logf):
            logmessage = "Running the ML classifier\n";
            writelog(logf,logmessage)
        
        cflag, tno_class, sim = run_and_MLclassify_TNO(des=des,clones=tno_results.clones,
                                    archivefile=tno_results.archivefile, deletefile=False,
                                    logfile=tno_results.logfile, classify_only=True)

        tno_results.tno_class = tno_class
        if(cflag < 1):
            logmessage = "Failed at the machine learning stage in sbdynt.analyze_tno_run\n"
            logmessage += "at tno_classifier.run_and_MLclassify_TNO"
            return 0, tno_results

        if(logf=='screen'):
            tno_class.print_results()

        if(run_proper and tno_class.most_common_class == 'scattering'):
            logmessage = "This is most likely a scattering TNO.\n"
            logmessage += "Proper a,e,sini will be 10 Myr averages.\n"
            logmessage += "If you ran a longer integraiton anyway and want the proper\n"
            logmessage += "elements, re-run this function with run_ML=False\n"
            writelog(logf,logmessage)  
            if(logf != 'screen'):
                print(logmessage)
            
            tno_results.proper_elements.proper_elements.a = tno_class.features.a_mean
            tno_results.proper_elements.proper_elements.e = tno_class.features.e_mean
            tno_results.proper_elements.proper_elements.sinI = np.sin(tno_class.features.i_mean)
        
            return 2, tno_results

    if(run_proper or run_stability):
        if(logf):
            logmessage = 'Reading TNO integration for Proper Elements and/or stability indicators\n'
            writelog(logf,logmessage)

        reflag, times, sb_elems, planet_elems, clone_elems, small_planets_flag = read_archive_for_pe(
                des=des,archivefile=tno_results.archivefile,clones=tno_results.clones,
                logfile=tno_results.logfile, object_type=tno_results.object_type)

        if(reflag < 1):
            logmessage ="Failed when reading in integrated simulation for proper elements\n"
            logmessage += "in sbdynt.analyze_tno_run at prop_elem.read_archive_for_pe\n"
            writelog(logf,logmessage)
            if(logf != 'screen'):
                print(logmessage)
            return 0, tno_results
    
        if(run_proper):
            if(logf):
                logmessage = 'Calculating proper elements\n'
                writelog(logf,logmessage)
            pflag, prope = calc_proper_elements(des=des, times=times, sb_elems=sb_elems, 
                                                planet_elems=planet_elems, clones=clones, clone_elems = clone_elems,
                                                small_planets_flag=small_planets_flag, 
                                                output_arrays=output_arrays)
            
            tno_results.proper_elements = prope
            
            if(pflag < 1):
                logmessage = "Failed at proper elements calculation stage in \n"
                logmessage += " sbdynt.analyze_tno_run at prop_elem.calc_proper_elements\n"
                writelog(logf,logmessage)
                if(logf != 'screen'):
                    print(logmessage)
                return 0, tno_results

            if(logf=='screen'):
                prope.print_results()
            
        if(run_stability):
            if(logf):
                logmessage = 'Running TNO Stability Indicators'
                writelog(logf,logmessage)
        
            stflag, tno_results.stability_indicators = compute_stability(des=des, times=times, 
                                                    sb_elems=sb_elems, clones=tno_results.clones, 
                                                    pe_obj=tno_results.proper_elements, 
                                                    clone_elems=clone_elems, 
                                                    output_arrays=output_arrays)
        
            if(stflag < 1):
                logmessage = "Failed at stability indicators calculation stage in \n"
                logmessage += " sbdynt.analyze_tno_run at stability_indicators.compute_stability\n"
                writelog(logf,logmessage)
                if(logf != 'screen'):
                    print(logmessage)
                return 0, tno_results
            
            if(logfile=='screen'):
                tno_results.stability_indicators.print_results()

    return 1, tno_results

#function to do a standard small body analysis run using given or default choices
def run_sb(des=None, object_type=None, clones=None, datadir='',archivefile=None,
           saveic = True, save_sbdb = True, logfile=False,deletefile=False):
    '''
    documentation here...
    '''
    if(des == None):
        print("The designation of a solar system small body must be provided")
        return None
        
    if(logfile==True):
        logf = log_file_name(des=des)
    else:
        logf = logfile

    if(datadir):
        check_datadir(datadir)

    if(datadir and logf and logf!='screen'):        
        logf = datadir + '/' +logf

    if(logf):
        logmessage = "Initializing a small body simulation instance by querying JPL"
        writelog(logf,logmessage)  

    if(object_type == None):
        #query JPL SBDB to see what kind of object it is
        flag, a = horizons_api.query_sbdb_for_a(des=des)
        if(flag<1):
            print("Object not found in JPL's small body database")
            return None
        if(a < 2.5):
            logmessage = "caution: SBDynT is not optimized for bodies interior to Mars"
            logmessage += "proceeding with an attempted default asteroid run"
            if(logf != 'screen'):
                print(logmessage)  
            if(logf):           
                writelog(logf,logmessage)
            object_type = 'ast'

        elif(a < 6.):
            object_type = 'ast'
        elif(a > 29.):
            object_type = 'tno'
        else:
            logmessage = "object is in the giant planet region"
            logmessage += "proceeding with tno defaults"
            if(logf != 'screen'):
                print(logmessage)  
            if(logf):           
                writelog(logf,logmessage)
            object_type = 'tno'
    
    sb_results = small_body(des,object_type,clones)
    iflag = 0
    if(object_type == 'ast'):
        iflag, sim, epoch, clones, cloning_method, weights = \
                setup_ast_integration(des=des, sb_results=sb_results, clones=clones, datadir=datadir,
                                              save_sbdb=save_sbdb,saveic=saveic,
                                              archivefile=archivefile,
                                              logfile=logfile)
    elif(object_type == 'tno'):
        iflag, sim, epoch, clones, cloning_method, weights = \
                setup_ast_integration(des=des, sb_results=sb_results, clones=clones, datadir=datadir,
                                              save_sbdb=save_sbdb,saveic=saveic,
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
                                        clones=clones,return_timeseries=True,logfile=logfile, output_arrays = output_arrays)

    if(pflag < 1):
        print("Failed at proper elements calculation stage")
        return sb_results

    sb_results.proper_elements = pe

    sb_results.stability_indicators = compute_stability(des=des, times = times, sb_elems = sb_elems, clones=clones, pe_obj = tno_results.proper_elements, clone_elems = clone_elems, output_arrays = output_arrays)


    return sb_results


def run_existing_sb(des=None, clones=None, datadir='',archivefile=None,
            logfile=False,deletefile=False, run_proper=False, run_stability=False, object_type=None, output_arrays = True):
    '''
    documentation here...
    '''
    if(des == None):
        print("The designation of the small body must be provided")
        return None

    if(logfile==True):
        logf = log_file_name(des=des)
    else:
        logf = logfile

    if(datadir):
        check_datadir(datadir)

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

            pflag, prope = calc_proper_elements(des=des, times = times, sb_elems = sb_elems, 
                                             planet_elems = planet_elems, small_planets_flag = small_planets_flag, output_arrays = output_arrays)

            if(pflag < 1):
                print("Failed at proper elements calculation stage")
                return tno_results

            tno_results.proper_elements = prope
            
    elif run_stability:
        reflag, times, sb_elems, planet_elems, clone_elems, small_planets_flag = read_archive_for_pe(des=des,datadir=datadir,archivefile=archivefile,
                                        clones=clones,logfile=logfile, object_type = tno_results.object_type)
    if run_stability:
        #print('clone elemes:', clone_elems.shape, clone_elems)
        tno_results.stability_indicators = compute_stability(des=des, times = times, sb_elems = sb_elems, clones=clones, pe_obj = tno_results.proper_elements, clone_elems = clone_elems, output_arrays = output_arrays)
        
   
        
    return tno_results

