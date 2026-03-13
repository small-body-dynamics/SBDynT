import tools
import run_reb
import numpy as np


def setup_default_ast_integration(des=None, clones=None, datadir='',save_sbdb=True,
                                  planets=['inner+outer'],
                                  saveic=True, archivefile=None,logfile=False):
    '''
    Initialize an asteroid small body simulation using standard defaults for SBDynT

    inputs:
        des (string): the designation for the object in the SBDB
        clones (optional, integer): number of clones. Default of None results in choosing
            two clones at approximately 3-sigma variations in seimajor axis
        datadir (optional, string): path for saving any files produced in this 
            function; defaults to the current directory
        saveic (optional, boolean or string): default True which saves a rebound file with 
            the initial simulation state that can be used to restart it later to a default 
            file name; if a string is provided, that is the file name used
            if False, nothing is saved
        archivefile (optional, str): Name of the Simulationarchive binary file rebound saves
            to. If nothing is passed, the default filename is <des>-simarchive.bin.
        logfile (optional, boolean or string): if True, log messages will be saved to a
            default log file name; if a string is passed that string is the file name;
            if 'screen' is passed, logmessages are printed to screen
        save_sbdb (optional, boolean or string): default True  will save a pickle file 
            with the results of the JPL SBDB query to a default file; of a string is
            passed, that is the save-file name name;
            if False, nothing is saved
        planets (optional): string list, list of planet names - defaults to Venus-Neptune
            

    outputs:
        flag (integer, 0 or 1): 0 indicates a failure, while 1 indicates success.
        sim (Rebound Simulation instance): the initialized Rebound Simulation containing
            the planets and small bodies
    '''
    flag = 0
    if(des == None):
        print("The designation of an Asteroid must be provided")
        print("failed at asteroid.setup_default_ast_integration()")
        return flag, None, None, None, None, None
 
    if(datadir):
        tools.check_datadir(datadir)

    if(logfile==True):
        logf = tools.log_file_name(des=des)
    else:
        logf = logfile
    if(datadir and logf and logf!='screen'):        
        logf = datadir + '/' +logf

    if(clones==None):
        #default to the best fit and 3-sigma clones in semi-major axis
        clones = 2
        cloning_method = 'find_3_sigma'
        if(logf):
            logmessage = "Clones were not specified, so the default behavior is to return\n"
            logmessage += "a best-fit and 3-sigma minimum and maximum semimajor axis clones\n"
            tools.writelog(logf,logmessage)  
        iflag, epoch, sim, weights = run_reb.initialize_simulation(planets=planets,
                          des=des, clones=clones, cloning_method= cloning_method,datadir=datadir,
                          logfile=logfile, save_sbdb=save_sbdb, saveic=saveic)
    else:
        cloning_method = 'Gaussian'
        iflag, epoch, sim = run_reb.initialize_simulation(planets=planets,
                          des=des, clones=clones, cloning_method= cloning_method,datadir=datadir,
                          logfile=logfile, save_sbdb=save_sbdb, saveic=saveic)
        weights = np.ones(clones+1)


    if(iflag < 1):
        print("failed at asteroid.setup_default_ast_integration()")
        return flag, None, None, None, None, None


    flag = 1
    return flag, sim, epoch, clones, cloning_method, weights
        
