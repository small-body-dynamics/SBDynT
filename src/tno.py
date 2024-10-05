import tools
import run_reb
import numpy as np


def setup_default_tno_integration(des=None, clones=None, datadir='',save_sbdb=False,
                                  saveic=False,archivefile=None,logfile=False):
    '''

    '''
    flag = 0
    if(des == None):
        print("The designation of a TNO must be provided")
        print("failed at tno.setup_default_tno_integration()")
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
        iflag, epoch, sim, weights = run_reb.initialize_simulation(planets=['outer'],
                          des=des, clones=clones, cloning_method= cloning_method,datadir=datadir,
                          logfile=logfile, save_sbdb=save_sbdb, saveic=saveic)
    else:
        cloning_method = 'Gaussian'
        iflag, epoch, sim = run_reb.initialize_simulation(planets=['outer'],
                          des=des, clones=clones, cloning_method= cloning_method,datadir=datadir,
                          logfile=logfile, save_sbdb=save_sbdb, saveic=saveic)
        weights = np.ones(clones+1)


    if(iflag < 1):
        print("failed at tno.setup_default_tno_integration()")
        return flag, None, None, None, None, None


    flag = 1
    return flag, sim, epoch, clones, cloning_method, weights
        
