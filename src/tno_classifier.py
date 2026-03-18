import sys
import rebound
import numpy as np
import pandas as pd
import tools
import run_reb
import tno
import resonances
import machine_learning

from os import path
from datetime import date
from pickle import dump
from pickle import load
from importlib import resources as impresources



class TNO_ML_outputs:
    # class that stores the information that comes out of the 
    # TNO machine learning classifier
    def __init__(self,clones=0):
        self.clones = clones
        # make an empty features instance
        self.features = machine_learning.TNO_ML_features(self.clones)
        # parameters related to the classifier
        self.classes_dictionary = None
        self.class_probs = np.zeros((self.clones+1,3))
        
        #clone-by-clone predicted classification and 
        #confidence level for that classification
        self.clone_classification = (self.clones+1)*[None]
        self.clone_confidence = np.zeros(self.clones+1)

        self.most_common_class = None
        self.fraction_most_common_class = None

        #resonance parameters for resonant argument
        # phi = p*lambda_tno - q*lambda_neptune - m*(long of perihelion)_tno - n*(long of ascending node)_tno
        self.res_p = np.zeros(self.clones+1,dtype=int)
        self.res_q = np.zeros(self.clones+1,dtype=int)
        self.res_m = np.zeros(self.clones+1,dtype=int)
        self.res_n = np.zeros(self.clones+1,dtype=int)

        self.res_phi_delta = np.zeros(self.clones+1)
        self.res_phi_std = np.zeros(self.clones+1)

        self.res_img_probability = -1.*np.ones(self.clones+1)


    def set_initial_clone_classification(self):
        # find the most probable class and associated confidence on a 
        # clone-by-clone basis
        for n in range (self.clones+1):        
            cn = np.argmax(self.class_probs[n])
            self.clone_classification[n] = self.classes_dictionary[cn]
            self.clone_confidence[n] = self.class_probs[n,cn]
            #run checks for objects outside the range of interest (mostly relevant
            #to cases where classifying the end state of a simulation
            if(self.features.a_mean[n] < 5.):
                self.clone_classification[n] = 'not_TNO'                
        return
    
    def correct_clone_classification(self):
        for n in range (self.clones+1):
            #do checks for the scattering/detached boundary
            if(self.clone_classification[n] == 'class-det'):
                #check if it meets the scattering requirement
                if(self.features.a_delta[n] > 1.5):
                    self.clone_classification[n] = 'scattering'
                elif(self.features.e_mean[n] < 0.24):
                    if(self.features.a_mean[n] < 39.4):
                        self.clone_classification[n] = 'classical_inner'
                    elif(self.features.a_mean[n] < 47.7):
                        self.clone_classification[n] = 'classical_main'
                    else:
                        self.clone_classification[n] = 'classical_outer'
                else:
                    self.clone_classification[n] = 'detached'
            elif(self.clone_classification[n] == 'scattering'):
                #check if it meets the scattering requirement
                if(self.features.a_delta[n] <= 1.5):
                    self.clone_classification[n] = 'detached'
                    if(self.features.e_mean[n] < 0.24):
                        if(self.features.a_mean[n] < 39.4):
                            self.clone_classification[n] = 'classical_inner'
                        elif(self.features.a_mean[n] < 47.7):
                            self.clone_classification[n] = 'classical_main'
                        else:
                            self.clone_classification[n] = 'classical_outer'
        return


    def determine_most_common_classification(self):
        classes = list(set(self.clone_classification))
        rate = np.zeros(len(classes))
        for i in range (len(classes)):
            rate[i] = self.clone_classification.count(classes[i])
        mp = np.argmax(rate)
        self.most_common_class = classes[mp]
        self.fraction_most_common_class = float(rate[mp])/float(self.clones+1)
        return

    def print_results(self):

        print("#Most common classification: %s" % self.most_common_class)
        percentage = 100*self.fraction_most_common_class
        print("#Shared by %f percent of clones\n#" % percentage)

        nclas = len(self.classes_dictionary)
        print("Clone_number, most_probable_G08_class, p, q, m, n, phi_std_rad, phi_delta_rad, res_image_probability, probability_of_primary_class, ",end ="")
        for n in range(nclas):
            class_string =  self.classes_dictionary[n]
            if(class_string == 'class-det'):
                class_string = "class_det"
            print("probability_%s, " % class_string,end ="")
        print("\n",end ="")
        format_string = "%d, %s, "
        for n in range(nclas-1):
            format_string+="%e, "
        format_string+="%e,\n"
        for n in range(0,self.clones+1):
            print("%d, %s, %d, %d, %d, %d, %e, %e, %e, %e, " % (n,self.clone_classification[n], 
                   self.res_p[n],  self.res_q[n], self.res_m[n], self.res_n[n],
                   self.res_phi_std[n], self.res_phi_delta[n], self.res_img_probability[n],
                   self.clone_confidence[n]),end ="")
            for j in range(nclas):
                print("%e, " % self.class_probs[n][j] ,end ="")
            print("\n",end ="")

    def print_results_detailed(self):
        print("#Most common classification: %s" % self.most_common_class)
        percentage = 100*self.fraction_most_common_class
        print("#Shared by %f percent of clones\n#" % percentage)

        nclas = len(self.classes_dictionary)
        print("Clone_number, most_probable_G08_class, p, q, m, n, phi_std_rad, phi_delta_rad, res_image_probability, probability_of_primary_class, ",end ="")
        print("probability_of_", end ="")
        for n in range(nclas):
            class_string =  self.classes_dictionary[n]
            if(class_string == 'class-det'):
                class_string = "class_det"
            print("probability_%s, " % class_string,end ="")
        print("mean_a, mean_e, mean_i, std_a, std_e, std_i ",end ="")
        print("\n",end ="")
        format_string = "%d, %s, "
        for n in range(nclas-1):
            format_string+="%e, "
        format_string+="%e,\n"
        for n in range(0,self.clones+1):
            print("%d, %s, %d, %d, %d, %d, %e, %e, %e, %e, " % (n,self.clone_classification[n], 
                   self.res_p[n],  self.res_q[n], self.res_m[n], self.res_n[n],
                   self.res_phi_std[n], self.res_phi_delta[n], self.res_img_probability[n],
                   self.clone_confidence[n]),end ="")
            for j in range(nclas):
                print("%e, " % self.class_probs[n][j] ,end ="")
            print("%e, %e, %e, " % (self.features.a_mean[n], self.features.e_mean[n], 
                                    self.features.i_mean[n]), end = "")
            print("%e, %e, %e, " % (self.features.a_stddev[n], self.features.e_stddev[n], 
                                    self.features.i_stddev[n]), end = "")
            print("\n",end ="")

#################################################################
def run_and_MLclassify_TNO(sim=None, des=None, clones=None,  saveic=True,
                           save_sbdb=True, datadir='', archivefile=None, 
                           deletefile=False, logfile=False,
                           classify_only=False, related_clones=True):
    '''
    inputs:
        sim (optional, Rebound Simulation instance): if the user has initialized a custom 
            simulation, this should contain the 4 giant planets plus a TNO and its clones
            Otherwise this routine will initialize a simulation on its own based on querying
            JPL's SBDB
        des: string, the designation for the object in the SBDB or the provided simulation
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
        classify_only (boolean): default False, which will result in running
            the correct integration for the classifier to analyse. 
            If True, it is assumed that archivefile contains the relevant simulation
            and only the classification will be run
        related_clones (boolean): default True which means the test particles in the
            simulation are all sampled from one observed object's covariance matrix.
            If one clone is determined to be resonant, this flag ensures that an even
            more thorough resonance check is done on all the clones. 
            If False (used for, e.g., a simulation with a bunch of unrelated test 
            particles), the decision to check resonance angles for one clone is 
            unaffected by the results for other clones

    outputs:
        flag (integer): 0 for failure, 1 for success
        tno_class: the tno classification results python class TNO_ML_outputs (see 
            above)
        sim: the rebound simulation instance at its final state
            
    '''
    flag = 0

    if(des == None):
        print("The designation of the small body must be provided")
        print("failed at machine_learning.run_and_MLclassify_TNO()")
        return flag, None, sim

    if(datadir and classify_only == False):
        tools.check_datadir(datadir)

    #define names/paths to all the files to be saved
    if(logfile==True):
        logf = tools.log_file_name(des=des)
        print(logf)
    else:
        logf = logfile
    if(datadir and logf and logf!='screen'):        
        logf = datadir + '/' +logf
    if(archivefile==None):
        archivefile = tools.archive_file_name(des)
    if(datadir):
        archivefile = datadir + '/' +archivefile
    ic_file = saveic
    if(saveic):
        if(saveic == True):
            ic_file = tools.ic_file_name(des=des)
        if(datadir):
            ic_file = datadir + '/'  +ic_file
    sbdb_file = save_sbdb
    if(save_sbdb):
        if(save_sbdb == True):
            sbdb_file = tools.orbit_solution_file(des)
        if(datadir):
            sbdb_file = datadir + '/' + sbdb_file


    if(sim == None and classify_only == False):
        #initialize a default simulation
        if(logf):
            logmessage = "No simulation was provided to machine_learning.run_and_MLclassify_TNO\n"
            logmessage += "so initializing a default TNO run for " + str(des) + "\n"
            tools.writelog(logf,logmessage) 

        iflag, sim, epoch, clones, cloning_method, weights = \
                tno.setup_default_tno_integration(des=des, clones=clones, 
                        save_sbdb=sbdb_file,saveic=ic_file,archivefile=archivefile,logfile=logf)
        if(iflag < 1):
            logmessage = "Failed at simulation initialization stage in\n"
            logmessage += "machine_learning.run_and_MLclassify_TNO at\n" 
            logmessage += "tno.setup_default_tno_integration\n"
            tools.writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)
            return flag, None, sim

    if(classify_only==True and clones==None):
        if(logf):
            logmessage = "Reading in the last simulation snapshot to determine how many\n"
            logmessage += "clones are in the simulation\n"
            tools.writelog(logf,logmessage) 
        try:
            sa = rebound.Simulationarchive(archivefile)
        except:
            logmessage = "tno_classifier.run_and_MLclassify_TNO failed because\n"
            logmessage += "there was a problem reading the simulation archive file: "
            logmessage += archivefile + "\n"
            tools.writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)
            return flag, None, sim

        ntp_max = sa[0].N - sa[0].N_active
        clones = ntp_max - 1

    #make an empty set of classification outputs 
    tno_class = TNO_ML_outputs(clones)

    #initialize the machine learning classifier
    clf = machine_learning.TNO_ML_classifier()
    cflag = clf.initialize_classifiers()
    if(cflag<1):
        logmessage = "failed to initialize machine learning classifier in\n"
        logmessage += "tno_classifier.run_and_MLclassify_TNO\n"
        tools.writelog(logf,logmessage) 
        if(logf != 'screen'):
            print(logmessage)
        return flag, None, sim
    res_index = -1
    for i in range(len(clf.classes_dictionary)):
        if (clf.classes_dictionary[i] == 'Nresonant'):
            res_index=i
    if(res_index < 0):
        logmessage = "failed to find 'Nresonant' in the classifier dictionary\n"
        logmessage += "failed in machine_learning.run_and_MLclassify_TNO()\n"
        tools.writelog(logf,logmessage) 
        if(logf != 'screen'):
            print(logmessage)        
        return flag, None, sim

    tno_class.classes_dictionary = clf.classes_dictionary

    #short integration first

    #run the short integration
    if(not classify_only):    
        tmin = sim.t
        tmax = sim.t + 0.5e6
        if(logf):
            logmessage = "Running the 0.5 Myr short classification integration\n"
            tools.writelog(logf,logmessage) 

        rflag, sim = run_reb.run_simulation(sim,des=des,tmax=tmax,tout=50.,archivefile=archivefile,
                                            deletefile=deletefile, logfile=logf)
        if(rflag < 1):
            logmessage = "The short integration for the TNO machine learning failed\n"
            logmessage += "failed in machine_learning.run_and_MLclassify_TNO()\n"
            logmessage += "at run_reb.run_simulation\n"
            tools.writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)                   
            return flag, None, sim
    else:
        tmin = 0.
        tmax = 0.5e6
    #read the short integration
    rflag, a_short, ec_short, inc_short, node_short, peri_short, ma_short, t_short = \
            tools.read_sa_for_sbody(des=des,archivefile=archivefile,clones=clones,tmin=tmin,tmax=tmax)
    if(rflag < 1):
        logmessage = "Unable to read the small body output for the short integration\n"
        logmessage += "failed in machine_learning.run_and_MLclassify_TNO()\n"
        logmessage += "at tools.read_sa_for_sbody\n"
        tools.writelog(logf,logmessage) 
        if(logf != 'screen'):
            print(logmessage)                   
        return flag, None, sim

    pomega_short = peri_short + node_short 
    lambda_short = ma_short + peri_short+ node_short 
    q_short = a_short*(1.-ec_short)
    
    rflag, apl_short, ecpl, incpl, nodepl, peripl, mapl, tpl = \
            tools.read_sa_by_hash(obj_hash='neptune',archivefile=archivefile,tmin=tmin,tmax=tmax)
    if(rflag < 1):
        logmessage = "Unable to read Neptune's output for the short integration\n"
        logmessage += "failed in machine_learning.run_and_MLclassify_TNO()\n"
        logmessage += "at tools.read_sa_by_hash\n"
        tools.writelog(logf,logmessage) 
        if(logf != 'screen'):
            print(logmessage)        
        return flag, None, sim

    lambda_pl_short = mapl + nodepl + peripl

    rflag, xr, yr, zr, vxr, vyr, vzr, tr = \
            tools.calc_rotating_frame(des=des,planet='neptune', 
                                      archivefile=archivefile,clones=clones,tmin=tmin,tmax=tmax)
    if(rflag < 1):
        logmessage ="Unable to calculate the rotating frame for the short integration\n"
        logmessage += "failed in machine_learning.run_and_MLclassify_TNO()\n"
        logmessage += "at tools.calc_rotating_frame\n"
        tools.writelog(logf,logmessage) 
        if(logf != 'screen'):
            print(logmessage)        
        return flag, None, sim

    rrf_short = np.sqrt(xr*xr + yr*yr + zr*zr)
    phirf_short = np.arctan2(yr, xr)
    if(a_short.all()):
        #no zero/missing values in a_short
        tiss_short = apl_short/a_short + 2.*np.cos(inc_short)*np.sqrt(a_short/apl_short*(1.-ec_short*ec_short))
    else:
        #loop through and avoid dividing by zero
        tiss_short = np.zeros_like(a_short)
        for n in range(0,clones+1):
            if(a_short[n].all()):
                tiss_short[n] = apl_short/a_short[n] + 2.*np.cos(inc_short[n])*np.sqrt(a_short[n]/apl_short*(1.-ec_short[n]*ec_short[n]))


    #do a quick check for severe scattering and Centaurs to save the time of doing
    #the longer integration
    a_min = np.amin(a_short,axis=1)
    a_max = np.amax(a_short,axis=1)
    a_mean = np.mean(a_short,axis=1)
    a_delta = a_max - a_min
    all_classified = 1
    for n in range(0,clones+1):
        if(a_short[n,0] < 28. or a_mean[n] < 29):
            tno_class.clone_classification[n] = 'not_TNO'
        elif(a_delta[n] > 3.5):
            tno_class.clone_classification[n] = 'scattering'
        else:
            all_classified = 0  
    if(all_classified):
        tno_class.determine_most_common_classification()
        logmessage ="All clones were classifiable as scattering or not TNOs based on the\n"
        logmessage += "short integration alone. Returning early from machine_learning.run_and_MLclassify_TNO\n";
        tools.writelog(logf,logmessage) 
        if(logf != 'screen'):
            print(logmessage)     
        return 2, tno_class, sim

    #continue at lower resolution to 10 Myr
    if(not classify_only):
        tmax = sim.t + 9.5e6
        tmin = sim.t + 0.001e6 
        if(logf):
            logmessage = "Running the classification integration to 10 Myr\n"
            tools.writelog(logf,logmessage)         
        rflag, sim = run_reb.run_simulation(sim,des=des,tmax=tmax,tout=1000.,archivefile=archivefile,
                                        deletefile=False,logfile=logf)
        if(rflag < 1):
            logmessage ="The long integration in machine_learning.run_and_MLclassify_TNO() failed\n"
            logmessage += "at run_reb.run_simulation\n";
            tools.writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)                
            return flag, None, sim
    else:
        tmin = .5001e6
        tmax = 10e6

    #read the new part of the integration

    rflag, a, ec, inc, node, peri, ma, t = \
            tools.read_sa_for_sbody(des=des,archivefile=archivefile,clones=clones,tmin=tmin,tmax=tmax)
    if(rflag < 1):
        logmessage = "Unable to read the small body output for the long integration\n"
        logmessage += "failed in machine_learning.run_and_MLclassify_TNO()\n"
        logmessage += "at tools.read_sa_for_sbody\n"
        tools.writelog(logf,logmessage) 
        if(logf != 'screen'):
            print(logmessage)                          
        return flag, None, sim    
    pomega = peri+ node 
    lambda_t = ma + peri+ node 

    rflag, apl, ecpl, incpl, nodepl, peripl, mapl, tpl = \
            tools.read_sa_by_hash(obj_hash='neptune',archivefile=archivefile,tmin=tmin,tmax=tmax)
    if(rflag < 1):
        logmessage = "Unable to read Neptune's output for the long integration\n"
        logmessage += "failed in machine_learning.run_and_MLclassify_TNO()\n"
        logmessage += "at tools.read_sa_by_hash\n"
        tools.writelog(logf,logmessage) 
        if(logf != 'screen'):
            print(logmessage)        
        return flag, None, sim
    lambda_pl = mapl + nodepl + peripl

    q = a*(1.-ec)
    rflag, xr, yr, zr, vxr, vyr, vzr, tr = \
            tools.calc_rotating_frame(des=des,planet='neptune',archivefile=archivefile,
                                      clones=clones,tmin=tmin,tmax=tmax)
    if(rflag < 1):
        logmessage ="Unable to calculate the rotating frame for the short integration\n"
        logmessage += "failed in machine_learning.run_and_MLclassify_TNO()\n"
        logmessage += "at tools.calc_rotating_frame\n"
        tools.writelog(logf,logmessage) 
        if(logf != 'screen'):
            print(logmessage)              
        return flag, None, sim

    rrf = np.sqrt(xr*xr + yr*yr + zr*zr)
    phirf = np.arctan2(yr, xr)
    if(a.all()):
        #no zer0/missing vlaues in a
        tiss = apl/a + 2.*np.cos(inc)*np.sqrt(a/apl*(1.-ec*ec))
    else:
        #loop through and avoid dividing by zero
        tiss = np.zeros_like(a)
        for n in range(0,clones+1):
            if(a[n].all()):
                tiss[n] = apl/a[n] + 2.*np.cos(inc[n])*np.sqrt(a[n]/apl*(1.-ec[n]*ec[n]))


            
    #concatenate the downsampled short integration with the rest of the long integration
    t_long = np.concatenate((t_short[::20],t))
    lambda_pl_long = np.concatenate((lambda_pl_short[::20],lambda_pl))
    a_pl_long = np.concatenate((apl_short[::20],apl))

    a_long = np.concatenate((a_short[:,::20],a),axis=1)
    lambda_long = np.concatenate((lambda_short[:,::20],lambda_t),axis=1)
    ec_long = np.concatenate((ec_short[:,::20],ec),axis=1)
    inc_long = np.concatenate((inc_short[:,::20],inc),axis=1)
    node_long = np.concatenate((node_short[:,::20],node),axis=1)
    peri_long = np.concatenate((peri_short[:,::20],peri),axis=1)
    pomega_long = np.concatenate((pomega_short[:,::20],pomega),axis=1)
    q_long = np.concatenate((q_short[:,::20],q),axis=1)
    rrf_long = np.concatenate((rrf_short[:,::20],rrf),axis=1)
    phirf_long = np.concatenate((phirf_short[:,::20],phirf),axis=1)
    tiss_long = np.concatenate((tiss_short[:,::20],tiss),axis=1)
    


    fflag, tno_class.features = machine_learning.calc_ML_features(t_long,a_long,ec_long,
                                   inc_long,node_long,peri_long,
                                   pomega_long,q_long,rrf_long,phirf_long,tiss_long,
                                   t_short,a_short,ec_short,inc_short,
                                   node_short,peri_short,pomega_short,q_short,
                                   rrf_short,phirf_short,tiss_short,clones=clones,
                                   logfile=logf)
    if (fflag<1):
        #check to see if this is just a wildly scattering object 
        #which is often why the above fails
        a_min = np.amin(a_long,axis=1)
        a_max = np.amax(a_long,axis=1)
        a_mean = np.mean(a_long,axis=1)
        a_delta = a_max - a_min
        all_classified = 1
        for n in range(0,clones+1):
            if(a_delta[n] > 3.5):
                tno_class.clone_classification[n] = 'scattering'
            else:
                all_classified = 0  
        if(all_classified):
            tno_class.determine_most_common_classification()
            logmessage ="All clones were classifiable as scattering or not TNOs based on the\n"
            logmessage += "basic time-series data (no ML needed).\n";
            tools.writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)    
            return 2, tno_class, sim
        
        logmessage = "failed to calculate data features\n"
        logmessage +="failed in machine_learning.run_and_MLclassify_TNO()\n"
        logmessage += "at machine_learning.calc_ML_features\n";
        tools.writelog(logf,logmessage) 
        if(logf != 'screen'):
            print(logmessage)            
        
        return flag, tno_class, sim

    if(logf):
        logmessage = "running the machine learning classifier on the simulation data features\n"
        tools.writelog(logf,logmessage) 

    #apply the base classifier
    try:
        tno_class.class_probs = clf.G08_classifier.predict_proba(tno_class.features.return_features_list())
    except:
        logmessage = "failed to successfully apply the base G08 classifier\n"
        logmessage += "failed in machine_learning.run_and_MLclassify_TNO()\n"
        logmessage += "at clf.G08_classifier.predict_proba()\n"
        tools.writelog(logf,logmessage) 
        if(logf != 'screen'):
            print(logmessage)            
        return flag, tno_class, sim

    tno_class.set_initial_clone_classification()

    #we will keep track of the list of resonance angles
    #identified for any clones
    pres = [0]
    qres = [0]
    mres = [0]
    nres = [0]

    res_check_performed = np.zeros(clones+1)
    for n in range(tno_class.clones+1):
        prob_res = tno_class.class_probs[n][res_index]
        predicted_class = tno_class.clone_classification[n]
        if(predicted_class == 'not_TNO'):
            #prevent the resonance angle classifier from being run
            res_check_performed[n] = 1.
        elif( predicted_class == 'Nresonant' or prob_res >= 1e-2 ):
            #run the resonance angle classifier
            res_check_performed[n] = 1.
            rflag, p_id, q_id, m_id, n_id, angle_prob, sigma_phi_id, delta_phi_id, phi = run_res_angle_classifier(
                    img_clf=clf.phi_classifier,
                    time=t_long, lambda_pl=lambda_pl_long, lambda_tp=lambda_long[n],
                    node=node_long[n], pomega=pomega_long[n], apl=a_pl_long, a = a_long[n],
                    incbar=tno_class.features.i_mean[n],
                    logfile=logf)
            if(rflag < 1):
                logmessage += "the resonance angle classifier failed at clone " +str(n) +"\n";
                logmessage += "failed in machine_learning.run_and_MLclassify_TNO()\n"
                logmessage += "at  machine_learning.run_res_angle_classifier\n"
                tools.writelog(logf,logmessage) 
                if(logf != 'screen'):
                    print(logmessage)                    
                return flag, tno_class, sim

            tno_class.res_p[n] = p_id
            tno_class.res_q[n] = q_id
            tno_class.res_m[n] = m_id
            tno_class.res_n[n] = n_id
            tno_class.res_img_probability[n] = angle_prob

            #add the identified resonance to the list of resonances
            #if it was a reasonably high probability
            if(angle_prob >= 0.999):
                if(not p_id in pres):
                    pres.append(p_id)
                    qres.append(q_id)
                    mres.append(m_id)
                    nres.append(n_id)
                else:
                    indices_of_p = [i for i, x in enumerate(pres) if x == p_id]
                    match = 0
                    for i in indices_of_p:
                        if(qres[i] == q_id and mres[i] == m_id and nres[i] == n_id):
                            match = 1
                    if(match < 1):
                        pres.append(p_id)
                        qres.append(q_id)
                        mres.append(m_id)
                        nres.append(n_id)

            tno_class.res_phi_std[n] = sigma_phi_id
            tno_class.res_phi_delta[n] = delta_phi_id

            #make adjustments to classifications based on the resonant angle check
            if(predicted_class != 'Nresonant' and angle_prob >= 0.999):
                #it sure looks resonant. As long as it isn't a more strongly scattering object
                if(not (predicted_class == 'scattering' and tno_class.features.a_delta[n] > 2.5) ):
                    if(prob_res > 0.25):
                        #switch to the resonant classification because that was reasonably probable anyway
                        tno_class.clone_classification[n] = 'Nresonant'
                        tno_class.clone_confidence[n] = -1
                    elif(sigma_phi_id < 1.7 or delta_phi_id < 6.2):
                        #for lower probability resonant objects,
                        #switch to resonant if the standard deviation or min/max angle range is reasonable
                        tno_class.clone_classification[n] = 'Nresonant'
                        tno_class.clone_confidence[n] = -1
            elif(predicted_class == 'Nresonant' and  prob_res <= 0.95 and angle_prob < 0.99):
                #if it wasn;t a super high-confidance resonant classification and the resonant
                #angle check is also not super confident,
                #take the next most probable class instead
                temp = np.argsort(tno_class.class_probs[n])
                tno_class.clone_classification[n] = clf.classes_dictionary[temp[1]]
                tno_class.clone_confidence[n] = -1
            elif(predicted_class == 'Nresonant' and  prob_res <= 0.99 and angle_prob < 1e-3):
                #it was a fairly high-confidence resonant identification, but the resonant angle check 
                #was very negative, so take the next most probable class instead
                temp = np.argsort(tno_class.class_probs[n])
                tno_class.clone_classification[n] = clf.classes_dictionary[temp[1]]
                tno_class.clone_confidence[n] = -1
            if(tno_class.clone_classification[n] == 'Nresonant' and tno_class.features.a_delta[n] > 2.5
                and delta_phi_id > 6.2):
                #switch to scattering if delta-a is very large and it's not a cleanly librating resonant angle
                tno_class.clone_classification[n] = 'scattering'
                tno_class.clone_confidence[n] = -1
    

    #For clones that are sampled from the same object, we will
    #reloop through the clones to check for the already confidently identified resonances
    #(this helps id some of the higher order resonances that don't always trigger the angle check)
    #This is skipped if the clones are just a simulation model and not actually related
    if(related_clones==True):
        for n in range(tno_class.clones+1):
            if(tno_class.clone_classification[n] != 'Nresonant' and tno_class.features.a_delta[n] < 2.5
                and res_check_performed[n] < 1.):
                for i in range(1,len(pres)):
                    phi = pres[i]*lambda_long[n] - qres[i]*lambda_pl_long - mres[i]*pomega_long[n] - nres[i]*node_long[n]
                    cflag, angle_prob, sigma_phi, delta_phi = machine_learning.check_angle(clf.phi_classifier,t_long,phi,-1.,logfile=logf)
                    if(angle_prob > 0.999 and (sigma_phi < 1.6 or delta_phi < 6.2)):
                        #change the classification to resonant
                        tno_class.clone_classification[n] = 'Nresonant'
                        tno_class.clone_confidence[n] = -1
                        tno_class.res_p[n] = pres[i]
                        tno_class.res_q[n] = qres[i]
                        tno_class.res_m[n] = mres[i]
                        tno_class.res_n[n] = nres[i]
                        tno_class.res_phi_std[n] = sigma_phi
                        tno_class.res_phi_delta[n] = delta_phi
                        tno_class.res_img_probability[n] = angle_prob
                    elif(angle_prob>0.5):
                        #record some of the resonance parameters without changing the classification
                        tno_class.res_p[n] = pres[i]
                        tno_class.res_q[n] = qres[i]
                        tno_class.res_m[n] = mres[i]
                        tno_class.res_n[n] = nres[i]
                        tno_class.res_phi_std[n] = sigma_phi
                        tno_class.res_phi_delta[n] = delta_phi
                        tno_class.res_img_probability[n] = angle_prob
    
    #run the minor corrections and assign clone-by-clone and most common classes
    tno_class.correct_clone_classification()
    tno_class.determine_most_common_classification()

   
    flag = 1
    return flag, tno_class, sim
#################################################################




################################################################
def run_res_angle_classifier(img_clf=None,time=None,lambda_pl=None,lambda_tp=None,node=None,
                             pomega=None,apl=None,a=None,incbar=None,logfile=False):
    '''
    Produces plots of a large number of resonant angles and runs them through a 
    ML classifier to determine if the resonant angle is librating or not
    '''

    logf=logfile

    flag = 0

    max_iterations = 50
    qlimit = 16
    mlimit = 30
    rescheck_soft_limit = 100
    max_reschecks = 150
    p_id = 0
    q_id = 0
    m_id = 0
    n_id = 0

    a_bar = np.mean(a)
    halfway = int(len(a)/2)
    a_bar_n = np.mean(apl)
    a_bar1 = np.mean(a[0:halfway])
    a_bar2 = np.mean(a[halfway::])

    obspr = np.power((a_bar/a_bar_n),(1.5))
    obspr1 = np.power((a_bar1/a_bar_n),(1.5))
    obspr2 = np.power((a_bar2/a_bar_n),(1.5))
    num = np.array([0, 1])
    denom = np.array([1, 1])
    prtol = 0.02
    prmax=1./((1.0-prtol)*min(obspr,obspr2,obspr1))
    prmin=1./((1.0+prtol)*max(obspr,obspr2,obspr1))


    res_flag=0
    iterations = 0
    reschecks = 0
    max_prob = 0.
    sigma_phi_id = 0.
    delta_phi_id = 0.

    phi = None


    if(prmax > 1.0 and prmin < 1.0):
        #check for coorbital resonance
        phi = lambda_tp - lambda_pl
        cflag, prob, sigma_phi, delta_phi = machine_learning.check_angle(img_clf,time,phi,max_prob,logfile=logf)
        if(not(cflag)):
            logmessage = "machine_learning.run_res_angle_classifier failed\n";
            logmessage += "at machine_learning.check_angle\n";
            tools.writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)                
            return flag, p_id, q_id, m_id, n_id, max_prob, sigma_phi_id, delta_phi_id, phi
        if(prob > 0.5):
            flag = 1
            return flag, 1, 1, 0, 0, prob, sigma_phi_id, delta_phi_id, phi
        elif(prob > max_prob):
            max_prob = prob
            sigma_phi_id = sigma_phi
            delta_phi_id = delta_phi
            p_id = 1
            q_id = 1
            m_id = 0
            n_id = 0

    #run through other possible resonances
    while(res_flag == 0):
        iterations+=1
        #get the next nearest, lowest order resonances to check
        ftflag, num, denom, new_check_q, new_check_p, n_check = resonances.farey_tree(num, denom, prmin, prmax)
        if(not(ftflag)):
            logmessage = "tno_classifier.run_res_angle_classifier had a problem at\n";
            logmessage += "resonances.farey_tree call (so resonance check might be incomplete)"
            tools.writelog(logf,logmessage) 
            if(logf != 'screen'):
                print(logmessage)                
            break
        if(ftflag == 2):
            #we've hit the end of the tree
            break
            #return flag, p_id, q_id, m_id, n_id, max_prob, sigma_phi_id, delta_phi_id, phi
        if(iterations > max_iterations or reschecks > max_reschecks): 
            #hit the reasonable limit of how many resonances to check
            flag = 2
            break
            #return flag, p_id, q_id, m_id, n_id, max_prob, sigma_phi_id, delta_phi_id
        if(reschecks > rescheck_soft_limit and max_prob > 0.1):
            #probably have the correct resonance and it's just messy
            #so we will go ahead and quit the search now
            flag = 2
            break
            #return flag, p_id, q_id, m_id, n_id, max_prob, sigma_phi_id, delta_phi_id

        for jk in range(n_check):
            if(res_flag > 0):
                #we've already identified a good resonance angle
                #exit the search
                flag = 1
                break
            if(reschecks > max_reschecks):
                #exit the search
                break

            pr = int(new_check_p[jk])
            qr = int(new_check_q[jk])
            mr_e = (pr-qr)
            if(reschecks > rescheck_soft_limit):
                if(mr_e > mlimit and qr > qlimit):
                    #skip this one as unlikely
                    continue
            nr=0.
            phi = pr*lambda_tp - qr*lambda_pl - mr_e*pomega
            cflag, prob, sigma_phi, delta_phi = machine_learning.check_angle(img_clf,time,phi,max_prob,logfile=logf)
            reschecks+=1
            if(not(cflag)):
                logmessage = "tno_classifier.run_res_angle_classifier failed\n";
                logmessage += "at machine_learning.check_angle for e-type resonance\n"
                tools.writelog(logf,logmessage) 
                if(logf != 'screen'):
                    print(logmessage)                    
                return flag, p_id, q_id, m_id, n_id, max_prob, sigma_phi_id, delta_phi_id, phi
            if(prob > 0.5):
                res_flag = 1
            if(prob > max_prob):
                p_id = pr
                q_id = qr
                m_id = mr_e
                n_id = nr
                max_prob = prob
                sigma_phi_id = sigma_phi
                delta_phi_id = delta_phi
            #check the n=2 case regardless of the outcome of the n=0:
            if(mr_e >=2):
                mr = mr_e - 2
                nr = 2                
                phi = pr*lambda_tp - qr*lambda_pl - mr*pomega - nr*node
                cflag, prob, sigma_phi, delta_phi = machine_learning.check_angle(img_clf,time,phi,max_prob,logfile=logf)
                reschecks+=1
                if(not(cflag)):
                    logmessage = "tno_classifier.run_res_angle_classifier failed\n";
                    logmessage += "at machine_learning.check_angle for n=2 resonance\n"
                    tools.writelog(logf,logmessage) 
                    if(logf != 'screen'):
                        print(logmessage)                       
                    return flag, p_id, q_id, m_id, n_id, max_prob, sigma_phi_id, delta_phi_id, phi
                
                if( (prob > 0.5 and res_flag == 0) or (prob > max_prob and res_flag == 1) ):
                    #the mixed mode resonance is better than the eccentricity type
                    res_flag = 2
                    p_id = pr
                    q_id = qr
                    m_id = mr
                    n_id = nr
                    max_prob = prob
                    sigma_phi_id = sigma_phi
                    delta_phi_id = delta_phi
                elif(prob == max_prob and sigma_phi < sigma_phi_id and 
                    ( (delta_phi_id > 355*np.pi/180. and delta_phi < 355*np.pi/180.) or delta_phi < 0.9*delta_phi_id) ):
                    if(prob > 0.5):
                        res_flag = 2
                    p_id = pr
                    q_id = qr
                    m_id = mr
                    n_id = nr
                    max_prob = prob
                    sigma_phi_id = sigma_phi
                    delta_phi_id = delta_phi
            #if the e-type resonance wasn't the winner, keep checking other mixed-mode resonances
            #if we are dealing with an at least moderately inclined orbit
            if(flag != 1 and mr_e >=4 and incbar > 7.*np.pi/180.):
                mr = mr_e - 4
                nr = 4
                while(mr>=0):
                    phi = pr*lambda_tp - qr*lambda_pl - mr*pomega - nr*node
                    cflag, prob, sigma_phi, delta_phi = machine_learning.check_angle(img_clf,time,phi,max_prob,logfile=logf)
                    reschecks+=1
                    if(not(cflag)):
                        print("tno_classifier.run_res_angle_classifier failed")
                        logmessage = "tno_classifier.run_res_angle_classifier failed\n";
                        logmessage += "at machine_learning.check_angle for high-order mixed resonance\n"
                        tools.writelog(logf,logmessage) 
                        if(logf != 'screen'):
                            print(logmessage)   
                        return flag, p_id, q_id, m_id, n_id, max_prob, sigma_phi_id, delta_phi_id, phi
                    if( (prob > 0.5 and res_flag == 0) or (prob > max_prob and res_flag == 2) ):
                        res_flag = 3
                    if(prob > max_prob):
                        p_id = pr
                        q_id = qr
                        m_id = mr
                        n_id = nr
                        max_prob = prob
                        sigma_phi_id = sigma_phi
                        delta_phi_id = delta_phi
                    elif(prob == max_prob and sigma_phi < sigma_phi_id and 
                        ( (delta_phi_id > 355*np.pi/180. and delta_phi < 355*np.pi/180.) or delta_phi<0.9*delta_phi_id) ):
                        max_prob = prob
                        p_id = pr
                        q_id = qr
                        m_id = mr
                        n_id = nr 
                        sigma_phi_id = sigma_phi
                        delta_phi_id = delta_phi
                        if(res_flag > 1):
                            res_flag = 3
                    mr = mr - 2
                    nr = nr + 2
                    if(nr > 8 and incbar < 30.*np.pi/180):
                        #move on to the next p/q resonance
                        break                    
                    if(reschecks > max_reschecks):
                        #go to the start of the loop, which will exist for 
                        #maxing out the number of checks
                        break

    if(res_flag == 0):
        flag = 2
    elif(flag == 0):
        flag = 1
    
    return flag, p_id, q_id, m_id, n_id, max_prob, sigma_phi_id, delta_phi_id, phi

