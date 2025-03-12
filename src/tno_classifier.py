import sys
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
        self.class_probs = None
        
        #clone-by-clone predicted classification and 
        #confidence level for that classification
        self.clone_classification = None
        self.clone_confidence = None

        self.most_common_class = None
        self.fraction_most_common_class = None

        #resonance parameters for resonant argument
        # phi = p*lambda_tno - q*lambda_neptune - m*(long of perihelion)_tno - n*(long of ascending node)_tno
        self.res_p = None
        self.res_q = None
        self.res_m = None
        self.res_n = None

    def determine_clone_classification(self):
        # find the most probable class and associated confidence on a 
        # clone-by-clone basis
        # make an empty list and empty array
        self.clone_classification = (self.clones+1)*[None]
        self.clone_confidence = np.zeros(self.clones+1)
        for n in range (self.clones+1):
            cn = np.argmax(self.class_probs[n])
            self.clone_classification[n] = self.classes_dictionary[cn]
            self.clone_confidence[n] = self.class_probs[n,cn]

            #do checks for the scattering/detached boundary
            if(self.clone_classification[n] == 'class-det'):
                #check if it meets the scattering requirement
                if(self.features.a_delta[n] > 1.5):
                    self.clone_classification[n] = 'scattering'
                elif(self.features.e_mean[n] < 0.24):
                    if(self.features.a_mean[n] < 39.4):
                        self.clone_classification[n] = 'classical-inner'
                    elif(self.features.a_mean[n] < 47.7):
                        self.clone_classification[n] = 'classical-main'
                    else:
                        self.clone_classification[n] = 'classical-outer'
                else:
                    self.clone_classification[n] = 'detached'
            elif(self.clone_classification[n] == 'scattering'):
                #check if it meets the scattering requirement
                if(self.features.a_delta[n] <= 1.5):
                    self.clone_classification[n] = 'detached'
                    if(self.features.e_mean[n] < 0.24):
                        if(self.features.a_mean[n] < 39.4):
                            self.clone_classification[n] = 'classical-inner'
                        elif(self.features.a_mean[n] < 47.7):
                            self.clone_classification[n] = 'classical-main'
                        else:
                            self.clone_classification[n] = 'classical-outer'

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

        print("Most common classification: %s\n" % self.most_common_class)
        percentage = 100*self.fraction_most_common_class
        print("Shared by %f percent of clones\n\n" % percentage)

        nclas = len(self.classes_dictionary)
        print("Clone number, most probable G08 class, probability of that class, ",end ="")
        print("probability of ", end ="")
        for n in range(nclas):
            print("%s, " % self.classes_dictionary[n],end ="")
        print("\n",end ="")
        format_string = "%d, %s, "
        for n in range(nclas-1):
            format_string+="%e, "
        format_string+="%e,\n"
        for n in range(0,self.clones+1):
            print("%d, %s, %e, " % (n,self.clone_classification[n], 
                                    self.clone_confidence[n]),end ="")
            for j in range(nclas):
                print("%e, " % self.class_probs[n][j] ,end ="")
            print("\n",end ="")







#################################################################
def run_and_MLclassify_TNO(sim=None, des=None, clones=None, 
                            datadir='', archivefile=None, 
                            deletefile=False,logfile=False):
    '''
    add documentation here...
    '''
    flag = 0

    if(des == None):
        print("The designation of the small body must be provided")
        print("failed at machine_learning.run_and_MLclassify_TNO()")
        return flag, None, sim


    if(sim == None):
        #initialize a default simulation
        iflag, sim, epoch, clones, cloning_method, weights = \
                tno.setup_default_tno_integration(des=des, clones=clones, datadir=datadir,
                        save_sbdb=False,saveic=False,archivefile=archivefile,logfile=logfile)
        if(iflag < 1):
            print("Failed at simulation initialization stage")
            print("failed at machine_learning.run_and_MLclassify_TNO()")    
            return flag, None, sim

    if(logfile==True):
        logf = tools.log_file_name(des=des)
    else:
        logf=logfile
    if(datadir and logf and logf!='screen'):        
        logf = datadir + '/' +logf

    flag = 0
    #make an empty set of classification outputs 
    tno_class = machine_learning.TNO_ML_outputs(clones)

    if(archivefile == None):
        archivefile = tools.archive_file_name(des=des)
    
    if(datadir):
        archivefile = datadir + "/" + archivefile

    #short integration first
    tmin = sim.t
    tmax = sim.t + 0.5e6
    #run the short integration
    rflag, sim = run_reb.run_simulation(sim,des=des,tmax=tmax,tout=50.,archivefile=archivefile,
                                        deletefile=deletefile, logfile=logfile)
    if(rflag < 1):
        print("The short integration for the TNO machine learning failed")
        print("failed at machine_learning.run_and_MLclassify_TNO()")
        return flag, None, sim
    #read the short integration
    rflag, a_short, ec_short, inc_short, node_short, peri_short, ma_short, t_short = \
            tools.read_sa_for_sbody(des=des,archivefile=archivefile,clones=clones,tmin=tmin,tmax=tmax)
    if(rflag < 1):
        print("Unable to read the output for the short integration")
        print("failed at machine_learning.run_and_MLclassify_TNO()")
        return flag, None, sim

    pomega_short = peri_short+ node_short 
    q_short = a_short*(1.-ec_short)
    
    rflag, apl, ecpl, incpl, nodepl, peripl, mapl, tpl = \
            tools.read_sa_by_hash(obj_hash='neptune',archivefile=archivefile,tmin=tmin,tmax=tmax)
    if(rflag < 1):
        return flag, None, sim

    rflag, xr, yr, zr, vxr, vyr, vzr, tr = \
            tools.calc_rotating_frame(des=des,planet='neptune', 
                                      archivefile=archivefile,clones=clones,tmin=tmin,tmax=tmax)
    if(rflag < 1):
        return flag, None, sim

    rrf_short = np.sqrt(xr*xr + yr*yr + zr*zr)
    phirf_short = np.arctan2(yr, xr)
    tiss_short = apl/a_short + 2.*np.cos(inc_short)*np.sqrt(a_short/apl*(1.-ec_short*ec_short))

    #continue at lower resolution to 10 Myr
    tmax = sim.t + 9.5e6
    tmin = sim.t + 0.001e6 
    rflag, sim = run_reb.run_simulation(sim,des=des,tmax=tmax,tout=1000.,archivefile=archivefile,
                                        deletefile=False,logfile=logfile)
    if(rflag < 1):
        return flag, None, sim 

    #read the new part of the integration

    rflag, a, ec, inc, node, peri, ma, t = \
            tools.read_sa_for_sbody(des=des,archivefile=archivefile,clones=clones,tmin=tmin,tmax=tmax)
    if(rflag < 1):
        return flag, None, sim    
    pomega = peri+ node 

    rflag, apl, ecpl, incpl, nodepl, peripl, mapl, tpl = \
            tools.read_sa_by_hash(obj_hash='neptune',archivefile=archivefile,tmin=tmin,tmax=tmax)
    if(rflag < 1):
        return flag, None, sim

    q = a*(1.-ec)
    rflag, xr, yr, zr, vxr, vyr, vzr, tr = \
            tools.calc_rotating_frame(des=des,planet='neptune',archivefile=archivefile,
                                      clones=clones,tmin=tmin,tmax=tmax)
    if(rflag < 1):
        return flag, None, sim

    rrf = np.sqrt(xr*xr + yr*yr + zr*zr)
    phirf = np.arctan2(yr, xr)
    tiss = apl/a + 2.*np.cos(inc)*np.sqrt(a/apl*(1.-ec*ec))


            
    #concatenate the downsampled short integration with the rest of the long integration
    t_long = np.concatenate((t_short[::20],t))
    a_long = np.concatenate((a_short[:,::20],a),axis=1)
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
                                    rrf_short,phirf_short,tiss_short,clones=clones)
    if (fflag<1):
        print("failed to calculate data features")
        print("failed at machine_learning.run_and_MLclassify_TNO()")
        return flag, None, sim

    cflag, classifier, tno_class.classes_dictionary = initialize_TNO_classifier()
    if(cflag<1):
        print("failed to initialize machine learning classifier")
        print("failed at machine_learning.run_and_MLclassify_TNO()")
        return flag, None, sim


    #apply the base classifier
    tno_class.class_probs = classifier.predict_proba(tno_class.features.return_features_list())





    #run the minor corrections and assign clone-by-clone and most common classes
    tno_class.determine_clone_classification()
    tno_class.determine_most_common_classification()

   
    flag = 1
    return flag, tno_class, sim
#################################################################












#################################################################
def run_res_angle_classifier(img_clf=None,mean_long_pl=None,mean_long=None,node=None,pomega=None,
                             pr_min=None,pr_max=None,incbar=None,clones=0):
    #if(img_clf=None):
    #    flag, img_clf = initialize_image_classifier()
    
    max_iterations = 50
    qlimit = 16
    mlimit = 30
    rescheck_soft_limit = 100
    max_reschecks = 150

    p_id = 0
    q_id = 0
    m_id = 0
    n_id = 0
    num = np.array([0, 1])
    denom = np.array([1, 1])
    prtol = 0.02

    res_flag=0
    iterations = 0
    reschecks = 0
    max_prob = 0.
    sigma_phi_id = 0.
    delta_phi_id = 0.


    if(prmax > 1.0 and prmin < 1.0):
        #check for coorbital resonance
        phi = mean_long - mean_long_pl
        cflag, prob, sigma_phi, delta_phi = machine_learning.check_angle(img_clf,phi,max_prob)
        if(not(cflag)):
            print("machine_learning.run_res_angle_classifier failed")
            return flag, p_id, q_id, m_id, n_id, max_prob
        if(prob > 0.5):
            flag = 1
            return flag, 1, 1, 0, 0, prob
        elif(prob > max_prob):
            max_prob = prob
            sigma_phi_id = sigma_phi
            delta_phi_id = delta_phi
            p_id = 1
            q_id = 1

    #run through other possible resonances
    while(res_flag == 0):
        iterations+=1
        #get the next nearest, lowest order resonances to check
        ftflag, num, denom, new_check_q, new_check_p, n_check = resonances.farey_tree(num, denom, prmin, prmax)
        if(not(ftflag)):
            print("quitting tno_classifier.run_res_angle_classifier early due to failure in")
            print("resonances.farey_tree call")
            flag = 2
            return flag, p_id, q_id, m_id, n_id, max_prob
        if(iterations > max_iterations or reschecks > max_reschecks): 
            #hit the reasonable limit of how many resonances to check
            flag = 2
            return flag, p_id, q_id, m_id, n_id, max_prob
        if(reschecks > rescheck_soft_limit and max_prob > 0.1):
            #probably have the correct resonance and it's just messy
            #so we will go ahead and quit the search now
            flag = 2
            return flag, p_id, q_id, m_id, n_id, max_prob

        for jk in range(n_check):
            if(res_flag > 0):
                #we've already identified a good resonance angle
                #exit the search
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
            phi = pr*mean_long - qr*mean_long_pl - mr_e*pomega
            cflag, prob, sigma_phi, delta_phi = machine_learning.check_angle(img_clf,phi,max_prob)
            reschecks+=1
            if(not(cflag)):
                print("tno_classifier.run_res_angle_classifier failed")
                return flag, p_id, q_id, m_id, n_id, max_prob
            if(prob > 0.5):
                res_flag = 1
            if(prob > max_prob):
                p_id = pr
                q_id = qr
                m_id = mr_e
                max_prob = prob
                sigma_phi_id = sigma_phi
                delta_phi_id = delta_phi
            #check the n=2 case regardless of the outcome of the n=0:
            if(mr_e >=2):
                mr = mr_e - 2
                nr = 2                
                phi = pr*mean_long - qr*mean_long_pl - mr*pomega - nr*node
                cflag, prob, sigma_phi, delta_phi = machine_learning.check_angle(img_clf,phi,max_prob)
                reschecks+=1
                if(not(cflag)):
                    print("tno_classifier.run_res_angle_classifier failed")
                    return flag, p_id, q_id, m_id, n_id, max_prob
                
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
                    (delta_phi_id > 355*np.pi/180. or delta_phi<0.9*delta_phi_id) ):
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
                    phi = pr*mean_long - qr*mean_long_pl - mr*pomega - nr*node
                    cflag, prob, sigma_phi, delta_phi = machine_learning.check_angle(img_clf,phi,max_prob)
                    reschecks+=1
                    if(not(cflag)):
                        print("tno_classifier.run_res_angle_classifier failed")
                        return flag, p_id, q_id, m_id, n_id, max_prob
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
                        (delta_phi_id > 355*np.pi/180. or delta_phi<0.9*delta_phi_id)):
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
        return flag, p_id, q_id, m_id, n_id, max_prob
    
    flag = 1
    return flag, p_id, q_id, m_id, n_id, max_prob

