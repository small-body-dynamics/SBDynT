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
        # make an empty list and empty array
        for n in range (self.clones+1):
            cn = np.argmax(self.class_probs[n])
            self.clone_classification[n] = self.classes_dictionary[cn]
            self.clone_confidence[n] = self.class_probs[n,cn]
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

        print("#Most common classification: %s" % self.most_common_class)
        percentage = 100*self.fraction_most_common_class
        print("#Shared by %f percent of clones\n#" % percentage)

        nclas = len(self.classes_dictionary)
        print("#Clone number, most probable G08 class, p, q, m, n, phi_std, phi_delta, res_image_prob, probability of that class, ",end ="")
        print("probability of ", end ="")
        for n in range(nclas):
            print("%s, " % self.classes_dictionary[n],end ="")
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




#################################################################
def run_and_MLclassify_TNO(sim=None, des=None, clones=None, 
                            datadir='', archivefile=None, 
                            deletefile=False,logfile=False,
                            classify_only=False):
    '''
    add documentation here...
    '''
    flag = 0

    if(des == None):
        print("The designation of the small body must be provided")
        print("failed at machine_learning.run_and_MLclassify_TNO()")
        return flag, None, sim


    if(sim == None and classify_only == False):
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
    tno_class = TNO_ML_outputs(clones)

    #initialize the machine learning classifier
    clf = machine_learning.TNO_ML_classifier()
    cflag = clf.initialize_classifiers()
    if(cflag<1):
        print("failed to initialize machine learning classifier")
        print("failed at tno_classifier.run_and_MLclassify_TNO()")
        return flag, None, sim
    res_index = -1
    for i in range(len(clf.classes_dictionary)):
        if (clf.classes_dictionary[i] == 'Nresonant'):
            res_index=i
    if(res_index < 0):
        print("failed to find 'Nresonant' in the classifier dictionary")
        print("failed at machine_learning.run_and_MLclassify_TNO()")
        return flag, None, sim

    tno_class.classes_dictionary = clf.classes_dictionary

    if(archivefile == None):
        archivefile = tools.archive_file_name(des=des)
    
    if(datadir):
        archivefile = datadir + "/" + archivefile

    #short integration first

    #run the short integration
    if(not classify_only):    
        tmin = sim.t
        tmax = sim.t + 0.5e6
        rflag, sim = run_reb.run_simulation(sim,des=des,tmax=tmax,tout=50.,archivefile=archivefile,
                                        deletefile=deletefile, logfile=logfile)
        if(rflag < 1):
            print("The short integration for the TNO machine learning failed")
            print("failed at machine_learning.run_and_MLclassify_TNO()")
            return flag, None, sim
    else:
        tmin = 0.
        tmax = 0.5e6
    #read the short integration
    rflag, a_short, ec_short, inc_short, node_short, peri_short, ma_short, t_short = \
            tools.read_sa_for_sbody(des=des,archivefile=archivefile,clones=clones,tmin=tmin,tmax=tmax)
    if(rflag < 1):
        print("Unable to read the output for the short integration")
        print("failed at machine_learning.run_and_MLclassify_TNO()")
        return flag, None, sim

    pomega_short = peri_short+ node_short 
    lambda_short = ma_short + peri_short+ node_short 
    q_short = a_short*(1.-ec_short)
    
    rflag, apl_short, ecpl, incpl, nodepl, peripl, mapl, tpl = \
            tools.read_sa_by_hash(obj_hash='neptune',archivefile=archivefile,tmin=tmin,tmax=tmax)
    if(rflag < 1):
        return flag, None, sim
    lambda_pl_short = mapl + nodepl + peripl

    rflag, xr, yr, zr, vxr, vyr, vzr, tr = \
            tools.calc_rotating_frame(des=des,planet='neptune', 
                                      archivefile=archivefile,clones=clones,tmin=tmin,tmax=tmax)
    if(rflag < 1):
        print("Unable to calculate the rotating frame for the short integration")
        return flag, None, sim

    rrf_short = np.sqrt(xr*xr + yr*yr + zr*zr)
    phirf_short = np.arctan2(yr, xr)
    tiss_short = apl_short/a_short + 2.*np.cos(inc_short)*np.sqrt(a_short/apl_short*(1.-ec_short*ec_short))

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
        return 2, tno_class, sim

    #continue at lower resolution to 10 Myr
    if(not classify_only):
        tmax = sim.t + 9.5e6
        tmin = sim.t + 0.001e6 
        rflag, sim = run_reb.run_simulation(sim,des=des,tmax=tmax,tout=1000.,archivefile=archivefile,
                                        deletefile=False,logfile=logfile)
        if(rflag < 1):
            return flag, None, sim
    else:
        tmin = .5001e6
        tmax = 10e6

    #read the new part of the integration

    rflag, a, ec, inc, node, peri, ma, t = \
            tools.read_sa_for_sbody(des=des,archivefile=archivefile,clones=clones,tmin=tmin,tmax=tmax)
    if(rflag < 1):
        return flag, None, sim    
    pomega = peri+ node 
    lambda_t = ma + peri+ node 

    rflag, apl, ecpl, incpl, nodepl, peripl, mapl, tpl = \
            tools.read_sa_by_hash(obj_hash='neptune',archivefile=archivefile,tmin=tmin,tmax=tmax)
    if(rflag < 1):
        return flag, None, sim
    lambda_pl = mapl + nodepl + peripl

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
                                    rrf_short,phirf_short,tiss_short,clones=clones)
    if (fflag<1):
        #check to see if this is just a wildly scattering object 
        #x, 2-d numpy array of size/shape ([clones+1,nout])
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
            return 2, tno_class, sim
        
        print("failed to calculate data features")
        print("failed at machine_learning.run_and_MLclassify_TNO()")
        return flag, None, sim

    #apply the base classifier
    try:
        tno_class.class_probs = clf.G08_classifier.predict_proba(tno_class.features.return_features_list())
    except:
        print("failed to apply the base G08 classifier")
        print("failed at machine_learning.run_and_MLclassify_TNO()")
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
        if(predicted_class == 'Nresonant' or prob_res >= 1e-2):
            #run the resonance angle classifier
            res_check_performed[n] = 1.
            rflag, p_id, q_id, m_id, n_id, angle_prob, sigma_phi_id, delta_phi_id, phi = run_res_angle_classifier(
                    img_clf=clf.phi_classifier,
                    time=t_long, lambda_pl=lambda_pl_long, lambda_tp=lambda_long[n],
                    node=node_long[n], pomega=pomega_long[n], apl=a_pl_long, a = a_long[n],
                    incbar=tno_class.features.i_mean[n])
            if(rflag < 1):
                print("failed to classify resonance angle for clone %d" % n)
                print("failed at machine_learning.run_and_MLclassify_TNO()")
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

            #make adjustments based on the resonant angle check
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
    
    #reloop through the clones to check for the already confidently idenfied resonances
    #(this helps id some of the higher order resonances that don't always trigger the angle check)
    for n in range(tno_class.clones+1):
        if(tno_class.clone_classification[n] != 'Nresonant' and tno_class.features.a_delta[n] < 2.5
            and res_check_performed[n] < 1.):
            for i in range(1,len(pres)):
                phi = pres[i]*lambda_long[n] - qres[i]*lambda_pl_long - mres[i]*pomega_long[n] - nres[i]*node_long[n]
                cflag, angle_prob, sigma_phi, delta_phi = machine_learning.check_angle(clf.phi_classifier,t_long,phi,-1.)
                #print(n, pres[i], qres[i], mres[i], nres[i], cflag, angle_prob, sigma_phi, delta_phi)
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
                             pomega=None,apl=None,a=None,incbar=None):

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
        cflag, prob, sigma_phi, delta_phi = machine_learning.check_angle(img_clf,time,phi,max_prob)
        if(not(cflag)):
            print("machine_learning.run_res_angle_classifier failed")
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
            print("in tno_classifier.run_res_angle_classifier, there was a failure in")
            print("resonances.farey_tree call")
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
            cflag, prob, sigma_phi, delta_phi = machine_learning.check_angle(img_clf,time,phi,max_prob)
            reschecks+=1
            if(not(cflag)):
                print("tno_classifier.run_res_angle_classifier failed")
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
                cflag, prob, sigma_phi, delta_phi = machine_learning.check_angle(img_clf,time,phi,max_prob)
                reschecks+=1
                if(not(cflag)):
                    print("tno_classifier.run_res_angle_classifier failed")
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
                    cflag, prob, sigma_phi, delta_phi = machine_learning.check_angle(img_clf,time,phi,max_prob)
                    reschecks+=1
                    if(not(cflag)):
                        print("tno_classifier.run_res_angle_classifier failed")
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
        #return flag, p_id, q_id, m_id, n_id, max_prob, sigma_phi_id, delta_phi_id
    elif(flag == 0):
        flag = 1
    
    return flag, p_id, q_id, m_id, n_id, max_prob, sigma_phi_id, delta_phi_id, phi

