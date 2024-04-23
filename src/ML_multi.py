import rebound
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../src')
import sbdynt as sbd
import tools
import machine_learning as ml
from datetime import date
import run_reb


def run_object(shortarchive='',longarchive='',tno='',clones=0):
    print(tno,shortarchive)
    flag, a, ec, inc, node, peri, ma, t = tools.read_sa_for_sbody(sbody=tno,archivefile=shortarchive,nclones=clones)
    pomega = peri+ node 
    flag, apl, ecpl, incpl, nodepl, peripl, mapl, tpl = tools.read_sa_by_hash(obj_hash='neptune',archivefile=shortarchive)
    q = a*(1.-ec)
    flag, xr, yr, zr, vxr, vyr, vzr, tr = tools.calc_rotating_frame(sbody=tno, planet='neptune', 
                                                                    archivefile=shortarchive, nclones=clones)
    rrf = np.sqrt(xr*xr + yr*yr + zr*zr)
    phirf = np.arctan2(yr, xr)
    tiss = apl/a + 2.*np.cos(inc)*np.sqrt(a/apl*(1.-ec*ec))

    flag, l_a, l_ec, l_inc, l_node, l_peri, l_ma, l_t = tools.read_sa_for_sbody(sbody=tno,archivefile=longarchive,nclones=clones)
    l_pomega = l_peri+ l_node 
    flag, apl, ecpl, incpl, nodepl, peripl, mapl, tpl = tools.read_sa_by_hash(obj_hash='neptune',archivefile=longarchive)
    l_q = l_a*(1.-l_ec)
    flag, xr, yr, zr, vxr, vyr, vzr, tr = tools.calc_rotating_frame(sbody=tno, planet='neptune', 
                                                                    archivefile=longarchive, nclones=clones)
    l_rrf = np.sqrt(xr*xr + yr*yr + zr*zr)
    l_phirf = np.arctan2(yr, xr)
    l_tiss = apl/l_a + 2.*np.cos(l_inc)*np.sqrt(l_a/apl*(1.-l_ec*l_ec))

    #first list is the always removed set
    index_remove = [25, 27, 32, 34, 39, 41, 46, 48, 53, 55, 93, 95, 116, 117, 118, 119, 120, 121, 122, 
                    123, 124, 125, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 147, 234, 236, 241, 
                    243, 248, 250, 255, 257, 262, 264, 302, 304, 325, 326, 327, 328, 329, 330, 331, 332, 
                    333, 334, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 356]

    #set to remove for current best classifier (subject to change)
    index_remove = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 25, 27, 32, 34, 39, 41, 46, 48, 
                    53, 55, 93, 95, 101, 102, 103, 104, 105, 106, 107, 110, 112, 114, 116, 117, 118, 119, 
                    120, 121, 122, 123, 124, 125, 127, 128, 130, 132, 134, 135, 136, 137, 138, 139, 140, 
                    141, 142, 143, 147, 149, 152, 183, 184, 186, 187, 188, 190, 234, 236, 241, 243, 248, 
                    250, 255, 257, 262, 264, 268, 270, 272, 273, 274, 275, 276, 280, 282, 287, 289, 302, 
                    304, 310, 311, 312, 313, 314, 315, 316, 319, 321, 323, 325, 326, 327, 328, 329, 330, 
                    331, 332, 333, 334, 336, 337, 339, 341, 343, 344, 345, 346, 347, 348, 349, 350, 351, 
                    352, 356, 358, 361, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 392, 
                    393, 396, 397]


    n=0
    short_features = ml.calc_ML_features(t,a[n],ec[n],inc[n],node[n],peri[n],
                                        pomega[n],q[n],rrf[n],phirf[n],tiss[n])
    long_features = ml.calc_ML_features(l_t,l_a[n],l_ec[n],l_inc[n],l_node[n],l_peri[n],
                                        l_pomega[n],l_q[n],l_rrf[n],l_phirf[n],l_tiss[n])

    all_features = np.concatenate((long_features, short_features),axis=0)
    temp_features = np.delete(all_features,index_remove)
    features = np.array([temp_features])

    for n in range(1,clones+1):
        short_features = ml.calc_ML_features(t,a[n],ec[n],inc[n],node[n],peri[n],
                                        pomega[n],q[n],rrf[n],phirf[n],tiss[n])
        long_features = ml.calc_ML_features(l_t,l_a[n],l_ec[n],l_inc[n],l_node[n],l_peri[n],
                                        l_pomega[n],l_q[n],l_rrf[n],l_phirf[n],l_tiss[n])
        all_features = np.concatenate((long_features, short_features),axis=0)
        temp_features = np.delete(all_features,index_remove)
        features = np.append(features,[temp_features],axis=0)
   
    return features
    
training_file = '../src/TNO-ML-training-data.csv'
data = pd.read_csv('../data/data_files/DES_objects.csv')
classes = []
probs = []
(clf, score, classes_train, classes_test, 
 features_train, features_test, 
 feature_names, classes_dictionary) = sbd.train_and_test_TNO_classifier(training_file)

#for i in range(4):
for i in range(len(data)):
    if i%20==0:
        print(i)
    tno = str(data['Name'][i])

    today = date.today()
    datestring = today.strftime("%b-%d-%Y")
    
    file = '../data/DES_objects/'+tno+'/archive_init.bin'
    shortarchive = '../data/DES_objects/'+tno+'/shortarchive.bin'
    longarchive = '../data/DES_objects/'+tno+'/archive.bin'
    sim = rebound.Simulation(file)
    flag, sim2 = run_reb.run_simulation(sim,tmax=0.5e6,tout=50.,filename=shortarchive,deletefile=True)
    
    
    '''
    import os
    for fname in os.listdir('../data/ML_archives/.'):    # change directory as needed
        if tno in fname:
            if 'short' in fname:
                shortarchive = str('../data/ML_archives/'+fname)
            elif 'long' in fname:
                longarchive = str('../data/ML_archives/'+fname)
    '''
    clones = 0
    try:
        features = run_object(shortarchive,longarchive,tno,clones)
        predicted_classes = clf.predict(features)
        classes.append(predicted_classes)
    except:
        print("Didn't read properly")
        classes.append(3)
    
    
    #class_probs = clf.predict_probs(features)
    #probs.append(class_probs)
classes_dictionary[3] = 'NaN'
data['Class_SBDynT'] = np.zeros(len(data))
for i in range(len(classes)):
#for i in range(4):
    pred = classes[i]
    pred = int(np.mean(pred))
    #cprob = probs[i]
    print(pred)
    data['Class_SBDynT'][i] = classes_dictionary[pred]
    #data['Class_prob'][i] = class_probs[i][cprob]

data.to_csv('ML_class_DES_objects_longcadence.csv')