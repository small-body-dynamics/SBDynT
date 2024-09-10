import rebound
import sys
#change the next line to reflect where you have downloaded the source code
#sys.path.insert(0, '/Users/kvolk/Documents/GitHub/SBDynT/src')
sys.path.insert(0, '../src')
import sbdynt as sbd
import pandas as pd
import os
import sys
import numpy as np
import functools
import schwimmbad
import json

def calc_ML(name,clf=None):
    clones = 0
    try:
        #new_features, short_simarchive_file, long_simarchive_file = sbd.run_TNO_integration_for_ML(tno=str(name),clones=clones)

        sim_init = rebound.Simulation('../data/YHuang_freei/'+str(name)+'/archive_init.bin')
    
        new_features, short_simarchive_file, long_simarchive_file = sbd.read_TNO_integration_for_ML(sim_init=sim_init,clones=clones,tno=str(name))
    
        predicted_classes = clf.predict(new_features)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        predicted_classes = [10,10,10,10,10]
    predicted_classes = list(predicted_classes)
    return predicted_classes

if __name__ == "__main__":
    
    from schwimmbad import MPIPool
    #print('schwimmbad in')
    with MPIPool() as pool:
        #print('mpipool pooled')    
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
            
        training_file = '../src/TNO-ML-training-data.csv'
        (clf, score, classes_train, classes_test, features_train, features_test, feature_names, classes_dictionary) = sbd.train_and_test_TNO_classifier(training_file)
        data = pd.read_csv('../data/data_files/YHuang_freei.csv')

        #for i in range(len(data)):
        des = np.array(data['Name'])

        run2 = functools.partial(calc_ML,clf=clf)
        classes = pool.map(run2, des)
        print(classes)
        c_df = pd.DataFrame(classes)
        c_df['median'] = np.median(classes,axis=1)
        
        c_df['Objname'] = des
        c_df.to_csv('../data/YHuang_freei_classes.csv')
