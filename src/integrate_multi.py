import rebound
import sys
#change the next line to reflect where you have downloaded the source code
#sys.path.insert(0, '/Users/kvolk/Documents/GitHub/SBDynT/src')
sys.path.insert(0, '../src')
import sbdynt as sbd

training_file = '../src/TNO-ML-training-data.csv'

(clf, score, classes_train, classes_test, 
 features_train, features_test, 
 feature_names, classes_dictionary) = sbd.train_and_test_TNO_classifier(training_file)

import pandas as pd
data = pd.read_csv('../data/data_files/DES_objects.csv')
classes = []
#for i in range(len(data)):
for i in range(652,700):
    if i%10 == 0:
        print(i)
    tno = data['Name'][i]
    clones = 2
    try:
        new_features, short_simarchive_file, long_simarchive_file = sbd.run_TNO_integration_for_ML(tno=tno,clones=clones)
    except:
        continue