import sys
import numpy as np
import pandas as pd
import tools
import run_reb
import MLdata
import tno
import resonances

from os import path
from os import remove

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier

from skimage.feature import hog
from skimage.io import imread

from datetime import date
from pickle import dump
from pickle import load
from importlib import resources as impresources




default_TNO_training_data = '09-06-2024-ML-features.csv'
#default_TNO_train_test_data = 'TNO_training_testing_set.pkl'








'''
Everything below this point is for development and testing only
'''


##########################################################################################################
# Helper functions for reading in the provided csv of pre-calculated and labeled features and then using
# those to train and test the classifier.
# Or for reading in pkl files with the pre-split training and testing data
##########################################################################################################

#################################################################
def read_TNO_training_data(training_file,default=False):
    '''
    Read in the csv file with all the TNO data features and labels
    removes any TNOs with a>1000 au or those that have drastic changes
    in semimajor axis (da>30 au in 0.5 Myr or da>100 au in 10 Myr)
    inputs:
        training_file, string: name of the csv file
        default, boolean: True if the default file is being read
    outputs:
        flag, integer: 0 for failure, 1 for success
        filtered_TNOs, pandas dataframe: the filtered TNO training/testing data
    '''
    
    flag = 0

    if(not default):
        print("machine_learning.read_TNO_training_data is only designed to read the default file")
        print("if you are providing a different file, it needs identical columns")
        print("or you need to write your own function")
        return flag, None

    try:
        all_TNOs = pd.read_csv(training_file, skipinitialspace=True, index_col=False, low_memory=False)
    except:
        print("machine_learning.read_TNO_training_data failed")
        print("could not read the training file: ")
        print(training_file)
        return flag, None
    
    #remove extremely large-a objects and those that scatter a lot in a
    try:
        filtered1_TNOs = all_TNOs[all_TNOs['a_mean']<1000.0].copy()
        filtered2_TNOs = filtered1_TNOs[filtered1_TNOs['a_delta_short']<30.0].copy()
        filtered_TNOs = filtered2_TNOs[filtered2_TNOs['a_delta']<100.0].copy()
    except:
        print("machine_learning.read_TNO_training_data failed")
        print("could not filter the TNOs based on expected column labels")
        return flag, None

    try:
        filtered_TNOs['simplified_G08'] = filtered_TNOs.apply(label_particle_simplifiedG08, axis=1)
    except:
        print("machine_learning.read_TNO_training_data failed")
        print("could not add the simplified G08 labels to the TNOs")
        return flag, filtered_TNOs

    flag = 1
    return flag, filtered_TNOs

##############################################################
def label_particle_simplifiedG08(row):
    newlabel = 'none'
    if (row['G08_class'] == 'detached' or row['G08_class'] == 'classical'):
        newlabel = 'class-det'
    else:
        newlabel= row['G08_class']
    return newlabel


def train_and_test_TNO_classifier(training_file=None):
    '''
    read in the provided csv file of labeled features
    and use it as both a training and testing set
    '''
    if (training_file == None):
        training_file = impresources.files(MLdata) / default_TNO_training_data


    dataset = read_TNO_training_data(training_file)

    #before training the classifier, we have to drop the labels from the dataset
    drop_columns = ['real_or_articial_TNO', 'designation', 'particle_id', 'simplified_G08',
                          'G08_class', 'res_character', 'res_p', 'res_q', 'res_m', 'res_n']


    feature_names = []#dataset.columns.to_list()
    #features = []
    for i in range(0,len(dataset.columns)):
        if(dataset.columns[i] not in (drop_columns)):
            feature_names.append(dataset.columns[i])
    
    clasfeat = 'simplified_G08'
    all_types = list( set(dataset[clasfeat]) )
    types_dict = { all_types[i] : i for i in range( len(all_types) ) }
    classes_dict = { i : all_types[i] for i in range( len(all_types) ) }
    classes = dataset[clasfeat].map(types_dict)

    rs=283
    features_train, features_test, classes_train, classes_test = train_test_split(
                        dataset, classes, test_size=0.333, random_state=rs)

    ids_train = features_train['particle_id'].to_numpy()
    ids_test = features_test['particle_id'].to_numpy()

    features_train.drop(drop_columns, axis=1, inplace=True)
    features_train = features_train.to_numpy()

    features_test.drop(drop_columns, axis=1, inplace=True)
    features_test = features_test.to_numpy()

    rs = 42
    clf = GradientBoostingClassifier(max_leaf_nodes = None, min_impurity_decrease=0.0, min_weight_fraction_leaf = 0.0, 
                                     min_samples_leaf = 1, min_samples_split=3, 
                                     criterion = 'friedman_mse',subsample = 0.9, learning_rate=0.15,
                                     max_depth=8, max_features='log2', 
                                     n_estimators=300, random_state=rs)
    clf.fit(features_train, classes_train)

    classes_predict = clf.predict(features_test)
    score = accuracy_score(classes_test, classes_predict)
    
    #added ids_train and ids_test for now (2-25-25)
    return clf, score, feature_names, classes_dict, ids_train, ids_test, features_test, classes_test


def train_and_test_TNO_classifiers_from_csv(training_file=None,savefile=None,default=False):
    '''
    read in the provided csv file of labeled features
    and use it as both a training and testing set
    Then save that training and testing set to a pickle file
    '''

    flag = 0
    #if (savefile == None):
    #    if(training_file != None):
    #        print("You must specify a savefile if you provide a training_file")
    #        return flag, None, None
    #    savefile = impresources.files(MLdata) / default_TNO_train_test_data

    if (training_file == None):
        default = True
        training_file = impresources.files(MLdata) / default_TNO_training_data

    if(not default):
        print("machine_learning.train_and_test_TNO_classifiers_from_csv is only designed to read the default file")
        print("if you are providing a different file, it needs identical columns")
        print("in which case you must pass default=True to this function")
        return flag, None, None

    rflag, dataset = read_TNO_training_data(training_file,default=default)
    if(rflag<1):
        print("machine_learning.train_and_test_TNO_classifiers_from_csv failed")
        print("could not read the training file")
        return flag, None, None

    #before training the classifier, we have to drop the labels from the dataset
    drop_columns = ['real_or_articial_TNO', 'designation', 'particle_id', 'simplified_G08',
                    'G08_class', 'res_character', 'res_p', 'res_q', 'res_m', 'res_n']

    feature_names = []
    for i in range(0,len(dataset.columns)):
        if(dataset.columns[i] not in (drop_columns)):
            feature_names.append(dataset.columns[i])
    
    clasfeat = 'simplified_G08'
    all_types = list( set(dataset[clasfeat]) )
    if(len(all_types)!=3):
        print("The training file has the wrong number of classes. There should be 3")
        print("The classes found from the file are:")
        print(all_types)
        print("The file is:")
        print(training_file)
        return flag, None, None
    types_dict = { all_types[i] : i for i in range( len(all_types) ) }
    classes_dict = { i : all_types[i] for i in range( len(all_types) ) }
    classes = dataset[clasfeat].map(types_dict)

    rs=283
    try:
        features_train, features_test, classes_train, classes_test = train_test_split(
                        dataset, classes, test_size=0.333, random_state=rs)
    except:
        print("machine_learning.train_and_test_TNO_classifiers_from_csv failed")
        print("could not split the training and testing data")
        return flag, None, None

    try:
        ids_train = features_train['particle_id'].to_numpy()
        ids_test = features_test['particle_id'].to_numpy()
    except:
        print("machine_learning.train_and_test_TNO_classifiers_from_csv failed")
        print("could not get the particle_ids from the training and testing data")
        return flag, None, None

    try:
        features_train.drop(drop_columns, axis=1, inplace=True)
        features_train = features_train.to_numpy()

        features_test.drop(drop_columns, axis=1, inplace=True)
        features_test = features_test.to_numpy()
    except:
        print("machine_learning.train_and_test_TNO_classifiers_from_csv failed")
        print("could not drop the expected columns from the training and testing data")
        return flag, None, None

    rs = 42
    clf = GradientBoostingClassifier(max_leaf_nodes = None, 
                                    min_impurity_decrease=0.0, 
                                    min_weight_fraction_leaf = 0.0, 
                                    min_samples_leaf = 1, 
                                    min_samples_split=3, 
                                    criterion = 'friedman_mse',
                                    subsample = 0.9, 
                                    learning_rate=0.15,
                                    max_depth=8, 
                                    max_features='log2', 
                                    n_estimators=300, 
                                    random_state=rs)
    try:
        clf.fit(features_train, classes_train)
    except:
        print("machine_learning.train_and_test_TNO_classifiers_from_csv failed")
        print("could not train the classifier")
        return flag, None, None

    try:
        classes_predict = clf.predict(features_test)
        score = accuracy_score(classes_test, classes_predict)
    except:
        print("machine_learning.train_and_test_TNO_classifiers_from_csv failed")
        print("could not test the classifier and retrieve a score")
        return flag, None, None

    if(score < 0.95 and default):
        print 
        print("The default classifier is less accurate than expected, something isn't right")
        flag = 2

    #try:
    #    with open(savefile, "wb") as f:
    #        dump([classes_dict,classes_train,features_train,classes_test,features_test], f, protocol=5)
    #except:
    #    print("could not save the training and testing data to:")
    #    print(savefile)
    #    flag = 2

    if(flag == 0):
        flag = 1

    return flag, clf, score, ids_train, ids_test



def train_TNO_classifier(training_file=None):
    '''
    read in the provided csv file of labeled features
    and use the entire file as the training set
    '''

    if (training_file == None):
        training_file = impresources.files(MLdata) / default_TNO_training_data


    dataset = read_TNO_training_data(training_file)

    #before training the classifier, we have to drop the labels from the dataset
    drop_columns = ['real_or_articial_TNO', 'designation', 'particle_id', 'simplified_G08',
                          'G08_class', 'res_character', 'res_p', 'res_q', 'res_m', 'res_n']


    feature_names = []#dataset.columns.to_list()
    #features = []
    for i in range(0,len(dataset.columns)):
        if(dataset.columns[i] not in (drop_columns)):
            feature_names.append(dataset.columns[i])
    
    clasfeat = 'simplified_G08'
    all_types = list( set(dataset[clasfeat]) )
    types_dict = { all_types[i] : i for i in range( len(all_types) ) }
    classes_dict = { i : all_types[i] for i in range( len(all_types) ) }
    
    classes_train = dataset[clasfeat].map(types_dict)
    features_train = dataset.copy()
    ids_train = dataset['particle_id'].to_numpy()
    features_train.drop(drop_columns, axis=1, inplace=True)
    features_train = features_train.to_numpy()

    rs = 42
    clf = GradientBoostingClassifier(max_leaf_nodes = None, min_impurity_decrease=0.0, min_weight_fraction_leaf = 0.0, 
                                     min_samples_leaf = 1, min_samples_split=3, 
                                     criterion = 'friedman_mse',subsample = 0.9, learning_rate=0.15,
                                     max_depth=8, max_features='log2', 
                                     n_estimators=300, random_state=rs)
    clf.fit(features_train, classes_train)

    return clf, classes_dict

def initialize_TNO_classifier(classifier_file=None,dict_file=None,training_file=None):
    '''
    read in a pre-trained classifier saved in a pickle file
    if this is the first time the user has used the TNO classifier, it
    will first have to train the classifier. The first ever call can be
    a bit slow
    inputs:
        classifier_file, string (optional): path to the saved classifier
                                 defaults are defined in this package
        dict_file, string (optional): path to the saved dictionary for the classifier
                                 defaults are defined in this package
    outputs:
        flag, integer: 0 if failed, 1 if sucessful
        classifier: scikitlearn GradientBoostingClassifier that provides
                    simplified Gladman et al. 2008 TNO classifications
        classifier_dictionary: dictionary that relates the integer classes
                    provided by classifier to the string classifications
    '''

    flag = 0
    default = 0
    if(classifier_file == None):
        classifier_file =  impresources.files(MLdata) / default_trained_classifier    
        default = 1
    if(dict_file == None):
        dict_file =  impresources.files(MLdata) / default_trained_classifier_dictionary    

    if (training_file == None):
        training_file = impresources.files(MLdata) / default_TNO_training_data

    if(not path.exists(classifier_file) or not path.exists(dict_file)):
        print("The saved classifier file and/or dictionary file do not exist.")
        print("We will train a classifier using the training file:")
        print(training_file)
        print("and save it and the dictionary to:")
        print(classifier_file)
        print(dict_file)
        print("This will take a moment, but future calls will be much much faster")
        
        # first do a training and testing run to check the accuracy and make sure the
        # training file features match those expected
        clf, score, feature_names, classes_dict = train_and_test_TNO_classifier(training_file=training_file)
        f = TNO_ML_features()
        expected_features = f.feature_names
        if not (expected_features == feature_names):
            print("the features in the specified training set do not match those expected!")
            print("failed at machine_learning.initialize_TNO_classifier()")
            print(expected_features)
            print(feature_names)
            return flag, None, None
        print("the trained classifier has a strict accuracy of %f percent\n" % (100*score))
        if(score < 0.95 and default == 1):
            print("The default classifier is less accurate than expected, something isn't right")
            print("This classifier will not be saved.")
            return flag, clf, classes_dict

        #everything looks ok, so we will re-train on all the data and then save things
        classifier, classes_dictionary = train_TNO_classifier(training_file=training_file)
        
        sflag = save_TNO_classifier(classifier = classifier, dictionary=classes_dictionary, 
                                    classifier_file=classifier_file, dict_file = dict_file)
        if(sflag == 0):
            print("Failed to save classifier")
        flag = 1
        
        return flag, classifier, classes_dictionary
    else:
        try:
            with open(classifier_file, "rb") as f:
                classifier = load(f)
        except: 
            print("Couldn't read in saved classifier file %s" % classifier_file)
            print("try deleting the file and trying again")
            print("failed at machine_learning.initialize_TNO_classifier()")
            return flag, None,None

        try:
            with open(dict_file, "rb") as f:
                dictionary = load(f)
        except: 
            print("Couldn't read in saved classifier dictionary file %s" % dict_file)
            print("try deleting the file and trying again")
            print("failed at machine_learning.initialize_TNO_classifier()")
            return flag, classifier,None

        flag = 1
    return flag, classifier, dictionary



def save_TNO_classifier(classifier=None,dictionary=None,classifier_file=None,dict_file=None):
    '''
    save a trained classifier to either specified or default file paths
    inputs:
        classifier, scikitlearn trained classifier to save
        dictionary: dictionary that relates the integer classes
                    provided by classifier to the string classifications
        classifier_file, string (optional): path to the saved classifier
                                 defaults are defined in this package
        dict_file, string (optional): path to the saved dictionary for the classifier
                                 defaults are defined in this package
    outputs:
        flag, integer: 0 if failed, 1 if successful

    '''
    
    flag = 0
    if(classifier_file == None):
        classifier_file =  impresources.files(MLdata) / default_trained_classifier    
    if(dict_file == None):
        dict_file =  impresources.files(MLdata) / default_trained_classifier_dictionary    

    if(classifier==None):
        print("must specify a classifier to save\n")
        return flag
    if(dictionary==None):
        print("must specify a classifier dictionary to save\n")
        return flag


    try:
        with open(classifier_file, "wb") as f:
            dump(classifier, f, protocol=5)
    except:
        print("failed in machine_learning.save_TNO_classifier()\n")
        print("cannot write the classifier to specified filepath\n")
        return flag
    
    try:
        with open(dict_file, "wb") as f:
            dump(dictionary, f)
    except:
        print("failed in machine_learning.save_TNO_classifier()\n")
        print("cannot write the dictionary to specified filepath\n")
        return flag

    flag = 1
    return flag



##########################################################################################################
# Helper functions for reading in the training set integrations, which are not provided in the SBDynT  
# GitHub repository because the dataset is ~200 GB. So these functions will not be useful to 99.9% of
# users and are not used in any of the examples/documentation. However, they are included for development
# purposes and for anyone who wishes to try to improve the machine learning classification process.
# If you are interested in the labeled dataset, please email Kat Volk (kvolk@psi.edu or kat.volk@gmail.com)
# and we can figure out how best to get the 200 GB of data copied to you.
##########################################################################################################

def read_trainingset_datafiles(fname):
    '''
    Load data from the machine learning trainingset integration outputs
    reads the short or long trainingset files
    '''

    #################################
    # define the id for Neptune and the test particle
    ######################################
    pl_id = -5
    tno_id = 1

    ########################################################
    # read in the data from the follow file
    ########################################################

    data = pd.read_csv(fname, skipinitialspace=True, comment='#')
    # file column names=['particle-n', 'particle-id','time', 'a', 'e', 'inc', 'Node', 'argperi','MA','phi'])

    data_pl = data[data['particle-n'] == pl_id]
    data_sb = data[data['particle-n'] == tno_id]

    lines = len(data_sb)
    a_sb = [data_sb['a'].to_numpy()]
    a_pl = data_pl['a'].to_numpy()
    time = data_sb['time'].to_numpy()
    e_sb = data_sb['e'].to_numpy()
    i_sb = data_sb['inc'].to_numpy()
    i_pl = data_pl['inc'].to_numpy()
    node_sb = data_sb['Node'].to_numpy()
    node_pl = data_pl['Node'].to_numpy()
    peri_sb = data_sb['argperi'].to_numpy()
    peri_pl = data_pl['argperi'].to_numpy()
    MA_sb = data_sb['MA'].to_numpy()
    MA_pl = data_pl['MA'].to_numpy()

    rrf = np.zeros(lines)
    phirf = np.zeros(lines)

    q_sb = a_sb*(1.-e_sb)
    tiss_sb = a_pl/a_sb + 2.*np.cos(i_sb)*np.sqrt((a_sb/a_pl)*(1.-e_sb*e_sb))
    pomega_sb = node_sb + peri_sb

    for j in range(0, lines):
        [flag, x, y, z, vx, vy, vz] = tools.aei_to_xv(
            GM=1., a=a_sb[j],e=e_sb[j],inc=i_sb[j],node=node_sb[j],
            argperi=peri_sb[j],ma=MA_sb[j])

        [xrf, yrf, zrf, vxrf, vyrf, vzrf] = tools.rotating_frame_cartesian(x=x, y=y, z=z,
            node=node_pl[j], inc=i_pl[j], argperi=peri_pl[j],ma=MA_pl[j])

        rrf[j] = np.sqrt(xrf*xrf + yrf*yrf + zrf*zrf)
        phirf[j] = np.arctan2(yrf, xrf)


    peri_sb = tools.arraymod2pi(peri_sb)
    node_sb = tools.arraymod2pi(node_sb)
    pomega_sb = tools.arraymod2pi(pomega_sb)
    phirf = tools.arraymod2pi(phirf)


    return time,a_sb,e_sb,i_sb,node_sb,peri_sb,pomega_sb,q_sb,rrf,phirf,tiss_sb



def calc_ML_features_from_trainingset_datafiles(short_fname, long_fname):
    '''
    return calculated data features from the trainingset integration outputs
    requires file names for both a short and long integration
    '''

    t,a,e,inc,node,peri,pomega,q,rrf,phirf,tn = read_trainingset_datafiles(long_fname)
    t_short,a_short,e_short,inc_short,node_short,peri_short,pomega_short,q_short,rrf_short, \
            phirf_short,tn_short = read_trainingset_datafiles(short_fname)

    features = calc_ML_features(t,a,e,inc,node,peri,pomega,q,rrf,phirf,tn, \
        t_short,a_short,e_short,inc_short,node_short,peri_short,\
        pomega_short,q_short,rrf_short,phirf_short,tn_short)

    return features




#################################################################
def print_TNO_ML_results(pred_class,classes_dictionary,class_probs,clones=2):
    nclas = len(classes_dictionary)
    print("Clone number, most probable class, probability of most probable class, ",end ="")
    for n in range(nclas):
        print("probability of %s," % classes_dictionary[n],end ="")
    print("\n",end ="")
    format_string = "%d, %s, "
    for n in range(nclas-1):
        format_string+="%e, "
    format_string+="%e,\n"
    for n in range(0,clones+1):
        print("%d, %s, %e, " % (n,classes_dictionary[pred_class[n]], class_probs[n][pred_class[n]]),end ="")
        for j in range(nclas):
            print("%e, " % class_probs[n][j] ,end ="")
        print("\n",end ="")

def print_TNO_ML_results_to_file(des,pred_class,classes_dictionary,class_probs,clones=2):
    outfile = des + "-classes.txt"
    out = open(outfile,"w")
    nclas = len(classes_dictionary)
    line = "Designation, clone number, most probable class, probability of most probable class, "
    for n in range(nclas):
        line+= ("probability of %s, " % classes_dictionary[n])
    line+="\n"
    out.write(line)
    format_string = "%d, %s, "
    for n in range(nclas-1):
        format_string+="%e, "
    format_string+="%e,\n"
    for n in range(0,clones+1):
        line = des + ', '
        line+=("%d, %s, %e, " % (n,classes_dictionary[pred_class[n]], class_probs[n][pred_class[n]]))
        for j in range(nclas):
            line+=("%e, " % class_probs[n][j])
        line+="\n"
        out.write(line)

    out.close()
#################################################################









