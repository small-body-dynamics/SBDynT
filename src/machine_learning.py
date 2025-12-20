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

import random
import string


'''
This file contains the details of all the machine learning classifiers and
feature calculations for the TNO classifiers.

The calls to the classifier are in tno_classifier.py to make it a bit easier
to see how to interact with the classifier and to find some of the logic flow 
for overriding the pure machine learning results. 
'''


# define the default file scheme for the machine learning datasets

# these files are provided with the package
default_TNO_training_data = 'G08_TNO_training_testing_set.pkl'
default_phi_training_data = 'phi_TNO_training_testing_set.pkl'


# these files will be generated the first time the ML is called
# so that the trained classifiers can be saved and used later
default_trained_classifier = 'trained-G08-TNO-classifier.pkl'
default_trained_phi_classifier = 'trained-phi-TNO-classifier.pkl'

class TNO_ML_classifier:
    '''
    class that stores the trained TNO machine learning classifier
    and the training data used to train it
    '''
    def __init__(self,G08_training_testing_data=None,G08_classifier_file=None,
                phi_training_testing_data=None,phi_classifier_file=None):
        
        #initialize the file names for the training/testing data and 
        #stored trained classifiers
        self.G08_training_testing_data = G08_training_testing_data
        self.G08_classifier_file = G08_classifier_file
        self.phi_training_testing_data = phi_training_testing_data
        self.phi_classifier_file = phi_classifier_file
        
        #initialize the main Gladman et al 2008 classifier
        self.G08_classifier = None
        self.classes_dictionary = None
        #initialize the main Gladman et al 2008 classifier
        self.phi_classifier = None

    def initialize_classifiers(self):
        #start with the Gladman et al 2008 classifier
        default = False
        if(self.G08_classifier_file is None):
            self.G08_classifier_file =  impresources.files(MLdata) / default_trained_classifier
        if(path.exists(self.G08_classifier_file)):
            try:
                with open(self.G08_classifier_file, 'rb') as f:
                    [self.classes_dictionary,self.G08_classifier] = load(f)
            except:
                print("Error loading the Gladman et al 2008 classifier from:")
                print(self.G08_classifier_file)
                print("Will try to load the training/testing data instead.")
        if(self.G08_classifier is None):
            if(self.G08_training_testing_data is None):
                self.G08_training_testing_data =  impresources.files(MLdata) / default_TNO_training_data
                default = True
            if(path.exists(self.G08_training_testing_data)):
                try:
                    with open(self.G08_training_testing_data, 'rb') as f:
                        [classes_dict,classes_train,features_train,classes_test,features_test] = load(f)
                except:
                    print("Error loading the training/testind data for the G08 classifier from:")
                    print(self.G08_training_testing_data)
                    print("Please check the file and try again.")
                #train the classifier
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
                    print("TNO_ML_classifier.initialize_classifiers() failed when trying to fit the G08 classifier")
                    return 0
                try:
                    classes_predict = clf.predict(features_test)
                    score = accuracy_score(classes_test, classes_predict)
                except:
                    print("TNO_ML_classifier.initialize_classifiers() failed")
                    print("could not test the classifier and retrieve a score")
                    return 0
                self.G08_classifier = clf
                self.classes_dictionary = classes_dict
                if(score < 0.95 and default):
                    print 
                    print("The default G08 classifier is less accurate than expected, something isn't right")
                    print("the score is: ", score)
                    print("This classifier will not be saved to be read in later")
                else:
                    #save the classifier
                    try:
                        with open(self.G08_classifier_file, 'wb') as f:
                            dump([self.classes_dictionary,self.G08_classifier],f,protocol=5)
                    except:
                        print("TNO_ML_classifier.initialize_classifiers() failed when trying to save the G08 classifier")
            else:
                print("The specified G08 classifier files were not found")
                return 0

        #initialize the phi classifier
        if(self.phi_classifier_file is None):
            self.phi_classifier_file =  impresources.files(MLdata) / default_trained_phi_classifier
        if(path.exists(self.phi_classifier_file)):
            try:
                with open(self.phi_classifier_file, 'rb') as f:
                    [self.phi_classifier] = load(f)
            except:
                print("Error loading the phi classifier from:")
                print(self.phi_classifier_file)
                print("Will try to load the training/testing data instead.")
        if(self.phi_classifier is None):
            if(self.phi_training_testing_data is None):
                self.phi_training_testing_data =  impresources.files(MLdata) / default_phi_training_data
                default = True
            if(path.exists(self.phi_training_testing_data)):
                try:
                    with open(self.phi_training_testing_data, 'rb') as f:
                        [label_train,hog_train,label_test,hog_test] = load(f)
                except:
                    print("Error loading the training/testind data for the phi classifier from:")
                    print(self.phi_training_testing_data)
                    print("Please check the file and try again.")
                    return 0
                #train the classifier
                img_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3, loss='log_loss')

                try:
                    img_clf.fit(hog_train,label_train)
                except:
                    print("TNO_ML_classifier.initialize_classifiers() failed when trying to fit the phi classifier")
                    return 0
                try:
                    label_predict = img_clf.predict(hog_test)
                    score = 100*np.sum(label_predict == label_test)/len(label_test)
                except:
                    print("TNO_ML_classifier.initialize_classifiers() failed")
                    print("could not test the phi classifier and retrieve a score")
                    return 0
                self.phi_classifier = img_clf
                if(score < 0.98 and default):
                    print 
                    print("The default phi classifier is less accurate than expected, something isn't right")
                    print("the score is: ", score)
                    print("This classifier will not be saved to be read in later")
                else:
                    #save the classifier
                    try:
                        with open(self.phi_classifier_file, 'wb') as f:
                            dump([self.phi_classifier],f,protocol=5)
                    except:
                        print("TNO_ML_classifier.initialize_classifiers() failed when trying to save the phi classifier")
            else:
                print("The specified phi classifier files were not found")
                return 0 
        return 1           


class TNO_ML_features:
    '''
    class that stores the pre-determined set of data features that 
    the TNO machine learning classifier uses
    non-obvious terms are defined the first time they appear
    '''
    def __init__(self,clones=0):
        self.clones = clones
        # initialize all of the data features

        # first the set based on the long (10Myr integration)
        # a = semimajor axis (au)
        self.a_mean = np.zeros(self.clones+1)
        self.a_stddev = np.zeros(self.clones+1)
        self.a_stddev_normed = np.zeros(self.clones+1) # == a_stddev/a_mean
        self.a_delta = np.zeros(self.clones+1) #maximum(a) - minimum(a)
        self.a_delta_normed = np.zeros(self.clones+1) # == a_delta/a_mean
        
        # adot is the change in a from one time output
        # to the next divided by the delta-t between outputs
        # (au/year)
        self.adot_min = np.zeros(self.clones+1)
        self.adot_mean = np.zeros(self.clones+1)
        self.adot_max = np.zeros(self.clones+1)
        self.adot_stddev = np.zeros(self.clones+1)
        self.adot_delta = np.zeros(self.clones+1)
        
        # e = eccentricity
        self.e_min = np.zeros(self.clones+1)
        self.e_mean = np.zeros(self.clones+1)
        self.e_max = np.zeros(self.clones+1)
        self.e_stddev = np.zeros(self.clones+1)
        self.e_delta = np.zeros(self.clones+1)
 
        # (per year)
        self.edot_min = np.zeros(self.clones+1)
        self.edot_mean = np.zeros(self.clones+1)
        self.edot_max = np.zeros(self.clones+1)
        self.edot_stddev = np.zeros(self.clones+1)
        self.edot_delta = np.zeros(self.clones+1)

        # i = inclination (radians)
        self.i_min = np.zeros(self.clones+1)
        self.i_mean = np.zeros(self.clones+1)
        self.i_max = np.zeros(self.clones+1)
        self.i_stddev = np.zeros(self.clones+1)
        self.i_delta = np.zeros(self.clones+1)

        # (radians/year)
        self.idot_min = np.zeros(self.clones+1)
        self.idot_mean = np.zeros(self.clones+1)
        self.idot_max = np.zeros(self.clones+1)
        self.idot_stddev = np.zeros(self.clones+1)
        self.idot_delta = np.zeros(self.clones+1)

        # Om = longitude of ascending node 
        # (radians/year)
        self.Omdot_min = np.zeros(self.clones+1)
        self.Omdot_mean = np.zeros(self.clones+1)
        self.Omdot_max = np.zeros(self.clones+1)
        self.Omdot_stddev = np.zeros(self.clones+1)
        self.Omdot_stddev_normed = np.zeros(self.clones+1)
        self.Omdot_delta = np.zeros(self.clones+1)
        self.Omdot_delta_normed = np.zeros(self.clones+1)

        # o = argument of perihelion (radians)
        self.o_min = np.zeros(self.clones+1)
        self.o_mean = np.zeros(self.clones+1)
        self.o_max = np.zeros(self.clones+1)
        self.o_stddev = np.zeros(self.clones+1)
        self.o_delta = np.zeros(self.clones+1)

        # (radians/year)
        self.odot_min = np.zeros(self.clones+1)
        self.odot_mean = np.zeros(self.clones+1)
        self.odot_max = np.zeros(self.clones+1)
        self.odot_stddev = np.zeros(self.clones+1)
        self.odot_stddev_normed = np.zeros(self.clones+1)
        self.odot_delta = np.zeros(self.clones+1)
        self.odot_delta_normed = np.zeros(self.clones+1)

        # po = longitude of perhelion
        # (radians/year)
        self.podot_min = np.zeros(self.clones+1)
        self.podot_mean = np.zeros(self.clones+1)
        self.podot_max = np.zeros(self.clones+1)
        self.podot_stddev = np.zeros(self.clones+1)
        self.podot_stddev_normed = np.zeros(self.clones+1)
        self.podot_delta = np.zeros(self.clones+1)
        self.podot_delta_normed = np.zeros(self.clones+1)

        # q = perhelion distance (au)
        self.q_min = np.zeros(self.clones+1)
        self.q_mean = np.zeros(self.clones+1)
        self.q_max = np.zeros(self.clones+1)
        self.q_stddev = np.zeros(self.clones+1)
        self.q_stddev_normed = np.zeros(self.clones+1)
        self.q_delta = np.zeros(self.clones+1)
        self.q_delta_normed = np.zeros(self.clones+1)

        # (au/year)
        self.qdot_min = np.zeros(self.clones+1)
        self.qdot_mean = np.zeros(self.clones+1)
        self.qdot_max = np.zeros(self.clones+1)
        self.qdot_stddev = np.zeros(self.clones+1)
        self.qdot_delta = np.zeros(self.clones+1)
        
        # tn = tisserand parameter with respect 
        # to Neptune
        self.tn_min = np.zeros(self.clones+1)
        self.tn_mean = np.zeros(self.clones+1)
        self.tn_max = np.zeros(self.clones+1)
        self.tn_stddev = np.zeros(self.clones+1)
        self.tn_delta = np.zeros(self.clones+1)

        '''
        summary statistics of the distribution
        of points in a grid of heliocentric distance
        vs angle from Neptune in the rotating frame
        with Neptune
        '''
        # number of grid spaces at the smallest heliocentric
        # distances that have no visits
        self.empty_peri_sec = np.zeros(self.clones+1)
        # number of grid spaces at the smallest heliocentric
        # distances surrounding Neptune that have no visits
        self.adj_empty_peri_sec = np.zeros(self.clones+1)
        # for the smallest heliocentric distance grid spaces, 
        # the standard deviation in visits across those bins
        # (normalized by the average number of visits across
        # all grid spaces)
        self.stddev_n_peri_sec = np.zeros(self.clones+1)
        # for the smallest heliocentric distance grid spaces, 
        # the difference in visits between the most and least
        # visited grid spaces
        # (normalized by the average number of visits across
        # all grid spaces)
        self.delta_n_peri_sec = np.zeros(self.clones+1)
        # for the smallest heliocentric distance grid spaces, 
        # the largest raleigh-Z parameter calculated for 
        # potential resonant configurations up to order 10
        self.rz_peri_max = np.zeros(self.clones+1)

        # same as above, but for the largest heliocentric
        # distance grid spaces
        self.empty_apo_sec = np.zeros(self.clones+1)
        self.stddev_apo_sec = np.zeros(self.clones+1)
        self.delta_n_apo_sec = np.zeros(self.clones+1)
        self.rz_apo_max = np.zeros(self.clones+1)

        # for the whole grid, the average number of visits
        # to all non-empty grid spaces 
        # (normalized by the average number of visits across
        # all grid spaces)
        self.nonzero_grid_avg = np.zeros(self.clones+1)
        # for the whole grid, the standard deviation in number
        # of visits to all non-empty grid spaces 
        # (normalized by the average number of visits across
        # all grid spaces)        
        self.nonzero_grid_stddev = np.zeros(self.clones+1)
        # for the whole grid, the standard deviation in number
        # of visits per grid space 
        # (normalized by the average number of visits across
        # all grid spaces)            
        self.grid_stddev = np.zeros(self.clones+1)
        # (nonzero_grid_stddev - grid_stddev)
        self.grid_delta_stddev = np.zeros(self.clones+1)
        # total number of empty grid spaces
        self.grid_n_empty = np.zeros(self.clones+1)


        # correlation coefficients between
        # a and e
        self.ae_correlation = np.zeros(self.clones+1)
        # a and i
        self.ai_correlation = np.zeros(self.clones+1)
        # e and i
        self.ei_correlation = np.zeros(self.clones+1)

        '''
        features based on FFTs
        '''
        # the spectral fraction of the semimajor axis
        self.a_spectral_fraction = np.zeros(self.clones+1)
        # normalized power in the dominant frequency
        self.a_maxpower = np.zeros(self.clones+1)
        # normalized power in the 3 most dominant frequencies
        self.a_maxpower3 = np.zeros(self.clones+1)
        # the three most dominant frequencies (yr^-1)
        self.a_frequency1 = np.zeros(self.clones+1)
        self.a_frequency2 = np.zeros(self.clones+1)
        self.a_frequency3 = np.zeros(self.clones+1)

        self.e_spectral_fraction = np.zeros(self.clones+1)
        self.e_maxpower = np.zeros(self.clones+1)
        self.e_maxpower3 = np.zeros(self.clones+1)
        self.e_frequency1 = np.zeros(self.clones+1)
        self.e_frequency2 = np.zeros(self.clones+1)
        self.e_frequency3 = np.zeros(self.clones+1)

        self.i_spectral_fraction = np.zeros(self.clones+1)
        self.i_maxpower = np.zeros(self.clones+1)
        self.i_maxpower3 = np.zeros(self.clones+1)
        self.i_frequency1 = np.zeros(self.clones+1)
        self.i_frequency2 = np.zeros(self.clones+1)
        self.i_frequency3 = np.zeros(self.clones+1)

        self.amd_spectral_fraction = np.zeros(self.clones+1)
        self.amd_maxpower = np.zeros(self.clones+1)
        self.amd_maxpower3 = np.zeros(self.clones+1)
        self.amd_frequency1 = np.zeros(self.clones+1)
        self.amd_frequency2 = np.zeros(self.clones+1)
        self.amd_frequency3 = np.zeros(self.clones+1)

        
        # ratio of the number of outputs near the extreme
        # (minimum and maximum) semimajor axis values to the 
        # number of outputs near the mean semimajor axis value
        self.a_minmax_to_mean_density_ratio = np.zeros(self.clones+1)
        # ratio of the number of outputs near the minumum a to 
        # the number of outputs near the maximum a
        self.a_min_to_max_density_ratio = np.zeros(self.clones+1)
        # maximum change in the above ratios over four,
        # non-overlapping time bins
        self.a_delta_minmax_to_mean_density_ratio = np.zeros(self.clones+1)
        self.a_delta_min_to_max_density_ratio = np.zeros(self.clones+1)

        self.e_minmax_to_mean_density_ratio = np.zeros(self.clones+1)
        self.e_min_to_max_density_ratio = np.zeros(self.clones+1)
        
        self.i_minmax_to_mean_density_ratio = np.zeros(self.clones+1)
        self.i_min_to_max_density_ratio = np.zeros(self.clones+1)


        #then the set based on the short, 0.5 Myr integration
        self.a_mean_short = np.zeros(self.clones+1)
        self.a_stddev_short = np.zeros(self.clones+1)
        self.a_stddev_normed_short = np.zeros(self.clones+1)
        self.a_delta_short = np.zeros(self.clones+1)
        self.a_delta_normed_short = np.zeros(self.clones+1)
        
        self.adot_min_short = np.zeros(self.clones+1)
        self.adot_mean_short = np.zeros(self.clones+1)
        self.adot_max_short = np.zeros(self.clones+1)
        self.adot_stddev_short = np.zeros(self.clones+1)
        self.adot_delta_short = np.zeros(self.clones+1)
        
        self.e_min_short = np.zeros(self.clones+1)
        self.e_mean_short = np.zeros(self.clones+1)
        self.e_max_short = np.zeros(self.clones+1)
        self.e_stddev_short = np.zeros(self.clones+1)
        self.e_delta_short = np.zeros(self.clones+1)
 
        self.edot_min_short = np.zeros(self.clones+1)
        self.edot_mean_short = np.zeros(self.clones+1)
        self.edot_max_short = np.zeros(self.clones+1)
        self.edot_stddev_short = np.zeros(self.clones+1)
        self.edot_delta_short = np.zeros(self.clones+1)

        self.i_min_short = np.zeros(self.clones+1)
        self.i_mean_short = np.zeros(self.clones+1)
        self.i_max_short = np.zeros(self.clones+1)
        self.i_stddev_short = np.zeros(self.clones+1)
        self.i_delta_short = np.zeros(self.clones+1)

        self.idot_min_short = np.zeros(self.clones+1)
        self.idot_mean_short = np.zeros(self.clones+1)
        self.idot_max_short = np.zeros(self.clones+1)
        self.idot_stddev_short = np.zeros(self.clones+1)
        self.idot_delta_short = np.zeros(self.clones+1)

        self.Omdot_min_short = np.zeros(self.clones+1)
        self.Omdot_mean_short = np.zeros(self.clones+1)
        self.Omdot_max_short = np.zeros(self.clones+1)
        self.Omdot_stddev_short = np.zeros(self.clones+1)
        self.Omdot_stddev_normed_short = np.zeros(self.clones+1)
        self.Omdot_delta_short = np.zeros(self.clones+1)
        self.Omdot_delta_normed_short = np.zeros(self.clones+1)

        self.odot_min_short = np.zeros(self.clones+1)
        self.odot_mean_short = np.zeros(self.clones+1)
        self.odot_max_short = np.zeros(self.clones+1)
        self.odot_stddev_short = np.zeros(self.clones+1)
        self.odot_stddev_normed_short = np.zeros(self.clones+1)
        self.odot_delta_short = np.zeros(self.clones+1)
        self.odot_delta_normed_short = np.zeros(self.clones+1)

        self.podot_min_short = np.zeros(self.clones+1)
        self.podot_mean_short = np.zeros(self.clones+1)
        self.podot_max_short = np.zeros(self.clones+1)
        self.podot_stddev_short = np.zeros(self.clones+1)
        self.podot_stddev_normed_short = np.zeros(self.clones+1)
        self.podot_delta_short = np.zeros(self.clones+1)
        self.podot_delta_normed_short = np.zeros(self.clones+1)

        self.q_min_short = np.zeros(self.clones+1)
        self.q_mean_short = np.zeros(self.clones+1)
        self.q_max_short = np.zeros(self.clones+1)
        self.q_stddev_short = np.zeros(self.clones+1)
        self.q_stddev_normed_short = np.zeros(self.clones+1)
        self.q_delta_short = np.zeros(self.clones+1)
        self.q_delta_normed_short = np.zeros(self.clones+1)

        self.qdot_min_short = np.zeros(self.clones+1)
        self.qdot_mean_short = np.zeros(self.clones+1)
        self.qdot_max_short = np.zeros(self.clones+1)
        self.qdot_stddev_short = np.zeros(self.clones+1)
        self.qdot_delta_short = np.zeros(self.clones+1)
        
        self.tn_min_short = np.zeros(self.clones+1)
        self.tn_mean_short = np.zeros(self.clones+1)
        self.tn_max_short = np.zeros(self.clones+1)
        self.tn_stddev_short = np.zeros(self.clones+1)
        self.tn_delta_short = np.zeros(self.clones+1)

        self.empty_peri_sec_short = np.zeros(self.clones+1)
        self.adj_empty_peri_sec_short = np.zeros(self.clones+1)
        self.stddev_n_peri_sec_short = np.zeros(self.clones+1)
        self.delta_n_peri_sec_short = np.zeros(self.clones+1)
        self.rz_peri_max_short = np.zeros(self.clones+1)

        self.empty_apo_sec_short = np.zeros(self.clones+1)
        self.stddev_apo_sec_short = np.zeros(self.clones+1)
        self.delta_n_apo_sec_short = np.zeros(self.clones+1)
        self.rz_apo_max_short = np.zeros(self.clones+1)

        self.nonzero_grid_avg_short = np.zeros(self.clones+1)
        self.nonzero_grid_stddev_short = np.zeros(self.clones+1)
        self.grid_stddev_short = np.zeros(self.clones+1)
        self.grid_delta_stddev_short = np.zeros(self.clones+1)
        self.grid_n_empty_short = np.zeros(self.clones+1)

        self.ae_correlation_short = np.zeros(self.clones+1)
        self.ai_correlation_short = np.zeros(self.clones+1)
        self.ei_correlation_short = np.zeros(self.clones+1)

        self.a_spectral_fraction_short = np.zeros(self.clones+1)
        self.a_maxpower_short = np.zeros(self.clones+1)
        self.a_maxpower3_short = np.zeros(self.clones+1)
        self.a_frequency1_short = np.zeros(self.clones+1)
        self.a_frequency2_short = np.zeros(self.clones+1)
        self.a_frequency3_short = np.zeros(self.clones+1)

        self.amd_spectral_fraction_short = np.zeros(self.clones+1)
        self.amd_maxpower_short = np.zeros(self.clones+1)
        self.amd_maxpower3_short = np.zeros(self.clones+1)
        self.amd_frequency1_short = np.zeros(self.clones+1)
        self.amd_frequency2_short = np.zeros(self.clones+1)
        self.amd_frequency3_short = np.zeros(self.clones+1)

        self.a_minmax_to_mean_density_ratio_short = np.zeros(self.clones+1)
        self.a_min_to_max_density_ratio_short = np.zeros(self.clones+1)
        self.a_delta_minmax_to_mean_density_ratio_short = np.zeros(self.clones+1)
        self.a_delta_min_to_max_density_ratio_short = np.zeros(self.clones+1)

        self.e_minmax_to_mean_density_ratio_short = np.zeros(self.clones+1)
        self.e_min_to_max_density_ratio_short = np.zeros(self.clones+1)
        
        self.i_minmax_to_mean_density_ratio_short = np.zeros(self.clones+1)
        self.i_min_to_max_density_ratio_short = np.zeros(self.clones+1)

        # list of all the features above
        feature_names = [None]*(len(vars(self)))
        i=0
        for d in vars(self):
            feature_names[i] = d
            i+=1
        #remove the clones variable from the list
        feature_names.pop(0)
        self.feature_names = feature_names


    def return_features_list(self):
        #return all the features as a numpy array 
        #in a fixed order that matches the stored training set
        #of labeled features
        features_list = np.zeros([(self.clones+1),len(self.feature_names)])
        i=0
        for t in self.feature_names:
            features_list[:,i] = self.__getattribute__(t)
            i+=1
        return features_list


    def print_features(self,n):
        features_list = np.zeros([(self.clones+1),len(self.feature_names)])
        for t in self.feature_names:
            flist = self.__getattribute__(t)
            print(t, ": ", flist[n])
        return 




def calc_ML_features(time,a,ec,inc,node,argperi,pomega,q,rh,phirf,tn,\
        time_short,a_short,ec_short,inc_short,node_short,argperi_short,\
        pomega_short,q_short,rh_short,phirf_short,tn_short,clones=0,):
    """
    calculate ML  data features from the short and long time-series
    of a TNO integration

    inputs:
        time: 1-d np array, time in years from 0-10 Myr sampled every 1000 years
        a: 2-d np array, semimajor axis in au at times=time 
        ec: 2-d np array, eccentricity at times=time 
        inc: 2-d np array, inclination in radians at times=time 
        node: 2-d np array, longitude of ascending node in radians at times=time 
        argperi: 2-d np array, argument of perihelion in radians at times=time 
        pomega: 2-d np array, longitude of perihelion in radians at times=time 
        q: 2-d np array, perihelion distance in au at times=time 
        rh: 2-d np array, heliocentric distance in au at times=time 
        phirf: 2-d np array, angle from Neptune in the rotating frame in radians at times=time 
        tn: 2-d np array, tisserand parameter with respect to Neptune at times=time 

        time_short: 1-d np array, time in years from 0-0.5 Myr sampled every 50 years
        a_short: 2-d np array, semimajor axis in au at times=time_short
        ec_short: 2-d np array, eccentricity at times=time_short
        inc_short: 2-d np array, inclination in radians at times=time_short 
        node_short: 2-d np array, longitude of ascending node in radians at times=time_short 
        argperi_short: 2-d np array, argument of perihelion in radians at times=time_short 
        pomega_short: 2-d np array, longitude of perihelion in radians at times=time_short 
        q_short: 2-d np array, perihelion distance in au at times=time_short 
        rh_short: 2-d np array, heliocentric distance in au at times=time_short 
        phirf_short: 2-d np array, angle from Neptune in the rotating frame in radians at times=time_short 
        tn_short: 2-d np array, tisserand parameter with respect to Neptune at times=time_short

    outputs:
        flag, integer: 0 if failed, 1 if suceeded
        features: python class TNO_ML_features containing all the ML data features

    """
    flag = 0
    #create the empty class for the features
    f = TNO_ML_features(clones)

    #make the arrays 2-d if needed (everything below assumes 2-d)
    if(len(a.shape)<2):
        a = np.array([a])
        ec = np.array([ec])
        inc = np.array([inc])
        node = np.array([node])
        argperi = np.array([argperi])
        pomega = np.array([pomega])
        q = np.array([q])
        rh = np.array([rh])
        phirf = np.array([phirf])
        tn = np.array([tn])
 
        a_short = np.array([a_short])
        ec_short = np.array([ec_short])
        inc_short = np.array([inc_short])
        node_short = np.array([node_short])
        argperi_short = np.array([argperi_short])
        pomega_short = np.array([pomega_short])
        q_short = np.array([q_short])
        rh_short = np.array([rh_short])
        phirf_short = np.array([phirf_short])
        tn_short = np.array([tn_short])
 

    #check if the time outputs are as expected
    delta_nlong = np.abs(len(time) - 10001)
    delta_nshort = np.abs(len(time_short) - 10001)
    delta_length_long = np.abs( (time[-1] - time[0]) - 1.e7)
    delta_length_short = np.abs( (time_short[-1] - time_short[0]) - 0.5e6)

    if(delta_nlong > 10 or delta_nshort > 10 or delta_length_long > 1e4 or delta_length_long > 5e2):
        print("The length and output cadence of the provided data series are not sufficiently close")
        print("to that expected for the TNO machine learning classifier. The classifier was trained")
        print("on two time series: 1) a short, 0.5 Myr integration with outputs every 50 years and")
        print("2) a longer, 10 Myr integration with outputs every 1000 years.")
        print("delta_nlong, delta_nshort, delta_length_long, delta_length_short")
        print(delta_nlong, delta_nshort, delta_length_long, delta_length_short)
        print("Failed at machine_learning.calc_ML_features()")
        return flag, None
    if(delta_nlong > 0 or delta_nshort > 0 or delta_length_long > 5. or delta_length_long > 100.):
        print("Warning: The length and output cadence of the provided data series are not identical")
        print("to that expected for the TNO machine learning classifier. The classifier was trained")
        print("on two time series: 1) a short, 0.5 Myr integration with outputs every 50 years and")
        print("2) a longer, 10 Myr integration with outputs every 1000 years.")
        print("delta_nlong, delta_nshort, delta_length_long, delta_length_short")
        print(delta_nlong, delta_nshort, delta_length_long, delta_length_short)
        print("The provided time series are close, so the code will proceed, but consider adjusting")
        print("your integrations to exactly match those criteria.")
        flag = 2


    
    ########################################################
    ########################################################
    #
    # Very basic time-series data features
    #
    ########################################################
    ########################################################
    
    #basic analysis of a, e, i, q, and T_nep
    a_min, a_max, f.a_mean, f.a_stddev, f.a_delta, f.a_stddev_normed, f.a_delta_normed = \
            basic_time_series_features(a)
    a_min_short, a_max_short, f.a_mean_short, f.a_stddev_short, f.a_delta_short, \
            f.a_stddev_normed_short, f.a_delta_normed_short = basic_time_series_features(a_short)
    
    f.e_min, f.e_max, f.e_mean, f.e_stddev, f.e_delta, junk, junk = basic_time_series_features(ec)
    f.e_min_short, f.e_max_short, f.e_mean_short, f.e_stddev_short, f.e_delta_short, junk, junk = \
            basic_time_series_features(ec_short)
    
    f.i_min, f.i_max, f.i_mean, f.i_stddev, f.i_delta, junk, junk = basic_time_series_features(inc)
    f.i_min_short, f.i_max_short, f.i_mean_short, f.i_stddev_short, f.i_delta_short, junk, junk = \
            basic_time_series_features(inc_short)

    f.q_min, f.q_max, f.q_mean, f.q_stddev, f.q_delta, f.q_stddev_normed, f.q_delta_normed = \
            basic_time_series_features(q)
    f.q_min_short, f.q_max_short, f.q_mean_short, f.q_stddev_short, f.q_delta_short, \
            f.q_stddev_normed_short, f.q_delta_normed_short = basic_time_series_features(q_short)

    f.tn_min, f.tn_max, f.tn_mean, f.tn_stddev, f.tn_delta, junk, junk = basic_time_series_features(tn)
    f.tn_min_short, f.tn_max_short, f.tn_mean_short, f.tn_stddev_short, f.tn_delta_short, junk, junk = \
            basic_time_series_features(tn_short)

    #make sure all angles are 0-2pi
    argperi = tools.arraymod2pi(argperi)
    node = tools.arraymod2pi(node)
    pomega = tools.arraymod2pi(pomega)

    argperi_short = tools.arraymod2pi(argperi_short)
    node_short = tools.arraymod2pi(node_short)
    pomega_short = tools.arraymod2pi(pomega_short)

    #basic analysis of argperi
    #long sims
    f.o_min, f.o_max, argperi_mean, argperi_std, argperi_del, junk, junk =  basic_time_series_features(argperi)
    #recenter arg peri around 0 and repeat
    argperi_zero = tools.arraymod2pi0(argperi)
    argperi_min2, argperi_max2, argperi_mean2, argperi_std2, argperi_del2, junk, junk =  \
            basic_time_series_features(argperi_zero)
    #take the better values for delta, mean, and standard deviation:
    #(with a 5 degree threshold requirement for the difference in deltas)
    for n in range(0,clones+1):
        if(argperi_del2[n] < argperi_del[n] and np.abs(argperi_del2[n] -argperi_del[n])>0.08726):
            f.o_delta[n] = argperi_del2[n]
            f.o_mean[n] = np.mod(argperi_mean2[n],2.*np.pi)
            f.o_stddev[n] = argperi_std2[n]
        else:
            f.o_delta[n] = argperi_del[n]
            f.o_mean[n] = argperi_mean[n]
            f.o_stddev[n] = argperi_std[n]


    #long simulations
    #calculate time derivatives
    dt = time[1:] - time[:-1] 
    try:
        adot = (a[:,1:] - a[:,:-1])/dt
    except:
        print("problem in calculating the long simulation time derivatives, probably")
        print("because the same time output is included in the arrays twice")
        return flag, f
    edot = (ec[:,1:] - ec[:,:-1])/dt
    idot = (inc[:,1:] - inc[:,:-1])/dt
    qdot = (q[:,1:] - q[:,:-1])/dt
    #unwrap the angles first to be sure we get proper differences
    temp = np.unwrap(argperi,axis=1)
    argperidot = (temp[:,1:] - temp[:,:-1])/dt
    temp = np.unwrap(node,axis=1)
    nodedot = (temp[:,1:] - temp[:,:-1])/dt
    temp = np.unwrap(pomega,axis=1)
    pomegadot = (temp[:,1:] - temp[:,:-1])/dt

    #basic analysis of the time derivatives:
    f.adot_min, f.adot_max, f.adot_mean, f.adot_stddev, f.adot_delta, junk, junk = basic_time_series_features(adot)
    f.edot_min, f.edot_max, f.edot_mean, f.edot_stddev, f.edot_delta, junk, junk = basic_time_series_features(edot)
    f.idot_min, f.idot_max, f.idot_mean, f.idot_stddev, f.idot_delta, junk, junk = basic_time_series_features(idot)
    f.qdot_min, f.qdot_max, f.qdot_mean, f.qdot_stddev, f.qdot_delta, junk, junk = basic_time_series_features(qdot)
    f.Omdot_min, f.Omdot_max, f.Omdot_mean, f.Omdot_stddev, f.Omdot_delta, f.Omdot_stddev_normed, f.Omdot_delta_normed = \
            basic_time_series_features(nodedot)
    f.odot_min, f.odot_max, f.odot_mean, f.odot_stddev, f.odot_delta, f.odot_stddev_normed, f.odot_delta_normed = \
            basic_time_series_features(argperidot)
    f.podot_min, f.podot_max, f.podot_mean, f.podot_stddev, f.podot_delta, f.podot_stddev_normed, f.podot_delta_normed = \
            basic_time_series_features(pomegadot)
    
    #short simulations
    #calculate time derivatives
    dt = time_short[1:] - time_short[:-1] 
    try:
        adot = (a_short[:,1:] - a_short[:,:-1])/dt
    except:
        print("problem in calculating the long simulation time derivatives, probably")
        print("because the same time output is included in the arrays twice")
        return flag, f

    edot = (ec_short[:,1:] - ec_short[:,:-1])/dt
    idot = (inc_short[:,1:] - inc_short[:,:-1])/dt
    qdot = (q_short[:,1:] - q_short[:,:-1])/dt
    #unwrap the angles first to be sure we get proper differences
    temp = np.unwrap(argperi_short,axis=1)
    argperidot = (temp[:,1:] - temp[:,:-1])/dt
    temp = np.unwrap(node_short,axis=1)
    nodedot = (temp[:,1:] - temp[:,:-1])/dt
    temp = np.unwrap(pomega_short,axis=1)
    pomegadot = (temp[:,1:] - temp[:,:-1])/dt

    #basic analysis of the time derivatives:
    f.adot_min_short, f.adot_max_short, f.adot_mean_short, f.adot_stddev_short, f.adot_delta_short, junk, junk = \
            basic_time_series_features(adot)
    f.edot_min_short, f.edot_max_short, f.edot_mean_short, f.edot_stddev_short, f.edot_delta_short, junk, junk = \
            basic_time_series_features(edot)
    f.idot_min_short, f.idot_max_short, f.idot_mean_short, f.idot_stddev_short, f.idot_delta_short, junk, junk = \
            basic_time_series_features(idot)
    f.qdot_min_short, f.qdot_max_short, f.qdot_mean_short, f.qdot_stddev_short, f.qdot_delta_short, junk, junk = \
            basic_time_series_features(qdot)
    f.Omdot_min_short, f.Omdot_max_short, f.Omdot_mean_short, f.Omdot_stddev_short, f.Omdot_delta_short, \
            f.Omdot_stddev_normed_short, f.Omdot_delta_normed_short = basic_time_series_features(nodedot)
    f.odot_min_short, f.odot_max_short, f.odot_mean_short, f.odot_stddev_short, f.odot_delta_short, \
            f.odot_stddev_normed_short, f.odot_delta_normed_short = basic_time_series_features(argperidot)
    f.podot_min_short, f.podot_max_short, f.podot_mean_short, f.podot_stddev_short, f.podot_delta_short, \
            f.podot_stddev_normed_short, f.podot_delta_normed_short = basic_time_series_features(pomegadot)


    ########################################################
    ########################################################
    #
    # Rotating Frame data features
    #
    ########################################################
    ########################################################

    phirf = tools.arraymod2pi(phirf)
    phirf_short = tools.arraymod2pi(phirf_short)
    
    for n in range(0,clones+1):
        f.grid_n_empty[n], f.nonzero_grid_avg[n], f.nonzero_grid_stddev[n], f.grid_stddev[n], f.grid_delta_stddev[n], \
            f.empty_peri_sec[n], f.adj_empty_peri_sec[n], f.stddev_n_peri_sec[n], f.delta_n_peri_sec[n], f.rz_peri_max[n], \
            f.empty_apo_sec[n], f.stddev_apo_sec[n], f.delta_n_apo_sec[n], f.rz_apo_max[n] \
            = rotating_frame_features(rh[n],phirf[n])

        f.grid_n_empty_short[n], f.nonzero_grid_avg_short[n], f.nonzero_grid_stddev_short[n], f.grid_stddev_short[n], \
            f.grid_delta_stddev_short[n], f.empty_peri_sec_short[n], f.adj_empty_peri_sec_short[n], \
            f.stddev_n_peri_sec_short[n], f.delta_n_peri_sec_short[n], f.rz_peri_max_short[n], \
            f.empty_apo_sec_short[n], f.stddev_apo_sec_short[n], f.delta_n_apo_sec_short[n], f.rz_apo_max_short[n] \
            = rotating_frame_features(rh_short[n],phirf_short[n])

    
    ########################################################
    ########################################################
    #
    # FFT data features
    #
    ########################################################
    ########################################################

    #calculate the correlations between a and e, a and i, and e and i
    #done in a loop because np.correlate wants 1-d arrays
    for n in range(0,clones+1):
        f.ae_correlation[n] =  max_corelation(a[n],ec[n])
        f.ai_correlation[n] =  max_corelation(a[n],inc[n])
        f.ei_correlation[n] =  max_corelation(ec[n],inc[n])
        f.ae_correlation_short[n] =  max_corelation(a_short[n],ec_short[n])
        f.ai_correlation_short[n] =  max_corelation(a_short[n],inc_short[n])
        f.ei_correlation_short[n] =  max_corelation(ec_short[n],inc_short[n])

    #calculate spectral fractions (done in a loop because the subroutines expect
    #1-d arrays
    deltat = time[2] - time[1]
    deltat_short = time_short[2] - time_short[1]
    for n in range(0,clones+1):
        #a
        f.a_spectral_fraction[n], f.a_maxpower[n], f.a_maxpower3[n], f.a_frequency1[n], f.a_frequency2[n], f.a_frequency3[n] \
            = spectral_characteristics(a[n],deltat)
        f.a_spectral_fraction_short[n], f.a_maxpower_short[n], f.a_maxpower3_short[n], f.a_frequency1_short[n], \
            f.a_frequency2_short[n], f.a_frequency3_short[n] = spectral_characteristics(a_short[n],deltat_short)
        # eccentricity, via e*sin(varpi)
        hec = ec[n]*np.sin(pomega[n])
        f.e_spectral_fraction[n], f.e_maxpower[n], f.e_maxpower3[n], f.e_frequency1[n], f.e_frequency2[n], f.e_frequency3[n] \
            = spectral_characteristics(hec,deltat)
        # inclination, via sin(i)sin(Omega)
        pinc = np.sin(inc[n])*np.sin(node[n])
        f.i_spectral_fraction[n], f.i_maxpower[n], f.i_maxpower3[n], f.i_frequency1[n], f.i_frequency2[n], f.i_frequency3[n] \
            = spectral_characteristics(pinc,deltat)
        #amd
        amd = 1. - np.sqrt(1.- ec[n]*ec[n])*np.cos(inc[n])
        amd = amd*np.sqrt(a[n])
        f.amd_spectral_fraction[n], f.amd_maxpower[n], f.amd_maxpower3[n], f.amd_frequency1[n], f.amd_frequency2[n], \
            f.amd_frequency3[n]  = spectral_characteristics(amd,deltat)

        amd = 1. - np.sqrt(1.- ec_short[n]*ec_short[n])*np.cos(inc_short[n])
        amd = amd*np.sqrt(a_short[n])
        f.amd_spectral_fraction_short[n], f.amd_maxpower_short[n], f.amd_maxpower3_short[n], \
            f.amd_frequency1_short[n], f.amd_frequency2_short[n], f.amd_frequency3_short[n] \
            = spectral_characteristics(amd,deltat_short)


    ########################################################
    ########################################################
    #
    # additional time-series based features
    #
    ########################################################
    ########################################################

    #Do some binning in the a, e, and i-distributions
    #compare visit distributions

    for n in range(0,clones+1):

        f.a_minmax_to_mean_density_ratio[n], f.a_min_to_max_density_ratio[n], \
            f.a_delta_minmax_to_mean_density_ratio[n], f.a_delta_min_to_max_density_ratio[n] \
            =  histogram_features(a[n],a_min[n],a_max[n],f.a_mean[n],f.a_stddev[n],delta=True)

        f.a_minmax_to_mean_density_ratio_short[n], f.a_min_to_max_density_ratio_short[n], \
            f.a_delta_minmax_to_mean_density_ratio_short[n], f.a_delta_min_to_max_density_ratio_short[n] \
            =  histogram_features(a_short[n],a_min_short[n],a_max_short[n],f.a_mean_short[n],f.a_stddev_short[n],delta=True)


        f.e_minmax_to_mean_density_ratio[n], f.e_min_to_max_density_ratio[n], junk, junk \
            =  histogram_features(ec[n],f.e_min[n],f.e_max[n],f.e_mean[n],f.e_stddev[n])
        f.e_minmax_to_mean_density_ratio_short[n], f.e_min_to_max_density_ratio_short[n], junk, junk \
            =  histogram_features(ec_short[n],f.e_min_short[n],f.e_max_short[n],f.e_mean_short[n],f.e_stddev_short[n])

        f.i_minmax_to_mean_density_ratio[n], f.i_min_to_max_density_ratio[n], junk, junk \
            =  histogram_features(inc[n],f.i_min[n],f.i_max[n],f.i_mean[n],f.i_stddev[n])
        f.i_minmax_to_mean_density_ratio_short[n], f.i_min_to_max_density_ratio_short[n], junk, junk \
            =  histogram_features(inc_short,f.i_min_short[n],f.i_max_short[n],f.i_mean_short[n],f.i_stddev_short[n])
 

      
    if(flag<1):
        flag = 1

    return flag, f


    
def check_angle(img_clf, time, phi, max_prob=0.):
    '''
    Generates the images and data features for the phi classifier
    then runs the classifier to get the probability the angle is
    librating. If that probability is sufficiently high, also 
    calculates the standard deviation and range of phi

    inputs:
        img_clf: the scikitlearn classifier 
        time, 1-d numpy array: time values from a 1e7 year integration
        phi, 1-d numpy array: phi values from a 1e7 year integration
        max_prob, float: the highest-probability resonance angle so far
    outputs: 
        flag, integer:  0 for failure, 1 for success
        prob, float: the probability that phi is librating according to img_clf
        sigma_phi, float: standard deviation in phi (minimally corrected for wrapping)
        delta_phi, float: max range in phi (minimally corrected for wrapping)
    '''
    flag = 0
    prob = 0.
    sigma_phi = 0.
    delta_phi = 0.


    
    #use random strings for the two image file names to prevent issues if multiple
    #instances of this subroutine end up running in the same directory
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
    img1 = random_string+'.png' #'0centered.png'
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
    img2 = random_string+'.png' #'180centered.png'


    iflag, phi, phi0 = make_ml_phi_plots(phi,time,img1,img2)
    if(not(iflag)):
        print("machine_learning.check_angle failed")
        return flag, prob, sigma_phi, delta_phi
    hflag, comb_hog = calc_ml_combined_hog(img1,img2)
    if(not(hflag)):
        print("machine_learning.check_angle failed")
        return flag, prob, sigma_phi, delta_phi
    
    try:
        class_probs = img_clf.predict_proba([comb_hog])
    except:
        print("machine_learning.check_angle failed at the image classifier")
        return flag, prob, sigma_phi, delta_phi

    prob = class_probs[0][1]
    if(prob >= max_prob):
        sigma_phi = min(np.std(phi),np.std(phi0))
        dp1 = np.max(phi) - np.min(phi)
        dp2 = np.max(phi0) - np.min(phi0)
        delta_phi = min(dp1,dp2)    

    #remove the image files
    remove(img1)
    remove(img2)

    flag = 1
    return flag, prob, sigma_phi, delta_phi
    

##########################################################################################################
# Helper functions for calculating the ML features
##########################################################################################################


#################################################################
# Functions needed for the resonance angle search
#################################################################
def make_ml_phi_plots(phi,time,img1,img2):
    '''
    Makes the plots that the ML image classifier uses to determine
    if a resonant angle is librating or not

    inputs:
        phi, 1-d numpy array of resonant angle values
        time, 1-d numpy array of time (0,1e7 years)
        img1, string: name/path for image file to be saved to
                      for phi vs time, phi running from -pi to pi
        img2, string: name/path for image file to be saved to
                      for phi vs time, phi running from 0 to 2pi
    outputs:
        flag, integer: 0 for failure, 1 for success
        phi, 1-d numpy array of resonant angle values from 0-2pi
        phi0, 1-d numpy array of resonant angle values from -pi-pi
    '''

    flag = 0

    phi = tools.arraymod2pi(phi)
    phi0 = tools.arraymod2pi0(phi)

    #none of these plot parameters should be changed!
    #the classifier was trained on images of this 
    #specific size, point type, etc!
    fig = plt.figure()
    fig.set_size_inches(5,3.6)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.set_ylim(-np.pi,np.pi)
    ax.set_xlim(0,1e7)
    try:
        ax.scatter(time,phi0,s=5,c='k')
    except:
        print("problem in machine_learning.make_ml_phi_plots")
        print("couldn't make the -pi to pi plot of phi")
        return flag, phi, phi0
    ax.set_axis_off()
    try:
        plt.savefig(img1,dpi=200)
    except:
        print("problem in machine_learning.make_ml_phi_plots")
        print("couldn't save the -pi to pi plot of phi to file:")
        print(img1)
        return flag, phi, phi0

    plt.close('all')

    fig = plt.figure()
    fig.set_size_inches(5,3.6)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.set_ylim(0,2.*np.pi)
    ax.set_xlim(0,1e7)
    try:
        ax.scatter(time,phi,s=5,c='k')
    except:
        print("problem in machine_learning.make_ml_phi_plots")
        print("couldn't make the 0 to 2pi plot of phi")
        return flag, phi, phi0
    ax.set_axis_off()
    try:
        plt.savefig(img2,dpi=200)
    except:
        print("problem in machine_learning.make_ml_phi_plots")
        print("couldn't save the 0 to 2pi plot of phi to file:")
        print(img2)
        return flag, phi, phi0        
    plt.close('all')
    flag = 1
    return flag, phi, phi0


def calc_ml_combined_hog(img1,img2):
    '''
    Calculates the HOG (histogram of oriented gradients) for the 
    resonant angle plots. This is the data feature the image 
    classifier uses to determine libration.

    inputs:
        img1, string: name/path for image file showing
                      for phi vs time, phi running from -pi to pi
        img2, string: name/path for image file showing
                      for phi vs time, phi running from 0 to 2pi
    outputs:
        flag, integer: 0 for failure, 1 for success
        comb_hog, numpy array: concatenated HOG values for the two
                      input images

    '''    

    flag = 0

    #do not change these values because the training set for the 
    #classifier uses these specific values
    ppc_1 = 40
    ppc_2 = 20
    ori=10

    try:
        read1 = imread(img1, as_gray=True)
    except:
        print("problem in machine_learning.calc_ml_combined_hog")
        print("could not read the first image: ")
        print(img1)
        return flag, [0.]

    try:
        hog1 = hog(read1, pixels_per_cell=(ppc_1,ppc_2), 
                   cells_per_block=(1, 1), 
                   orientations=ori, visualize=False, 
                   block_norm='L1-sqrt')
    except:
        print("problem in machine_learning.calc_ml_combined_hog")
        print("could not calculate the HOG for the first image: ")
        print(img1)
        return flag, [0.]

    try:    
        read2 = imread(img2, as_gray=True)
    except:
        print("problem in machine_learning.calc_ml_combined_hog")
        print("could not read the second image: ")
        print(img2)
        return flag, [0.]

    try:
        hog2 = hog(read2, pixels_per_cell=(ppc_1,ppc_2), 
                   cells_per_block=(1, 1), 
                   orientations=ori, visualize=False, 
                   block_norm='L1-sqrt')
    except:
        print("problem in machine_learning.calc_ml_combined_hog")
        print("could not calculate the HOG for the second image: ")
        print(img2)
        return flag, [0.]

    comb_hog = np.append(hog2,hog1)
    flag = 1

    return flag, comb_hog
########################################################################


#################################################################
# Functions needed for the main classifier
#################################################################
def basic_time_series_features(x):
    '''
    input:
        x, 2-d numpy array of size/shape ([clones+1,nout])
    '''
    x_min = np.amin(x,axis=1)
    x_max = np.amax(x,axis=1)
    x_mean = np.mean(x,axis=1)
    x_std = np.std(x,axis=1)
    x_del = x_max - x_min
    #test for divide by zero and remove issues there
    if(x_mean.any(0)):
        x_std_norm = np.zeros(len(x_mean))
        x_del_norm = np.zeros(len(x_mean))
        for i, val in enumerate(x_mean):
            if(val==0):
                x_std_norm[i] = x_std[i]
                x_del_norm[i] = x_del[i]
            else:
                x_std_norm[i] = x_std[i]/val
                x_del_norm[i] = x_del[i]/val
    else:
        x_std_norm = x_std/x_mean
        x_del_norm = x_del/x_mean

    return x_min, x_max, x_mean, x_std, x_del, x_std_norm, x_del_norm


def max_corelation(d1, d2):
    '''
    input: 
        d1, 1-d np array
        d2, 1-d np array
    '''
    if(len(d1)<1):
        return 0
    std_d1 = np.std(d1)
    std_d2 = np.std(d2)
    if(std_d1 == 0 or std_d2 == 0):
        nd1 = (d1 - np.mean(d1))
        nd2 = (d2 - np.mean(d2))  
    else:
        nd1 = (d1 - np.mean(d1)) / (std_d1)
        nd2 = (d2 - np.mean(d2)) / (std_d2)  
    cmax = (np.correlate(nd1, nd2, 'full')/len(d1)).max()
    return cmax


def spectral_characteristics(data,dt):
    Y = np.fft.rfft(data)
    n = len(data)
    freq = np.fft.rfftfreq(n,d=dt)
    jmax = len(Y)
    Y = Y[1:jmax]
    freq = freq[1:jmax]
    Y = np.abs(Y)**2.
    arr1 = Y.argsort()    
    sorted_Y = Y[arr1[::-1]]
    sorted_freq = freq[arr1[::-1]]
    f1 = sorted_freq[0]
    f2 = sorted_freq[1]
    f3 = sorted_freq[2]
    ytot = 0.
    for Y in (sorted_Y):
        ytot+=Y

    if(sorted_Y[0] == 0 or ytot == 0):
        return 0.,0.,0.,0.,0.,0.
        
    maxnorm_Y = sorted_Y/sorted_Y[0]
    count=0
    for j in range(0,jmax-1):
        if(maxnorm_Y[j] > 0.05):
            count+=1
    sf = 1.0*count/(jmax-1.)
    maxpower = sorted_Y[0]/ytot
    max3 = (sorted_Y[0] + sorted_Y[1] + sorted_Y[2])/ytot
    return sf, maxpower, max3, f1, f2, f3


def histogram_features(x,xmin,xmax,xmean,xstd,delta=False):
    dx = (xmax-xmin)/8.
    if(dx == 0):
        return 0.,0.,0.,0.
    x1 = xmin
    x2 = xmin + 2.*dx
    x3 = x2 + dx
    x4 = x3 + 2.*dx
    x5 = x4 + dx
    x6 = xmax
    xbins = [x1,x2,x3,x4,x5,x6]
    xcounts, tbins = np.histogram(x,bins=xbins)

    #average ratio of extreme-x density to middle-x density
    if(xcounts[2] == 0):
        xcounts[2] = 1 #avoid a nan
    em_x = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
    #ratio of extreme-low-x density to extreme high-x density
    if(xcounts[4] == 0):
        xcounts[4] = 1
    lh_x = (xcounts[0]/xcounts[4])

    if(delta):
        #repeat across a couple time bins 
        dj = x.size//4
        xcounts, tbins = np.histogram(x[0:dj],bins=xbins)
        if(xcounts[2] == 0):
            xcounts[2] = 1
        em1 = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
        if(xcounts[4] == 0):
            xcounts[4] = 1
        lh1 = (xcounts[0]/xcounts[4])
 
        xcounts, tbins = np.histogram(x[dj:2*dj],bins=xbins)
        if(xcounts[2] == 0):
            xcounts[2] = 1
        em2 = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
        if(xcounts[4] == 0):
            xcounts[4] = 1
        lh2 = (xcounts[0]/xcounts[4])
        
        xcounts, tbins = np.histogram(x[2*dj:3*dj],bins=xbins)
        if(xcounts[2] == 0):
            xcounts[2] = 1
        em3 = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
        if(xcounts[4] == 0):
            xcounts[4] = 1
        lh3 = (xcounts[0]/xcounts[4])


        xcounts, tbins = np.histogram(x[3*dj:4*dj],bins=xbins)
        if(xcounts[2] == 0):
            xcounts[2] = 1
        em4 = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
        if(xcounts[4] == 0):
            xcounts[4] = 1
        lh4 = (xcounts[0]/xcounts[4])

        min_em_x = min(em1,em2,em3,em4)
        max_em_x = max(em1,em2,em3,em4)

        delta_em_x = max_em_x - min_em_x

        min_lh_x = min(lh1,lh2,lh3,lh4)
        max_lh_x = max(lh1,lh2,lh3,lh4)

        delta_lh_x = max_lh_x - min_lh_x
    else:
        delta_em_x=0.
        delta_lh_x=0.

    return  em_x, lh_x, delta_em_x, delta_lh_x



def rotating_frame_features(rh,phirf):
    #divide heliocentric distance into 10 bins and theta_n
    #into 20 bins
    qmin = np.amin(rh) - 0.01
    Qmax = np.amax(rh) + 0.01

    if( (Qmax - qmin) == 0.02 ):
        return 0,0,0,0,0,0,0,0,0,0,0,0,0,0

    nrbin = 10.
    nphbin = 20.
    dr = (Qmax - qmin) / nrbin
    dph = (2. * np.pi) / nphbin
    # center on the planet in phi (so when we bin, we will
    # add the max and min bins together since they're really
    # half-bins
    phmin = -dph / 2.

    # radial plus aziumthal binning
    # indexing is radial bin, phi bin: rph_count[rbin,phibin]
    rph_count = np.zeros((int(nrbin), int(nphbin)))
    # radial only binning
    r_count = np.zeros(int(nrbin))

    # for calculating the average sin(ph) and cos(ph)
    # indexing is   sinphbar[rbin,resorder]
    resorder_max = 10
    sinphbar = np.zeros((int(nrbin), resorder_max+1))
    cosphbar = np.zeros((int(nrbin), resorder_max+1))

    # divide into radial and azimuthal bins
    nmax = len(rh)
    for n in range(0, nmax):
        rbin = int(np.floor((rh[n] - qmin) / dr))
        for resorder in range(1,resorder_max+1):
            tcos = np.cos(float(resorder)*phirf[n])
            tsin = np.sin(float(resorder)*phirf[n])
            sinphbar[rbin,resorder]+=tsin
            cosphbar[rbin,resorder]+=tcos
        r_count[rbin]+=1.
        phbin = int(np.floor((phirf[n] - phmin) / dph))
        if (phbin == int(nphbin)):
            phbin = 0
        rph_count[rbin, phbin] += 1

    # perihelion/aphelion distance bin stats
    bins = [0,int(nrbin)-1] #only going to do these two bins
    nempty = np.zeros(int(nrbin))
    nadjempty = np.zeros(int(nrbin))
    rbinavg = np.zeros(int(nrbin))
    rbinstd = np.zeros(int(nrbin))
    rbinmax = np.zeros(int(nrbin))
    rbinmin = np.zeros(int(nrbin))

    for nr in bins:
        rbinmax[nr] = 0.
        rbinmin[nr] = 1e9
        for resorder in range(1,resorder_max+1):
            sinphbar[nr,resorder] = sinphbar[nr,resorder]/r_count[nr]
            cosphbar[nr,resorder] = cosphbar[nr,resorder]/r_count[nr]
        for n in range(0, int(nphbin)):
            if (rph_count[nr, n] == 0):
                nempty[nr] += 1
            if (rph_count[nr, n] < rbinmin[nr]):
                rbinmin[nr] = rph_count[nr, n]
            if (rph_count[nr, n] > rbinmax[nr]):
                rbinmax[nr] = rph_count[nr, n]
            rbinavg[nr] += rph_count[nr, n]
        rbinavg[nr] = rbinavg[nr] / nphbin

        for n in range(0, int(nphbin)):
            rbinstd[nr] += (rph_count[nr, n] - rbinavg[nr]) * (
                        rph_count[nr, n] - rbinavg[nr])
        if (not (rbinavg[nr] == 0)):
            rbinstd[nr] = np.sqrt(rbinstd[nr] / nphbin) #/ rbinavg[nr]
        else:
            rbinstd[nr] = 0.

        if (rph_count[nr, 0] == 0):
            nadjempty[nr] = 1
            for n in range(1, int(np.floor(nphbin / 2.)) + 1):
                if (rph_count[nr, n] == 0):
                    nadjempty[nr] += 1
                if (rph_count[nr, n] != 0):
                    break
            for n in range(int(nphbin) - 1, int(np.floor(nphbin / 2.)), -1):
                if (rph_count[nr, n] == 0):
                    nadjempty[nr] += 1
                if (rph_count[nr, n] != 0):
                    break


    n_peri_empty = nempty[0]
    nadj_peri_empty = nadjempty[-1]
    nstd_peri = rbinstd[0]
    ndel_peri = rbinmax[0] - rbinmin[0]

    n_apo_empty = nempty[-1]
    nstd_apo = rbinstd[-1]
    ndel_apo = rbinmax[-1] - rbinmin[-1]
 
    #rayleigh z-test statistics at perihelion and aphelion
    rz_peri = np.zeros(resorder_max+1)
    rz_apo = np.zeros(resorder_max+1)
    for resorder in range(1, resorder_max+1):
        rz_peri[resorder] = np.sqrt(sinphbar[0,resorder]*sinphbar[0,resorder] +
                       cosphbar[0,resorder]*cosphbar[0,resorder])
        rz_apo[resorder] = np.sqrt(sinphbar[-1,resorder]*sinphbar[-1,resorder] +
                       cosphbar[-1,resorder]*cosphbar[-1,resorder])

    rzperi_max = np.amax(rz_peri[1:resorder_max])
    rzapo_max = np.amax(rz_apo[1:resorder_max])


    spatial_counts = rph_count.flatten()
    #grid_avg is used to normalize all the rest of the counts
    grid_avg =  np.mean(spatial_counts)
    grid_nz_avg = np.mean(spatial_counts[np.nonzero(spatial_counts)])
    grid_nz_std = np.std(spatial_counts[np.nonzero(spatial_counts)])
    grid_std =  np.std(spatial_counts)
    grid_deltastd = grid_std - grid_nz_std
    
    n_empty=0
    for n in range(0,len(spatial_counts)):
        if(spatial_counts[n]==0):
            n_empty += 1

    #normalize all the grid counts:
    grid_nz_avg = grid_nz_avg/grid_avg
    grid_nz_std = grid_nz_std/grid_avg
    grid_std = grid_std/grid_avg
    grid_deltastd = grid_deltastd/grid_avg
    nstd_apo = nstd_apo/grid_avg
    ndel_apo = ndel_apo/grid_avg
    nstd_peri = nstd_peri/grid_avg
    ndel_peri = ndel_peri/grid_avg


    return n_empty, grid_nz_avg, grid_nz_std, grid_std, grid_deltastd, n_peri_empty,  nadj_peri_empty, \
            nstd_peri, ndel_peri, rzperi_max, n_apo_empty, nstd_apo, ndel_apo, rzapo_max









