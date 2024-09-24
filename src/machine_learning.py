import sys
import numpy as np
import pandas as pd
import tools
import run_reb
import MLdata
from os import path

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from datetime import date
from pickle import dump
from pickle import load
from importlib import resources as impresources


# define the default file scheme for the machine learning datasets

# this file is provided with the package
default_TNO_training_data = '09-06-2024-ML-features.csv'

# these files will be generated the first time the ML is called
# so that the trained classifier can be saved and used later
default_trained_classifier = 'trained-TNO-classifier.pkl'
default_trained_classifier_dictionary = 'trained-TNO-classifier-dictionary.pkl'


class TNO_ML_outputs:
    # class that stores the information that comes out of the 
    # TNO machine learning classifier
    def __init__(self,clones=0):
        self.clones = clones
        # make an empty features instance
        self.features = TNO_ML_features(self.clones)
        # parameters related to the classifier
        self.classes_dictionary = None
        self.class_probs = None
        
        #clone-by-clone predicted classification and 
        #confidence level for that classification
        self.clone_classification = None
        self.clone_confidence = None

        self.most_common_class = None
        self.fraction_most_common_class = None

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


class TNO_ML_features:
    # class that stores the pre-determined set of data features that 
    # the TNO machine learning classifier uses
    # non-obvious terms are defined the first time they appear
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

        ##########
        # summary statistics of the distribution
        # of points in a grid of heliocentric distance
        # vs angle from Neptune in the rotating frame
        ##########
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

        ##########
        # features based on FFTs
        #########
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
        print("Failed at machine_learning.calc_ML_features()")
        return flag, None
    if(delta_nlong > 0 or delta_nshort > 0 or delta_length_long > 5. or delta_length_long > 100.):
        print("Warning: The length and output cadence of the provided data series are not identical")
        print("to that expected for the TNO machine learning classifier. The classifier was trained")
        print("on two time series: 1) a short, 0.5 Myr integration with outputs every 50 years and")
        print("2) a longer, 10 Myr integration with outputs every 1000 years.")
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




def setup_and_run_TNO_integration_for_ML(des=None, clones=None, datadir='',
                                         archivefile=None,logfile=False):
    '''
    '''
    flag = 0


    if(des == None):
        print("The designation of a TNO must be provided")
        print("failed at machine_learning.setup_and_run_TNO_integration_for_ML()")
        return flag, None, None


    if(logfile==True):
        logf = tools.log_file_name(des=des)
    else:
        logf=logfile
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
                          des=des, clones=clones, cloning_method= cloning_method,
                          logfile=logfile)
    else:
        cloning_method = 'Gaussian'
        iflag, epoch, sim = run_reb.initialize_simulation(planets=['outer'],
                          des=des, clones=clones, cloning_method= cloning_method,
                          logfile=logfile)

    if(iflag < 1):
        return flag, None, sim


    fflag, tno_class, sim = run_and_MLclassify_TNO(sim,des=des,clones=clones,datadir=datadir,
                                                  archivefile=archivefile,deletefile=True,
                                                  logfile=logfile)

    if(fflag < 1):
        return flag, tno_class, sim

    flag = 1
    return flag, tno_class, sim



def run_and_MLclassify_TNO(sim=None, des=None,clones=None, 
                           datadir='', archivefile=None, deletefile=False,
                           logfile=False):
    '''
    '''

    if(sim == None):
        print("No initialized rebound simulation provided")
        print("use the function setup_and_run_TNO_integration_for_ML instead or")
        print("initialize the simulation and try again")
        print("failed at machine_learning.run_and_MLclassify_TNO()")
        return flag, tno_class, sim

    if(des == None):
        print("The designation of the small body must be provided")
        print("failed at machine_learning.run_and_MLclassify_TNO()")
        return flag, tno_class, sim

    if(logfile==True):
        logf = tools.log_file_name(des=des)
    else:
        logf=logfile
    if(datadir and logf and logf!='screen'):        
        logf = datadir + '/' +logf

    flag = 0
    #make an empty set of classification outputs 
    tno_class = TNO_ML_outputs(clones)

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
    


    fflag, tno_class.features = calc_ML_features(t_long,a_long,ec_long,inc_long,node_long,peri_long,
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


    #apply the classifier
    tno_class.class_probs = classifier.predict_proba(tno_class.features.return_features_list())

    #run the minor corrections and assign clone-by-clone and most common classes
    tno_class.determine_clone_classification()
    tno_class.determine_most_common_classification()

   
    flag = 1
    return flag, tno_class, sim



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




##########################################################################################################
# Helper functions for calculating the ML features
##########################################################################################################

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
    x_std_norm = x_std/x_mean
    x_del_norm = x_del/x_mean
    return x_min, x_max, x_mean, x_std, x_del, x_std_norm, x_del_norm


def max_corelation(d1, d2):
    d1 = (d1 - np.mean(d1)) / (np.std(d1))
    d2 = (d2 - np.mean(d2)) / (np.std(d2))  
    cmax = (np.correlate(d1, d2, 'full')/len(d1)).max()
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
    norm_Y = sorted_Y/ytot
    count=0
    maxnorm_Y = sorted_Y/sorted_Y[0]
    for j in range(0,jmax-1):
        if(maxnorm_Y[j] > 0.05):
            count+=1
    sf = 1.0*count/(jmax-1.)
    maxpower = sorted_Y[0]/ytot
    max3 = (sorted_Y[0] + sorted_Y[1] + sorted_Y[2])/ytot
    return sf, maxpower, max3, f1, f2, f3


def histogram_features(x,xmin,xmax,xmean,xstd,delta=False):
    dx = (xmax-xmin)/8.
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


##########################################################################################################
##########################################################################################################


##########################################################################################################
# Helper functions for reading in the provided csv of pre-calculated and labeled features and then using
# those to train and test the classifier.
##########################################################################################################

def read_TNO_training_data(training_file):
    '''
    Read in the csv file with all the TNO data features and labels
    removes any TNOs with a>1000 au or those that have drastic changes
    in semimajor axis (da>30 au in 0.5 Myr or da>100 au in 10 Myr)
    '''
    all_TNOs = pd.read_csv(training_file, skipinitialspace=True, index_col=False, low_memory=False)
    
    #remove extremely large-a objects and those that scatter a lot in a
    filtered1_TNOs = all_TNOs[all_TNOs['a_mean']<1000.0].copy()
    filtered2_TNOs = filtered1_TNOs[filtered1_TNOs['a_delta_short']<30.0].copy()
    filtered_TNOs = filtered2_TNOs[filtered2_TNOs['a_delta']<100.0].copy()

    filtered_TNOs['simplified_G08'] = filtered_TNOs.apply(label_particle_simplifiedG08, axis=1)

    return filtered_TNOs


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
    

    return clf, score, feature_names, classes_dict



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









