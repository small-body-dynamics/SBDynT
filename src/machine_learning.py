import sys
import numpy as np
import pandas as pd
import tools
import run_reb

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from datetime import date


class TNO_ML_features:
    # class that stores the pre-determined set of data features that 
    # the TNO machine learning classifier uses
    # non-obvious terms are defined the first time they appear
    def __init__(self):
        # initialize all of the data features

        # first the set based on the long (10Myr integration)
        # a = semimajor axis (au)
        self.a_mean = 0.
        self.a_stddev = 0.
        self.a_stddev_normed = 0. # == a_stddev/a_mean
        self.a_delta = 0. #maximum(a) - minimum(a)
        self.a_delta_normed = 0. # == a_delta/a_mean
        
        # adot is the change in a from one time output
        # to the next divided by the delta-t between outputs
        # (au/year)
        self.adot_min = 0.
        self.adot_mean = 0.
        self.adot_max = 0.
        self.adot_stddev = 0.
        self.adot_delta = 0.
        
        # e = eccentricity
        self.e_min = 0.
        self.e_mean = 0.
        self.e_max = 0.
        self.e_stddev = 0.
        self.e_delta = 0.
 
        # (per year)
        self.edot_min = 0.
        self.edot_mean = 0.
        self.edot_max = 0.
        self.edot_stddev = 0.
        self.edot_delta = 0.

        # i = inclination (radians)
        self.i_min = 0.
        self.i_mean = 0.
        self.i_max = 0.
        self.i_stddev = 0.
        self.i_delta = 0.

        # (radians/year)
        self.idot_min = 0.
        self.idot_mean = 0.
        self.idot_max = 0.
        self.idot_stddev = 0.
        self.idot_delta = 0.

        # Om = longitude of ascending node 
        # (radians/year)
        self.Omdot_min = 0.
        self.Omdot_mean = 0.
        self.Omdot_max = 0.
        self.Omdot_stddev = 0.
        self.Omdot_stddev_normed = 0.
        self.Omdot_delta = 0.
        self.Omdot_delta_normed = 0.

        # o = argument of perihelion (radians)
        self.o_min = 0.
        self.o_mean = 0.
        self.o_max = 0.
        self.o_stddev = 0.
        self.o_delta = 0.

        # (radians/year)
        self.odot_min = 0.
        self.odot_mean = 0.
        self.odot_max = 0.
        self.odot_stddev = 0.
        self.odot_stddev_normed = 0.
        self.odot_delta = 0.
        self.odot_delta_normed = 0.

        # po = longitude of perhelion
        # (radians/year)
        self.podot_min = 0.
        self.podot_mean = 0.
        self.podot_max = 0.
        self.podot_stddev = 0.
        self.podot_stddev_normed = 0.
        self.podot_delta = 0.
        self.podot_delta_normed = 0.

        # q = perhelion distance (au)
        self.q_min = 0.
        self.q_mean = 0.
        self.q_max = 0.
        self.q_stddev = 0.
        self.q_stddev_normed = 0.
        self.q_delta = 0.
        self.q_delta_normed = 0.

        # (au/year)
        self.qdot_min = 0.
        self.qdot_mean = 0.
        self.qdot_max = 0.
        self.qdot_stddev = 0.
        self.qdot_delta = 0.
        
        # tn = tisserand parameter with respect 
        # to Neptune
        self.tn_min = 0.
        self.tn_mean = 0.
        self.tn_max = 0.
        self.tn_stddev = 0.
        self.tn_delta = 0.

        ##########
        # summary statistics of the distribution
        # of points in a grid of heliocentric distance
        # vs angle from Neptune in the rotating frame
        ##########
        # number of grid spaces at the smallest heliocentric
        # distances that have no visits
        self.empty_peri_sec = 0.
        # number of grid spaces at the smallest heliocentric
        # distances surrounding Neptune that have no visits
        self.adj_empty_peri_sec = 0.
        # for the smallest heliocentric distance grid spaces, 
        # the standard deviation in visits across those bins
        # (normalized by the average number of visits across
        # all grid spaces)
        self.stddev_n_peri_sec = 0.
        # for the smallest heliocentric distance grid spaces, 
        # the difference in visits between the most and least
        # visited grid spaces
        # (normalized by the average number of visits across
        # all grid spaces)
        self.delta_n_peri_sec = 0.
        # for the smallest heliocentric distance grid spaces, 
        # the largest raleigh-Z parameter calculated for 
        # potential resonant configurations up to order 10
        self.rz_peri_max = 0.

        # same as above, but for the largest heliocentric
        # distance grid spaces
        self.empty_apo_sec = 0.
        self.stddev_apo_sec = 0.
        self.delta_n_apo_sec = 0.
        self.rz_apo_max = 0.

        # for the whole grid, the average number of visits
        # to all non-empty grid spaces 
        # (normalized by the average number of visits across
        # all grid spaces)
        self.nonzero_grid_avg = 0.
        # for the whole grid, the standard deviation in number
        # of visits to all non-empty grid spaces 
        # (normalized by the average number of visits across
        # all grid spaces)        
        self.nonzero_grid_stddev = 0.
        # for the whole grid, the standard deviation in number
        # of visits per grid space 
        # (normalized by the average number of visits across
        # all grid spaces)            
        self.grid_stddev = 0.
        # (nonzero_grid_stddev - grid_stddev)
        self.grid_delta_stddev = 0.
        # total number of empty grid spaces
        self.grid_n_empty = 0.


        # correlation coefficients between
        # a and e
        self.ae_correlation = 0.
        # a and i
        self.ai_correlation = 0.
        # e and i
        self.ei_correlation = 0.

        ##########
        # features based on FFTs
        #########
        # the spectral fraction of the semimajor axis
        self.a_spectral_fraction = 0.
        # normalized power in the dominant frequency
        self.a_maxpower = 0.
        # normalized power in the 3 most dominant frequencies
        self.a_maxpower3 = 0.
        # the three most dominant frequencies (yr^-1)
        self.a_frequency1 = 0.
        self.a_frequency2 = 0.
        self.a_frequency3 = 0.

        self.e_spectral_fraction = 0.
        self.e_maxpower = 0.
        self.e_maxpower3 = 0.
        self.e_frequency1 = 0.
        self.e_frequency2 = 0.
        self.e_frequency3 = 0.

        self.i_spectral_fraction = 0.
        self.i_maxpower = 0.
        self.i_maxpower3 = 0.
        self.i_frequency1 = 0.
        self.i_frequency2 = 0.
        self.i_frequency3 = 0.

        self.amd_spectral_fraction = 0.
        self.amd_maxpower = 0.
        self.amd_maxpower3 = 0.
        self.amd_frequency1 = 0.
        self.amd_frequency2 = 0.
        self.amd_frequency3 = 0.

        
        # ratio of the number of outputs near the extreme
        # (minimum and maximum) semimajor axis values to the 
        # number of outputs near the mean semimajor axis value
        self.a_minmax_to_mean_density_ratio = 0.
        # ratio of the number of outputs near the minumum a to 
        # the number of outputs near the maximum a
        self.a_min_to_max_density_ratio = 0.
        # maximum change in the above ratios over four,
        # non-overlapping time bins
        self.a_delta_minmax_to_mean_density_ratio = 0.
        self.a_delta_min_to_max_density_ratio = 0.

        self.e_minmax_to_mean_density_ratio = 0.
        self.e_min_to_max_density_ratio = 0.
        
        self.i_minmax_to_mean_density_ratio = 0.
        self.i_min_to_max_density_ratio = 0.


        #then the set based on the short, 0.5 Myr integration
        self.a_mean_short = 0.
        self.a_stddev_short = 0.
        self.a_stddev_normed_short = 0.
        self.a_delta_short = 0.
        self.a_delta_normed_short = 0.
        
        self.adot_min_short = 0.
        self.adot_mean_short = 0.
        self.adot_max_short = 0.
        self.adot_stddev_short = 0.
        self.adot_delta_short = 0.
        
        self.e_min_short = 0.
        self.e_mean_short = 0.
        self.e_max_short = 0.
        self.e_stddev_short = 0.
        self.e_delta_short = 0.
 
        self.edot_min_short = 0.
        self.edot_mean_short = 0.
        self.edot_max_short = 0.
        self.edot_stddev_short = 0.
        self.edot_delta_short = 0.

        self.i_min_short = 0.
        self.i_mean_short = 0.
        self.i_max_short = 0.
        self.i_stddev_short = 0.
        self.i_delta_short = 0.

        self.idot_min_short = 0.
        self.idot_mean_short = 0.
        self.idot_max_short = 0.
        self.idot_stddev_short = 0.
        self.idot_delta_short = 0.

        self.Omdot_min_short = 0.
        self.Omdot_mean_short = 0.
        self.Omdot_max_short = 0.
        self.Omdot_stddev_short = 0.
        self.Omdot_stddev_normed_short = 0.
        self.Omdot_delta_short = 0.
        self.Omdot_delta_normed_short = 0.

        self.odot_min_short = 0.
        self.odot_mean_short = 0.
        self.odot_max_short = 0.
        self.odot_stddev_short = 0.
        self.odot_stddev_normed_short = 0.
        self.odot_delta_short = 0.
        self.odot_delta_normed_short = 0.

        self.podot_min_short = 0.
        self.podot_mean_short = 0.
        self.podot_max_short = 0.
        self.podot_stddev_short = 0.
        self.podot_stddev_normed_short = 0.
        self.podot_delta_short = 0.
        self.podot_delta_normed_short = 0.

        self.q_min_short = 0.
        self.q_mean_short = 0.
        self.q_max_short = 0.
        self.q_stddev_short = 0.
        self.q_stddev_normed_short = 0.
        self.q_delta_short = 0.
        self.q_delta_normed_short = 0.

        self.qdot_min_short = 0.
        self.qdot_mean_short = 0.
        self.qdot_max_short = 0.
        self.qdot_stddev_short = 0.
        self.qdot_delta_short = 0.
        
        self.tn_min_short = 0.
        self.tn_mean_short = 0.
        self.tn_max_short = 0.
        self.tn_stddev_short = 0.
        self.tn_delta_short = 0.

        self.empty_peri_sec_short = 0.
        self.adj_empty_peri_sec_short = 0.
        self.stddev_n_peri_sec_short = 0.
        self.delta_n_peri_sec_short = 0.
        self.rz_peri_max_short = 0.

        self.empty_apo_sec_short = 0.
        self.stddev_apo_sec_short = 0.
        self.delta_n_apo_sec_short = 0.
        self.rz_apo_max_short = 0.

        self.nonzero_grid_avg_short = 0.
        self.nonzero_grid_stddev_short = 0.
        self.grid_stddev_short = 0.
        self.grid_delta_stddev_short = 0.
        self.grid_n_empty_short = 0.

        self.ae_correlation_short = 0.
        self.ai_correlation_short = 0.
        self.ei_correlation_short = 0.

        self.a_spectral_fraction_short = 0.
        self.a_maxpower_short = 0.
        self.a_maxpower3_short = 0.
        self.a_frequency1_short = 0.
        self.a_frequency2_short = 0.
        self.a_frequency3_short = 0.

        self.amd_spectral_fraction_short = 0.
        self.amd_maxpower_short = 0.
        self.amd_maxpower3_short = 0.
        self.amd_frequency1_short = 0.
        self.amd_frequency2_short = 0.
        self.amd_frequency3_short = 0.

        self.a_minmax_to_mean_density_ratio_short = 0.
        self.a_min_to_max_density_ratio_short = 0.
        self.a_delta_minmax_to_mean_density_ratio_short = 0.
        self.a_delta_min_to_max_density_ratio_short = 0.

        self.e_minmax_to_mean_density_ratio_short = 0.
        self.e_min_to_max_density_ratio_short = 0.
        
        self.i_minmax_to_mean_density_ratio_short = 0.
        self.i_min_to_max_density_ratio_short = 0.

        # list of all the features above
        feature_names = [None]*(len(vars(self)))
        i=0
        for d in vars(self):
            feature_names[i] = d
            i+=1
        self.feature_names = feature_names


    def return_features_list(self):
        #return all the features as a numpy array 
        #in a fixed order that matches the stored training set
        #of labeled features
        features_list = np.zeros(len(self.feature_names))
        i=0
        for t in self.feature_names:
            features_list[i] = self.__getattribute__(t)
            i+=1
        return features_list



def calc_ML_features(time,a,ec,inc,node,argperi,pomega,q,rh,phirf,tn,\
        time_short,a_short,ec_short,inc_short,node_short,argperi_short,\
        pomega_short,q_short,rh_short,phirf_short,tn_short):
    """
    calculate ML  data features from the short and long time-series
    of a TNO integration

    inputs:
        time: 1-d np array, time in years from 0-10 Myr sampled every 1000 years
        a: 1-d np array, semimajor axis in au at times=time 
        ec: 1-d np array, eccentricity at times=time 
        inc: 1-d np array, inclination in radians at times=time 
        node: 1-d np array, longitude of ascending node in radians at times=time 
        argperi: 1-d np array, argument of perihelion in radians at times=time 
        pomega: 1-d np array, longitude of perihelion in radians at times=time 
        q: 1-d np array, perihelion distance in au at times=time 
        rh: 1-d np array, heliocentric distance in au at times=time 
        phirf: 1-d np array, angle from Neptune in the rotating frame in radians at times=time 
        tn: 1-d np array, tisserand parameter with respect to Neptune at times=time 

        time_short: 1-d np array, time in years from 0-0.5 Myr sampled every 50 years
        a_short: 1-d np array, semimajor axis in au at times=time_short
        ec_short: 1-d np array, eccentricity at times=time_short
        inc_short: 1-d np array, inclination in radians at times=time_short 
        node_short: 1-d np array, longitude of ascending node in radians at times=time_short 
        argperi_short: 1-d np array, argument of perihelion in radians at times=time_short 
        pomega_short: 1-d np array, longitude of perihelion in radians at times=time_short 
        q_short: 1-d np array, perihelion distance in au at times=time_short 
        rh_short: 1-d np array, heliocentric distance in au at times=time_short 
        phirf_short: 1-d np array, angle from Neptune in the rotating frame in radians at times=time_short 
        tn_short: 1-d np array, tisserand parameter with respect to Neptune at times=time_short

    outputs:
        features: python class TNO_ML_features containing all the ML data features

    """
    #create the empty class for the features
    f = TNO_ML_features()
    
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
    if(argperi_del2 < argperi_del and np.abs(argperi_del2 -argperi_del)>0.08726):
        f.o_delta = argperi_del2
        f.o_mean = tools.mod2pi(argperi_mean2)
        f.o_stddev = argperi_std2
    else:
        f.o_delta = argperi_del
        f.o_mean = argperi_mean
        f.o_stddev = argperi_std


    #long simulations
    #calculate time derivatives
    dt = time[1:] - time[:-1] 
    adot = (a[1:] - a[:-1])/dt
    edot = (ec[1:] - ec[:-1])/dt
    idot = (inc[1:] - inc[:-1])/dt
    qdot = (q[1:] - q[:-1])/dt
    #unwrap the angles first to be sure we get proper differences
    temp = np.unwrap(argperi)
    argperidot = (temp[1:] - temp[:-1])/dt
    temp = np.unwrap(node)
    nodedot = (temp[1:] - temp[:-1])/dt
    temp = np.unwrap(pomega)
    pomegadot = (temp[1:] - temp[:-1])/dt

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
    adot = (a_short[1:] - a_short[:-1])/dt
    edot = (ec_short[1:] - ec_short[:-1])/dt
    idot = (inc_short[1:] - inc_short[:-1])/dt
    qdot = (q_short[1:] - q_short[:-1])/dt
    #unwrap the angles first to be sure we get proper differences
    temp = np.unwrap(argperi_short)
    argperidot = (temp[1:] - temp[:-1])/dt
    temp = np.unwrap(node_short)
    nodedot = (temp[1:] - temp[:-1])/dt
    temp = np.unwrap(pomega_short)
    pomegadot = (temp[1:] - temp[:-1])/dt

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
    
    f.grid_n_empty, f.nonzero_grid_avg, f.nonzero_grid_stddev, f.grid_stddev, f.grid_delta_stddev, \
            f.empty_peri_sec, f.adj_empty_peri_sec, f.stddev_n_peri_sec, f.delta_n_peri_sec, f.rz_peri_max, \
            f.empty_apo_sec, f.stddev_apo_sec, f.delta_n_apo_sec, f.rz_apo_max \
            = rotating_frame_features(rh,phirf)

    f.grid_n_empty_short, f.nonzero_grid_avg_short, f.nonzero_grid_stddev_short, f.grid_stddev_short, \
            f.grid_delta_stddev_short, f.empty_peri_sec_short, f.adj_empty_peri_sec_short, \
            f.stddev_n_peri_sec_short, f.delta_n_peri_sec_short, f.rz_peri_max_short, \
            f.empty_apo_sec_short, f.stddev_apo_sec_short, f.delta_n_apo_sec_short, f.rz_apo_max_short \
            = rotating_frame_features(rh_short,phirf_short)

    
    ########################################################
    ########################################################
    #
    # FFT data features
    #
    ########################################################
    ########################################################

    #calculate the correlations between a and e, a and i, and e and i
    f.ae_correlation =  max_corelation(a,ec)
    f.ai_correlation =  max_corelation(a,inc)
    f.ei_correlation =  max_corelation(ec,inc)
    f.ae_correlation_short =  max_corelation(a_short,ec_short)
    f.ai_correlation_short =  max_corelation(a_short,inc_short)
    f.ei_correlation_short =  max_corelation(ec_short,inc_short)

    #calculate spectral fractions
    deltat = time[2] - time[1]
    deltat_short = time_short[2] - time_short[1]
    #a
    f.a_spectral_fraction, f.a_maxpower, f.a_maxpower3, f.a_frequency1, f.a_frequency2, f.a_frequency3 \
            = spectral_characteristics(a,deltat)
    f.a_spectral_fraction_short, f.a_maxpower_short, f.a_maxpower3_short, f.a_frequency1_short, \
            f.a_frequency2_short, f.a_frequency3_short = spectral_characteristics(a_short,deltat_short)
    # eccentricity, via e*sin(varpi)
    hec = ec*np.sin(pomega)
    f.e_spectral_fraction, f.e_maxpower, f.e_maxpower3, f.e_frequency1, f.e_frequency2, f.e_frequency3 \
            = spectral_characteristics(hec,deltat)
    # inclination, via sin(i)sin(Omega)
    pinc = np.sin(inc)*np.sin(node)
    f.i_spectral_fraction, f.i_maxpower, f.i_maxpower3, f.i_frequency1, f.i_frequency2, f.i_frequency3 \
            = spectral_characteristics(pinc,deltat)
    #amd
    amd = 1. - np.sqrt(1.- ec*ec)*np.cos(inc)
    amd = amd*np.sqrt(a)
    f.amd_spectral_fraction, f.amd_maxpower, f.amd_maxpower3, f.amd_frequency1, f.amd_frequency2, \
            f.amd_frequency3  = spectral_characteristics(amd,deltat)

    amd = 1. - np.sqrt(1.- ec_short*ec_short)*np.cos(inc_short)
    amd = amd*np.sqrt(a_short)
    f.amd_spectral_fraction_short, f.amd_maxpower_short, f.amd_maxpower3_short, \
            f.amd_frequency1_short, f.amd_frequency2_short, f.amd_frequency3_short \
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


    f.a_minmax_to_mean_density_ratio, f.a_min_to_max_density_ratio, \
            f.a_delta_minmax_to_mean_density_ratio, f.a_delta_min_to_max_density_ratio \
            =  histogram_features(a,a_min,a_max,f.a_mean,f.a_stddev,delta=True)

    f.a_minmax_to_mean_density_ratio_short, f.a_min_to_max_density_ratio_short, \
            f.a_delta_minmax_to_mean_density_ratio_short, f.a_delta_min_to_max_density_ratio_short \
            =  histogram_features(a_short,a_min_short,a_max_short,f.a_mean_short,f.a_stddev_short,delta=True)


    f.e_minmax_to_mean_density_ratio, f.e_min_to_max_density_ratio, junk, junk \
            =  histogram_features(ec,f.e_min,f.e_max,f.e_mean,f.e_stddev)
    f.e_minmax_to_mean_density_ratio_short, f.e_min_to_max_density_ratio_short, junk, junk \
            =  histogram_features(ec_short,f.e_min_short,f.e_max_short,f.e_mean_short,f.e_stddev_short)

    f.i_minmax_to_mean_density_ratio, f.i_min_to_max_density_ratio, junk, junk \
            =  histogram_features(inc,f.i_min,f.i_max,f.i_mean,f.i_stddev)
    f.i_minmax_to_mean_density_ratio_short, f.i_min_to_max_density_ratio_short, junk, junk \
            =  histogram_features(inc_short,f.i_min_short,f.i_max_short,f.i_mean_short,f.i_stddev_short)
 

      
    return f






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


def train_and_test_TNO_classifier(training_file):

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
    int_dict = { i : all_types[i] for i in range( len(all_types) ) }
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

    return clf, score, classes_train, classes_test, features_train, features_test, feature_names, int_dict


def label_particle_simplifiedG08(row):
    newlabel = 'none'
    if (row['G08_class'] == 'detached' or row['G08_class'] == 'classical'):
        newlabel = 'class-det'
    else:
        newlabel= row['G08_class']
    return newlabel



def setup_and_run_TNO_integration_for_ML(tno='',clones=2,datadir='./',archivefile=None,logfile=None):
    '''
    '''

    if(clones==2):
        find_3_sigma=True
    else:
        find_3_sigma=False

    today = date.today()
    datestring = today.strftime("%b-%d-%Y")

    flag, epoch, sim = run_reb.initialize_simulation(planets=['jupiter', 'saturn', 'uranus', 'neptune'],
                          des=tno, clones=clones, find_3_sigma=find_3_sigma)

<<<<<<< Updated upstream
    sim_2 = sim.copy()

    shortarchive = datestring + "-" + tno + "-short-archive.bin"
    flag, sim = run_reb.run_simulation(sim,tmax=0.5e6,tout=50.,filename=shortarchive,deletefile=True)
    
    longarchive = datestring + "-" + tno + "-long-archive.bin"
    flag, sim_2 = run_reb.run_simulation(sim_2,tmax=10e6,tout=1000.,filename=longarchive,deletefile=True)
=======

    features = run_sim_for_TNO_ML(sim,des=tno,clones=clones,datadir=datadir,archivefile=archivefile)

    return features



def run_sim_for_TNO_ML(sim, des='',clones=0,datadir='./',archivefile=None):
    '''
    '''

    if(archivefile == None):
        archivefile = datadir + des + '-simarchive.bin'


    #run the short integration
    flag, sim = run_reb.run_simulation(sim,tmax=0.5e6,tout=50.,filename=archivefile,deletefile=True)
    #read the short integration
    flag, a_short, ec_short, inc_short, node_short, peri_short, ma_short, t_short = tools.read_sa_for_sbody(
            sbody=des,archivefile=archivefile,nclones=clones)
    pomega_short = peri_short+ node_short 
    q_short = a_short*(1.-ec_short)
    
    flag, apl, ecpl, incpl, nodepl, peripl, mapl, tpl = tools.read_sa_by_hash(obj_hash='neptune',archivefile=archivefile)
    flag, xr, yr, zr, vxr, vyr, vzr, tr = tools.calc_rotating_frame(sbody=des, planet='neptune', 
                                                                    archivefile=archivefile, nclones=clones)
    rrf_short = np.sqrt(xr*xr + yr*yr + zr*zr)
    phirf_short = np.arctan2(yr, xr)
    tiss_short = apl/a_short + 2.*np.cos(inc_short)*np.sqrt(a_short/apl*(1.-ec_short*ec_short))
>>>>>>> Stashed changes

    #continue at lower resolution to 10 Myr
    flag, sim = run_reb.run_simulation(sim,tmax=10e6,tout=1000.,filename=archivefile,deletefile=False)

<<<<<<< Updated upstream
    flag, a, ec, inc, node, peri, ma, t = tools.read_sa_for_sbody(sbody=tno,archivefile=shortarchive,nclones=clones)
=======
    #read the new part of the integration
    flag, a, ec, inc, node, peri, ma, t = tools.read_sa_for_sbody(sbody=des,archivefile=archivefile,nclones=clones,tmin=0.501e6)
>>>>>>> Stashed changes
    pomega = peri+ node 
    flag, apl, ecpl, incpl, nodepl, peripl, mapl, tpl = tools.read_sa_by_hash(obj_hash='neptune',archivefile=archivefile,tmin=0.501e6)
    q = a*(1.-ec)
<<<<<<< Updated upstream
    flag, xr, yr, zr, vxr, vyr, vzr, tr = tools.calc_rotating_frame(sbody=tno, planet='neptune', 
                                                                    archivefile=shortarchive, nclones=clones)
=======
    flag, xr, yr, zr, vxr, vyr, vzr, tr = tools.calc_rotating_frame(sbody=des, planet='neptune', 
                                                                    archivefile=archivefile, nclones=clones,tmin=0.501e6)
>>>>>>> Stashed changes
    rrf = np.sqrt(xr*xr + yr*yr + zr*zr)
    phirf = np.arctan2(yr, xr)
    tiss = apl/a + 2.*np.cos(inc)*np.sqrt(a/apl*(1.-ec*ec))

<<<<<<< Updated upstream
    flag, l_a, l_ec, l_inc, l_node, l_peri, l_ma, l_t = tools.read_sa_for_sbody(sbody=tno,archivefile=longarchive,nclones=clones)
    l_pomega = l_peri+ l_node 
    flag, apl, ecpl, incpl, nodepl, peripl, mapl, tpl = tools.read_sa_by_hash(obj_hash='neptune',archivefile=longarchive)
    l_q = l_a*(1.-l_ec)
    flag, xr, yr, zr, vxr, vyr, vzr, tr = tools.calc_rotating_frame(sbody=tno, planet='neptune', 
                                                                    archivefile=longarchive, nclones=clones)
    l_rrf = np.sqrt(xr*xr + yr*yr + zr*zr)
    l_phirf = np.arctan2(yr, xr)
    l_tiss = apl/l_a + 2.*np.cos(l_inc)*np.sqrt(l_a/apl*(1.-l_ec*l_ec))
=======
    features = [None]*(clones+1)
>>>>>>> Stashed changes

    for n in range(0,clones+1):
            
        #concatenate the downsampled short integration with the rest of the long integration
        a_long = np.concatenate((a_short[n,::20],a[n,:]))
        ec_long = np.concatenate((ec_short[n,::20],ec[n,:]))
        inc_long = np.concatenate((inc_short[n,::20],inc[n,:]))
        node_long = np.concatenate((node_short[n,::20],node[n,:]))
        peri_long = np.concatenate((peri_short[n,::20],peri[n,:]))
        pomega_long = np.concatenate((pomega_short[n,::20],pomega[n,:]))
        t_long = np.concatenate((t_short[::20],t))
        q_long = np.concatenate((q_short[n,::20],q[n,:]))
        rrf_long = np.concatenate((rrf_short[n,::20],rrf[n,:]))
        phirf_long = np.concatenate((phirf_short[n,::20],phirf[n,:]))
        tiss_long = np.concatenate((tiss_short[n,::20],tiss[n,:]))
    


        features[n] = calc_ML_features(t_long,a_long,ec_long,inc_long,node_long,peri_long,
                                    pomega_long,q_long,rrf_long,phirf_long,tiss_long,
                                    t_short,a_short[n,:],ec_short[n,:],inc_short[n,:],
                                    node_short[n,:],peri_short[n,:],pomega_short[n,:],q_short[n,:],
                                    rrf_short[n,:],phirf_short[n,:],tiss_short[n,:])
   
    return features



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

<<<<<<< Updated upstream
=======
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
    x_min = np.amin(x)
    x_max = np.amax(x)
    x_mean = np.mean(x)
    x_std = np.std(x)
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
    a_sb = data_sb['a'].to_numpy()
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



##########################################################################################################
##########################################################################################################

def farey_tree(num, denom, prmin, prmax):
    order_max = 20
    # Initialize fractions
    flag = 0
    oldnum = num.copy()  
    olddenom = denom.copy()
    nfractions = len(oldnum)
    if nfractions == 1:
        # Only one fraction left, can't keep building the tree
        return flag, num, denom,np.array([0.]),np.array([0.]), 0
        
    # the next layer in the farey tree will have nfraction-1 new fractions
    newnum = np.zeros(nfractions-1)  
    newdenom = np.zeros(nfractions-1)

    # the full set of numbers will have 2*nfractions -1 entries
    num = np.zeros(2*nfractions-1)  
    denom = np.zeros(2*nfractions-1)
    
    nn = 0
    new_n = 0
    for n in range(0,nfractions-1):
        num[nn] = oldnum[n]
        denom[nn] = olddenom[n] 
        nn+=1
        num[nn] = oldnum[n] + oldnum[n + 1]
        denom[nn] = olddenom[n] + olddenom[n + 1]
        if(num[nn]<=order_max):
            nn+=1
            newnum[new_n] = oldnum[n] + oldnum[n + 1]
            newdenom[new_n] = olddenom[n] + olddenom[n + 1]
            new_n+=1


    num[nn] = oldnum[nfractions-1]
    denom[nn] = olddenom[nfractions-1]

    if(new_n <1):
        flag = 0
    else:
        flag = 1
    
    newnum = newnum[0:new_n]
    newdenom = newdenom[0:new_n]
    num = num[0:nn+1]
    denom = denom[0:nn+1]
    
    
    left = 0
    right = nn+1
    for n in range(0,nn):
        if(prmin > float(num[n]/denom[n])):# and left ==0):
            left = n

    for n in range(nn,0,-1):
        if(prmax < float(num[n]/denom[n])):# and right==nn):
            right = n+1
    num = num[left:right]
    denom = denom[left:right]

    new_check_q = np.empty(0)
    new_check_p = np.empty(0)

    for n in range(0,new_n):
        pr = float(newnum[n]/newdenom[n])
        if(pr>=prmin and pr<=prmax):
            new_check_q = np.append(new_check_q,int(newnum[n]))
            new_check_p = np.append(new_check_p,int(newdenom[n]))
            

    n_check = len(new_check_p)
    return flag, num, denom,new_check_q, new_check_p, n_check 





>>>>>>> Stashed changes
