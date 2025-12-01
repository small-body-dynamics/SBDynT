import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, '../src')
import rebound
import numpy as np
import horizons_api
import tools
import warnings
warnings.filterwarnings("ignore")
import scipy.signal as signal

import run_reb
import tools

import functools
import schwimmbad

from prop_elem import prop_calc

if __name__ == "__main__":
    filename = str(sys.argv[1])
    
    from schwimmbad import MPIPool
    #print('schwimmbad in')
    with MPIPool() as pool:
        #print('mpipool pooled')    
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        names_df = pd.read_csv('../data/data_files/'+filename+'.csv')
        data = []
        
        objname = np.array(names_df['Name'])
        #objname = np.array(names_df['Name'].iloc[:4])
        windows=5

        fullfile = '../data/'+filename+'/'+str(objname[0])+'/archive.bin'
        home = str(os.path.expanduser("~"))
        fullfile = home+'/nobackup/archive/SBDynT_sims/'+filename+'/'+str(objname[0])+'/archive.bin'
        #fullfile=home+'/../../../hdd/haumea-data/djspenc/SBDynT_sims/'+filename+'/'+str(objname)+'/archive.bin'

        s = rebound.Simulation(fullfile)
        time_archive = s.t

        time_run = -150e6


        if time_run == 0 or abs(time_run) > abs(time_archive):
            time_run = time_archive
        
        rms = True
        run = functools.partial(prop_calc, filename=filename,windows=windows, time_run = time_run, rms = rms)
        #run = functools.partial(prop_calc, filename=filename,windows=windows)


        #j = range(len(names_df))
        #begin = datetime.now()
        data = pool.map(run, objname)
        #print(data,len(data),len(data[0]))
        column_names = ['Objname','ObsSMA','ObsEcc','ObsSin(Inc)','Obs_omega','Obs_Omega','Obs_M','MeanSMA','MeanEcc','MeanSin(Inc)','PropSMA','PropEcc','PropSin(Inc)','Prop_omega','Prop_Omega']

        for i in range(windows):
            numrange = str(i)+'_'+str(i+2)+'PE'
            column_names.append(numrange+'_a')
            column_names.append(numrange+'_e')
            column_names.append(numrange+'_sinI')
            #print(numrange)
        column_names.append('RMS_err_a')
        column_names.append('RMS_err_e')
        column_names.append('RMS_err_sinI')
        column_names.append('Delta_a')
        column_names.append('Delta_e')
        column_names.append('Delta_sinI')
        column_names.append('g("/yr)')
        column_names.append('s("/yr)')
        column_names.append('Res_e')
        column_names.append('Res_I')
        column_names.append('Ecc_0_bin_%')
        column_names.append('Inc_0_bin_%')
        column_names.append('Ecc_osculating_amplitude')
        column_names.append('Sin(Inc)_osculating_amplitude')
        column_names.append('Ecc_filtered_amplitude')
        column_names.append('Sin(Inc)_filtered_amplitude')
        column_names.append('Secular Resonant Angle')
        column_names.append('Median Librating Angle (rad)')
        column_names.append('Angle Entropy')
        column_names.append('Phi Fraction')

        #print(len(column_names))
        #print(data)
        #print(data.shape)
        data_df = pd.DataFrame(data,columns=column_names)
        if rms: rms_str = 'rms'
        else: rms_str = 'std'
        data_df.to_csv('../data/results/'+filename+'_prop_elem_multi_' + rms_str + '_' + str(int(abs(time_run/1e6)))+ 'myr.csv')
        
    
