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
        #objname = np.array(names_df['Name'].iloc[:20])
        windows=5
        run = functools.partial(prop_calc, filename=filename,windows=windows)

        #j = range(len(names_df))
        #begin = datetime.now()
        data = pool.map(run, objname)
        print(data,len(data),len(data[0]))
        column_names = ['Objname','ObsSMA','ObsEcc','ObsSin(Inc)','Obs_h','PropSMA','PropEcc','PropSin(Inc)','Prop_h']

        for i in range(windows):
            numrange = str(i)+'_'+str(i+2)+'PE'
            column_names.append(numrange+'_a')
            column_names.append(numrange+'_e')
            column_names.append(numrange+'_sinI')
            column_names.append(numrange+'_h')
            #print(numrange)
        column_names.append('RMS_err_a')
        column_names.append('RMS_err_e')
        column_names.append('RMS_err_sinI')
        column_names.append('RMS_err_h')
        column_names.append('Delta_a')
        column_names.append('Delta_e')
        column_names.append('Delta_sinI')
        column_names.append('Delta_h')
        print(len(column_names))
        data_df = pd.DataFrame(data,columns=column_names)
        data_df.to_csv('../data/results/'+filename+'_prop_elem_multi_1e8_nomax.csv')
        
    
