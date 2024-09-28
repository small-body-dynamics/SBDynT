import sys
import os
import numpy as np
import pandas as pd
import rebound
import run_reb
import horizons_api
import tools
import integrate

import schwimmbad
import functools
    
if __name__ == "__main__":
    
    from schwimmbad import MPIPool
    with MPIPool() as pool:   
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        if len(sys.argv) < 2:
            print("Usage: python integrate_mmult.py <Filename>")
            sys.exit(1)

        objtype = str(sys.argv[1])
        names_df = pd.read_csv('../data/data_files/' + objtype + '.csv')
        
        run = functools.partial(integrate.integrate, objtype=objtype)

        des = np.array(names_df['Name'])
        
        pool.map(run, des)

            
