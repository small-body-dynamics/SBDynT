import sys
import os
import numpy as np
import pandas as pd
import rebound
import run_reb
import horizons_api
import tools

import schwimmbad
import functools


def integrate(objname, tmax=1e7, tout=1e3, objtype='Single'):
    try:
        # Construct the file path
        file = '../data/' + objtype + '/' + str(objname)
        
        # Load the simulation from the archive
        print(file)
        sim2 = rebound.Simulation(file + "/archive.bin")
        
        # Uncomment if you need to print simulation information
        # print(sim2, sim2.particles)
        
    except Exception as error:
        # Raise a specific exception with an informative error message
        raise ValueError(f"Failed to integrate {objtype} {objname}. Error: {error}")

    # Rest of the integration code

    sim = run_reb.run_simulation(sim2, tmax=tmax, tout=tout, filename=file + "/archive.bin", deletefile=True)

    
def integrate_multi(filename,tmax = 1e7,tout=1e3):
    names_df = pd.read_csv('../data/data_files/' + filename + '.csv')
        
        # Iterate over each object and integrate
    for i in range(len(names_df)):
        print('Obj #',i)
        integrate(str(names_df['Name'].iloc[i]),tmax,tout,filename)
    
if __name__ == "__main__":
    
    from schwimmbad import MPIPool
    #print('schwimmbad in')
    with MPIPool() as pool:
        #print('mpipool pooled')    
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    # Check if the required command-line arguments are provided
        if len(sys.argv) < 2:
            print("Usage: python integrate.py <Filename>")
            sys.exit(1)

        objtype = str(sys.argv[1])
        # Load data file for the given objtype
        names_df = pd.read_csv('../data/data_files/' + objtype + '.csv')
        
        run = functools.partial(integrate, tmax=1e8, tout=1e3, objtype=objtype)

        des = np.array(names_df['Name'])

        #j = range(22,23)
        #begin = datetime.now()
        pool.map(run, des)

            