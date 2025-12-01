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
import h5py

def convert(objname, objtype='Single'):
    """
    Integrate the given archive.bin file which has been prepared.

    Parameters:
        objname (str or int): Index of the celestial body in the names file.
        tmax (float): The number of years the integration will run for. Default set for 10 Myr.
        tmin (float): The interval of years at which to save. Default set to save every 1000 years.  
        objtype (str): Name of the file containing the list of names, and the directory containing the archive.bin files. 

    Returns:
        None
        
    """   
    try:
        # Construct the file path
        file = '../data/' + objtype + '/' + str(objname)
        
        # Load the simulation from the archive
        
        flag, a, e, inc, node, aperi, ma, t = tools.read_sa_for_sbody(des = objname, archivefile=file+'/archive.bin', clones = 0, tmax=0.,tmin=-5e9)
        
        f = h5py.File(file+'/archive.hdf5','w')
        f.create_dataset(objname,(7,len(a)))
        f[objname][:] = [a, e, inc, node, aperi, ma, t]
        for i in range(9):
            clonename = 'clone'+str(i+1)
            f.create_dataset(clonename,(7,len(a)))
            f[clonename][:] = [a, e, inc, node, aperi, ma, t]
            
        f.close()
    except Exception as error:
        # Raise a specific exception with an informative error message
        raise ValueError(f"Failed to save {objtype} {objname} as h5df file. Error: {error}")


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
        
        run = functools.partial(convert, objtype=objtype)

        des = np.array(names_df['Name'])
        
        pool.map(run, des)

            
