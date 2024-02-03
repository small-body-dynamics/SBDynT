import sys
import os
import numpy as np
import pandas as pd
import rebound
import run_reb
import horizons_api
import tools

def integrate(objtype, objnum):
    try:
        # Construct the file path
        filetype = '../data/' + objtype + '/' + str(objnum)
        
        # Load the simulation from the archive
        sim2 = rebound.Simulation(filetype + "/archive.bin")
        
        # Uncomment if you need to print simulation information
        # print(sim2, sim2.particles)
        
    except Exception as error:
        # Raise a specific exception with an informative error message
        raise ValueError(f"Failed to integrate {objtype} {objnum}. Error: {error}")

    # Rest of the integration code
    tmax = 1e6
    tout = 1e3
    sim = run_reb.run_simulation(sim2, tmax=tmax, tout=tout, filename=filetype + "/archive.bin", deletefile=True)

if __name__ == "__main__":
    # Check if the required command-line arguments are provided
    if len(sys.argv) < 2:
        print("Usage: python integrate.py <Filename>")
        sys.exit(1)

    objtype = str(sys.argv[1])

    if objtype == "Generic":
        objname = str(sys.argv[2])
        sbody = objname
        # Add specific handling for Generic type if needed
    else:
        # Load data file for the given objtype
        names_df = pd.read_csv('../data/data_files/' + objtype + '_data.csv')
        
        # Iterate over each object and integrate
        for i in range(len(names_df)):
            print('Obj #',i)
            integrate(objtype, i)
            