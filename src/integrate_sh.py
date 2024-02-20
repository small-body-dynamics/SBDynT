import sys
import os
import numpy as np
import pandas as pd
import rebound
import run_reb
import horizons_api
import tools

def integrate(objname, objtype):
    try:
        # Construct the file path
        file = '../data/' + objtype + '/' + str(objname)
        
        # Load the simulation from the archive
        print(file)
        sim2 = rebound.Simulation(file + "/archive.bin")
        sma = sim2[0].particles[str(objname)].a
        
        if sma > 25:
            tmax = 1e8
            tout = 1e3
        else:
            tmax = 1e7
            tout = 1e3
        
        # Uncomment if you need to print simulation information
        # print(sim2, sim2.particles)
        
    except Exception as error:
        # Raise a specific exception with an informative error message
        raise ValueError(f"Failed to integrate {objtype} {objname}. Error: {error}")

    # Rest of the integration code

    sim = run_reb.run_simulation(sim2, tmax=tmax, tout=tout, filename= "/tmp/archive_$CASE_NUM.bin", deletefile=True)

if __name__ == "__main__":
    # Check if the required command-line arguments are provided
    if len(sys.argv) < 2:
        print("Usage: python integrate.py <Filename>")
        sys.exit(1)

    objtype = str(sys.argv[1])
    objnum = sys.argv[2]
    
    integrate(objnum, objtype)