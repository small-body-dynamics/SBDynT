import sys
import os
import numpy as np
import pandas as pd
import rebound
import run_reb
import horizons_api
import tools

def integrate(objname, tmax=1e7, tout=1e3, objtype='Single'):
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
        print(file)
        sim2 = rebound.Simulation(file + "/archive_init.bin")
        
        # Uncomment if you need to print simulation information
        # print(sim2, sim2.particles)
        
    except Exception as error:
        # Raise a specific exception with an informative error message
        raise ValueError(f"Failed to integrate {objtype} {objname}. Error: {error}")

    # Rest of the integration code

    sim = run_reb.run_simulation(sim2, tmax=-tmax, tout=tout, filename=file + "/archive.bin", deletefile=True)

    
def integrate_multi(filename,tmax = True,tout=1e3):
    names_df = pd.read_csv('../data/data_files/' + filename + '.csv')
    file = str(names_df['Name'].iloc[i])
    sim2 = rebound.Simulation(file + "/archive_init.bin")
    
    if tmax == True:
        rockp = False
        #Check if rocky planets are included. If they are, default tmax = 1e7, otherwise tmax = 1e8.
        try:
            earth = sim2.particles['earth']
            rockp = True
        except:
            rockp = False
        
        if rockp == True:
            tmax = 1e7
        else:
            tmax = 1e8
            
    # Iterate over each object and integrate
    for i in range(len(names_df)):
        print('Obj #',i)
        integrate(str(names_df['Name'].iloc[i]),tmax,tout,filename)
    
if __name__ == "__main__":
    # Check if the required command-line arguments are provided
    if len(sys.argv) < 2:
        print("Usage: python integrate.py <Filename>")
        sys.exit(1)

    objtype = str(sys.argv[1])
    print(objtype)
    if objtype == "Single":
        objname = str(sys.argv[2])
        print(objname)
        sbody = objname
        integrate(objname, objtype=objtype, tmax=1e7,tout=1e2)
        # Add specific handling for Generic type if needed
    else:
        # Load data file for the given objtype
        names_df = pd.read_csv('../data/data_files/' + objtype + '.csv')
        
        # Iterate over each object and integrate
        for i in range(len(names_df)):
            print('Obj #',i)
            integrate_multi(objtype)
            
