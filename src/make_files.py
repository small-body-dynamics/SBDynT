import rebound
import numpy as np
import sys
sys.path.insert(0, '../src')
import os
import horizons_api
import pandas as pd
import run_reb


def make(des,rocky_planets=False,clones=0,filename='Single'):
    """
    Initialize a given object's archive.bin Simulation.

    Parameters:
        des (str or int): Name/Designation of the celestial body in the names file.
        rocky_planets (boolean): If True, will initizliae the simulation with all 8 planets. Otherwise will only initizlie the gas giants.
        clones (int): The number of desired clones in the simulation.  
        filename (str): Name of the file containing the list of names, and the directory containing the archive.bin files. 

    Returns:
        None. Produces archive_init.bin in the data/filename directory.
        
    """ 
    if rocky_planets:
        planet_id = {1: 'mercury', 2: 'venus', 3: 'earth', 4: 'mars', 5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune'}
    else:
        planet_id = {5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune'}
    
    obj_directory = '../data/'+filename+'/'+str(des)
    #print(obj_directory,filename)
    os.makedirs(obj_directory, exist_ok=True)
    flag, epoch, sim = run_reb.initialize_simulation(planets=list(planet_id.values()), des=str(des), clones=clones)
    
    # Save the initial state to an archive file
    archive_file = os.path.join(obj_directory, "archive_init.bin")
    sim.save(archive_file)
    sim = None


def make_multi(filename='Single',rockp=False,clones=0):
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
    name_list = pd.read_csv('../data/data_files/' + filename + '.csv')
    
    # Create main data directory if it doesn't exist
    main_data_directory = os.path.join('../data', filename)
    os.makedirs(main_data_directory, exist_ok=True)
    #print(name_list)
      
    for i, objname in enumerate(name_list['Name']):
        #for i in range(len(name_list)):
        #print(objname)
    
        # Create directory for each object
        obj_directory = os.path.join(main_data_directory, str(objname))
        os.makedirs(obj_directory, exist_ok=True)
    
        des = objname
        ntp = 1 + clones
        sbx = np.zeros(ntp)
        sby = np.zeros(ntp)
        sbz = np.zeros(ntp)
        sbvx = np.zeros(ntp)
        sbvy = np.zeros(ntp)
        sbvz = np.zeros(ntp)
    
        # Initialize simulation
        make(str(des),rockp,clones,filename)
    
if __name__ == "__main__":
    filetype = str(sys.argv[1])
    
    if filetype != 'Single':
        if len(sys.argv > 2):
            rockp = sys.argv[2]
        else: 
            rockp = False

        if len(sys.argv > 3):
            clones = int(sys.argv[3])
        else:
            clones = 0
            
        make_multi(filetype,rockp,clones)
           
    else:
        objname = str(sys.argv[2])
        
        if len(sys.argv > 3):
            rockp = sys.argv[3]
        else: 
            rockp = False

        if len(sys.argv > 4):
            clones = int(sys.argv[4])
        else:
            clones = 0
        obj_directory = os.path('../data/Single/'+str(objname))
        os.makedirs(obj_directory, exist_ok=True)
    
        des = objname
        ntp = 1 + clones
        sbx = np.zeros(ntp)
        sby = np.zeros(ntp)
        sbz = np.zeros(ntp)
        sbvx = np.zeros(ntp)
        sbvy = np.zeros(ntp)
        sbvz = np.zeros(ntp)
    
            # Initialize simulation
        make(str(des),rockp,clones)
        
        
