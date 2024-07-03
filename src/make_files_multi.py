import rebound
import numpy as np
import sys
sys.path.insert(0, '../src')
import os
import horizons_api
import pandas as pd
import run_reb

import schwimmbad
import functools

def make(i,names,rocky_planets=False,clones=0,filename='Single'):
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
        planet_id = {2: 'venus', 3: 'earth', 4: 'mars', 5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune'}
    else:
        planet_id = {5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune'}
        
    des = str(names['Name'].iloc[i])
    #print(des)
    obj_directory = '../data/'+filename+'/'+str(des)
    #print(obj_directory,filename)
    os.makedirs(obj_directory, exist_ok=True)
    
    #print(list(planet_id.values()),str(des),clones)
    #Start Simulation using JPL Horizons
    flag, epoch, sim = run_reb.initialize_simulation(planets=list(planet_id.values()), des=str(des), clones=clones)
    
    #Start simulation using state vector
    
    
    '''
    epoch = 2457388.50000
    x = (names['x'].iloc[i])
    y = (names['y'].iloc[i])
    z = (names['z'].iloc[i])
    vx = (names['vx'].iloc[i])
    vy = (names['vy'].iloc[i])
    vz = (names['vz'].iloc[i])

    sb = [epoch,x,y,z,vx,vy,vz]
    #print(sb)
    flag, epoch, sim = run_reb.initialize_simulation_from_sv(planets=list(planet_id.values()), des=str(des), clones=clones,sb=sb)
    '''
    # Save the initial state to an archive file
    archive_file = os.path.join(obj_directory, "archive_init.bin")
    sim.save(archive_file)
    sim = None



if __name__ == "__main__":
    filetype = str(sys.argv[1])

    clones = 0
    rocky_planets = False

    # Dictionary for planet IDs
    
    from schwimmbad import MPIPool
    #print('schwimmbad in')
    with MPIPool() as pool:
        #print('mpipool pooled')    
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        name_list = pd.read_csv('../data/data_files/' + filetype + '.csv')
    
        # Create main data directory if it doesn't exist
        main_data_directory = os.path.join('../data', filetype)
        os.makedirs(main_data_directory, exist_ok=True)


        multi = functools.partial(make, names=name_list, rocky_planets=rocky_planets, clones=clones, filename=filetype)
        j = range(len(name_list))
        #print(name_list)
        #j = range(22,23)
        #begin = datetime.now()
        pool.map(multi, j)
        