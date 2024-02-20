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

def make(ind, names, filename, clones=0):
    planet_id = {1: 'mercury', 2: 'venus', 3: 'earth', 4: 'mars', 5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune'}
    des = names['Name'].iloc[ind]
    obj_directory = '../data/'+filename+'/'+str(des)
    os.makedirs(obj_directory, exist_ok=True)
    flag, epoch, sim = run_reb.initialize_simulation(planets=list(planet_id.values()), des=str(des), clones=clones)
    
    # Save the initial state to an archive file
    archive_file = os.path.join(obj_directory, "archive.bin")
    sim.save(archive_file)
    sim = None
    return None



if __name__ == "__main__":
    filetype = str(sys.argv[1])

    clones = 0

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


        multi = functools.partial(make, names=name_list, filename=filetype, clones=clones)
        j = range(len(name_list))
        #j = range(22,23)
        #begin = datetime.now()
        pool.map(multi, j)
        