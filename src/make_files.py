import rebound
import numpy as np
import sys
sys.path.insert(0, '../src')
import os
import horizons_api
import pandas as pd
import run_reb

filetype = str(sys.argv[1])

clones = 0

# Dictionary for planet IDs
planet_id = {1: 'mercury', 2: 'venus', 3: 'earth', 4: 'mars', 5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune'}

if filetype != 'Generic':
    name_list = pd.read_csv('../data/data_files/' + filetype + '_data.csv')
    
    # Create main data directory if it doesn't exist
    main_data_directory = os.path.join('../data', filetype)
    os.makedirs(main_data_directory, exist_ok=True)

    print(name_list)
    
    for i, objname in enumerate(name_list['Name']):
    #for i in range(len(name_list)):
        #print(objname)

        # Create directory for each object
        obj_directory = os.path.join(main_data_directory, str(i))
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
        flag, epoch, sim = run_reb.initialize_simulation(planets=list(planet_id.values()), des=des, clones=clones)

        # Save the initial state to an archive file
        archive_file = os.path.join(obj_directory, "archive.bin")
        sim.save(archive_file)
        sim = None
