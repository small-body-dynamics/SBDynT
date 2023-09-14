import sys
import commentjson as json
sys.path.insert(0, '../src')
import run_reb
import rebound
import numpy as np
import horizons_api
import tools
import pandas as pd
import os


class ReadJson(object):
    def __init__(self, filename):
        print('Read the runprops.txt file')
        self.data = json.load(open(filename))
    def outProps(self):
        return self.data
objtype = str(sys.argv[1])

if objtype == "Generic":
    objname = str(sys.argv[2])
    sbody = objname
else:
    objnum = int(sys.argv[2])
    names_df = pd.read_csv('data_files/'+objtype+'_data.csv')
    objname = str(names_df['Name'].iloc[objnum])
    sbody = objname
    
sim= rebound.Simulation()
try:
    filetype = 'Sims/' + objtype + '/' + objname
    if not os.path.isdir(filetype):
        os.mkdir(filetype)
    runtype = str(sys.argv[3])
    if runtype == '4planet':
        flag, epoch, sim = run_reb.initialize_simulation(planets=['jupiter','saturn','uranus','neptune'],des=objname,clones=0, folder = objtype)
        print(flag, epoch, sim)
    elif runtype == '8planet':
        flag, epoch, sim = run_reb.initialize_simulation(planets=['mercury','venus','earth','mars','jupiter','saturn','uranus','neptune'],des=objname,clones=0, folder = objtype)
        print(flag, epoch, sim)    
    
    com = sim.calculate_com()
    p = sim.particles[sbody+"_bf"]
except:
    print(runtype)
    runprops = {}
    runprops['objname'] = objname
    runprops['err_message'] = 'Simulation failed during initialization. Might not be findable in JPL Horizons'
    print(runprops.get('err_message'))
    runprops['run_success'] = False
    runpath = 'Sims/'+ objtype + '/'+objname+'/runprops.txt'
    with open(runpath, 'w') as file:
        file.write(json.dumps(runprops, indent = 4))
    sys.exit()


o = p.calculate_orbit(com)
r2d = 180./np.pi
    
tmax = 1e7
tout = 1e3

runprops = {}
runprops['tmax'] = tmax
runprops['tout'] = tout
runprops['objname'] = objname
runprops['runtype'] = runtype
runprops['run_success'] = True

runpath = filetype+'/runprops.txt'
with open(runpath, 'w') as file:
    file.write(json.dumps(runprops, indent = 4))

sim = run_reb.run_simulation(sim, tmax=tmax, tout=tout,filename=filetype+"/archive.bin",deletefile=True,mindist=20.)
