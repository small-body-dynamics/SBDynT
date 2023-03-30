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

astdys = pd.read_csv('astdys_tnos.csv')

#for i in range(len(astdys)):
#for i in range(10):
objnum = int(sys.argv[1])
print(objnum)

if isinstance(objnum,int):
#Produce tno file based on astdys list
    astdys = pd.read_csv('astdys_tnos.csv')
    objname = astdys['Name'].iloc[objnum]

#Produce tno file based on given filename
else if isinstance(objnum,str):
    objname = objnum

filename = 'TNOs/' + objname
if not os.path.isdir(filename):
    os.mkdir(filename)
sbody = objname    
sim= rebound.Simulation()

try:
    flag, epoch, sim = run_reb.initialize_simulation(planets=['jupiter','saturn','uranus','neptune'],des=objname,clones=0)
    com = sim.calculate_com()
    p = sim.particles[sbody+"_bf"]
except:
    runprops = {}
    runprops['objname'] = sbody
    runprops['objnum'] = objnum
    runprops['objtype'] = 'TNO'
    runprops['err_message'] = 'Simulation failed during initialization. Might not be findable in JP Horizons'
    runprops['run_success'] = False
    runpath = 'TNOs/'+sbody+'/runprops.txt'
    with open(runpath, 'w') as file:
        file.write(json.dumps(runprops, indent = 4))
    sys.exit()


o = p.calculate_orbit(com)
r2d = 180./np.pi
    
tmax = 1e8
tout = 1e4

runprops = {}
runprops['tmax'] = 1e8
runprops['tout'] = 1e4
runprops['objname'] = sbody
runprops['objnum'] = objnum
runprops['objtype'] = 'TNO'
runprops['run_success'] = True

runpath = 'TNOs/'+sbody+'/runprops.txt'
with open(runpath, 'w') as file:
    file.write(json.dumps(runprops, indent = 4))

sim = run_reb.run_simulation(sim, tmax=tmax, tout=tout,filename=filename+"/archive.bin",deletefile=True,mindist=20.)
