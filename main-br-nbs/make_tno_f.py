import sys
sys.path.insert(0, '../src')
import run_reb
import rebound
import numpy as np
import horizons_api
import tools
import pandas as pd
import os

astdys = pd.read_csv('astdys_tnos.csv')

#for i in range(len(astdys)):
#for i in range(10):
objnum = int(sys.argv[1])
print(objnum)
objname = astdys['Name'].iloc[objnum]
filename = 'TNOs/' + objname
if not os.path.isdir(filename):
    os.mkdir(filename)
sbody = objname    
sim= rebound.Simulation()
flag, epoch, sim = run_reb.initialize_simulation(planets=['jupiter','saturn','uranus','neptune'],des=objname,clones=0)
com = sim.calculate_com()
p = sim.particles[sbody+"_bf"]
o = p.calculate_orbit(com)
r2d = 180./np.pi
    
tmax = 1e8
tout = 1e4

sim = run_reb.run_simulation(sim, tmax=tmax, tout=tout,filename=filename+"/archive.bin",deletefile=True,mindist=20.)

    
