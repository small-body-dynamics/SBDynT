import sys
sys.path.insert(0, '../src')
import run_reb
import rebound
import numpy as np
import horizons_api
import tools
import pandas as pd
import os

astdys = pd.read_csv('data_files/astdys_ast.csv')

#for i in range(len(astdys)):
#for i in range(10):
objnum = int(sys.argv[1])
print(objnum)
objname = str(astdys['Name'].iloc[objnum])
filename = 'Asteroids/' + objname
if not os.path.isdir(filename):
    os.mkdir(filename)
sbody = objname    
sim= rebound.Simulation()
flag, epoch, sim = run_reb.initialize_simulation(planets=['mercury','venus','earth','mars','jupiter','saturn','uranus','neptune'],des=objname,clones=0)
com = sim.calculate_com()
p = sim.particles[sbody+"_bf"]
o = p.calculate_orbit(com)
r2d = 180./np.pi
    
tmax = 5e6
tout = 1e2

sim = run_reb.run_simulation(sim, tmax=tmax, tout=tout,filename=filename+"/archive.bin",deletefile=True,mindist=20.)

    
