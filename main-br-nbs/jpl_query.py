import rebound
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '../src')
import os
import horizons_api
import pandas as pd

astdys = pd.read_csv('astdys_tnos.csv')
clones = 0 
#for i in range(len(astdys)):
#for i in range(10):
#objnum = int(sys.argv[1])
#print(objnum)
vals = np.zeros((len(astdys),8))
plan_vals = np.zeros((1,64))

plan_cols = []
for i in range(8):
   plan_cols.append('mass_'+str(i+1))
   plan_cols.append('radius_'+str(i+1))
   plan_cols.append('x_'+str(i+1))
   plan_cols.append('y_'+str(i+1))
   plan_cols.append('z_'+str(i+1))
   plan_cols.append('vx_'+str(i+1))
   plan_cols.append('vy_'+str(i+1))
   plan_cols.append('vz_'+str(i+1))


#vals = [flag,epoch,sbx,sby,sbz,sbvx,sbvy,sbvz]
horizon_data = pd.DataFrame(vals,columns=['flag','epoch','sbx','sby','sbz','sbvx','sbvy','sbvz'])
horizon_planets = pd.DataFrame(plan_vals, columns=plan_cols)
planet_id = {1: 'mercury', 2: 'venus', 3:'earth', 4:'mars', 5: 'jupiter', 6 : 'saturn', 7 : 'uranus', 8 : 'neptune'}

for i in range(len(astdys)):
#for i in range(2):
    objname = astdys['Name'].iloc[i]
    filename = 'TNOs/' + objname
    if not os.path.isdir(filename):
        os.mkdir(filename)
    sbody = objname    
    des=sbody
    ntp = 1 + clones
    sbx = np.zeros(ntp)
    sby = np.zeros(ntp)
    sbz = np.zeros(ntp)
    sbvx = np.zeros(ntp)
    sbvy = np.zeros(ntp)
    sbvz = np.zeros(ntp)

    flag, epoch, sbx, sby, sbz, sbvx, sbvy, sbvz = horizons_api.query_sb_from_jpl(des=des,clones=clones)
    horizon_data['flag'][i] = flag
    horizon_data['epoch'][i] = epoch
    horizon_data['sbx'][i] = sbx
    horizon_data['sby'][i] = sby
    horizon_data['sbz'][i] = sbz
    horizon_data['sbvx'][i] = sbvx
    horizon_data['sbvy'][i] = sbvy
    horizon_data['sbvz'][i] = sbvz
    
    horizon_data.to_csv(filename+'/horizon_data.csv')  


    notplanets = [1,2,3,4]
    planets = [5,6,7,8]
                

    for pl in notplanets:
        flag, mass, radius, [x, y, z], [vx, vy, vz] = horizons_api.query_horizons_planets(obj=planet_id[pl],epoch=epoch)
        horizon_planets['mass_'+str(pl)] = mass
        horizon_planets['radius_'+str(pl)] = radius
        horizon_planets['x_'+str(pl)] = x
        horizon_planets['y_'+str(pl)] = y
        horizon_planets['z_'+str(pl)] = z
        horizon_planets['vx_'+str(pl)] = vx
        horizon_data['vy_'+str(pl)] = vy
        horizon_data['vz_'+str(pl)] = vz

    for pl in planets:
        flag, mass, radius, [x, y, z], [vx, vy, vz] = horizons_api.query_horizons_planets(obj=planet_id[pl], epoch=epoch)
        horizon_planets['mass_'+str(pl)] = mass
        horizon_planets['radius_'+str(pl)] = radius
        horizon_planets['x_'+str(pl)] = x
        horizon_planets['y_'+str(pl)] = y
        horizon_planets['z_'+str(pl)] = z
        horizon_planets['vx_'+str(pl)] = vx
        horizon_planets['vy_'+str(pl)] = vy
        horizon_planets['vz_'+str(pl)] = vz

    horizon_planets.to_csv(filename+'/horizon_planets.csv')
    
