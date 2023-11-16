import rebound
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '../src')
import os
import horizons_api
import pandas as pd

filetype = str(sys.argv[1])

clones = 1 
#for i in range(len(astdys)):
#for i in range(10):
#objnum = int(sys.argv[1])
#print(objnum)
vals = np.zeros((clones+1,8))
plan_vals = np.zeros((1,64))
horizon_data = pd.DataFrame(vals,columns=['flag','epoch','sbx','sby','sbz','sbvx','sbvy','sbvz'])
planet_id = {1: 'mercury', 2: 'venus', 3:'earth', 4:'mars', 5: 'jupiter', 6 : 'saturn', 7 : 'uranus', 8 : 'neptune'}
   
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
    
horizon_planets = pd.DataFrame(plan_vals, columns=plan_cols)    
if filetype != 'Generic':
    astdys = pd.read_csv('data_files/'+filetype+'_data.csv')
    
    #print(astdys)
    for i in range(1465,len(astdys)):
    #for i in range(1465,1500):
    #for i in arange:
        #print(i)
        objname = str(astdys['Name'].iloc[i])
        print(objname)
        #objname = str(objects[i])
        filename = 'Sims/'+filetype + '/' + objname
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
        
        horizon_data = pd.DataFrame(vals,columns=['flag','epoch','sbx','sby','sbz','sbvx','sbvy','sbvz'])
        flag, epoch, sbx, sby, sbz, sbvx, sbvy, sbvz = horizons_api.query_sb_from_jpl(des=des,clones=clones)
        #print(sbx)
        #print(horizon_data['sbx'])
        horizon_data['flag'][:] = flag
        horizon_data['epoch'][:] = epoch
        horizon_data['sbx'] = sbx
        horizon_data['sby'] = sby
        horizon_data['sbz'] = sbz
        horizon_data['sbvx'] = sbvx
        horizon_data['sbvy'] = sbvy
        horizon_data['sbvz'] = sbvz
        
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
            horizon_planets['vy_'+str(pl)] = vy
            horizon_planets['vz_'+str(pl)] = vz
    
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
        
    
elif filetype == "DEEP":
    astdys = pd.read_csv('data_files/'+filetype+'/_data.csv')
    
    print(astdys)
    for i in range(len(astdys)):
    #for i in range(len(objects)):
    #for i in arange:
        #print(i)
        objname = str(astdys['Name'].iloc[i])
        print(objname)
        #objname = str(objects[i])
        filename = filetype + '/' + objname
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
        horizon_data['flag'][0] = flag
        horizon_data['epoch'][0] = epoch
        horizon_data['sbx'][0] = sbx
        horizon_data['sby'][0] = sby
        horizon_data['sbz'][0] = sbz
        horizon_data['sbvx'][0] = sbvx
        horizon_data['sbvy'][0] = sbvy
        horizon_data['sbvz'][0] = sbvz
        
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
            horizon_planets['vy_'+str(pl)] = vy
            horizon_planets['vz_'+str(pl)] = vz
    
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
        
    
    
    #vals = [flag,epoch,sbx,sby,sbz,sbvx,sbvy,sbvz]
    horizon_data = pd.DataFrame(vals,columns=['flag','epoch','sbx','sby','sbz','sbvx','sbvy','sbvz'])
    horizon_planets = pd.DataFrame(plan_vals, columns=plan_cols)
    planet_id = {1: 'mercury', 2: 'venus', 3:'earth', 4:'mars', 5: 'jupiter', 6 : 'saturn', 7 : 'uranus', 8 : 'neptune'}
    
    print(astdys)
    for i in range(len(astdys)):
    #for i in range(len(objects)):
    #for i in arange:
        #print(i)
        objname = str(astdys['Name'].iloc[i])
        print(objname)
        #objname = str(objects[i])
        filename = filetype + '/' + objname
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
        horizon_data['flag'] = [flag]
        horizon_data['epoch'] = epoch
        horizon_data['sbx'] = sbx
        horizon_data['sby'] = sby
        horizon_data['sbz'] = sbz
        horizon_data['sbvx'] = sbvx
        horizon_data['sbvy'] = sbvy
        horizon_data['sbvz'] = sbvz
        
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
            horizon_planets['vy_'+str(pl)] = vy
            horizon_planets['vz_'+str(pl)] = vz
    
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