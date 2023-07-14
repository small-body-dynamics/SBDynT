import rebound
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '../src')
import os
import horizons_api
import pandas as pd

filetype = 'Distance'

if filetype == 'Asteroids':
    astdys = pd.read_csv('data_files/astdys_ast.csv')
elif filetype == 'TNOs':
    astdys = pd.read_csv('data_files/astdys_tnos.csv')
elif filetype == 'Distance':
    astdys = pd.read_csv('data_files/Distance.txt')

clones = 0 
#for i in range(len(astdys)):
#for i in range(10):
#objnum = int(sys.argv[1])
#print(objnum)
vals = np.zeros((1,8))
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

#arange = range(300,340)
#objects = ['D4860', 'K13SA0W', 'I3595', 'K20K53G', 'c5437', 'K15G57B', 'o0839', 'K15G58Z', 'q3615', 'K15RR9T', 'q3615', 'K15VG9A', 'q3780', 'K15G58A', 'r1076', 'K20B63Q', 'r5018', 'K21D15N', 'r6919', 'K21L43Y', 'v8833', 'z7103', 'y9222', 'K01QT7Z', 'z2026', 'K15VH3F', 'z2049', 'K15VH0V', 'z2092', 'K15VH3B', 'z2157', 'K15G57G', 'z2174', 'K16R82H', 'z2213', 'K02PH0Z', 'z2333', 'K01QT7Z', 'z2333', 'K15VG8R', 'z2388', 'K13SA0X', 'z2549', 'K15VH2Z', 'z2573', 'K03Q90T', 'z2581', 'K15Ta1P', 'J95K01J', 'K04O12L', 'J99D08E', 'K02G33B', 'J99D08L', 'K01XP4W', 'J99H11S', 'K02G32K', 'J99O03Z', 'K13SA2T', 'K00CA4M', 'K02CF4Y', 'K00CB4N', 'K15VG9S', 'K00F08H', 'K04M08T', 'K00F53S', 'K13SA1C', 'K00GE6X', 'K01OA9G', 'K00P30M', 'K09M10A', 'K00P30M', 'K15VH3M', 'K00P30N', 'K04XJ0X', 'K00P30N', 'K15RS0D', 'K00QM6F', 'K02VD1C', 'K00SX1G', 'K01QT7V', 'K00SX1G', 'K02TU1A', 'K00W12V', 'K03Q91D', 'K01K76Y', 'K15VG5N', 'K01OA8K', 'K13SA0V', 'K01OA8K', 'K15G59C', 'K01OA8Y', 'K13V46L', 'K01OA8Z', 'K05JH9O', 'K01OA8Z', 'K15G56M', 'K01QT7Z', 'K15RR9P', 'K01RE3W', 'K16R82S', 'K02CF4S', 'K13SA1F', 'K02PH0Y', 'K15VH1Y', 'K03H57H', 'K03YH9J', 'K03H57H', 'K07DA1S', 'K03Q91D', 'K15VH2X', 'K03Q91F', 'K15VH3C', 'K03Q91L', 'K15VH3A', 'K04H79K', 'K15G57A', 'K04L32W', 'K21D15Q', 'K04PA7X', 'K07H90V', 'K04PB7W', 'K13TH2L', 'K04U10D', 'K16R82G', 'K04V75Z', 'K20K56A', 'K05B49W', 'K15G56X', 'K05P23H', 'K13RC4J', 'K06A98N', 'K15VH0U', 'K06QI1B', 'K16R82R', 'K06QI1O', 'K15G58C', 'K06QI1O', 'K15VH2F', 'K06UW1K', 'K15VH0O', 'K06UW1S', 'K15G56W', 'K07C66J', 'K15VH1Y', 'K13RF6O', 'K15VH1A', 'K13RF8T', 'K13TM9S', 'K13SA0V', 'K15VH2Z', 'K13SB2M', 'K15VH3G', 'K13TH2L', 'K15G58L', 'K13TI7H', 'K15VH1O', 'K13TI7U', 'K13TI8B', 'K13TM7M', 'K13U18W', 'K13TM8S', 'K15VG9S', 'K13TM8V', 'K20K54J', 'K13TM9O', 'K15VH2Q', 'K13U17L', 'K15VG8Y', 'K13U17R', 'K15VH2A', 'K13U17W', 'K15VG8P', 'K14UR7N', 'K21L43U', 'K15G56X', 'K15VG9B', 'K15G57T', 'K15G58T', 'K15G57T', 'K15VH3F', 'K15G57U', 'K15G58R', 'K15G57U', 'K15VH1B', 'K15G57Z', 'K15VH2E', 'K15G58Y', 'K15VH2H', 'K15G58Y', 'K15VH3G', 'K15G59H', 'K15VH3N', 'K15VG9K', 'K15VH3M', 'K19P02R', 'K19Q06R', 'K20K54F', 'K20K55C']
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
    
