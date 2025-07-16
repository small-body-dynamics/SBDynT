import numpy as np
import pandas as pd
import os
import rebound
os.chdir('../src')

import tools
import run_reb
import hard_coded_constants as C

def build_init_files(filetype,epoch,obj_num=1):
    data = pd.read_csv('../data/data_files/'+str(filetype)+'.csv')
    filename=str(filetype)
    planet_id = {1: 'mercury', 2: 'venus', 3: 'earth', 4: 'mars', 5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune'}
    planet_id = {5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune'}
    clones=0
    GM_SS = C.SS_GM[0]
    AU = 1.496e8
    yr = 365*24*60*60
    dt = C.dt
    GM_new = GM_SS/AU**3*yr**2
    clones=0
    x = []
    y = []
    z = []
    vx = []
    vy = []
    vz = []
    objnames = []

    sim, sx, sy, sz, svx, svy, svz = run_reb.initialize_simulation_from_sv(planets=list(planet_id.values()), des='0', clones=clones,sb=[epoch,x,y,z,vx,vy,vz],return_sxyz=True)

    for i, objname in enumerate(data['Name']):   
    #for i, objname in enumerate(data['Name'].iloc[34200:]):     
        #if i%1000:
        #    print(i)
        if len(x) == 0:
            name1 = objname
            sim2 = sim.copy()
        objnames.append(objname)                          
        smas = data['a'].iloc[i]
        eccs = data['e'].iloc[i]
        incs = data['i'].iloc[i]/180*np.pi
        aops = data['aop'].iloc[i]/180*np.pi
        lans = data['lan'].iloc[i]/180*np.pi
    #epoch = data['epoch'].iloc[i]
    #epoch = data['epoch'].iloc[i]
        #epoch=2453157.5
    #Tps = data['Tps'].iloc[i]
    #P = smas**1.5*365*24

    #M = 2*np.pi*(epoch-Tps)/P
        M = data['M'].iloc[i]/180*np.pi
        flag, sbx, sby, sbz, sbvx, sbvy, sbvz = tools.aei_to_xv(GM=GM_new, a=smas, e=eccs, inc=incs, node=lans, argperi=aops, ma=M)
        x.append(sx)
        y.append(sy)
        z.append(sz)
        vx.append(svx)
        vy.append(svy)
        vz.append(svz)

        sbx += sx; sby += sy; sbz += sz
        sbvx += svx; sbvy += svy; sbvz += svz
        #sbhash = str(des) 
        #print(des)
        sbhash = str(objname) 
        sim2.add(m=0., x=sbx, y=sby, z=sbz,
                vx=sbvx, vy=sbvy, vz=sbvz, hash=sbhash)

        sim2.move_to_com()
        
        if (len(x)%obj_num == 0) or (i == len(data)):
            objs = str(name1)+'_'+str(objname)
            
            obj_directory = '../data/'+filename+'/'+str(objs)
            print(obj_directory)
            os.makedirs(obj_directory, exist_ok=True)
            #epochin = np.ones(len(x))*epoch
            '''
            epochin=epoch
            x=np.array(x)
            y=np.array(y)
            z=np.array(z)
            vx=np.array(vx)
            vy=np.array(vy)
            vz=np.array(vz)
            flag, epochout, sim = run_reb.initialize_simulation_from_sv(planets=list(planet_id.values()), des=objnames, clones=clones,sb=[epochin,x,y,z,vx,vy,vz])
            '''
            x = []
            y = []
            z = []
            vx = []
            vy = []
            vz = []
            objnames = []
    
    # Save the initial state to an archive file
            archive_file = os.path.join(obj_directory, "archive_init.bin")
            sim2.save_to_file(archive_file)

build_init_files('CFEPS',2453157.5,20)