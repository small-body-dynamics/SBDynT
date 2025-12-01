import numpy as np
import pandas as pd
import os
import rebound
os.chdir('../src')

import tools
import run_reb
import hard_coded_constants as C

def build_init_files(filetype,epoch,clones=1,cov=''):
    data = pd.read_csv('../data/data_files/'+str(filetype)+'.csv')
    filename=str(filetype)
    #planet_id = {1: 'mercury', 2: 'venus', 3: 'earth', 4: 'mars', 5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune'}
    planet_id = {5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune'}

    GM_SS = C.SS_GM[0]
    AU = 1.496e8
    yr = 365*24*60*60
    dt = C.dt
    GM_new = GM_SS/AU**3*yr**2 #GM in units of AU, yr/ kg
    
    cov_run = False
    cov_mat = []
    if len(cov) > 0:
        cov_data = pd.read_csv(cov,index_col=0)
        cov_run = True
        print(cov_data)
    x = []
    y = []
    z = []
    vx = []
    vy = []
    vz = []
    objnames = []
    
    #for i, objname in enumerate(data['Name']):   
    for i, objname in enumerate(data['OrbitID'].iloc[90:]):    
        print(i)
        if len(x) == 0:
            name1 = objname
        objnames.append(str(objname))                         
        smas = data['a'].iloc[i]
        eccs = data['e'].iloc[i]
        incs = data['inc'].iloc[i]
        aops = data['omega'].iloc[i]
        lans = data['Omega'].iloc[i]
    #epoch = data['epoch'].iloc[i]
    #epoch = data['epoch'].iloc[i]
        #epoch=2453157.5
    #Tps = data['Tps'].iloc[i]
    #P = smas**1.5*365*24

    #M = 2*np.pi*(epoch-Tps)/P
        M = data['M'].iloc[i]
        T = data['T'].iloc[i]
        
        flag, xin, yin, zin, vxin, vyin, vzin = tools.aei_to_xv(GM=GM_new, a=smas, e=eccs, inc=incs, node=lans, argperi=aops, ma=M)
        x.append(xin)
        y.append(yin)
        z.append(zin)
        vx.append(vxin)
        vy.append(vyin)
        vz.append(vzin)
        
        objs = str(name1)
        obj_directory = '../data/'+filename+'/'+str(objs)
        os.makedirs(obj_directory, exist_ok=True)
        #epochin = np.ones(len(x))*epoch
        epochin=epoch
        x=np.array(x)
        y=np.array(y)
        z=np.array(z)
        vx=np.array(vx)
        vy=np.array(vy)
        vz=np.array(vz)

        if cov_run:
            cov_ind = np.where(cov_data['OrbitID'] == objname)[0][0]
            cov_mat = np.array(cov_data.iloc[cov_ind,1:]).reshape(6,6)
        #print(cov_mat)
        #print(objnames)
        flag, epochout, sim = run_reb.initialize_simulation_from_sv(planets=list(planet_id.values()), des=objname, clones=clones,sb=[epochin,x,y,z,vx,vy,vz],sb_cov=cov_mat,mean=[smas,eccs,incs,lans,aops,T])
        x = []
        y = []
        z = []
        vx = []
        vy = []
        vz = []
        objnames = []

        #print(hi)
    # Save the initial state to an archive file
        archive_file = os.path.join(obj_directory, "archive_init.bin")
        sim.save_to_file(archive_file)

#df = pd.read_csv('../data/data_files/a0_orbs.csv')
df = pd.read_csv('../data/data_files/DEEP_fakes.csv')

build_init_files('DEEP_fakes',df['Epoch(JD)'].iloc[0],clones=0,cov='')
print('Done')