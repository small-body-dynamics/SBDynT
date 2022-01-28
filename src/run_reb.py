import rebound
import numpy as np
from pathlib import Path
import os


#local 
import horizons_api

def initialize_simulation(planets=['Jupiter','Saturn','Uranus','Neptune'], des='', clones=0):
    '''
    inputs:
        (optional) list of planets - defaults to JSUN
        small body designation
        (optional) number of clones - defaults to none
    outputs:
        flag (integer: 0 if failed, 1 if successful)
        epoch of the simulation start (JD)
        rebound simulation instance with planets and test particles added
    '''
    
    #make all planet names lowercase
    planets = [pl.lower() for pl in planets]


    # initialize simulation variable
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')

    #set up small body variables
    ntp = 1 + clones
    sbx = np.zeros(ntp)
    sby = np.zeros(ntp)
    sbz = np.zeros(ntp)
    sbvx = np.zeros(ntp)
    sbvy = np.zeros(ntp)
    sbvz = np.zeros(ntp)

    #get the small body's position and velocity
    flag, epoch, sbx, sby, sbz, sbvx, sbvy, sbvz = horizons_api.query_sb_from_jpl(des=des,clones=clones)
    if(flag<1):
        print("initialize_simulation failed at horizons_api.query_sb_from_jpl")
        return 0, 0., sim
    
    #set up massive body variables    
    npl = len(planets) + 1 #for the sun

    #define the planet-id numbers used by Horizons for the barycenters of each
    #major planet in the solar system
    planet_id = {1: 'mercury', 2: 'venus', 3:'earth', 4:'mars', 5: 'jupiter', 6:'saturn', 7:'uranus', 8:'neptune'}

    #array of GM values queried January 2022
    #(there isn't a way to get this from Horizons, so we just have to hard code it)
    #values for giant planet systems are from Park et al. 2021 DE440 and DE441, 
    #https://doi.org/10.3847/1538-3881/abd414
    #all in km^3 kg^–1 s^–2
    #G = 6.6743015e-20 #in km^3 kg^–1 s^–2
    SS_GM = np.zeros(9)
    SS_GM[0] = 132712440041.93938 #Sun
    SS_GM[1] = 22031.86855 #Mercury
    SS_GM[2] = 324858.592 #Venus
    SS_GM[3] = 403503.235502 #Earth-Moon
    SS_GM[4] = 42828.375214 #Mars
    SS_GM[5] = 126712764.10 #Jupiter system
    SS_GM[6] = 37940584.8418 #Saturn system
    SS_GM[7] = 5794556.4 #Uranus system
    SS_GM[8] = 6836527.10058 #Neptune system

    #set of reasonable whfast simulation timesteps for each planet
    #(1/20 of its orbital period for terrestrial planets, 1/30 for giants)
    dt = [0.012,0.03,0.05,0.09,0.4,0.98,2.7,5.4]

    #add the mass of any not-included planets to the sun
    msun = SS_GM[0]
    #start with shortest timestep
    sim.dt = dt[0]
    for i in range(1,9):
        if (not(planet_id[i] in planets )):
            msun+=SS_GM[i]
            #make the timestep bigger
            sim.dt = dt[i]

    #sun's augmented mass in solar masses
    msun = msun/SS_GM[0]
    radius = 695700.*6.68459e-9
    sim.add(m=msun,x=0.,y=0.,z=0.,vx=0.,vy=0.,vz=0.,hash='sun')
    
    for pl in planets:
        flag, mass, radius, [x, y, z], [vx, vy, vz] = horizons_api.query_horizons_planets(obj=pl,epoch=epoch)
        if(flag<1):
            print("initialize_simulation failed at horizons_api.query_horizons_planets for ", pl)
            return 0, 0., sim
        sim.add(m=mass,r=radius,x=x,y=y,z=z,vx=vx,vy=vy,vz=vz,hash=rebound.hash(pl))

    sim.N_active = npl

    if(clones>0):
        for i in range(0,ntp):
            if(i==0):
                sbhash = des + '_bf'
            else:
                sbhash = str(des) + '_' + str(i)
            sim.add(m=0.,x=sbx[i],y=sby[i],z=sbz[i],vx=sbvx[i],vy=sbvy[i],vz=sbvz[i],hash=sbhash)
    else:
        sbhash = des + '_bf'
        sim.add(m=0.,x=sbx,y=sby,z=sbz,vx=sbvx,vy=sbvy,vz=sbvz,hash=sbhash)

    sim.move_to_com()


    return 1, epoch, sim




def run_simulation(sim, tmax=0, tout=0,filename="archive.bin",deletefile=True,maxdist=1500,mindist=4.):
    '''
    run a mercurius simulation saving to a simulation archive every tout
    '''
    sim.automateSimulationArchive("archive.bin",interval=tout,deletefile=deletefile)
    sim.integrator = 'mercurius'
    sim.collision = "direct"
    sim.ri_mercurius.hillfac = 3.
    sim.collision_resolve = "merge"
    

    #set up a c-based heartbeat function 
    path1 = str(rebound.__libpath__)
    path2 = str(Path(rebound.__file__).parent / "rebound.h")
    com = 'cp ' + path1 + ' librebound.so'
    os.system(com)
    com = 'cp ' + path2 + ' rebound.h'
    os.system(com)

    if(os.path.exists('heartbeat.c')):
        os.system('rm heartbeat.c')
    if(os.path.exists('heartbeat.o')):
        os.system('rm heartbeat.o')
    if(os.path.exists('heartbeat.so')):
        os.system('rm heartbeat.so')
    
    heartbeat_file = '''
#include "rebound.h"
void heartbeat(struct reb_simulation* r){
	int N = r->N;
	for (int i=N-1;i>=r->N_active;i+=-1){
        double rh = r->particles[i].x*r->particles[i].x;
        rh+= r->particles[i].y*r->particles[i].y;
        rh+=r->particles[i].z*r->particles[i].z;
        rh = sqrt(rh);
        if(rh > '''+str(maxdist)+''' || rh < '''+str(mindist)+'''){
            reb_remove(r, i, 1);
	        FILE* of = fopen("removal-log.txt","a"); 
            fprintf(of,"removing particle %d at time %e\\n",i, r->t);
            fclose(of);
        }
    }
}
'''

    with open('heartbeat.c', mode='a') as file:
        file.write(heartbeat_file)
    os.system("gcc -c -O3 -fPIC heartbeat.c -o heartbeat.o")
    os.system("gcc -L. -shared heartbeat.o -o heartbeat.so -lrebound")
    from ctypes import cdll
    clibheartbeat = cdll.LoadLibrary("heartbeat.so")
    sim.heartbeat = clibheartbeat.heartbeat

    sim.integrate(tmax)
    return sim
