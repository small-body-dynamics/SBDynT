import numpy as np
import pandas as pd
import rebound
import integrate_multi as im
import sys
import os
import functools
import schwimmbad
import run_reb
import json
    
def integrate_chaos(objname, tmax1=5e4, tout1=1e3, tmax2=1e6, tout2=2e4, objtype='Single'):
    """
    Integrate the given archive.bin file which has been prepared.

    Parameters:
        objname (str or int): Index of the celestial body in the names file.
        tmax (float): The number of years the integration will run for. Default set for 10 Myr.
        tmin (float): The interval of years at which to save. Default set to save every 1000 years.  
        objtype (str): Name of the file containing the list of names, and the directory containing the archive.bin files. 

    Returns:
        None
        
    """   
    try:
        #print(objname)
        # Construct the file path
        file = '../data/' + objtype + '/' + str(objname)
        
        # Load the simulation from the archive
        #print(file)
        if os.path.isfile(file+'/archive_init.bin') == False:
            print('No init file') 
            return 0
        sim2 = rebound.Simulation(file + "/archive_init.bin")
        #print('read init')
        sim2.integrator = 'whfast'
        #print(sim2.integrator)
        sim2.ri_whfast.safe_mode = 0

        try:
            earth = sim2[0].particles['earth']
            small_planets_flag = True
        except:
            small_planets_flag = False
        
        clones_flag = False
        #print(small_planets_flag,len(sim2.particles))
        if small_planets_flag and len(sim2.particles) > 10:
            clones_flag = True
            numclones = len(sim2.particles)-10
        elif not small_planets_flag and len(sim2.particles) > 6:
            #print('yes')
            clones_flag = True
            numclones = len(sim2.particles)-6
        else:
            #print('no')
            numclones = 0
        numparts = len(sim2.particles)
        #if numparts <= 6:
        #    print(objname)
        for i in range(numclones):
            sim2.remove(numparts-i-1)
            
        sim2.init_megno()
        #print('init megno')
        #print(sim2.integrator
        
        sim1 = run_reb.run_simulation(sim2, tmax=tmax1, tout=tout1, filename=file + "/archive_chaos_short.bin", deletefile=True,integrator=sim2.integrator)
        print('short made')
        
        
        sim4 = rebound.Simulation(file + "/archive_init.bin")

        print(clones_flag)
        if clones_flag:
            print(objname)
            sim4.integrator = 'whfast'
            #sim4.init_megno()
            sim3 = run_reb.run_simulation(sim4, tmax=tmax2, tout=tout2, filename=file + "/archive_chaos_long.bin", deletefile=True,integrator=sim2.integrator)
        # Uncomment if you need to print simulation information
        # print(sim2, sim2.particles)
        
    except Exception as error:
        # Raise a specific exception with an informative error message
        #raise ValueError(f"Failed to integrate {objtype} {objname}. Error: {error}")
        print(error)
        return 0

    # Rest of the integration code

    return 1
 
def calc_chaos(objname,objtype,prop_vals=None):
    #print(objname)
    file = '../data/' + objtype + '/' + str(objname)
    try:
        short_sim = rebound.Simulationarchive(file + "/archive_chaos_short.bin") 
        MEGNO = short_sim[-1].calculate_megno()
    except:
        MEGNO = 0
    
    if os.path.exists(file + "/archive_chaos_long.bin"):
        long_sim = rebound.Simulationarchive(file + "/archive_chaos_long.bin")

        try:
            earth = archive[0].particles['earth']
            small_planets_flag = True
        except:
            small_planets_flag = False
        
        clones_flag = False
        if small_planets_flag and len(long_sim[-1].particles) > 10:
            clones_flag = True
            numclones = len(sim2.particles)-10
            objnum = 9
        elif not small_planets_flag and len(long_sim[-1].particles) > 6:
            clones_flag = True
            numclones = len(long_sim[-1].particles)-6
            #print('nuclones',numclones)
            objnum = 5
        else:
            numclones = 0
        sb = long_sim[-1].particles[5]
        #print(sb)
            
        diff = np.zeros((numclones,3))        
        CloneDist = np.zeros(4)
        GreatestDist = np.zeros(3)
        
        a_p = np.zeros(len(long_sim))
        e_p = np.zeros(len(long_sim))
        i_p = np.zeros(len(long_sim))
        
        for i in range(len(long_sim)):
            a_p[i] = long_sim[-1].particles[5].a
            e_p[i] = long_sim[-1].particles[5].e
            i_p[i] = long_sim[-1].particles[5].inc
        
        
        for i in range(numclones):          
            a_clone = np.zeros(len(long_sim))
            e_clone = np.zeros(len(long_sim))
            i_clone = np.zeros(len(long_sim))
            
            for j in range(len(long_sim)):
                a_clone[j] = long_sim[j].particles[objnum+i].a
                e_clone[j] = long_sim[j].particles[objnum+i].e
                i_clone[j] = long_sim[j].particles[objnum+i].inc
            
            #print(abs(sb.a - long_sim[-1].particles[objnum+i].a) > GreatestDist[0])
            if np.max(abs(a_p - a_clone)) > GreatestDist[0]:
                GreatestDist[0] = np.max(abs(np.mean(a_p - a_clone)))
            if np.max(abs(e_p - e_clone)) > GreatestDist[1]:
                GreatestDist[1] = np.max(abs(e_p - e_clone))
            if np.max(abs(i_p - i_clone)) > GreatestDist[2]:
                GreatestDist[2] = np.max(abs(np.sin(i_p) - np.sin(i_clone)))

            diff[i][0] = np.mean(((sb.a - long_sim[-1].particles[objnum+i].a)/sb.a)**2)
            diff[i][1] = np.mean((sb.e - long_sim[-1].particles[objnum+i].e)**2)
            diff[i][2] = np.mean((np.sin(sb.inc) - np.sin(long_sim[-1].particles[objnum+i].inc))**2)

        #print(sb.a,sb.e,sb.inc)
        #print(long_sim[-1].particles[objnum+i].a,long_sim[-1].particles[objnum+i].e,long_sim[-1].particles[objnum+i].inc)
        CloneDist[0:3] = np.sqrt(np.nanmean(diff,axis=0))
        if np.isnan(CloneDist[0]):
            print(diff)
    else:
        CloneDist = np.array([0,0,0,0])
        GreatestDist = np.array([0,0,0])
        
    prop_err = np.array([0,0,0])
    prop_delta = np.array([0,0,0])
    if os.path.exists("../data/results/"+objtype+"_prop_elem_multi.csv"):
        prop_elem = pd.read_csv("../data/results/"+objtype+"_prop_elem_multi.csv",index_col=1)
        prop_err = [prop_elem.loc[objname]['RMS_err_a']/prop_elem.loc[objname]['PropSMA'],prop_elem.loc[objname]['RMS_err_e'],prop_elem.loc[objname]['RMS_err_sinI']]
        prop_delta = [prop_elem.loc[objname]['Delta_a']/prop_elem.loc[objname]['PropSMA'],prop_elem.loc[objname]['Delta_e'],prop_elem.loc[objname]['Delta_sinI']]
          
    #print(prop_vals)
    if prop_vals != None:
        #print('making prop_err and delta')
        prop_err = [prop_vals[-8]/prop_vals[5],prop_vals[-7],prop_vals[-6]]
        prop_delta = [prop_vals[-4]/prop_vals[5],prop_vals[-3],prop_vals[-2]]
        
    if os.path.exists("../data/"+objtype+"/"+objname+"/archive.bin"):
        sim = rebound.Simulationarchive("../data/"+objtype+"/"+objname+"/archive.bin")
        a = np.zeros(len(sim))
        e = np.zeros(len(sim))
        inc = np.zeros(len(sim))
        try:
            for i in range(len(sim)):
                a[i] = sim[i].particles[-1].a
                e[i] = sim[i].particles[-1].e
                inc[i] = sim[i].particles[-1].inc        

            num_bins_a = 100
            num_bins_e = 100
            #num_bins_i = 100
            #hist, edges = np.histogramdd((a, e, inc), bins=(num_bins_a, num_bins_e, num_bins_i))
            hist, a_edges, e_edges = np.histogram2d(a, e, bins=[num_bins_a, num_bins_e])
            #print(a_edges)
            #hist, a_edges = np.histogram(a_clone, bins=num_bins_a)
    
            # Normalize to get probabilities
            #print(hist,hist.sum())
            hist = hist / hist.sum()

            # Calculate the Shannon entropy
            
            entropy = -np.sum(hist * np.log10(hist+1e-12))  # adding a small value to avoid log(0)
            #print(hist,hist.sum(),entropy)
            #print(len(sim))
            CloneDist[3] = entropy
        except:
            CloneDist[3] = 0
        
    M_flag = (MEGNO > 2.5)
    P_flag = (prop_err[0] > 0.01)
    D_flag = (CloneDist[0] > 0.01)
    E_flag = (CloneDist[3] < 0.9*3.94)        
        
    return [MEGNO,CloneDist[0],CloneDist[1],CloneDist[2],CloneDist[3],GreatestDist[0],GreatestDist[1],GreatestDist[2],prop_err[0],prop_err[1],prop_err[2],prop_delta[0],prop_delta[1],prop_delta[2],M_flag,P_flag,D_flag,E_flag]
    
if __name__ == "__main__":
    
    from schwimmbad import MPIPool
    #print('schwimmbad in')
    with MPIPool() as pool:
        #print('mpipool pooled')    
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    # Check if the required command-line arguments are provided
        if len(sys.argv) < 2:
            print("Usage: python integrate.py <Filename>")
            sys.exit(1)

        objtype = str(sys.argv[1])
        # Load data file for the given objtype
        names_df = pd.read_csv('../data/data_files/' + objtype + '.csv')
        
        run1 = functools.partial(integrate_chaos, objtype=objtype)

        des = np.array(names_df['Name'])
        #des = np.array(names_df['Name'][:980])
        
        #pool.map(run1, des)
        
        run2 = functools.partial(calc_chaos, objtype=objtype)
        data = pool.map(run2, des)
        
        chaos_df = pd.DataFrame(data,columns=['MEGNO','Div_RMS_a','Div_RMS_e','Div_RMS_sinI','Info Entropy','Delta_a','Delta_e','Delta_sinI','Prop_RMS_a','Prop_RMS_e','Prop_RMS_sinI','Prop_Delta_a','Prop_Delta_e','Prop_Delta_sinI','MEGNO_flag','Proper_SMA_flag','Clone_SMA_flag','Entropy_flag'])
        
        entropy_flag = np.where(chaos_df['Info Entropy'] < 0.9*np.median(chaos_df['Info Entropy']))[0]
        PSMA_flag = np.where(chaos_df['Prop_RMS_a'] > 0.01)[0]
        PSMA_flag = np.where(chaos_df['Div_RMS_a'] > 0.01)[0]
        MEGNO_flag = np.where(chaos_df['MEGNO'] > 2.5)[0]
        
        chaos_df['MEGNO_flag'] = np.zeros(len(data))
        chaos_df['Proper_SMA_flag'] = np.zeros(len(data))
        chaos_df['Entropy_flag'] = np.zeros(len(data))
        chaos_df['Clone_SMA_flag'] = np.zeros(len(data))
        
        chaos_df['MEGNO_flag'].iloc[MEGNO_flag] = 1
        chaos_df['Proper_SMA_flag'].iloc[PSMA_flag] = 1
        chaos_df['Clone_SMA_flag'].iloc[PSMA_flag] = 1
        chaos_df['Entropy_flag'].iloc[entropy_flag] = 1
        
        
        chaos_df.to_csv('../data/results/'+objtype+'_chaos.csv')
        

            
