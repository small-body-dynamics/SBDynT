import rebound
import numpy as np
# local
import horizons_api
import tools
from datetime import datetime


def add_planets(sim, planets=['all'],
                epoch=2459580.5):
    """
    inputs:
        sim: empty rebound simulation instance
        planets (optional): string list, list of planet names - defaults to all
        epoch (optional): float, epoch in JD, defaults to Jan 1, 2022       
    outputs:
        flag: integer, 0 if failed, 1 if successful
        sim: rebound simulation instance with sun and planets added
             with adjustments for missing major perturbers
        sx: float, cartesian position correction for missing perturbers (au)
        sy: float, cartesian position correction for missing perturbers (au)
        sz: float, cartesian position correction for missing perturbers (au)
        svx: float, cartesian velocity correction for missing perturbers (au/yr)
        svy: float, cartesian velocity correction for missing perturbers (au/yr)
        svz: float, cartesian velocity correction for missing perturbers (au/yr)
    """
    flag = 0

    #check to see if the sim already has particles in it
    if(sim.N > 0):
        print("run_reb.add_planets failed")
        print("This rebound simulation instance already has particles in it!")
        print("run_reb.add_planets can only accept an empty rebound simulation instance")
        return flag, sim, 0.,0.,0.,0.,0.,0.

    sim.units = ('yr', 'AU', 'Msun')

    # make sure planets is a list and make all planet names lowercase
    if not (type(planets) is list):
        planets = [planets]
    planets = [pl.lower() for pl in planets]
    if(planets == ['outer']):
        planets = ['jupiter', 'saturn', 'uranus', 'neptune']
    if(planets == ['all']):
        planets = ['mercury', 'venus', 'earth', 'mars','jupiter', 'saturn', 'uranus', 'neptune']


    # create an array of planets not included in the simulation
    # will be used to correct the simulation's barycenter for their absence
    notplanets = []
    
    # set up massive body variables
    npl = len(planets) + 1  # for the sun

    # define the planet-id numbers used by Horizons for the barycenters of each
    # major planet in the solar system
    planet_id = {1: 'mercury', 2: 'venus', 3: 'earth', 4: 'mars',
                 5: 'jupiter', 6: 'saturn', 7: 'uranus', 8: 'neptune'}

    # import the following hard-coded constants:
    # Planet physical parameters
    # SS_GM[0:9] in km^3 s^–2
    # SS_r[0:9] all in au
    # reasonable integration timesteps for each planet:
    # dt[0:9]
    import hard_coded_constants as const

    sim.dt = const.dt[0]

    # add the mass of any not-included planets to the sun
    msun = const.SS_GM[0]
    # work from Neptune in to also set the largest allowable timestep
    for i in range(8, 0, -1):
        if (not(planet_id[i] in planets)):
            msun += const.SS_GM[i]
            notplanets.append(planet_id[i])
        else:
            # reset the timestep for that planet
            sim.dt = const.dt[i]

    # sun's augmented mass in solar masses
    msun = msun/const.SS_GM[0]
    # sun's radius in au
    radius = const.SS_r[0]
    sim.add(m=msun, r=radius, x=0., y=0., z=0.,
            vx=0., vy=0., vz=0., hash='sun')

    # set the initial correction for the included planets'
    # position and velocities to zero
    sx = 0.; sy = 0.; sz = 0.; svx = 0; svy = 0.; svz = 0.

    # calculate the correction
    if(len(notplanets) > 0):
        # create a temporary simulation to calculate the barycenter of
        # the sun+not included planets so their mass can be added to
        # the sun in the simulation
        tsim = rebound.Simulation()
        tsim.units = ('yr', 'AU', 'Msun')
        tsim.add(m=1.0, x=0., y=0., z=0., vx=0., vy=0., vz=0.)
        for pl in notplanets:
            pflag, mass, radius, [x, y, z], [vx, vy, vz] = \
                horizons_api.query_horizons_planets(obj=pl, epoch=epoch)
            if(pflag < 1):
                print("run_reb.add_planets failed at \
                    horizons_api.query_horizons_planets for ", pl)
                return flag, sim, 0.,0.,0.,0.,0.,0.
            tsim.add(m=mass, r=radius, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
        # calculate the barycenter of the sun + missing planets
        com = tsim.com()
        # reset the corrections to the positions and velocities
        sx = -com.x; sy = -com.y; sz = -com.z
        svx = -com.vx; svy = -com.vy; svz = -com.vz
    
    #free up the temporary simulation variable
    tsim = None

    # add each included planet to the simulation and correct for the
    # missing planets
    for pl in planets:
        pflag, mass, radius, [x, y, z], [vx, vy, vz] = \
            horizons_api.query_horizons_planets(obj=pl, epoch=epoch)
        if(pflag < 1):
            print("run_reb.add_planets failed failed at \
                  horizons_api.query_horizons_planets for ", pl)
            return flag, sim, 0.,0.,0.,0.,0.,0.
        # correct for the missing planets
        x += sx; y += sy; z += sz
        vx += svx; vy += svy; vz += svz
        sim.add(m=mass, r=radius, x=x, y=y, z=z,
                vx=vx, vy=vy, vz=vz, hash=pl)

    sim.N_active = npl
    flag = 1
    return flag, sim, sx, sy, sz, svx, svy, svz


def initialize_simulation(planets=['all'], des=None, clones=0, cloning_method='Gaussian',
                          datadir='', saveic=False, logfile=False, save_sbdb=False):
    """
    inputs:
        planets (optional): string list, list of planet names - defaults to all
        des: string, small body designation
        clones (optional): integer, number of clones. Defaults to 0
        cloning_method (optional): string,  defaults to standard Guassian sampling
                           if set to 'find_3_sigma' the first two
                           returned clones will be approximately 3-
                           sigma min and max semimajor axis clones
                           if clones>2, the rest will be sampled in a Guassian manner
        datadir (optional): string, path for saving any files produced in this 
                           function; defaults to the current directory
        saveic (optional): boolean or string; 
                           if True:  will save a rebound file with the simulation 
                           state that can be used to restart later either to a default 
                           file name or to a file with the name equal to the string passed
                           (default) if False nothing is saved
        logfile (optional): boolean or string; 
                            if True:  will save some messages to adefault log file name
                            or to a file with the name equal to the string passed or
                            to the screen if 'screen' is passed 
                            (default) if False nothing is saved
        save_sbdb (optional): boolean or string; 
                           if True:  will save a pickle file with the results of the 
                           JPL SBDB query either to a default file name or to a file
                           with the name equal to the string passed
                           (default) if False nothing is saved
                           

    outputs:
        flag: integer, 0 if failed, 1 if successful
        epoch: float, date of the simulation start (JD)
        sim: rebound simulation instance with planets and test particles added
             with adjustments for missing major perturbers
        weights (optional output, triggered by use of non-default cloning_method):
            numpy array of weights for the clones added to the simulation
            in the default sampling method, clones are equally weighted, so we need
            not output weights
            when cloning_method = 'find_3_sigma' and nclones > 2, the weights of the
            two extreme clones are set to 0 and the rest to 1
    """
    
    flag = 0
    epoch = None

    if(des == None):
        print("The designation of the small body must be provided")
        print("failed in horizons_api.initialize_simulation()")
        return flag, epoch, sim

    if(logfile==True):
        logfile = tools.log_file_name(des=des)
    if(datadir and logfile and logfile!='screen'):        
        logfile = datadir + '/' +logfile


    # make sure planets is a list and make all planet names lowercase
    if not (type(planets) is list):
        planets = [planets]
    planets = [pl.lower() for pl in planets]
    if(planets == ['outer']):
        planets = ['jupiter', 'saturn', 'uranus', 'neptune']
    if(planets == ['all']):
        planets = ['mercury', 'venus', 'earth', 'mars','jupiter', 'saturn', 'uranus', 'neptune']


    # initialize simulation variable
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')

    # set up small body variables
    ntp = 1 + clones
    sbx = np.zeros(ntp)
    sby = np.zeros(ntp)
    sbz = np.zeros(ntp)
    sbvx = np.zeros(ntp)
    sbvy = np.zeros(ntp)
    sbvz = np.zeros(ntp)

    # get the small body's position and velocity
    sflag, epoch, sbx, sby, sbz, sbvx, sbvy, sbvz, weights = \
            horizons_api.query_sb_from_jpl(des=des, clones=clones, 
                                           cloning_method=cloning_method,
                                           datadir=datadir, logfile=logfile, 
                                           save_sbdb=save_sbdb)
    if(sflag < 1):
        print("run_reb.initialize_simulation failed at horizons_api.query_sb_from_jpl")
        return flag, 0., sim
    
    if(logfile):
        logmessage = "simulation epoch: " + str(epoch) + "\n"
        tools.writelog(logfile,logmessage)


    # add the planets and return the position/velocity corrections for
    # missing planets
    apflag, sim, sx, sy, sz, svx, svy, svz = add_planets(sim, planets=planets,
                epoch=epoch)
    if(apflag < 1):
        print("run_reb.initialize_simulation failed at run_reb.add_planets")
        return flag, 0., sim
    
    if(clones > 0):
        for i in range(0, ntp):
            if(i == 0):
                #first clone is always just the best-fit orbit
                #and the hash is not numbered
                sbhash = str(des)
            else:
                sbhash = str(des) + '_' + str(i)
            # correct for the missing planets
            sbx[i] += sx; sby[i] += sy; sbz[i] += sz
            sbvx[i] += svx; sbvy[i] += svy; sbvz[i] += svz
            sim.add(m=0., x=sbx[i], y=sby[i], z=sbz[i],
                    vx=sbvx[i], vy=sbvy[i], vz=sbvz[i], hash=sbhash)
    else:
        sbx += sx; sby += sy; sbz += sz
        sbvx += svx; sbvy += svy; sbvz += svz
        sbhash = des 
        sim.add(m=0., x=sbx, y=sby, z=sbz,
                vx=sbvx, vy=sbvy, vz=sbvz, hash=sbhash)

    sim.move_to_com()

    if(saveic):
        if(saveic == True):
            ic_file = tools.ic_file_name(des=des)
        else:
            ic_file = saveic
        if(datadir):
            ic_file = datadir + '/'  +ic_file
        sim.save_to_file(ic_file)
        if(logfile):
            logmessage = "Rebound simulation initial conditions saved to " + ic_file + "\n"
            tools.writelog(logfile,logmessage)
    flag = 1

    if(cloning_method=='Gaussian'):
        return 1, epoch, sim
    else:
        return 1, epoch, sim, weights

def initialize_simulation_at_epoch(planets=['all'], des=None, epoch=2459580.5,
                                   datadir='', saveic=False, logfile=False):
    """
    inputs:
        planets (optional): string list, list of planet names - defaults to all
        des: string or list of strings, small body designation or list of designations
        epoch (optional): float, epoch in JD, defaults to Jan 1, 2022
        datadir (optional): string, path for saving any files produced in this 
                           function; defaults to the current directory
        saveic (optional): boolean or string; 
                           if True:  will save a rebound file with the simulation 
                           state that can be used to restart later either to a default 
                           file name or to a file with the name equal to the string passed
                           (default) if False nothing is saved
        logfile (optional): boolean or string; 
                            if True:  will save some messages to adefault log file name
                            or to a file with the name equal to the string passed or
                            to the screen if 'screen' is passed 
                            (default) if False nothing is saved
        
    outputs:
        flag (integer): 0 if failed, 1 if successful
        epoch: float, epoch of the simulation start (JD)
        sim: rebound simulation instance with planets and test particles added
             with adjustments for missing major perturbers
    """
    flag = 0

    if(des == None):
        print("The designation of one or more small bodies must be provided")
        print("failed in run_reb.initialize_simulation_at_epoch()")
        return flag, 0.,sim

    if(logfile==True):
        logfile = tools.log_file_name(des=des[0])
    if(datadir and logfile and logfile!='screen'):
        logfile = datadir + '/' +logfile

    if(logfile):
        logmessage = "simulation epoch: " + epoch + "\n"
        tools.writelog(logfile,logmessage)

    # make sure planets is a list and make all planet names lowercase
    if not (type(planets) is list):
        planets = [planets]
    planets = [pl.lower() for pl in planets]
    if(planets == ['outer']):
        planets = ['jupiter', 'saturn', 'uranus', 'neptune']
    if(planets == ['all']):
        planets = ['mercury', 'venus', 'earth', 'mars','jupiter', 'saturn', 'uranus', 'neptune']
    
    # initialize simulation variable
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')

    # set up small body variables
    # if the user provided just a single string as the designation
    # turn it into a list first
    if not (type(des) is list):
        des = [des]
    ntp = len(des)
    sbx = np.zeros(ntp)
    sby = np.zeros(ntp)
    sbz = np.zeros(ntp)
    sbvx = np.zeros(ntp)
    sbvy = np.zeros(ntp)
    sbvz = np.zeros(ntp)

    # get the small body's position and velocity
    flag, sbx, sby, sbz, sbvx, sbvy, sbvz = \
        horizons_api.query_sb_from_horizons(des=des, epoch=epoch)
    if (flag < 1):
        print("run_reb.initialize_simulation_at_epoch failed at "
              "horizons_api.query_sb_from_horizons")
        return flag, 0., sim

    # add the planets and return the position/velocity corrections for
    # missing planets
    apflag, sim, sx, sy, sz, svx, svy, svz = add_planets(sim, planets=planets,
                epoch=epoch)
    if(apflag < 1):
        print("run_reb.initialize_simulation_at_epoch failed at run_reb.add_planets")
        return flag, 0., sim


    for i in range(0, ntp):
        sbhash = str(des[i])
        # correct for the missing planets
        sbx[i] += sx; sby[i] += sy; sbz[i] += sz
        sbvx[i] += svx; sbvy[i] += svy; sbvz[i] += svz
        sim.add(m=0., x=sbx[i], y=sby[i], z=sbz[i],
                
                vx=sbvx[i], vy=sbvy[i], vz=sbvz[i], hash=sbhash)

    sim.move_to_com()

    if(saveic):
        if(saveic == True):
            ic_file = tools.ic_file_name(des=des[0])
        else:
            ic_file = saveic
        if(datadir):
            ic_file = datadir + '/' +ic_file
        sim.save_to_file(ic_file)
        if(logfile):
            logmessage = "Rebound simulation initial conditions saved to " + ic_file + "\n"
            tools.writelog(logfile,logmessage)    

    return 1, epoch, sim


def run_simulation(sim, des=None, tmax=0, tout=0, archivefile=None,
                   deletefile=False,integrator='mercurius',
                   datadir='', logfile=False):
    """
    run a simulation saving to a simulation archive every tout
    inputs:
        sim (rebound simulation instance): initialized with all the
            planets and small bodies
        des: string or list of strings, small body designation or list of designations            
        tmax (float; years): simulation stopping time
        tout (float; years): interval for saving simulation outputs to
            the simulation archive file
        archivefile (str; optional): name for the simulation
            archive file that rebound will generate
        deletefile (bool; optional): if set to True and a file with
            the name/path of filename exists, it will be deleted before
            the new simulation archive file is created. The default
            is False, which means new data will be appended to filename
        integrator (rebound integrator; optional): the desired integrator
            rebound will use to integrate from sim.t to sim.tmax. The
            default is mercurius with a direct collision search, hill
            switchover at 3 hill radii, and the collision resolve set to
            merge. Currently, sbdynt is configured to allow the following
            integrator options: whfast, mercurius, ias15
        datadir (optional): string, path for saving any files produced in this 
            function; defaults to the current directory   
        logfile (optional): boolean or string; 
            if True:  will save some messages to adefault log file name
            or to a file with the name equal to the string passed or
            to the screen if 'screen' is passed 
            (default) if False nothing is saved
                          
    outputs:
        flag (int): 0 if something went wrong, 1 if the integration succeded
        sim (rebound simulation instance): contains the full simulation
            state at tmax
    """
    flag = 0
    if(archivefile==None):
        if(des == None):
            print("You must provide either an archivefile name or a designation ")
            print("that can be used to generate a default archivefile name")
            print("failed at run_reb.run_simulation")
            return flag, sim
        archivefile = tools.archive_file_name(des)
    
    if(datadir):
        archivefile = datadir + '/' +archivefile

    if(logfile==True):
        if(des == None):
            print("You must provide either a logfile name (or logfile='screen') or ")
            print("a designation that can be used to generate a default logfile name")
            print("failed at run_reb.run_simulation")
            return flag, sim        
        logfile = tools.log_file_name(des=des)
    if(datadir and logfile and logfile!='screen'):
        logfile = datadir + '/' +logfile


    #check for integrator choice and set any required extra parameters
    if(integrator.lower == 'mercurius'.lower):
        sim.integrator = 'mercurius'
        sim.collision = "direct"
        sim.ri_mercurius.hillfac = 3.
        sim.collision_resolve = "merge"
    elif(integrator.lower == 'whfast'.lower):
        sim.integrator = 'whfast'
    elif(integrator == 'ias15'):
        sim.integrator = 'ias15'
    else:
        print("chosen integrator type not currently supported here \
                    options are whfast, mercurius, ias15")
        return flag, sim

    tmin = sim.t

    #set up the simulation archive 
    sim.save_to_file(archivefile, interval=tout,
                     delete_file=deletefile)
    if(logfile):
        logmessage = "Running " + des + " from " + str(tmin) + " to " + str(tmax) +" years \n"
        logmessage +="using " + integrator + " outputting every " + str(tout) +" years \n"
        logmessage += "to simulation archivefile " + archivefile + "\n"
        now = datetime.now()
        logmessage +="starting at " + str(now) + "\n"
        tools.writelog(logfile,logmessage)
        logmessage=''

    #run until tmax
    sim.integrate(tmax)

    if(logfile):
        now = datetime.now()
        logmessage ="finishing at " + str(now) + "\n"
        tools.writelog(logfile,logmessage)

    flag = 1

    return flag, sim


def initialize_simulation_from_simarchive(des=None, archivefile=None,
                                          datadir='', logfile=False):
    """
    read in a simulation archive to initialize a simulation instance
    inputs:
        des: string or list of strings, small body designation or list of designations
        archivefile (str; optional): name/path for the simulation archive file that 
            you want to generate the rebound simulation from
            If it is not provided, a default file name will be searched for
        datadir (optional): string, path for saving any files produced in this 
            function; defaults to the current directory   
        logfile (optional): boolean or string; 
            if True:  will save some messages to a default log file name
            or to a file with the name equal to the string passed or
            to the screen if 'screen' is passed 
            (default) if False nothing is saved
    outputs:
        flag (int): 0 if something went very wrong, 1 if successful
                    2 if only some of the exected small bodies were found
        sim (rebound simulation instance): contains the simulation
            state in the last snapshot of the archivefile
        clones (int): number of clones of "des" in the simulation
    """
    flag = 0

    if(des == None):
        print("The designation of one or more small bodies must be provided")
        print("run_reb.initialize_simulation_from_simarchive failed")
        return flag, None, 0

    if(logfile==True):
        logfile = tools.log_file_name(des=des)
    if(datadir and logfile and logfile!='screen'):
        logfile = datadir + '/' +logfile
    logmessage = ''

    #try all the potential default file names if the archive file is not 
    #specified
    if(archivefile==None):
        archivefile = tools.archive_file_name(des)
        if(datadir):
            archivefile = datadir + '/' +archivefile

        try:
            sim = rebound.Simulation(archivefile)
        except:
            #that didn't work, see if there's a standard initial conditions file
            archivefile2 = tools.ic_file_name(des)
            if(datadir):
                archivefile2 = datadir + '/' +archivefile2
            try:
                sim = rebound.Simulation(archivefile2)
            except RuntimeError:
                print("run_reb.initialize_simulation_from_simarchive failed")
                print("couldn't read the simulation archive file from either default: ")
                print(archivefile)
                print(archivefile2)
                return flag, None, 0

    else:
        #try the specified archive file
        if(datadir):
            archivefile = datadir + '/' +archivefile
        try:
            sim = rebound.Simulation(archivefile)
        except RuntimeError:
            print("run_reb.initialize_simulation_from_simarchive failed")
            print("couldn't read the simulation archive file: ")
            print(archivefile)
            return flag, None, 0

    if(logfile):
        time = sim.t
        logmessage += "Loaded integration for " + str(des) + " from " + archivefile + "\n"
        logmessage += "simulation is at time " + str(time) + " years\n";
        logmessage +="integrator is " + sim.integrator + "\n"
        tools.writelog(logfile,logmessage)
        logmessage = ''


    ntp = sim.N - sim.N_active
    #check to make sure the expected object(s) are in the file:
    nfound = 0
    #first if we are doing a list of objects
    if(type(des) is list):
        clones = 0
        for d in des:
            nfound+=1
            try:
                p = sim.particles[str(d)]
            except:
                print("failed to find the following object in the simulation:")
                print(str(d))
                logmessage+= str(d) + " not in the simulation\n"
                nfound+=-1
        if(nfound == 0):
            print("run_reb.initialize_simulation_from_simarchive failed")
            print("couldn't find any of the small bodies in the simulation from: ")
            print(archivefile)
            if(logfile):
                logmessage+="couldn't find any of the small bodies in the simulation\n"
                tools.writelog(logfile,logmessage)
            return flag, None, 0
        elif(nfound < ntp):
            flag = 2
            if(logfile):
                tools.writelog(logfile,logmessage)
            return flag, sim, clones
        else:
            flag = 1
            if(logfile):
                tools.writelog(logfile,logmessage)            
            return flag, sim, clones

    else:
        clones = ntp-1
        nfound+=1
        try:
            p = sim.particles[str(des)]
        except:
            print("failed to find the best-fit in the simulation")
            logmessage+="Failed to find best-fit of " + str(des) + " in the simulation\n"
            nfound+=-1
        for i in range(1,ntp):
            sbhash = str(des) + '_' + str(i)
            nfound+=1
            try:
                p = sim.particles[sbhash]
            except:
                print("missing clone %d from the simulation" % j)
                logmessage+="missing clone " + str(j) + "from the simulation\n"
                n_found+=-1

    if(nfound == 0):
        print("run_reb.initialize_simulation_from_simarchive failed")
        print("couldn't find any of the small bodies in the simulation from: ")
        print(archivefile)
        if(logfile):
            logmessage+="couldn't find any of the small bodies in the simulation\n"
            tools.writelog(logfile,logmessage)
        return flag, None, 0
    elif(nfound < ntp):
        flag = 2
        if(logfile):
            logmessage+="some of the small bodies were missing\n"            
            tools.writelog(logfile,logmessage)
        return flag, sim, clones
    
    #everything went as expected
    flag = 1
    logmessage+="Found " + str(des) + " and " + str(clones) + " clones in the simulation\n"
    if(logfile):
        tools.writelog(logfile,logmessage)            
    return flag, sim, clones
    
