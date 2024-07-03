import rebound
import numpy as np
# local
import horizons_api


def add_planets(sim, planets=['mercury', 'venus', 'earth', 'mars',
                'jupiter', 'saturn', 'uranus', 'neptune'],
                epoch=2459580.5):
    """
    inputs:
        sim: empty rebound simulation instance
        planets (optional): string list, list of planet names - defaults to all
        epoch (optional): float, epoch in JD, defaults to Jan 1, 2022\        
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
    #check to see if the sim already has particles in it
    if(sim.N > 0):
        print("run_reb.add_planets failed")
        print("This rebound simulation instance already has particles in it!")
        print("run_reb.add_planets can only accept an empty rebound simulation instance")
        return 0, sim, 0.,0.,0.,0.,0.,0.

    sim.units = ('yr', 'AU', 'Msun')

    # make all planet names lowercase
    planets = [pl.lower() for pl in planets]
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
            flag, mass, radius, [x, y, z], [vx, vy, vz] = horizons_api.query_horizons_planets(obj=pl, epoch=epoch)
            if(flag < 1):
                print("run_reb.add_planets failed at \
                    horizons_api.query_horizons_planets for ", pl)
                return 0, sim, 0.,0.,0.,0.,0.,0.
            tsim.add(m=mass, r=radius, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
        # calculate the barycenter of the sun + missing planets
        com = tsim.calculate_com()
        # reset the corrections to the positions and velocities
        sx = -com.x; sy = -com.y; sz = -com.z
        svx = -com.vx; svy = -com.vy; svz = -com.vz
    
    #free up the temporary simulation variable
    tsim = None

    # add each included planet to the simulation and correct for the
    # missing planets
    for pl in planets:
        flag, mass, radius, [x, y, z], [vx, vy, vz] = \
            horizons_api.query_horizons_planets(obj=pl, epoch=epoch)
        if(flag < 1):
            print("run_reb.add_planets failed failed at \
                  horizons_api.query_horizons_planets for ", pl)
            return 0, sim, 0.,0.,0.,0.,0.,0.
        # correct for the missing planets
        x += sx; y += sy; z += sz
        vx += svx; vy += svy; vz += svz
        sim.add(m=mass, r=radius, x=x, y=y, z=z,
                vx=vx, vy=vy, vz=vz, hash=pl)

    sim.N_active = npl
    
    return 1, sim, sx, sy, sz, svx, svy, svz


def initialize_simulation(planets=['mercury', 'venus', 'earth', 'mars',
                                   'jupiter', 'saturn', 'uranus', 'neptune'],
                          des='', clones=0):
    """
    inputs:
        planets (optional): string list, list of planet names - defaults to all
        des: string, small body designation
        clones (optional): integer, number of clones - defaults to 0
    outputs:
        flag: integer, 0 if failed, 1 if successful
        epoch: float, date of the simulation start (JD)
        sim: rebound simulation instance with planets and test particles added
             with adjustments for missing major perturbers
    """
    
    # make all planet names lowercase
    planets = [pl.lower() for pl in planets]
    
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
    flag, epoch, sbx, sby, sbz, sbvx, sbvy, sbvz = \
        horizons_api.query_sb_from_jpl(des=des, clones=clones)
    if(flag < 1):
        print("run_reb.initialize_simulation failed at horizons_api.query_sb_from_jpl")
        return 0, 0., sim
    
    # add the planets and return the position/velocity corrections for
    # missing planets
    apflag, sim, sx, sy, sz, svx, svy, svz = add_planets(sim, planets=planets,
                epoch=epoch)
    if(apflag < 1):
        print("run_reb.initialize_simulation failed at run_reb.add_planets")
        return 0, 0., sim
    
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
        sbhash = str(des) 
        sim.add(m=0., x=sbx, y=sby, z=sbz,
                vx=sbvx, vy=sbvy, vz=sbvz, hash=sbhash)

    sim.move_to_com()

    return 1, epoch, sim

def initialize_simulation_at_epoch(
        planets=['mercury', 'venus', 'earth', 'mars',
                 'jupiter', 'saturn', 'uranus', 'neptune'],
        des=[''], epoch=2459580.5):
    """
    inputs:
        planets (optional): string list, list of planet names - defaults to all
        des: string or list of strings, small body designation or list of designations
        epoch (optional): float, epoch in JD, defaults to Jan 1, 2022
    outputs:
        flag (integer): 0 if failed, 1 if successful
        epoch: float, epoch of the simulation start (JD)
        sim: rebound simulation instance with planets and test particles added
             with adjustments for missing major perturbers
    """

    # make all planet names lowercase
    planets = [pl.lower() for pl in planets]
    
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
        return 0, 0., sim

    # add the planets and return the position/velocity corrections for
    # missing planets
    apflag, sim, sx, sy, sz, svx, svy, svz = add_planets(sim, planets=planets,
                epoch=epoch)
    if(apflag < 1):
        print("run_reb.initialize_simulation_at_epoch failed at run_reb.add_planets")
        return 0, 0., sim


    for i in range(0, ntp):
        sbhash = str(des[i])
        # correct for the missing planets
        sbx[i] += sx; sby[i] += sy; sbz[i] += sz
        sbvx[i] += svx; sbvy[i] += svy; sbvz[i] += svz
        sim.add(m=0., x=sbx[i], y=sby[i], z=sbz[i],
                
                vx=sbvx[i], vy=sbvy[i], vz=sbvz[i], hash=sbhash)

    sim.move_to_com()

    return 1, epoch, sim

def initialize_simulation_from_sv(planets=['mercury', 'venus', 'earth', 'mars',
                                   'jupiter', 'saturn', 'uranus', 'neptune'],
                          des='', clones=0, sb=[0,0,0,0,0,0,0]):
    """
    inputs:
        planets (optional): string list, list of planet names - defaults to all
        des: string, small body designation
        clones (optional): integer, number of clones - defaults to 0
        sb: list of floats, [epoch,x,y,z,vx,vy,vz] of the small body object. This is 
        effectively a way to start a simulation using data outside of Horizons.
    outputs:
        flag: integer, 0 if failed, 1 if successful
        epoch: float, date of the simulation start (JD)
        sim: rebound simulation instance with planets and test particles added
             with adjustments for missing major perturbers
    """
    
    # make all planet names lowercase
    planets = [pl.lower() for pl in planets]
    
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
    flag, epoch, sbx, sby, sbz, sbvx, sbvy, sbvz = 1,sb[0],sb[1],sb[2],sb[3],sb[4],sb[5],sb[6]
    if(flag < 1):
        print("run_reb.initialize_simulation failed at horizons_api.query_sb_from_jpl")
        return 0, 0., sim
    
    # add the planets and return the position/velocity corrections for
    # missing planets
    print(planets)
    apflag, sim, sx, sy, sz, svx, svy, svz = add_planets(sim, planets=planets,
                epoch=epoch)
    if(apflag < 1):
        print("run_reb.initialize_simulation failed at run_reb.add_planets")
        return 0, 0., sim
    
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
        sbhash = str(des) 
        sim.add(m=0., x=sbx, y=sby, z=sbz,
                vx=sbvx, vy=sbvy, vz=sbvz, hash=sbhash)

    sim.move_to_com()

    return 1, epoch, sim


def run_simulation(sim, tmax=0, tout=0, filename="archive.bin",
                   deletefile=False,integrator='mercurius'):
    """
    run a simulation saving to a simulation archive every tout
    inputs:
        sim (rebound simulation instance): initialized with all the
            planets and small bodies
        tmax (float; years): simulation stopping time
        tout (float; years): interval for saving simulation outputs to
            the simulation archive file
        filename (str; optional): name/path for the simulation
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
    outputs:
        flag (int): 0 if something went wrong, 1 if the integration succeded
        sim (rebound simulation instance): contains the full simulation
            state at tmax
    """

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
        return 0, sim

    #set up the simulation archive 
    sim.automateSimulationArchive(filename, interval=tout,
                                  deletefile=deletefile)

    #run until tmax
    try:
        sim.integrate(tmax)
    except:
        print('Particle ejected')
        return 1, sim
    return 1, sim


def initialize_simulation_from_simarchive(sim, filename=" "):
    """
    read in a simulation archive to initialize a simulation instance
    inputs:
        sim (rebound simulation instance): must be empty
        filename (str; optional): name/path for the simulation
            archive file that rebound will generate
    outputs:
        flag (int): 0 if something went wrong, 1 if sucessful
        sim (rebound simulation instance): contains the simulation
            state in the last snapshot of the archivefile
    """
    if(sim.N > 0):
        print("run_reb.initialize_simulation_from_simarchive failed")
        print("This rebound simulation instance already has particles in it!")
        print("can only accept an empty rebound simulation instance")
        return 0, sim

    try:
        sim = rebound.Simulation(filename)
    except RuntimeError:
        print("run_reb.initialize_simulation_from_simarchive failed")
        print("couldn't read the simulation archive file")
        return 0, sim

    return 1, sim
