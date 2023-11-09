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

    # array of GM values queried January 2022
    # (there isn't a way to get this from Horizons, so we just have to hard
    # code it) values for giant planet systems are from Park et al. 2021 DE440
    # and DE441, https://doi.org/10.3847/1538-3881/abd414
    # all in km^3 kg^–1 s^–2
    # G = 6.6743015e-20 #in km^3 kg^–1 s^–2
    SS_GM = np.zeros(9)
    SS_GM[0] = 132712440041.93938  # Sun
    SS_GM[1] = 22031.86855  # Mercury
    SS_GM[2] = 324858.592  # Venus
    SS_GM[3] = 403503.235502  # Earth-Moon
    SS_GM[4] = 42828.375214  # Mars
    SS_GM[5] = 126712764.10  # Jupiter system
    SS_GM[6] = 37940584.8418  # Saturn system
    SS_GM[7] = 5794556.4  # Uranus system
    SS_GM[8] = 6836527.10058  # Neptune system

    # set of reasonable whfast simulation timesteps for each planet
    # (1/20 of its orbital period for terrestrial planets, 1/30 for giants)
    # dt[0] is a placeholder since the planets are indexed starting
    # at 1 instad of at 0
    dt = [0.00001, 0.012, 0.03, 0.05, 0.09, 0.4, 0.98, 2.7, 5.4]
    sim.dt = dt[0]

    # add the mass of any not-included planets to the sun
    msun = SS_GM[0]
    # work from Neptune in to also set the largest allowable timestep
    for i in range(8, 0, -1):
        if (not(planet_id[i] in planets)):
            msun += SS_GM[i]
            notplanets.append(planet_id[i])
        else:
            # reset the timestep for that planet
            sim.dt = dt[i]

    # sun's augmented mass in solar masses
    msun = msun/SS_GM[0]
    radius = 695700.*6.68459e-9
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
            flag, mass, radius, [x, y, z], [vx, vy, vz] = \
                horizons_api.query_horizons_planets(obj=pl, epoch=epoch)
            if(flag < 1):
                print("initialize_simulation failed at \
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
            print("initialize_simulation failed at \
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
        print("initialize_simulation failed at horizons_api.query_sb_from_jpl")
        return 0, 0., sim
    
    # add the planets and return the position/velocity corrections for
    # missing planets
    apflag, sim, sx, sy, sz, svx, svy, svz = add_planets(sim, planets=planets,
                epoch=epoch)
    if(apflag < 1):
        print("initialize_simulation failed at run_reb.add_planets")
        return 0, 0., sim
    
    if(clones > 0):
        for i in range(0, ntp):
            if(i == 0):
                sbhash = str(des) + '_bf'
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
        sbhash = des + '_bf'
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
        print("initialize_simulation failed at "
              "horizons_api.query_sb_from_horizons")
        return 0, 0., sim

    # add the planets and return the position/velocity corrections for
    # missing planets
    apflag, sim, sx, sy, sz, svx, svy, svz = add_planets(sim, planets=planets,
                epoch=epoch)
    if(apflag < 1):
        print("initialize_simulation failed at run_reb.add_planets")
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


def run_simulation(sim, tmax=0, tout=0, filename="archive.bin",
                   deletefile=False,integrator='mercurius'):
    """
    run a mercurius simulation saving to a simulation archive every tout

    inputs:
        sim: rebound simulation instance with all bodies added and simulation
             timestep set
        tmax: float, desired stopping time for the integration in years
        tout: float, desired interval for saving the simulation output in years
        filename: string, file the simulation archive should be saved to
        deletefile (optional): Boolean, whether filename should be deleted
                               if it already exists (default False to not delete
                               the file)
        integrator (optional): rebound integrator to use. Currently doesn't do 
                               anything because only mercurius is set
    """
    sim.automateSimulationArchive(filename, interval=tout,
                                  deletefile=deletefile)

    sim.integrator = 'mercurius'
    sim.collision = "direct"
    sim.ri_mercurius.hillfac = 3.
    sim.collision_resolve = "merge"

    sim.integrate(tmax)
    return sim
