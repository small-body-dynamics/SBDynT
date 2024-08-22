import numpy as np

# internal modules
import tools
import horizons_api
import run_reb

def initialize_from_heliocentric_Find_Orb_orbit(sim, des = '',nclones=0,
                                                a=1.,e=0.,inc=0.,
                                                node=0.,aperi=0.,ma=0.,
                                                planets=['mercury', 'venus', 
                                                'earth', 'mars','jupiter', 'saturn', 
                                                'uranus', 'neptune'],
                                                epoch=0.):
    """
    inputs:
        sim: empty rebound simulation instance
        des (string): designation/name for object 
        nclones (integer): number of clones, if 0 only best-fit orbit is input
        a (float): heliocentric semimajor axis (au)
        e (float): heliocentric eccentricity
        inc (float): heliocentric ecliptic inclination (rad)
        node (float): heliocentric longitude of ascending node (rad)
        aperi (float): heliocentric argument of perihelion (rad)
        ma (float): heliocentric mean anomaly (rad)
        planets (optional): string list, list of planet names - defaults to all
        epoch: float, epoch of the orbit fit in JD        
    outputs:
        flag: integer, 0 if failed, 1 if successful
        sim: rebound simulation instance with the sun and planets added and
             a test particle with the desired heliocentric orbit added
             (all are adjusted for missing major perturbers)
    """
    #check to see if the sim already has particles in it
    if(sim.N > 0):
        print("add_orbits.initialize_from_heliocentric_Find_Orb_orbit failed")
        print("This rebound simulation instance already has particles in it!")
        print("This can only accept an empty rebound simulation instance")
        return 0, sim

    if(des==''):
        print("add_orbits.initialize_from_heliocentric_Find_Orb_orbit failed")
        print("you must provide a designation (used to label the particle)")
        return 0, sim

    if(nclones == 0 and np.isscalar(a)):
        #reshape the arrays since everything assumes 2-d
        a = np.array([a])
        e = np.array([e])
        inc = np.array([inc])
        node = np.array([node])
        aperi = np.array([aperi])
        ma = np.array([ma])
    elif (nclones == 0):
        print("add_orbits.initialize_from_heliocentric_Find_Orb_orbit failed")
        print("the number of clones specified is 0, but more than one set of orbital")
        print("elements were provided. Please specify the number of clones")
        return 0, sim



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
    # The value of GM for the sun assumed in Find_orb (not the same as JPL!):
    # find_orb_sunGM
    # Planet physical parameters
    # SS_GM[0:9] in km^3 s^–2
    # SS_r[0:9] all in au
    # reasonable integration timesteps for each planet:
    # dt[0:9]
    import hard_coded_constants as const

    # set up the rebound simulation, adding the planets first
    # add the planets and return the position/velocity corrections for
    # missing planets

    apflag, sim, sx, sy, sz, svx, svy, svz = run_reb.add_planets(sim, planets=planets,
                epoch=epoch)
    if(apflag < 1):
        print("add_orbits.initialize_from_heliocentric_Find_Orb_orbit failed at run_reb.add_planets")
        return 0, sim



    # First, we need to convert the Find_Orb orbit to heliocentric
    # cartesian variables using Find_Orb's assumed solar GM
    # which is in km^2/s^2, so have to convert a to km first
    a = a/const.kmtoau
    for n in range(0,nclones+1):
        i, x, y, z, vx, vy, vz = tools.aei_to_xv(GM=const.find_orb_sunGM, 
                        a=a[n], e=e[n], inc=inc[n], node=node[n], argperi=aperi[n], ma=ma[n])
        # those positions and velocities are in km and km/s, so need to convert
        # to au/year (defining a year as 365.25 days
        x=x*const.kmtoau
        y=y*const.kmtoau
        z=z*const.kmtoau
        vx=vx*const.kmtoau/const.stoyear
        vy=vy*const.kmtoau/const.stoyear
        vz=vz*const.kmtoau/const.stoyear

        #now we can apply the corrections from add_planets to the particle
        x+=sx; y+=sy; z+=sz;
        vx+=svx; vy+=svy; vz+=svz;
    
        #add a test particle to sim with that corrected orbit:
        if(n == 0):
            #first clone is always just the best-fit orbit
            #and the hash is not numbered
            sbhash = str(des)
        else:
            sbhash = str(des) + '_' + str(i)
        sim.add(m=0., x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, hash=sbhash)
    
    sim.move_to_com()

    return 1, sim

