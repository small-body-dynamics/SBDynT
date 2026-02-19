import numpy as np

# internal modules
import tools
import horizons_api
import run_reb


def initialize_from_heliocentric_DE440_orbit(sim, des = '',
                                                a=1.,e=0.,inc=0.,
                                                node=0.,aperi=0.,ma=0.,
                                                planets=['mercury', 'venus', 
                                                'earth', 'mars','jupiter', 'saturn', 
                                                'uranus', 'neptune'],
                                                epoch=0., ecl_or_inv='ecl', clones=0, cov_orb=[]):
    """
    inputs:
        sim: empty rebound simulation instance
        des (string): designation/name for object 
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
        print("add_orbits.initialize_from_heliocentric_DE440_orbit failed")
        print("This rebound simulation instance already has particles in it!")
        print("This can only accept an empty rebound simulation instance")
        return 0, sim

    if(des==''):
        print("add_orbits.initialize_from_heliocentric_DE440_orbit failed")
        print("you must provide a designation (used to label the particle)")
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
    # The value of GM for the sun assumed in DE440 and DE441 (not the same as JPL!):
    # Planet physical parameters
    # SS_GM[0:9] in km^3 s^–2
    # SS_r[0:9] all in au
    # reasonable integration timesteps for each planet:
    # dt[0:9]
    import hard_coded_constants as const


    i, x, y, z, vx, vy, vz = tools.aei_to_xv(GM=const.SS_GM[0]*const.kmtoau**3/const.stoyear**2, 
                    a=a, e=e, inc=inc, node=node, argperi=aperi, ma=ma)


    if clones > 0:
        if len(cov_orb) == 0:
            print('No covariance matrix was provided, so SBdynT cannot produce clones for this particle. Submit a covariance matrix with columns and rows associated with [a,e,inc,node,argperi,ma], or set clones to 0.')
        mean = np.array([a, e, inc, node, aperi, ma])
        ca, ce, cinc, cnode, caperi, cma = np.random.multivariate_normal(mean, cov_orb, clones).T
        ca = ca

        #print(ca, ce, cinc, cnode, caperi, cma)

        if clones == 1:
            ca = [ca]; ce = [ce]; cinc = [cinc]
            cnode = [cnode]; caperi = [caperi]; cma = [cma]
            
        cxl = np.zeros(clones); cyl = np.zeros(clones); czl = np.zeros(clones)
        cvxl = np.zeros(clones); cvyl = np.zeros(clones); cvzl = np.zeros(clones)
        for i in range(clones):
            ci, cx, cy, cz, cvx, cvy, cvz = tools.aei_to_xv(GM=const.SS_GM[0]*const.kmtoau**3/const.stoyear**2, 
                    a=ca[i], e=ce[i], inc=cinc[i], node=cnode[i], argperi=caperi[i], ma=cma[i])
            
            cxl[i] = cx; cyl[i] = cy; czl[i] = cz
            cvxl[i] = cvx; cvyl[i] = cvy; cvzl[i] = cvz

        

    # now we can set up the rebound simulation, adding the planets first
    # add the planets and return the position/velocity corrections for
    # missing planets
    apflag, sim, sx, sy, sz, svx, svy, svz = run_reb.add_planets(sim, planets=planets,
                epoch=epoch)
    if(apflag < 1):
        print("add_orbits.initialize_from_heliocentric_DE440_orbit failed at run_reb.add_planets")
        return 0, sim

    #now we can apply the corrections from add_planets to the particle
    x+=sx; y+=sy; z+=sz;
    vx+=svx; vy+=svy; vz+=svz;
    
    #add a test particle to sim with that corrected orbit:
    sbhash = des

    
    # DS: THIS FEATURE IS CURRENTLY NOT TOTALLY CORRECT, SINCE THE CENTER-OF-MASS OFFSET FROM MISSING PLANETS
    # IS WITH RESPECT TO THE ECLIPTIC STILL. THIS PRODUCES ERRORS OFFSETS IN THE STATE VECTOR ON THE ORDER OF 1e-6 AU.
    # For simulating a set of synthetic TNOs (which is why I wrote this feature), this is
    # pretty negligible, so you can still create a synthetic catalog of objects with orbits relative to the invaribale plane rather safely, but be aware that the orbits will be modified very slightly. 
    #However, I still do not recommend using this for anything, and if you must use it, just use it for synthetic object initialization.
    if ecl_or_inv == 'inv':
        rot = rebound.Rotation.to_new_axes(newz=sim.angular_momentum())
        sim.rotate(rot)
    
    sim.add(m=0., x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, hash=des)

    if clones > 0:

        
        cxl+=sx; cyl+=sy; czl+=sz;
        cvxl+=svx; cvyl+=svy; cvzl+=svz;

        for i in range(clones):
            hash = str(des) + '_' + str(i+1)
            sim.add(m=0., x=cxl[i], y=cyl[i], z=czl[i], vx=cvxl[i], vy=cvyl[i], vz=cvzl[i], hash=hash)
    
    sim.move_to_com()

    return 1, sim

def initialize_from_heliocentric_Find_Orb_orbit(sim, des = '',
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

    # First, we need to convert the Find_Orb orbit to heliocentric
    # cartesian variables using Find_Orb's assumed solar GM
    # which is in km^2/s^2, so have to convert a to km first
    a = a/const.kmtoau
    i, x, y, z, vx, vy, vz = tools.aei_to_xv(GM=const.find_orb_sunGM, 
                    a=a, e=e, inc=inc, node=node, argperi=aperi, ma=ma)
    # those positions and velocities are in km and km/s, so need to convert
    # to au/year (defining a year as 365.25 days
    x=x*const.kmtoau
    y=y*const.kmtoau
    z=z*const.kmtoau
    vx=vx*const.kmtoau/const.stoyear
    vy=vy*const.kmtoau/const.stoyear
    vz=vz*const.kmtoau/const.stoyear

    # now we can set up the rebound simulation, adding the planets first
    # add the planets and return the position/velocity corrections for
    # missing planets
    apflag, sim, sx, sy, sz, svx, svy, svz = run_reb.add_planets(sim, planets=planets,
                epoch=epoch)
    if(apflag < 1):
        print("add_orbits.initialize_from_heliocentric_Find_Orb_orbit failed at run_reb.add_planets")
        return 0, sim

    #now we can apply the corrections from add_planets to the particle
    x+=sx; y+=sy; z+=sz;
    vx+=svx; vy+=svy; vz+=svz;
    
    #add a test particle to sim with that corrected orbit:
    sbhash = des
    sim.add(m=0., x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, hash=des)
    
    sim.move_to_com()

    return 1, sim

def initialize_from_heliocentric_destnosim(sim, des = '',
                                                a=1.,e=0.,inc=0.,
                                                node=0.,aperi=0.,ma=0.,
                                                planets=['mercury', 'venus', 
                                                'earth', 'mars','jupiter', 'saturn', 
                                                'uranus', 'neptune'],
                                                epoch=0., sb_cov=[], clones=0):
    """
    inputs:
        sim: empty rebound simulation instance
        des (string): designation/name for object 
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
        print("add_orbits.initialize_from_heliocentric_destnosim failed")
        print("This rebound simulation instance already has particles in it!")
        print("This can only accept an empty rebound simulation instance")
        return 0, sim

    if(des==''):
        print("add_orbits.initialize_from_heliocentric_destnosim failed")
        print("you must provide a designation (used to label the particle)")
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

    # First, we need to convert the DESTNOSIM orbit to heliocentric
    # cartesian variables using DESTNOSIM's assumed solar GM
    # which is in km^3/s^2, so have to convert a to km first
    i, x, y, z, vx, vy, vz = tools.aei_to_xv(GM=const.destnosim_GM[0], 
                    a=a, e=e, inc=inc, node=node, argperi=aperi, ma=ma)

    # now we can set up the rebound simulation, adding the planets first
    # add the planets and return the position/velocity corrections for
    # missing planets
    apflag, sim, sx, sy, sz, svx, svy, svz = run_reb.add_planets(sim, planets=planets,
                epoch=epoch)
    if(apflag < 1):
        print("add_orbits.initialize_from_heliocentric_Find_Orb_orbit failed at run_reb.add_planets")
        return 0, sim

    #now we can apply the corrections from add_planets to the particle
    x+=sx; y+=sy; z+=sz;
    vx+=svx; vy+=svy; vz+=svz;
    
    #add a test particle to sim with that corrected orbit:
    sbhash = des
    sim.add(m=0., x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, hash=des)

    mean = np.array([a,e,inc,node,aperi,ma])

    a_c, e_c, inc_c, node_c, argperi_c, M_c = np.random.multivariate_normal(mean, sb_cov, clones).T

    for i in range(clones):
        flag, xc, yc, zc, vxc, vyc, vzc = tools.aei_to_xv(GM=const.destnosim_GM[0], 
                    a=a_c[i], e=e_c[i], inc=inc_c[i], node=node_c[i], argperi=argperi_c[i], ma=M_c[i])
        sim.add(m=0, x=xc, y=yc, z=zc, vx = vxc, vy=vyc, vz=vzc, hash = str(des)+'_'+str(i))
    
    sim.move_to_com()

    return 1, sim