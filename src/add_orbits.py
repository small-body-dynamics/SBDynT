import numpy as np
import rebound

# internal modules
import tools
import horizons_api
import run_reb
import hard_coded_constants as const



def initialize_from_heliocentric_Find_Orb_orbit(des=None,clones=None,
                                                a=1.,e=0.,inc=0.,
                                                node=0.,aperi=0.,ma=0.,
                                                planets=['all'],
                                                epoch=None,
                                                datadir='', saveic=False,
                                                logfile=False):
    """
    inputs:
        sim: empty rebound simulation instance
        des (string): designation/name for object 
        clones (integer, optional): number of clones,
            defaults to use all of the provided orbital elements
            if 0 only the first provided orbit is used
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

    if(logfile==True):
        logfile = tools.log_file_name(des=des)
    if(datadir and logfile and logfile!='screen'):
        logfile = datadir + '/' + logfile

    sim = rebound.Simulation()
    if(des==None):
        print("add_orbits.initialize_from_heliocentric_Find_Orb_orbit failed")
        print("you must provide a designation (used to label the particle within rebound)")
        return 0, sim

    if(np.isscalar(a)):
        #reshape the arrays since everything assumes arrays
        a = np.array([a])
        e = np.array([e])
        inc = np.array([inc])
        node = np.array([node])
        aperi = np.array([aperi])
        ma = np.array([ma])
    
    ntp_avail = len(a)
    if(clones == None):
        clones = ntp_avail-1
    elif (clones > ntp_avail-1):
        print("add_orbits.initialize_from_heliocentric_Find_Orb_orbit failed")
        print("the number of clones specified is more than than the length of orbital")
        print("element arrays that were provided.")
        return 0, sim



    # make sure planets is a list and make all planet names lowercase
    if not (type(planets) is list):
        planets = [planets]
    planets = [pl.lower() for pl in planets]
    if(planets == ['outer']):
        planets = ['jupiter', 'saturn', 'uranus', 'neptune']
    if(planets == ['all']):
        planets = ['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']    

    # Note: the value of GM for the sun assumed in Find_orb (not the same as JPL!):
    # is coded as const.find_orb_sunGM

    # set up the rebound simulation, adding the planets first
    # add the planets and return the position/velocity corrections for
    # missing planets

    apflag, sim, sx, sy, sz, svx, svy, svz = run_reb.add_planets(sim,planets=planets,epoch=epoch)
    if(apflag < 1):
        print("add_orbits.initialize_from_heliocentric_Find_Orb_orbit failed at run_reb.add_planets")
        return 0, sim

    # First, we need to convert the Find_Orb orbit to heliocentric
    # cartesian variables using Find_Orb's assumed solar GM
    # which is in km^2/s^2, so have to convert a to km first
    a = a/const.kmtoau
    for n in range(0,clones+1):
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
            sbhash = str(des) + '_' + str(n)
        sim.add(m=0., x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, hash=sbhash)
    
    sim.move_to_com()

    if(saveic):
        if(saveic == True):
            ic_file = tools.ic_file_name(des=des)
            sim.save_to_file(ic_file)
        else:
            ic_file = saveic

        if(datadir):
            ic_file = datadir + '/' +ic_file
        sim.save_to_file(ic_file)
        if(logfile):
            logmessage = "Rebound simulation initial conditions saved to " + ic_file + "\n"
            tools.writelog(logfile,logmessage)       

    return 1, sim





def initialize_from_heliocentric_DE440_orbit(des=None,clones=None, 
                                             a=1.,e=0.,inc=0.,
                                             node=0.,aperi=0.,ma=0.,
                                             planets=['all'],
                                             epoch=None, ecl_or_inv='ecl', 
                                             cov_orb=[],
                                             datadir='', saveic=False,
                                             logfile=False):
    """
    inputs:
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


    if(logfile==True):
        logfile = tools.log_file_name(des=des)
    if(datadir and logfile and logfile!='screen'):
        logfile = datadir + '/' + logfile

    sim = rebound.Simulation()
    
    if(des==None):
        print("add_orbits.initialize_from_heliocentric_DE440_orbit failed")
        print("you must provide a designation (used to label the particle)")
        return 0, sim



    # make sure planets is a list and make all planet names lowercase
    if not (type(planets) is list):
        planets = [planets]
    planets = [pl.lower() for pl in planets]
    if(planets == ['outer']):
        planets = ['jupiter', 'saturn', 'uranus', 'neptune']
    if(planets == ['all']):
        planets = ['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']    


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

    if(clones == None):
        clones=0

    if(clones > 0):
        if(len(cov_orb) == 0):
            print('No covariance matrix was provided, so SBdynT cannot produce clones for this particle. ')
            print('Submit a covariance matrix with columns and rows associated with [a,e,inc,node,argperi,ma] ')
            print('or set clones to None.')
        mean = np.array([a, e, inc, node, aperi, ma])
        ca, ce, cinc, cnode, caperi, cma = np.random.multivariate_normal(mean, cov_orb, clones).T
        ca = ca
        if(clones == 1):
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
    sbhash = str(des)

    
    # DS: THIS FEATURE IS CURRENTLY NOT TOTALLY CORRECT, SINCE THE CENTER-OF-MASS OFFSET FROM MISSING PLANETS
    # IS WITH RESPECT TO THE ECLIPTIC STILL. THIS PRODUCES ERRORS OFFSETS IN THE STATE VECTOR ON THE ORDER OF 1e-6 AU.
    # For simulating a set of synthetic TNOs (which is why I wrote this feature), this is
    # pretty negligible, so you can still create a synthetic catalog of objects with orbits relative to the invaribale 
    # plane rather safely, but be aware that the orbits will be modified very slightly. 
    # However, I still do not recommend using this for anything, and if you must use it, just use it for synthetic object initialization.
    if ecl_or_inv == 'inv':
        rot = rebound.Rotation.to_new_axes(newz=sim.angular_momentum())
        sim.rotate(rot)
    
    sim.add(m=0., x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, hash=sbhash)

    if(clones > 0):
        cxl+=sx; cyl+=sy; czl+=sz;
        cvxl+=svx; cvyl+=svy; cvzl+=svz;

        for i in range(clones):
            sbhash = str(des) + '_' + str(i+1)
            sim.add(m=0., x=cxl[i], y=cyl[i], z=czl[i], vx=cvxl[i], vy=cvyl[i], vz=cvzl[i], hash=sbhash)
    sim.move_to_com()

    if(saveic):
        if(saveic == True):
            ic_file = tools.ic_file_name(des=des)
            sim.save_to_file(ic_file)
        else:
            ic_file = saveic

        if(datadir):
            ic_file = datadir + '/' +ic_file
        sim.save_to_file(ic_file)
        if(logfile):
            logmessage = "Rebound simulation initial conditions saved to " + ic_file + "\n"
            tools.writelog(logfile,logmessage)       


    return 1, sim




def initialize_from_heliocentric_destnosim(des=None, clones=None,
                                           a=1.,e=0.,inc=0.,
                                           node=0.,aperi=0.,ma=0.,
                                           planets=['all'],
                                           epoch=0., sb_cov=[],
                                           datadir='', saveic=False,
                                           logfile=False):

    """
    inputs:
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


    if(logfile==True):
        logfile = tools.log_file_name(des=des)
    if(datadir and logfile and logfile!='screen'):
        logfile = datadir + '/' + logfile

    if(des==None):
        print("add_orbits.initialize_from_heliocentric_destnosim failed")
        print("you must provide a designation (used to label the particle)")
        return 0, sim

    # make sure planets is a list and make all planet names lowercase
    if not (type(planets) is list):
        planets = [planets]
    planets = [pl.lower() for pl in planets]
    if(planets == ['outer']):
        planets = ['jupiter', 'saturn', 'uranus', 'neptune']
    if(planets == ['all']):
        planets = ['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']    

    import hard_coded_constants as const

    # First, we need to convert the DESTNOSIM orbit to heliocentric
    # cartesian variables using DESTNOSIM's assumed solar GM
    # which is in km^3/s^2, so have to convert a to km first
    i, x, y, z, vx, vy, vz = tools.aei_to_xv(GM=const.destnosim_GM[0], 
                    a=a, e=e, inc=inc, node=node, argperi=aperi, ma=ma)


    if(clones == None):
        clones=0
    if(clones > 0):
        if(len(cov_orb) == 0):
            print('No covariance matrix was provided, so SBdynT cannot produce clones for this particle. ')
            print('Submit a covariance matrix with columns and rows associated with [a,e,inc,node,argperi,ma] ')
            print('or set clones to None.')


    sim = rebound.Simulation()
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
    sbhash = str(des)
    sim.add(m=0., x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, hash=sbhash)

    if(clones > 0):
        mean = np.array([a,e,inc,node,aperi,ma])

        a_c, e_c, inc_c, node_c, argperi_c, M_c = np.random.multivariate_normal(mean, sb_cov, clones).T

        for i in range(clones):
            flag, xc, yc, zc, vxc, vyc, vzc = tools.aei_to_xv(GM=const.destnosim_GM[0], 
                    a=a_c[i], e=e_c[i], inc=inc_c[i], node=node_c[i], argperi=argperi_c[i], ma=M_c[i])
            xc+=sx; yc+=sy; zc+=sz;
            vxc+=svx; vyc+=svy; vzc+=svz;
            sbhash = str(des) + '_' + str(i+1)
            sim.add(m=0, x=xc, y=yc, z=zc, vx = vxc, vy=vyc, vz=vzc, hash=sbhash)
    
    sim.move_to_com()


    if(saveic):
        if(saveic == True):
            ic_file = tools.ic_file_name(des=des)
            sim.save_to_file(ic_file)
        else:
            ic_file = saveic

        if(datadir):
            ic_file = datadir + '/' +ic_file
        sim.save_to_file(ic_file)
        if(logfile):
            logmessage = "Rebound simulation initial conditions saved to " + ic_file + "\n"
            tools.writelog(logfile,logmessage)       



    return 1, sim
