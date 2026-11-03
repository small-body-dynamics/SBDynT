import numpy as np
import rebound
import time
import requests
from pickle import dump


# internal modules
from sbdynt import tools
from sbdynt import horizons_api
from sbdynt import run_reb
from sbdynt import hard_coded_constants as const




def get_mpc_cartesian(desig = "None",logfile=False, orbit_file=False):
    """
    Get the heliocentric position and velocity with covariance matrix
    from the Minor Planet Center via API request

    inputs:
        desig: string, minor planet designation
        logfile (optional): string or boolean
        orbit_file (optional): string or boolean

    outputs:
        flag: integer, 0 if nothing was queried,
                       1 if query is successful
        xv: np array, cartesian heliocentric positions
            and velocities (in au and au/day)
        cov_mat: np array, covariance matrix for xv
        epoch: JD epoch of the MPC data
    """

    flag = 0
    
    pdes, destype = tools.mpc_designation_translation(desig)
    response = requests.get(
        "https://data.minorplanetcenter.net/api/get-orb",
        json={"desig": pdes}  # Famous close-approach asteroid
    )
    #adding a sleep command to avoid violating the MPC's api query limits
    time.sleep(0.25)
    
    cov_mat = None
    xv = None
    epoch = None
    
    if not response.ok:
        logmessage = "add_orbits.get_mpc_cartesian failed\n"
        if logfile != "screen":
            print(logmessage)
        if logfile:
            tools.writelog(logfile, logmessage)
        return flag, xv, cov_mat, epoch

    try:
        orbit = response.json()[0]['mpc_orb'][0]
    except:
        logmessage = "add_orbits.get_mpc_cartesian failed\n"
        if logfile != "screen":
            print(logmessage)
        if logfile:
            tools.writelog(logfile, logmessage)
        return flag, xv, cov_mat, epoch


    # save the MPC query results using pickle if that's desired
    if orbit_file:
        try:
            with open(orbit_file, "wb") as f:
                dump(orbit, f)
        except:
            logmessage = "unable to write the MPC query results to a file"
            logmessage += "tried to write to " + str(orbit_file)
            if logfile != "screen":
                print(logmessage)
            if logf:
                tools.writelog(logfile, logmessage)
            return flag, xv, cov_mat, epoch   

        if logfile:
            logmessage = "MPC query results saved to " + orbit_file + "\n"
            tools.writelog(logfile, logmessage)


    try:
        coords = { name:value for name,value in 
               zip(orbit["CAR"]['coefficient_names'],
                   orbit["CAR"]['coefficient_values'])}
    except:
        logmessage = "add_orbits.get_mpc_cartesian failed\n"
        logmessage += "couldn't extract orbit field values\n"
        if logfile != "screen":
            print(logmessage)
        if logfile:
            tools.writelog(logfile, logmessage)
        return flag, xv, cov_mat, epoch        
        
    try:
        cov = orbit["CAR"]['covariance']
    except:
        logmessage = "add_orbits.get_mpc_cartesian failed\n"
        logmessage += "couldn't extract orbit covariance values\n"
        if logfile != "screen":
            print(logmessage)
        if logfile:
            tools.writelog(logfile, logmessage)
        return flag, xv, cov_mat, epoch    

    #put the covariance matrix into the format expected for the cloning
    cov_mat = np.array([
        [cov['cov00'],cov['cov01'],cov['cov02'],cov['cov03'],cov['cov04'],cov['cov05']],
        [cov['cov01'],cov['cov11'],cov['cov12'],cov['cov13'],cov['cov14'],cov['cov15']],
        [cov['cov02'],cov['cov12'],cov['cov22'],cov['cov23'],cov['cov24'],cov['cov25']],
        [cov['cov03'],cov['cov13'],cov['cov23'],cov['cov33'],cov['cov34'],cov['cov35']],
        [cov['cov04'],cov['cov14'],cov['cov24'],cov['cov34'],cov['cov44'],cov['cov45']],
        [cov['cov05'],cov['cov15'],cov['cov25'],cov['cov35'],cov['cov45'],cov['cov55']]
    ])

    #put the cartesian position and velocity into the expected format
    xv = np.array([coords['x'],coords['y'],coords['z'],coords['vx'],coords['vy'],coords['vz']])    

    try:
        MJD_epoch = orbit['epoch_data']['epoch']
    except:
        logmessage = "add_orbits.get_mpc_cartesian failed\n"
        logmessage += "couldn't extract orbit epoch\n"
        if logfile != "screen":
            print(logmessage)
        if logfile:
            tools.writelog(logfile, logmessage)
        return flag, xv, cov_mat, epoch 
        
    if(orbit['epoch_data']['timeform'] == 'MJD'):
        epoch = MJD_epoch + 2400000.5
    else:
        epoch = MJD_epoch


    flag = 1
    
    return flag, xv, cov_mat, epoch



def initialize_from_mpc_cartesian(
    des=None,
    clones=None,
    datadir="",
    save_sbdb=True,
    saveic=True,
    planets=['all'],
    archivefile=None,
    logfile=False,
):

    """
    Initialize a small body simulation using data from the MPC

    inputs:
        des (string): the designation for the object in the MPC
        clones (optional, integer): number of clones to be generated in a Guassian
            manner from the covariance matrix
        datadir (optional, string): path for saving any files produced in this
            function; defaults to the current directory
        saveic (optional, boolean or string): default True which saves a rebound file with
            the initial simulation state that can be used to restart it later to a default
            file name; if a string is provided, that is the file name used
            if False, nothing is saved
        archivefile (optional, str): Name of the Simulationarchive binary file rebound saves
            to. If nothing is passed, the default filename is <des>-simarchive.bin.
        logfile (optional, boolean or string): if True, log messages will be saved to a
            default log file name; if a string is passed that string is the file name;
            if 'screen' is passed, logmessages are printed to screen
        save_sbdb (optional, boolean or string): default True  will save a pickle file
            with the results of the MPC query to a default file; of a string is
            passed, that is the save-file name;
            if False, nothing is saved
        planets (optional): string list, list of planet names - defaults to Mercury-Neptune

    outputs:
        flag (integer, 0 or 1): 0 indicates a failure, while 1 indicates success.
        sim (Rebound Simulation instance): the initialized Rebound Simulation containing
            the planets and small bodies
        epoch (real): JD date of the orbit fit

    """


    flag = 0
    if des == None:
        print("The designation of a small body must be provided")
        print("failed at add_orbits.initialize_from_mpc_cartesian()")
        return flag, None, None, None, None, None

    if logfile == True:
        logf = tools.log_file_name(des=des)
    else:
        logf = logfile

    if datadir:
        tools.check_datadir(datadir)

    if datadir and logf and logf != "screen":
        logf = datadir + "/" + logf

    orbit_file = False
    if save_sbdb:
        if save_sbdb == True:
            orbit_file = tools.orbit_solution_file(des)
        else:
            orbit_file = save_sbdb

        if datadir:
            orbit_file = datadir + "/" + orbit_file

    if(datadir and logf and logf!='screen'):
        logf = datadir + '/' + logf

    if clones == None:
        #no clones will be generated
        clones = 0

    #initialize empty rebound simulation
    sim = rebound.Simulation()


    #pull data from the MPC
    query_flag, xv, cov_mat, epoch = get_mpc_cartesian(des,logf,orbit_file)

    if(query_flag < 1):
        logmessage = "add_orbits.initialize_from_mpc_cartesian failed\n"
        logmessage += "at the MPC API query\n"
        if logf != "screen":
            print(logmessage)
        if logf:
            tools.writelog(logf, logmessage)
        return flag, None, None, None, None, None


    if(clones == 0):
        x = np.array([xv[0]])
        y = np.array([xv[1]])
        z = np.array([xv[2]])
        vx = np.array([xv[3]])
        vy = np.array([xv[4]])
        vz = np.array([xv[5]])
    else:
        #generate clones from the covariance matrix
        x, y, z, vx, vy, vz = np.random.multivariate_normal(xv, xv_cov, clones).T
        #put the best-fit orbit at the start of the array
        x = np.insert(x, 0, xv[0], axis=0)
        y = np.insert(y, 0, xv[1], axis=0)
        z = np.insert(z, 0, xv[2], axis=0)
        vx = np.insert(vx, 0, xv[3], axis=0)
        vy = np.insert(vy, 0, xv[4], axis=0)
        vz = np.insert(vz, 0, xv[5], axis=0)

    # make sure planets is a list and make all planet names lowercase
    if not (type(planets) is list):
        planets = [planets]
    planets = [pl.lower() for pl in planets]
    if(planets == ['outer']):
        planets = ['jupiter', 'saturn', 'uranus', 'neptune']
    if(planets == ['inner+outer']):
        planets = ['venus', 'earth', 'mars','jupiter', 'saturn', 'uranus', 'neptune']
    if(planets == ['all']):
        planets = ['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']    

    # set up the rebound simulation, adding the planets first
    # add the planets and return the position/velocity corrections for
    # missing planets

    apflag, sim, sx, sy, sz, svx, svy, svz = run_reb.add_planets(
            sim,planets=planets,epoch=epoch,logfile=logf)
    if(apflag < 1):
        logmessage = "initialize_from_mpc_cartesian failed at run_reb.add_planets"
        if(logf!='screen'):
            print(logmessage)
        if(logf):
            tools.writelog(logf,logmessage)        
        return 0, sim

    # those velocities are au/day, so need to convert
    # to au/year (defining a year as 365.25 days

    vx=vx*365.25
    vy=vy*365.25
    vz=vz*365.25

    #now we can apply the corrections from add_planets to the particle
    x+=sx; y+=sy; z+=sz;
    vx+=svx; vy+=svy; vz+=svz;

    for n in range(0,clones+1):
        #add a test particle to sim with that corrected orbit:
        if(n == 0):
            #first clone is always just the best-fit orbit
            #and the hash is not numbered
            sbhash = str(des)
        else:
            sbhash = str(des) + '_' + str(n)
        sim.add(m=0., x=x[n], y=y[n], z=z[n], vx=vx[n], vy=vy[n], vz=vz[n], hash=sbhash)
    
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
        if(logf):
            logmessage = "Rebound simulation initial conditions saved to " + ic_file + "\n"
            tools.writelog(logf,logmessage)       

    flag = 1

    return flag, sim, epoch



def initialize_from_heliocentric_cartesian(
    des=None,
    epoch = None,
    clones=None,
    x = None,
    y = None,
    z = None,
    vx = None,
    vy = None,
    vz = None,
    datadir="",
    saveic=True,
    planets=['all'],
    archivefile=None,
    logfile=False,
):

    """
    Initialize a small body simulation with user-input cartesian positions 
    and velocities

    inputs:
        des (string): the designation for the object in the MPC
        epoch (real): JD epoch of the cartesian inputs
        clones (optional, integer): number of clones (if None, will be taken
            from the length of the xv vectors
        x, y, z (real or array of reals): heliocentric cartesian positions in 
            units of au
        vx, vy, vz (real or array of reals): heliocentric cartesian velocities in 
            units of au/year
        datadir (optional, string): path for saving any files produced in this
            function; defaults to the current directory
        saveic (optional, boolean or string): default True which saves a rebound file with
            the initial simulation state that can be used to restart it later to a default
            file name; if a string is provided, that is the file name used
            if False, nothing is saved
        archivefile (optional, str): Name of the Simulationarchive binary file rebound saves
            to. If nothing is passed, the default filename is <des>-simarchive.bin.
        logfile (optional, boolean or string): if True, log messages will be saved to a
            default log file name; if a string is passed that string is the file name;
            if 'screen' is passed, logmessages are printed to screen
        planets (optional): string list, list of planet names - defaults to Mercury-Neptune

    outputs:
        flag (integer, 0 or 1): 0 indicates a failure, while 1 indicates success.
        sim (Rebound Simulation instance): the initialized Rebound Simulation containing
            the planets and small bodies
        epoch (real): JD date of the orbit fit

    """


    flag = 0
    if des == None:
        print("The designation of the small body must be provided (used to label particles)")
        print("failed at add_orbits.initialize_from_heliocentric_cartesian()")
        return flag, None, None, None, None, None

    if epoch == None:
        print("The JD epoch of the user-provided positions/velocities must be provided")
        print("failed at add_orbits.initialize_from_heliocentric_cartesian()")
        return flag, None, None, None, None, None

    if (x == None or y==None or z==None or vx==None or vy==None or:
        print("The JD epoch of the user-provided positions/velocities must be provided")
        print("failed at add_orbits.initialize_from_heliocentric_cartesian()")
        return flag, None, None, None, None, None


    if logfile == True:
        logf = tools.log_file_name(des=des)
    else:
        logf = logfile

    if datadir:
        tools.check_datadir(datadir)

    if datadir and logf and logf != "screen":
        logf = datadir + "/" + logf

    if save_sbdb:
        if save_sbdb == True:
            orbit_file = tools.orbit_solution_file(des)
        else:
            orbit_file = save_sbdb

        if datadir:
            orbit_file = datadir + "/" + orbit_file

    if(datadir and logf and logf!='screen'):
        logf = datadir + '/' + logf

    if clones == None:
        #no clones will be generated
        clones = 0

    #initialize empty rebound simulation
    sim = rebound.Simulation()


    #pull data from the MPC
    query_flag, xv, cov_mat, epoch = get_mpc_cartesian(des,logf,orbit_file)

    if(query_flag < 1):
        logmessage = "add_orbits.initialize_from_mpc_cartesian failed\n"
        logmessage += "at the MPC API query\n"
        if logf != "screen":
            print(logmessage)
        if logf:
            tools.writelog(logf, logmessage)
        return flag, None, None, None, None, None


    if(clones == 0):
        x = np.array([xv[0]])
        y = np.array([xv[1]])
        z = np.array([xv[2]])
        vx = np.array([xv[3]])
        vy = np.array([xv[4]])
        vz = np.array([xv[5]])
    else:
        #generate clones from the covariance matrix
        x, y, z, vx, vy, vz = np.random.multivariate_normal(xv, xv_cov, clones).T
        #put the best-fit orbit at the start of the array
        x = np.insert(x, 0, xv[0], axis=0)
        y = np.insert(y, 0, xv[1], axis=0)
        z = np.insert(z, 0, xv[2], axis=0)
        vx = np.insert(vx, 0, xv[3], axis=0)
        vy = np.insert(vy, 0, xv[4], axis=0)
        vz = np.insert(vz, 0, xv[5], axis=0)

    # make sure planets is a list and make all planet names lowercase
    if not (type(planets) is list):
        planets = [planets]
    planets = [pl.lower() for pl in planets]
    if(planets == ['outer']):
        planets = ['jupiter', 'saturn', 'uranus', 'neptune']
    if(planets == ['inner+outer']):
        planets = ['venus', 'earth', 'mars','jupiter', 'saturn', 'uranus', 'neptune']
    if(planets == ['all']):
        planets = ['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']    

    # set up the rebound simulation, adding the planets first
    # add the planets and return the position/velocity corrections for
    # missing planets

    apflag, sim, sx, sy, sz, svx, svy, svz = run_reb.add_planets(
            sim,planets=planets,epoch=epoch,logfile=logf)
    if(apflag < 1):
        logmessage = "initialize_from_mpc_cartesian failed at run_reb.add_planets"
        if(logf!='screen'):
            print(logmessage)
        if(logf):
            tools.writelog(logf,logmessage)        
        return 0, sim

    # those velocities are au/day, so need to convert
    # to au/year (defining a year as 365.25 days

    vx=vx*365.25
    vy=vy*365.25
    vz=vz*365.25

    #now we can apply the corrections from add_planets to the particle
    x+=sx; y+=sy; z+=sz;
    vx+=svx; vy+=svy; vz+=svz;

    for n in range(0,clones+1):
        #add a test particle to sim with that corrected orbit:
        if(n == 0):
            #first clone is always just the best-fit orbit
            #and the hash is not numbered
            sbhash = str(des)
        else:
            sbhash = str(des) + '_' + str(n)
        sim.add(m=0., x=x[n], y=y[n], z=z[n], vx=vx[n], vy=vy[n], vz=vz[n], hash=sbhash)
    
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
        if(logf):
            logmessage = "Rebound simulation initial conditions saved to " + ic_file + "\n"
            tools.writelog(logf,logmessage)       

    flag = 1

    return flag, sim, epoch


def initialize_from_heliocentric_Find_Orb_orbit(
    des=None,
    clones=None,
    a=1.0,
    e=0.0,
    inc=0.0,
    node=0.0,
    aperi=0.0,
    ma=0.0,
    planets=["all"],
    epoch=None,
    datadir="",
    saveic=False,
    logfile=False,
):
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

    if logfile == True:
        logf = tools.log_file_name(des=des)
    else:
        logf = logfile

    if datadir:
        tools.check_datadir(datadir)

    if datadir and logf and logf != "screen":
        logf = datadir + "/" + logf

    sim = rebound.Simulation()
    if des == None:
        logmessage = "add_orbits.initialize_from_heliocentric_Find_Orb_orbit failed\n"
        logmessage += "you must provide a designation (used to label the particle within rebound)"
        if logf != "screen":
            print(logmessage)
        if logf:
            tools.writelog(logf, logmessage)
        return 0, sim

    if np.isscalar(a):
        # reshape the arrays since everything assumes arrays
        a = np.array([a])
        e = np.array([e])
        inc = np.array([inc])
        node = np.array([node])
        aperi = np.array([aperi])
        ma = np.array([ma])

    ntp_avail = len(a)
    if clones == None:
        clones = ntp_avail - 1
    elif clones > ntp_avail - 1:
        logmessage = "add_orbits.initialize_from_heliocentric_Find_Orb_orbit failed\n"
        logmessage += "the number of clones specified is more than than the length of orbital\n"
        logmessage += "element arrays that were provided."
        if logf != "screen":
            print(logmessage)
        if logf:
            tools.writelog(logf, logmessage)
        return 0, sim

    # make sure planets is a list and make all planet names lowercase
    if not (type(planets) is list):
        planets = [planets]
    planets = [pl.lower() for pl in planets]
    if planets == ["outer"]:
        planets = ["jupiter", "saturn", "uranus", "neptune"]
    if planets == ["inner+outer"]:
        planets = ["venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]
    if planets == ["all"]:
        planets = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]

    # Note: the value of GM for the sun assumed in Find_orb (not the same as JPL!):
    # is coded as const.find_orb_sunGM

    # set up the rebound simulation, adding the planets first
    # add the planets and return the position/velocity corrections for
    # missing planets

    apflag, sim, sx, sy, sz, svx, svy, svz = run_reb.add_planets(
        sim, planets=planets, epoch=epoch, logfile=logf
    )
    if apflag < 1:
        logmessage = "add_orbits.initialize_from_heliocentric_Find_Orb_orbit failed at run_reb.add_planets"
        if logf != "screen":
            print(logmessage)
        if logf:
            tools.writelog(logf, logmessage)
        return 0, sim

    # First, we need to convert the Find_Orb orbit to heliocentric
    # cartesian variables using Find_Orb's assumed solar GM
    # which is in km^2/s^2, so have to convert a to km first
    a = a / const.kmtoau
    for n in range(0, clones + 1):
        i, x, y, z, vx, vy, vz = tools.aei_to_xv(
            GM=const.find_orb_sunGM, a=a[n], e=e[n], inc=inc[n], node=node[n], argperi=aperi[n], ma=ma[n]
        )
        if i < 1:
            logmessage = "add_orbits.initialize_from_heliocentric_Find_Orb_orbit failed\n"
            logmessage += "at tools.aei_to_xv for clone" + str(n)
            if logf != "screen":
                print(logmessage)
            if logf:
                tools.writelog(logf, logmessage)
            return 0, sim

        # those positions and velocities are in km and km/s, so need to convert
        # to au/year (defining a year as 365.25 days
        x = x * const.kmtoau
        y = y * const.kmtoau
        z = z * const.kmtoau
        vx = vx * const.kmtoau / const.stoyear
        vy = vy * const.kmtoau / const.stoyear
        vz = vz * const.kmtoau / const.stoyear

        # now we can apply the corrections from add_planets to the particle
        x += sx
        y += sy
        z += sz
        vx += svx
        vy += svy
        vz += svz

        # add a test particle to sim with that corrected orbit:
        if n == 0:
            # first clone is always just the best-fit orbit
            # and the hash is not numbered
            sbhash = str(des)
        else:
            sbhash = str(des) + "_" + str(n)
        sim.add(m=0.0, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, hash=sbhash)

    sim.move_to_com()

    if saveic:
        if saveic == True:
            ic_file = tools.ic_file_name(des=des)
            sim.save_to_file(ic_file)
        else:
            ic_file = saveic

        if datadir:
            ic_file = datadir + "/" + ic_file
        sim.save_to_file(ic_file)
        if logf:
            logmessage = "Rebound simulation initial conditions saved to " + ic_file + "\n"
            tools.writelog(logf, logmessage)

    return 1, sim
















###############################################################################
###############################################################################
###############################################################################
## Subroutines below are still in development. Use at your own risk!
###############################################################################
###############################################################################

def initialize_from_heliocentric_DE440_orbit(
    des=None,
    clones=None,
    a=1.0,
    e=0.0,
    inc=0.0,
    node=0.0,
    aperi=0.0,
    ma=0.0,
    planets=["all"],
    epoch=None,
    ecl_or_inv="ecl",
    cov_orb=None,
    datadir="",
    saveic=False,
    logfile=False,
):
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

    if logfile == True:
        logf = tools.log_file_name(des=des)
    else:
        logf = logfile

    if datadir:
        tools.check_datadir(datadir)

    if datadir and logf and logf != "screen":
        logf = datadir + "/" + logf

    sim = rebound.Simulation()

    if epoch == None:
        logmessage = "add_orbits.initialize_from_heliocentric_DE440_orbit failed\n"
        logmessage += "you must provide an epoch for the orbits"
        if logf != "screen":
            print(logmessage)
        if logf:
            tools.writelog(logf, logmessage)
        return 0, sim

    if des == None:
        logmessage = "add_orbits.initialize_from_heliocentric_DE440_orbit failed\n"
        logmessage += "you must provide a designation (used to label the particle)"
        if logf != "screen":
            print(logmessage)
        if logf:
            tools.writelog(logf, logmessage)
        return 0, sim

    # make sure planets is a list and make all planet names lowercase
    if not (type(planets) is list):
        planets = [planets]
    planets = [pl.lower() for pl in planets]
    if planets == ["outer"]:
        planets = ["jupiter", "saturn", "uranus", "neptune"]
    if planets == ["inner+outer"]:
        planets = ["venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]
    if planets == ["all"]:
        planets = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]

    # import the following hard-coded constants:
    # The value of GM for the sun assumed in DE440 and DE441 (not the same as JPL!):
    # Planet physical parameters
    # SS_GM[0:9] in km^3 s^–2
    # SS_r[0:9] all in au
    # reasonable integration timesteps for each planet:
    # dt[0:9]

    i, x, y, z, vx, vy, vz = tools.aei_to_xv(
        GM=const.SS_GM[0] * const.kmtoau**3 / const.stoyear**2,
        a=a,
        e=e,
        inc=inc,
        node=node,
        argperi=aperi,
        ma=ma,
    )

    if clones == None:
        clones = 0

    if clones > 0:
        if cov_orb == None:
            logmessage = "add_orbits.initialize_from_heliocentric_DE440_orbit failed\n"
            logmessage += (
                "No covariance matrix was provided, so SBdynT cannot produce clones for this particle.\n"
            )
            logmessage += (
                "Submit a covariance matrix with columns and rows associated with [a,e,inc,node,argperi,ma]\n"
            )
            logmessage += "or set clones to None.\n"
            if logf != "screen":
                print(logmessage)
            if logf:
                tools.writelog(logf, logmessage)
            return 0, sim
        mean = np.array([a, e, inc, node, aperi, ma])
        ca, ce, cinc, cnode, caperi, cma = np.random.multivariate_normal(mean, cov_orb, clones).T
        ca = ca
        if clones == 1:
            ca = [ca]
            ce = [ce]
            cinc = [cinc]
            cnode = [cnode]
            caperi = [caperi]
            cma = [cma]

        cxl = np.zeros(clones)
        cyl = np.zeros(clones)
        czl = np.zeros(clones)
        cvxl = np.zeros(clones)
        cvyl = np.zeros(clones)
        cvzl = np.zeros(clones)
        for i in range(clones):
            ci, cx, cy, cz, cvx, cvy, cvz = tools.aei_to_xv(
                GM=const.SS_GM[0] * const.kmtoau**3 / const.stoyear**2,
                a=ca[i],
                e=ce[i],
                inc=cinc[i],
                node=cnode[i],
                argperi=caperi[i],
                ma=cma[i],
            )
            if ci < 1:
                logmessage = "add_orbits.initialize_from_heliocentric_DE440_orbit failed\n"
                logmessage += "at tools.aei_to_xv for clone" + str(i)
                if logf != "screen":
                    print(logmessage)
                if logf:
                    tools.writelog(logf, logmessage)
                return 0, sim

            cxl[i] = cx
            cyl[i] = cy
            czl[i] = cz
            cvxl[i] = cvx
            cvyl[i] = cvy
            cvzl[i] = cvz

    # now we can set up the rebound simulation, adding the planets first
    # add the planets and return the position/velocity corrections for
    # missing planets
    apflag, sim, sx, sy, sz, svx, svy, svz = run_reb.add_planets(
        sim, planets=planets, epoch=epoch, logfile=logf
    )
    if apflag < 1:
        logmessage = "add_orbits.initialize_from_heliocentric_DE440_orbit failed at run_reb.add_planets"
        if logf != "screen":
            print(logmessage)
        if logf:
            tools.writelog(logf, logmessage)
        return 0, sim

    # now we can apply the corrections from add_planets to the particle
    x += sx
    y += sy
    z += sz
    vx += svx
    vy += svy
    vz += svz

    # add a test particle to sim with that corrected orbit:
    sbhash = str(des)

    # DS: THIS FEATURE IS CURRENTLY NOT TOTALLY CORRECT, SINCE THE CENTER-OF-MASS OFFSET FROM MISSING PLANETS
    # IS WITH RESPECT TO THE ECLIPTIC STILL. THIS PRODUCES ERRORS OFFSETS IN THE STATE VECTOR ON THE ORDER OF 1e-6 AU.
    # For simulating a set of synthetic TNOs (which is why I wrote this feature), this is
    # pretty negligible, so you can still create a synthetic catalog of objects with orbits relative to the invaribale
    # plane rather safely, but be aware that the orbits will be modified very slightly.
    # However, I still do not recommend using this for anything, and if you must use it, just use it for synthetic object initialization.
    if ecl_or_inv == "inv":
        rot = rebound.Rotation.to_new_axes(newz=sim.angular_momentum())
        sim.rotate(rot)

    sim.add(m=0.0, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, hash=sbhash)

    if clones > 0:
        cxl += sx
        cyl += sy
        czl += sz
        cvxl += svx
        cvyl += svy
        cvzl += svz

        for i in range(clones):
            sbhash = str(des) + "_" + str(i + 1)
            sim.add(m=0.0, x=cxl[i], y=cyl[i], z=czl[i], vx=cvxl[i], vy=cvyl[i], vz=cvzl[i], hash=sbhash)
    sim.move_to_com()

    if saveic:
        if saveic == True:
            ic_file = tools.ic_file_name(des=des)
            sim.save_to_file(ic_file)
        else:
            ic_file = saveic

        if datadir:
            ic_file = datadir + "/" + ic_file
        sim.save_to_file(ic_file)

        if logf:
            logmessage = "Rebound simulation initial conditions saved to " + ic_file + "\n"
            tools.writelog(logf, logmessage)

    return 1, sim


def initialize_from_heliocentric_destnosim(
    des=None,
    clones=None,
    a=1.0,
    e=0.0,
    inc=0.0,
    node=0.0,
    aperi=0.0,
    ma=0.0,
    planets=["all"],
    epoch=0.0,
    sb_cov=None,
    datadir="",
    saveic=False,
    logfile=False,
):
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

    if logfile == True:
        logf = tools.log_file_name(des=des)
    else:
        logf = logfile

    if datadir:
        tools.check_datadir(datadir)

    if datadir and logf and logf != "screen":
        logf = datadir + "/" + logf

    if des == None:
        logmessage = "add_orbits.initialize_from_heliocentric_destnosim failed\n"
        logmessage += "you must provide a designation (used to label the particle)"
        if logf != "screen":
            print(logmessage)
        if logf:
            tools.writelog(logf, logmessage)
        return 0, sim

    # make sure planets is a list and make all planet names lowercase
    if not (type(planets) is list):
        planets = [planets]
    planets = [pl.lower() for pl in planets]
    if planets == ["outer"]:
        planets = ["jupiter", "saturn", "uranus", "neptune"]
    if planets == ["inner+outer"]:
        planets = ["venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]
    if planets == ["all"]:
        planets = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]

    # First, we need to convert the DESTNOSIM orbit to heliocentric
    # cartesian variables using DESTNOSIM's assumed solar GM
    # which is in km^3/s^2, so have to convert a to km first
    flag, x, y, z, vx, vy, vz = tools.aei_to_xv(
        GM=const.destnosim_GM[0], a=a, e=e, inc=inc, node=node, argperi=aperi, ma=ma
    )
    if flag < 1:
        logmessage = "add_orbits.initialize_from_heliocentric_destnosim failed\n"
        logmessage += "at first call to tools.aei_to_xv"
        if logf != "screen":
            print(logmessage)
        if logf:
            tools.writelog(logf, logmessage)
        return 0, sim

    if clones == None:
        clones = 0
    if clones > 0:
        if cov_orb == None:
            logmessage = "add_orbits.initialize_from_heliocentric_destnosim failed\n"
            logmessage += (
                "No covariance matrix was provided, so SBdynT cannot produce clones for this particle.\n"
            )
            logmessage += (
                "Submit a covariance matrix with columns and rows associated with [a,e,inc,node,argperi,ma]\n"
            )
            logmessage += "or set clones to None.\n"
            if logf != "screen":
                print(logmessage)
            if logf:
                tools.writelog(logf, logmessage)
            return 0, sim

    sim = rebound.Simulation()
    # now we can set up the rebound simulation, adding the planets first
    # add the planets and return the position/velocity corrections for
    # missing planets
    apflag, sim, sx, sy, sz, svx, svy, svz = run_reb.add_planets(
        sim, planets=planets, epoch=epoch, logfile=logf
    )
    if apflag < 1:
        logmessage = "add_orbits.initialize_from_heliocentric_Find_Orb_orbit failed at run_reb.add_planets"
        if logf != "screen":
            print(logmessage)
        if logf:
            tools.writelog(logf, logmessage)
        return 0, sim

    # now we can apply the corrections from add_planets to the particle
    x += sx
    y += sy
    z += sz
    vx += svx
    vy += svy
    vz += svz

    # add a test particle to sim with that corrected orbit:
    sbhash = str(des)
    sim.add(m=0.0, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, hash=sbhash)

    if clones > 0:
        mean = np.array([a, e, inc, node, aperi, ma])

        a_c, e_c, inc_c, node_c, argperi_c, M_c = np.random.multivariate_normal(mean, sb_cov, clones).T

        for i in range(clones):
            flag, xc, yc, zc, vxc, vyc, vzc = tools.aei_to_xv(
                GM=const.destnosim_GM[0],
                a=a_c[i],
                e=e_c[i],
                inc=inc_c[i],
                node=node_c[i],
                argperi=argperi_c[i],
                ma=M_c[i],
            )
            if flag < 1:
                logmessage = "add_orbits.initialize_from_heliocentric_destnosim failed\n"
                logmessage += "at tools.aei_to_xv for clone" + str(i)
                if logf != "screen":
                    print(logmessage)
                if logf:
                    tools.writelog(logf, logmessage)
                return 0, sim

            xc += sx
            yc += sy
            zc += sz
            vxc += svx
            vyc += svy
            vzc += svz
            sbhash = str(des) + "_" + str(i + 1)
            sim.add(m=0, x=xc, y=yc, z=zc, vx=vxc, vy=vyc, vz=vzc, hash=sbhash)

    sim.move_to_com()

    if saveic:
        if saveic == True:
            ic_file = tools.ic_file_name(des=des)
            sim.save_to_file(ic_file)
        else:
            ic_file = saveic

        if datadir:
            ic_file = datadir + "/" + ic_file
        sim.save_to_file(ic_file)
        if logf:
            logmessage = "Rebound simulation initial conditions saved to " + ic_file + "\n"
            tools.writelog(logf, logmessage)

    return 1, sim
