# external modules to be imported
from astroquery.jplsbdb import SBDB
import numpy as np
import requests
import json
import textwrap

# internal modules
import tools


def query_horizons_planets(obj='', epoch=2459580.5):
    """
    Get the heliocentric position and velocity of a major planet from 
    JPL Horizons via web API request

    inputs:
        obj: string, major planet name (not case sensitive)
        epoch (optional): float, JD date, defaults to Jan 1, 2022

    outputs:
        flag: integer, 0 if nothing was queried, 
                       1 if query is successful)
        m: float, object mass (in solar masses)
        r: float, object radius (in au)
        x: np array, cartesian heliocentric positions (in au)
        v: np array, cartesian heliocentric velocities (in au/year)
        all return values set to 0 if unsuccessful
    """

    obj = obj.lower()
    # define the planet-id numbers used by Horizons for the barycenter
    # of each major planet in the solar system
    planet_id = {'sun': 0, 'mercury': 1, 'venus': 2, 'earth': 3, 'mars': 4,
                 'jupiter': 5, 'saturn': 6, 'uranus': 7, 'neptune': 8}
    # initialize the x and v return variables to zero.
    x = np.zeros(3)
    v = np.zeros(3)

    # exit if the object isn't one of the major planets
    try:
        des = str(planet_id[obj])
    except KeyError:
        print("KeyError: provided object is not one of the major planets")
        raise

    # array of GM values queried January 2022 (there isn't a way to get
    # this from Horizons via API, so we just have to hard code it)
    # values for giant planet systems are from Park et al. 2021 DE440
    # and DE441, https://doi.org/10.3847/1538-3881/abd414
    # all in km^3 kg^–1 s^–2
    # G = 6.6743015e-20 #in km^3 kg^–1 s^–2
    SS_GM = np.zeros(9)
    SS_GM[0] = 132712440041.93938  # Sun
    SS_GM[1] = 22031.868551  # Mercury
    SS_GM[2] = 324858.592  # Venus
    SS_GM[3] = 398600.435507 + 4902.800118  # Earth + Moon
    SS_GM[4] = 42828.375816  # Mars system
    SS_GM[5] = 126712764.10  # Jupiter system
    SS_GM[6] = 37940584.8418  # Saturn system
    SS_GM[7] = 5794556.4  # Uranus system
    SS_GM[8] = 6836527.10058  # Neptune system

    # array of physical radius values queried January 2022
    # (again, not possible to pull directly via API)
    kmtoau = (1000./149597870700.)  # 1000m/(jpl's au in m) = 6.68459e-9
    SS_r = np.zeros(9)
    SS_r[0] = 695700.*kmtoau  # Sun
    SS_r[1] = 2440.53*kmtoau  # Mercury
    SS_r[2] = 6051.8*kmtoau  # Venus
    SS_r[3] = 6378.136*kmtoau  # Earth
    SS_r[4] = 3396.19*kmtoau  # Mars
    SS_r[5] = 71492.*kmtoau  # Jupiter system
    SS_r[6] = 60268.*kmtoau  # Saturn system
    SS_r[7] = 25559.*kmtoau  # Uranus system
    SS_r[8] = 24764.*kmtoau  # Neptune system

    # calculate planet masses in solar masses
    mass = SS_GM[planet_id[obj]]/SS_GM[planet_id['sun']]
    rad = SS_r[planet_id[obj]]

    # we don't actually need to query for the Sun because
    # we are working in heliocentric coordinates
    if(obj == 'sun'):
        return 1, mass, rad, x, v    
    
    # build the url to query horizons
    start_time = "JD"+str(epoch)
    stop_time = "JD"+str(epoch+1)
    url = ("https://ssd.jpl.nasa.gov/api/horizons.api"
        + "?format=json&EPHEM_TYPE=Vectors&OBJ_DATA=YES&CENTER="
        + "'@Sun'&OUT_UNITS='AU-D'&COMMAND="
        + des + "&START_TIME=" + start_time + "&STOP_TIME=" + stop_time)
    
    # run the query and exit if it fails
    response = requests.get(url)
    try:
        data = json.loads(response.text)
    except ValueError:
        print("Unable to decode JSON results from Horizons API request")
        return 0, mass, rad, x, v
    # pull the lines we need from the resulting plain text return
    try:
        xvline = data["result"].split("X =")[1].split("\n")
    except:
        print("Unable to find \"X =\" in Horizons API request result:")
        print(data["result"])
        return 0, mass, rad, x, v

    try:
        # heliocentric positions:
        x[0] = float(xvline[0].split()[0])
        x[1] = float(xvline[0].split("Y =")[1].split()[0])
        x[2] = float(xvline[0].split("Z =")[1].split()[0])

        # heliocentric velocities converted from au/d to au/yr
        v[0] = float(xvline[1].split("VX=")[1].split()[0])*365.25
        v[1] = float(xvline[1].split("VY=")[1].split()[0])*365.25
        v[2] = float(xvline[1].split("VZ=")[1].split()[0])*365.25
    except:
        print("Unable to find Y,Y,Z, VX, VY, VZ in Horizons API request result")
        return 0, mass, rad, x, v

    # the query was successful, return the results!
    return 1, mass, rad, x, v
###############################################################################


def query_sb_from_jpl(des='', clones=0):
    """
    Get the orbit and covariance matrix of a small body from JPL's small
    body database browser, query Horizons for the value of GM that goes
    with that orbit, then convert the best-fit and clones (if desired)
    to heliocentric cartesian positions and velocities

    inputs:
        des: string, the designation for the object in the SBDB
        clones (optional): integer, number of times to clone using the
                           covariance matrix

    outputs:
        flag: integer, 1 if query worked, 0 otherwise)
        x: np array (size clones+1), cartesian heliocentric x (au)
        y: np array (size clones+1), cartesian heliocentric y (au)
        z: np array (size clones+1), cartesian heliocentric z (au)
        vx: np array (size clones+1), cartesian heliocentric vx (au)
        vy: np array (size clones+1), cartesian heliocentric vy (au)
        vz: np array (size clones+1), cartesian heliocentric vz (au)
        all return values set to 0 if unsuccessful
    """
    pdes, destype = tools.mpc_designation_translation(des)

    try:
        # query the JPL small body database browser for the best-fit
        # orbit and associated covariance matrix
        obj = SBDB.query(pdes, full_precision=True, covariance='mat', phys=True)
    except:
        print("JPL small body database browser query failed")
        return 0, 0., 0., 0., 0., 0., 0., 0.
        
    deg2rad = np.pi/180.

    try:
        # pull the best-fit orbit that was calculated at the same
        # epoch as the covariance matrix
        objcov = obj['orbit']['covariance']
        epoch = np.float64(str(objcov['epoch']).split()[0])
        bfecc = np.float64(str(objcov['elements']['e']).split()[0])
        bfq = np.float64(str(objcov['elements']['q']).split()[0])
        bfinc = np.float64(str(objcov['elements']['i']).split()[0])
        bfnode = np.float64(str(objcov['elements']['om']).split()[0])
        bfargperi = np.float64(str(objcov['elements']['w']).split()[0])
        bftp = np.float64(str(objcov['elements']['tp']).split()[0])
    except:
        # the covariance matrix wasn't there or it didn't have a best
        # fit orbit attached in the expected data structure
        try:
            # check if the best fit orbit reported higher-up in the
            # data structure is for the same epoch as the covariance
            # matrix
            objcov = obj['orbit']['covariance']
            cepoch = np.float64(str(objcov['epoch']).split()[0])
            oepoch = np.float64(str(obj['orbit']['epoch']).split()[0])
            if(cepoch != oepoch and clones > 0):
                warningstring = ("JPL small body database browser query did not"
                               + "return a best fit orbit at the same epoch as "
                               + "the covariance matrix. Query Failed.")
                print(textwrap.fill(warningstring, 80))
                return 0, 0., 0., 0., 0., 0., 0., 0.
            arc = np.float64(str(obj['orbit']['data_arc'].split()[0]))
            if(arc < 30. and clones == 0):
                warningstring = ("WARNING!!! The object's observational arc is "
                              + "less than 30 days which probably means the "
                              + "orbit is of too low quality for useful "
                              + "dynamical analysis and it's not possible to "
                              + "produce useful clones. "
                              + "This best-fit orbit will still be run, but "
                              + "the results should be used with caution")
                print(textwrap.fill(warningstring, 80))
            elif(arc < 30.):
                warningstring = ("WARNING!!! The object's observational arc is "
                              + "less than 30 days which probably means the "
                              + "orbit is of too low quality for useful "
                              + "dynamical analysis and it's not possible to "
                              + "produce useful clones. "
                              + "This object can be re-run, but only for "
                              + "clones=0 and even then he results should be "
                              + "used with caution.")
                print(textwrap.fill(warningstring, 80))
                return 0, 0., 0., 0., 0., 0., 0., 0.
            epoch = oepoch
            objorbit = obj['orbit']['elements']
            bfecc = np.float64(str(objorbit['e']).split()[0])
            bfq = np.float64(str(objorbit['q']).split()[0])
            bfinc = np.float64(str(objorbit['i']).split()[0])
            bfnode = np.float64(str(objorbit['om']).split()[0])
            bfargperi = np.float64(str(objorbit['w']).split()[0])
            bftp = np.float64(str(objorbit['tp']).split()[0])
        except:
            warningstring = ("JPL small body database browser query did not "
                          + "return the expected data for the orbit and "
                          + "covariance matrix")
            print(textwrap.fill(warningstring, 80))
            return 0, 0., 0., 0., 0., 0., 0., 0.
    
    if(bfecc >= 1. or bfecc < 0.):
        print("orbital eccentricity not between 0 and 1, cannot proceed")
        return 0, 0., 0., 0., 0., 0., 0., 0.
    
    # We have to query JPL horizons to find out what exact value of GM
    # was used for the orbit fit above (this should be in the SBDB but
    # alas it is not!)
    
    # build the url to query horizons
    # if the designation being used is a provisional one, we will
    # translate it to a packed designation for cleaner searching
    url = 'https://ssd.jpl.nasa.gov/api/horizons.api'
    start_time = 'JD'+str(epoch)
    stop_time = 'JD'+str(epoch+1)
    url += "?format=json&EPHEM_TYPE=ELEMENTS&OBJ_DATA=YES&CENTER='@Sun'"
    if(destype == 'provisional'):
        url += "&OUT_UNITS='AU-D'&COMMAND='DES="
        url += pdes + "'&START_TIME=" + start_time + "&STOP_TIME=" + stop_time
    elif(destype == 'other'):
        url += "&OUT_UNITS='AU-D'&COMMAND='DES="
        url += pdes + "%3BCAP%3BNOFRAG'&START_TIME=" + start_time + "&STOP_TIME=" + stop_time
    else:
        url += "&OUT_UNITS='AU-D'&COMMAND='"
        url += pdes + "'&START_TIME=" + start_time + "&STOP_TIME=" + stop_time

    
    # run the query and exit if it fails
    response = requests.get(url)
    try:
        data = json.loads(response.text)
    except ValueError:
        print("Unable to decode JSON results from Horizons API request")
        return 0, 0., 0., 0., 0., 0., 0., 0.
    
    # this is the GM in au^2/day^2
    try:
        gmpart = data["result"].split("Keplerian GM")[1]
        gm = np.float64(gmpart.split("\n")[0].split()[1])
    except:
        print("\nunable to pull the GM value from the horizons results:\n")
        print(data["result"])
        return 0, 0., 0., 0., 0., 0., 0., 0.
    
    # calculate the more standard orbital elements for the best-fit orbit
    a0 = bfq/(1.-bfecc)
    i0 = bfinc*deg2rad
    O0 = bfnode*deg2rad
    w0 = bfargperi*deg2rad
    mm = gm/(a0*a0*a0)  # mean motion
    mm = np.sqrt(mm)
    ma0 = mm*(epoch-bftp)  # translate time of perihelion to mean anomaly
    
    i, x0, y0, z0, vx0, vy0, vz0 = tools.aei_to_xv(
                    GM=gm, a=a0, e=bfecc, inc=i0, node=O0, argperi=w0, ma=ma0)
    if(i < 1):
        print("failed to convert to cartesian inside query_sb_from_jpl")
        return 0, 0., 0., 0., 0., 0., 0., 0.

    if(clones > 0):
        covmat = (obj['orbit']['covariance']['data'])
        mean = [bfecc, bfq, bftp, bfnode, bfargperi, bfinc]
        # sample the covariance matrix into temporary arrays
        ecc, q, tp, node, argperi, inc = \
            np.random.multivariate_normal(mean, covmat, clones).T
        node = node*deg2rad
        argperi = argperi*deg2rad
        inc = inc*deg2rad
        
        # set up output arrays
        x = np.zeros(clones+1)
        y = np.zeros(clones+1)
        z = np.zeros(clones+1)
        vx = np.zeros(clones+1)
        vy = np.zeros(clones+1)
        vz = np.zeros(clones+1)
        x[0] = x0
        y[0] = y0
        z[0] = z0
        vx[0] = vx0
        vy[0] = vy0
        vz[0] = vz0
        
        # convert clones into standarcd elements then cartesian coordinates
        for j in range(clones):
            a = q[j]/(1.-ecc[j])
            mm = gm/(a*a*a)  # mean motion
            mm = np.sqrt(mm)
            ma = mm*(epoch-tp[j])  # translate time of peri to mean anomaly
            i, x[j+1], y[j+1], z[j+1], vx[j+1], vy[j+1], vz[j+1] = \
                        tools.aei_to_xv(GM=gm, a=a, e=ecc[j], inc=inc[j],
                                node=node[j], argperi=argperi[j], ma=ma)
            if(i < 1):
                print("failed to convert to cartesian "
                      + "inside cloning part of query_sb_from_jpl")
                return 0, 0., 0., 0., 0., 0., 0., 0.
        # convert from au/d to au/yr
        vx = vx*365.25
        vy = vy*365.25
        vz = vz*365.25       
        return 1, epoch, x, y, z, vx, vy, vz
    else:
        # send back just the best-fit
        # after converting from au/d to au/yr
        vx0 = vx0*365.25
        vy0 = vy0*365.25
        vz0 = vz0*365.25
        return 1, epoch, x0, y0, z0, vx0, vy0, vz0


def query_sb_from_horizons(des=[''], epoch=2459580.5):
    """
    Get the orbit of a small body (or list of small bodies) from
    Horizons at a specific epoch, returning heliocentric cartesian
    positions and velocities

    inputs:
        des: string or list of strings, the designation for the
             object or list of objects
        epoch (optional): (JD) defaults to Jan 1, 2022

    outputs:
        flag: integer, 1 if query worked, 0 otherwise
        x: np array (size=len(des)), cartesian heliocentric x (au)
        y: np array (size=len(des)), cartesian heliocentric y (au)
        z: np array (size=len(des)), cartesian heliocentric z (au)
        vx: np array (size=len(des)), cartesian heliocentric vx (au)
        vy: np array (size=len(des)), cartesian heliocentric vy (au)
        vz: np array (size=len(des)), cartesian heliocentric vz (au)
        all return values set to 0 if unsuccessful
    """

    # if the user provided just a single string as the designation
    # turn it into a list
    if not (type(des) is list):
        des = [des]
    ntp = len(des)
    # initialize the position
    x = np.zeros(ntp)
    y = np.zeros(ntp)
    z = np.zeros(ntp)
    vx = np.zeros(ntp)
    vy = np.zeros(ntp)
    vz = np.zeros(ntp)

    for n in range(0,ntp):
        # build the url to query horizons
        # if the designation being used is a provisional one, we will
        # translate it to a packed designation for cleaner searching.
        # Numbered objects and temporary designation objects have to
        # be searched slightly differently

        pdes, destype = tools.mpc_designation_translation(des[n])
        start_time = 'JD' + str(epoch)
        stop_time = 'JD' + str(epoch + 1)
        url = ("https://ssd.jpl.nasa.gov/api/horizons.api"
               + "?format=json&EPHEM_TYPE=Vectors&OBJ_DATA=YES&CENTER='@Sun'")
        if(destype == 'provisional'):
            url += "&OUT_UNITS='AU-D'&COMMAND='DES="
            url += pdes + "'&START_TIME=" + start_time + "&STOP_TIME=" + stop_time
        elif(destype == 'other'):
            url += "&OUT_UNITS='AU-D'&COMMAND='DES="
            url += pdes + "%3BCAP%3BNOFRAG'&START_TIME=" + start_time + "&STOP_TIME=" + stop_time
        else:
            url += "&OUT_UNITS='AU-D'&COMMAND='"
            url += pdes + "'&START_TIME=" + start_time + "&STOP_TIME=" + stop_time

        # run the query and exit if it fails
        response = requests.get(url)
        try:
            data = json.loads(response.text)
        except ValueError:
            print("Unable to decode JSON results from Horizons API request for %s"
                  % (des[n]))
            return 0, x, y, z, vx, vy, vz

        try:
            data = json.loads(response.text)
        except ValueError:
            print("Unable to decode JSON results from Horizons API request for %s"
                   % (des[n]))
            return 0, x, y, z, vx, vy, vz

        # pull the lines we need from the resulting plain text return
        try:
            xvline = data["result"].split("X =")[1].split("\n")
        except:
            print("Unable to find \"X =\" in Horizons API request result for %s:"
                  % (des[n]))
            print(data["result"])
            return 0, x, y, z, vx, vy, vz

        try:
            # heliocentric positions:
            x[n] = float(xvline[0].split()[0])
            y[n] = float(xvline[0].split("Y =")[1].split()[0])
            z[n] = float(xvline[0].split("Z =")[1].split()[0])

            # heliocentric velocities converted from au/d to au/yr
            vx[n] = float(xvline[1].split("VX=")[1].split()[0]) * 365.25
            vy[n] = float(xvline[1].split("VY=")[1].split()[0]) * 365.25
            vz[n] = float(xvline[1].split("VZ=")[1].split()[0]) * 365.25
        except:
            print("Unable to find Y,Y,Z, VX, VY, VZ in Horizons API "
                  "request result for %s" %(des[n]))
            return 0, x, y, z, vx, vy, vz

    return 1, x, y, z, vx, vy, vz
