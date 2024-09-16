# external modules to be imported
from astroquery.jplsbdb import SBDB
import numpy as np
import requests
import json
import textwrap
from pickle import dump


# internal modules
import tools


def query_horizons_planets(obj=None, epoch=2459580.5):
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

    flag = 0

    if(obj == None):
        print("A planet name must be provided")
        print("horizons_api.query_horizons_planets failed")
        return flag, 0., 0., [0.,0.,0.], [0.,0.,0.]

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
        print("horizons_api.query_horizons_planets failed")
        print("KeyError: provided object is not one of the major planets")
        raise

    # import the following hard-coded constants:
    # Planet physical parameters
    # SS_GM[0:9] in km^3 s^–2
    # SS_r[0:9] all in au
    import hard_coded_constants as const

    # calculate planet masses in solar masses
    mass = const.SS_GM[planet_id[obj]]/const.SS_GM[planet_id['sun']]
    rad = const.SS_r[planet_id[obj]]

    # we don't actually need to query for the Sun because
    # we are working in heliocentric coordinates
    if(obj == 'sun'):
        flag = 1
        return flag, mass, rad, x, v    
    
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
        print("horizons_api.query_horizons_planets failed")
        print("Unable to decode JSON results from Horizons API request")
        return flag, mass, rad, x, v
    # pull the lines we need from the resulting plain text return
    try:
        xvline = data["result"].split("X =")[1].split("\n")
    except:
        print("horizons_api.query_horizons_planets failed")
        print("Unable to find \"X =\" in Horizons API request result:")
        print(data["result"])
        return flag, mass, rad, x, v

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
        print("horizons_api.query_horizons_planets failed")
        print("Unable to find Y,Y,Z, VX, VY, VZ in Horizons API request result")
        return flag, mass, rad, x, v

    # the query was successful, return the results!
    flag = 1
    return flag, mass, rad, x, v
###############################################################################


def query_sb_from_jpl(des=None, clones=0, cloning_method='Gaussian',
                      logfile=False, save_sbdb=False, datadir='./'):
    """
    Get the orbit and covariance matrix of a small body from JPL's small
    body database browser, query Horizons for the value of GM that goes
    with that orbit, then convert the best-fit and clones (if desired)
    to heliocentric cartesian positions and velocities

    inputs:
        des: string, the designation for the object in the SBDB
        clones (optional): integer, number of times to clone using the
                           covariance matrix
        cloning_method (optional): string,  defaults to standard Guassian 
                           sampling if set to 'find_3_sigma' the first two
                           returned clones will be approximately 3-sigma
                           min and max semimajor axis clones
                           if clones>2, the rest will be sampled in a 
                           Guassian manner
        logfile (optional): boolean or string; 
                            if True:  will save some messages to adefault log file name
                            or to a file with the name equal to the string passed
                            (default) if False nothing is saved
        save_sbdb (optional): boolean or string; 
                           if True:  will save a pickle file with the results of the 
                           JPL SBDB query either to a default file name or to a file
                           with the name equal to the string passed
                           (default) if False nothing is saved
                           
    outputs:
        flag: integer, 1 if query worked, 0 otherwise)
        x: np array (size clones+1), cartesian heliocentric x (au)
        y: np array (size clones+1), cartesian heliocentric y (au)
        z: np array (size clones+1), cartesian heliocentric z (au)
        vx: np array (size clones+1), cartesian heliocentric vx (au)
        vy: np array (size clones+1), cartesian heliocentric vy (au)
        vz: np array (size clones+1), cartesian heliocentric vz (au)
        weights:
            numpy array of weights for the clones added to the simulation
            in the default sampling method, clones are equally weighted, 
            so it's an array of ones
            when cloning_method = 'find_3_sigma' and nclones > 2, the weights 
            of the two extreme clones are set to 0 and the rest to 1

        all return values set to 0 if unsuccessful
    """

    flag = 0
    if(cloning_method == 'find_3_sigma'):
        find_3_sigma=True
    else:
        find_3_sigma=False

    if(not cloning_method == 'find_3_sigma' and not cloning_method == 'Gaussian'):
        print("unsupported cloning method!")
        print("Right now only 'Gaussian' and 'find_3_sigma' are implemented")
        print("horizons_api.query_sb_from_jpl failed")
        return flag, 0., 0., 0., 0., 0., 0., 0., 0.

    if(find_3_sigma and clones < 2):
        print("horizons_api.query_sb_from_jpl failed")
        print("if using cloning_method='find_3_sigma', clones must >= 2")
        return flag, 0., 0., 0., 0., 0., 0., 0., 0.

    if(logfile==True):
        logfile = tools.log_file_name(des=des)
        logfile = datadir + '/' + logfile

    pdes, destype = tools.mpc_designation_translation(des)

    try:
        # query the JPL small body database browser for the best-fit
        # orbit and associated covariance matrix
        obj = SBDB.query(pdes, full_precision=True, covariance='mat', phys=True)
    except:
        print("horizons_api.query_sb_from_jpl failed")
        print("first attempted JPL small body database browser query failed, returning:")
        print(obj)
        return flag, 0., 0., 0., 0., 0., 0., 0., 0.
    

    #some objects can't be found with their packed designation, 
    #so let's be sure the above didn't return an error code
    errorcode = None
    try:
        errorcode = obj['code']
    except KeyError:
        errorcode = None
    
    if(errorcode == 200):
        #try querying from the user-input version of the designation
        try:
            # query the JPL small body database browser for the best-fit
            # orbit and associated covariance matrix
            obj = SBDB.query(des, full_precision=True, covariance='mat', phys=True)
        except:
            print("horizons_api.query_sb_from_jpl failed")
            print("second attempted JPL small body database browser query failed, returning:")
            print(obj)
            return flag, 0., 0., 0., 0., 0., 0., 0., 0.

    #check to see if the user-provided designation is the same type as the primary one
    #if the user gave a provisional designation, but the object is numbered, the SBDB
    #query won't return the most up-to-date orbit (even though the darned system knows
    #the provisional designation corresponds to the numbered object...grr

    sbdbpdes, sbdbdestype = tools.mpc_designation_translation(obj['object']['des'])
    if(sbdbdestype != destype):
        try:
            newdes = obj['object']['des']
            obj = SBDB.query(newdes, full_precision=True, covariance='mat', phys=True)   
        except:
            print("horizons_api.query_sb_from_jpl failed")
            print("The user-provided designation was not the most up to date designation")            
            print("third attempted JPL small body database browser query failed, returning:")
            print(obj) 
            return flag, 0., 0., 0., 0., 0., 0., 0., 0.

    #save the SBDB query results using pickle if that's desired
    if(save_sbdb):
        if(save_sbdb == True):
            orbit_file = tools.orbit_solution_file(des)
            orbit_file = datadir + '/' + orbit_file
        else:
            orbit_file = datadir + '/' + save_sbdb
        try:
            with open(orbit_file, "wb") as f:
                dump(obj, f)
        except:
            print("unable to write the SBDB query to a file")
            print("tried to write to %s" % orbit_file)
            return flag, 0., 0., 0., 0., 0., 0., 0., 0.
        if(logfile):
            logmessage = "SBDB query results saved to " + orbit_file + "\n"
            tools.writelog(logfile,logmessage)


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
            try:
                cepoch = np.float64(str(objcov['epoch']).split()[0])
            except:
                #there isn't a covariance matrix
                cepoch = 0.
            oepoch = np.float64(str(obj['orbit']['epoch']).split()[0])
            if(cepoch != oepoch and clones > 0 and cepoch != 0.):
                print("horizons_api.query_sb_from_jpl failed")
                warningstring = ("JPL small body database browser query did not"
                               + "return a best fit orbit at the same epoch as "
                               + "the covariance matrix. Query Failed.")
                print(textwrap.fill(warningstring, 80))
            if(cepoch != oepoch and clones > 0 and cepoch == 0.):
                return flag, 0., 0., 0., 0., 0., 0., 0., 0.
                print("horizons_api.query_sb_from_jpl failed")
                warningstring = ("JPL small body database browser query did not "
                              + "return the expected data for the orbit and "
                              + "covariance matrix")
                print(textwrap.fill(warningstring, 80))
                print(obj)
                return flag, 0., 0., 0., 0., 0., 0., 0., 0.

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
                flag = 2
                if(logfile):
                    logmessage = "best-fit-orbit has a <30 day arc!\n"
                    tools.writelog(logfile,logmessage)

            elif(arc < 30.):
                print("horizons_api.query_sb_from_jpl failed")
                warningstring = ("WARNING!!! The object's observational arc is "
                              + "less than 30 days which probably means the "
                              + "orbit is of too low quality for useful "
                              + "dynamical analysis and it's not possible to "
                              + "produce useful clones. "
                              + "This object can be re-run, but only for "
                              + "clones=0 and even then he results should be "
                              + "used with caution.")
                print(textwrap.fill(warningstring, 80))
                return flag, 0., 0., 0., 0., 0., 0., 0., 0.
            #no clones, so we can just use the other best-fit orbit instead
            epoch = oepoch
            objorbit = obj['orbit']['elements']
            bfecc = np.float64(str(objorbit['e']).split()[0])
            bfq = np.float64(str(objorbit['q']).split()[0])
            bfinc = np.float64(str(objorbit['i']).split()[0])
            bfnode = np.float64(str(objorbit['om']).split()[0])
            bfargperi = np.float64(str(objorbit['w']).split()[0])
            bftp = np.float64(str(objorbit['tp']).split()[0])
        except:
            print("horizons_api.query_sb_from_jpl failed")
            warningstring = ("JPL small body database browser query did not "
                          + "return the expected data for the orbit and/or "
                          + "covariance matrix")
            print(textwrap.fill(warningstring, 80))
            print(obj)
            return flag, 0., 0., 0., 0., 0., 0., 0., 0.
    
    if(bfecc >= 1. or bfecc < 0.):
        print("horizons_api.query_sb_from_jpl failed")
        print("orbital eccentricity not between 0 and 1, cannot proceed")
        return flag, 0., 0., 0., 0., 0., 0., 0., 0.
    
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
        url += pdes + "%3B'&START_TIME=" + start_time + "&STOP_TIME=" + stop_time

    
    # run the query and exit if it fails
    response = requests.get(url)
    try:
        data = json.loads(response.text)
    except ValueError:
        print("horizons_api.query_sb_from_jpl failed")
        print("Unable to decode JSON results from Horizons API request")
        flag = 0
        return flag, 0., 0., 0., 0., 0., 0., 0., 0.
    
    # this is the GM in au^2/day^2
    try:
        gmpart = data["result"].split("Keplerian GM")[1]
        gm = np.float64(gmpart.split("\n")[0].split()[1])
    except:
        print("horizons_api.query_sb_from_jpl failed")
        print("\nunable to pull the GM value from the horizons results:\n")
        print(data["result"])
        flag = 0
        return flag, 0., 0., 0., 0., 0., 0., 0., 0.
    
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
        print("horizons_api.query_sb_from_jpl failed")
        print("failed to convert to cartesian inside query_sb_from_jpl")
        flag = 0
        return flag, 0., 0., 0., 0., 0., 0., 0., 0.

    weights = np.ones(clones+1)
    if(clones > 0):
        covmat = (obj['orbit']['covariance']['data'])
        mean = [bfecc, bfq, bftp, bfnode, bfargperi, bfinc]
        # sample the covariance matrix into temporary arrays
        if(find_3_sigma):
            #sample the covariance matrix 6000 times, sort by semimajor
            #axis and pull the top and bottom ~0.1% as 3-sigma values
            tecc, tq, ttp, tnode, targperi, tinc = \
                np.random.multivariate_normal(mean, covmat, 6000).T
            tempa = tq/(1.-tecc)
            sorted_a_index = np.argsort(tempa)
            #check to make sure the 3-sigma orbits are e=0-1
            #if not, find ones that are
            if(tecc[sorted_a_index[8]] < 0):
                stop=0
                for i in range(0,5991):
                    if(ecc[sorted_a_index[i]] >= 0 and stop == 0):
                        c1 = sorted_a_index[i]
                        stop = 1
            else:
                c1 = sorted_a_index[8]
            if(tecc[sorted_a_index[5991]] > 1.):
                stop=0
                for i in range(5999,8,-1):
                    if(ecc[sorted_a_index[i]] < 1. and stop == 0):
                        c2 = sorted_a_index[i]
                        stop = 1
            else:
                c2 = sorted_a_index[5991]
            
            ecc = np.array([tecc[c1],tecc[c2]])
            q = np.array([tq[c1],tq[c2]])
            tp = np.array([ttp[c1],ttp[c2]])
            node = np.array([tnode[c1],tnode[c2]])
            argperi = np.array([targperi[c1],targperi[c2]])
            inc = np.array([tinc[c1],tinc[c2]])
            
            if(clones > 2):
                #sample the rest of the clones 
                tecc, tq, ttp, tnode, targperi, tinc = \
                    np.random.multivariate_normal(mean, covmat, clones).T
                tecc[0:2] = ecc[0:2]
                tq[0:2] = q[0:2]
                ttp[0:2] = tp[0:2]
                tnode[0:2] = node[0:2]
                targperi[0] = argperi[0:2]
                tinc[0:2] = inc[0:2]
                ecc = tecc.copy()
                q = tq.copy()
                tp = ttp.copy()
                argperi = targperi.copy()
                node = tnode.copy()
                inc = tinc.copy()
                #set the weights of the 3-sigma clones to zero so the gaussian
                #clones can still be used later to calculate clone statistics
                weights[1] = 0.
                weights[2] = 0.
        else:
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
        
        # convert clones into standard elements then cartesian coordinates
        for j in range(clones):
            a = q[j]/(1.-ecc[j])
            mm = gm/(a*a*a)  # mean motion
            mm = np.sqrt(mm)
            ma = mm*(epoch-tp[j])  # translate time of peri to mean anomaly
            i, x[j+1], y[j+1], z[j+1], vx[j+1], vy[j+1], vz[j+1] = \
                        tools.aei_to_xv(GM=gm, a=a, e=ecc[j], inc=inc[j],
                                node=node[j], argperi=argperi[j], ma=ma)
            if(i < 1):
                print("horizons_api.query_sb_from_jpl failed")
                print("failed to convert to cartesian "
                      + "inside cloning part of query_sb_from_jpl")
                flag = 0
                return flag, 0., 0., 0., 0., 0., 0., 0., 0.
        # convert from au/d to au/yr
        vx = vx*365.25
        vy = vy*365.25
        vz = vz*365.25 
        if(flag<1):
            flag = 1
        return flag, epoch, x, y, z, vx, vy, vz, weights
    else:
        # send back just the best-fit
        # after converting from au/d to au/yr
        vx0 = vx0*365.25
        vy0 = vy0*365.25
        vz0 = vz0*365.25
        if(flag<1):
            flag = 1        
        return flag, epoch, x0, y0, z0, vx0, vy0, vz0, weights


def query_sb_from_horizons(des=None, epoch=2459580.5):
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

    flag = 0

    if(des == None):
        print("The designation of one or more small bodies must be provided")
        print("failed in horizons_api.query_sb_from_horizons()")
        return flag, 0.,0.,0.,0.,0.,0.

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
            url += pdes + "%3B'&START_TIME=" + start_time + "&STOP_TIME=" + stop_time

        # run the query and exit if it fails
        response = requests.get(url)
        try:
            data = json.loads(response.text)
        except ValueError:
            print("horizons_api.query_sb_from_horizons failed")
            print("Unable to decode JSON results from Horizons API request for %s"
                  % (des[n]))
            return flag, x, y, z, vx, vy, vz

        try:
            data = json.loads(response.text)
        except ValueError:
            print("horizons_api.query_sb_from_horizons failed")
            print("Unable to decode JSON results from Horizons API request for %s"
                   % (des[n]))
            return flag, x, y, z, vx, vy, vz

        # pull the lines we need from the resulting plain text return
        try:
            xvline = data["result"].split("X =")[1].split("\n")
        except:
            print("horizons_api.query_sb_from_horizons failed")
            print("Unable to find \"X =\" in Horizons API request result for %s:"
                  % (des[n]))
            print(data["result"])
            return flag, x, y, z, vx, vy, vz

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
            print("horizons_api.query_sb_from_horizons failed")
            print("Unable to find Y,Y,Z, VX, VY, VZ in Horizons API "
                  "request result for %s" %(des[n]))
            return flag, x, y, z, vx, vy, vz

    flag = 1
    if(ntp == 1):
        #return just single values instead of numpy arrays
        return flag, x[0], y[0], z[0], vx[0], vy[0], vz[0]
    else:
        return flag, x, y, z, vx, vy, vz
