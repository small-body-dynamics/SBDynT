import numpy as np
import re
import rebound
from datetime import date

#define the default file-naming schemes
def archive_file_name(des=None):
    '''
    if the user doesn't provide a simulation archive filename
    this function will be used to make the default
    input: 
        des, string: small body designation
    output:
        archivefile, string: filename for the simulation archive
    '''

    if(des == None):
        archivefile = 'simarchive.bin'

    #for the default file names, use the first designation
    #if this is a list of objects instead of just one
    if (type(des) is list):
        pdes = des[0]
    else:
        pdes = des

    #make the filename while removing any whitespaces from des
    archivefile = "".join(pdes.split()) + '-simarchive.bin'

    return archivefile

def ic_file_name(des=None):
    '''
    if the user doesn't provide an initial conditions filename
    this function will be used to make the default
    input: 
        des, string: small body designation
    output:
        icfile, string: filename for the simulation archive
                        that stores the initial conditions
    '''

    if(des == None):
        icfile = 'ic.bin'
   
    #for the default file names, use the first designation
    #if this is a list of objects instead of just one
    if (type(des) is list):
        pdes = des[0]
    else:
        pdes = des

    #make the filename while removing any whitespaces from des
    icfile = "".join(pdes.split()) + '-ic.bin'

    return icfile

def log_file_name(des=None):
    '''
    if the user doesn't provide a log filename
    this function will be used to make the default
    input: 
        des, string: small body designation
    output:
        icfile, string: filename for the log file
    '''

    if(des == None):
        logfile = 'log.txt'

    #for the default file names, use the first designation
    #if this is a list of objects instead of just one
    if (type(des) is list):
        pdes = des[0]
    else:
        pdes = des


    #make the filename while removing any whitespaces from des
    logfile = "".join(pdes.split()) + '-log.txt'

    return logfile

def writelog(logfile,logmessage):
    '''
    append to the log file
    inputs:
        logfile, bool or string
        logmessage, string
    '''
    if(logfile=='screen'):
        print(logmessage)
    else:
        with open(logfile,"a") as f:
            f.write(logmessage)

def orbit_solution_file(des):
    '''
    if the user doesn't provide a filename to save the orbit solution
    queried from JPL's SBDB, this function will be used to make the default
    input: 
        des, string: small body designation
    output:
        orbit_file, string: filename for the log file
    '''
    
    # this file will be date stamped to be sure it doesn't overwrite
    # a pre-existing saved orbit solution
    today = date.today()
    datestring = today.strftime("%b-%d-%Y")

    #for the default file names, use the first designation
    #if this is a list of objects instead of just one
    if (type(des) is list):
        pdes = des[0]
    else:
        pdes = des


    orbit_file = "".join(pdes.split()) + "-" + datestring + '.pkl'

    return orbit_file

#################################################################
#################################################################
# Convert orbital elements to cartesian coordinates
#################################################################
def aei_to_xv(GM=1., a=1, e=0., inc=0., node=0., argperi=0., ma=0.):
    """
    inputs:
        GM the value of GM for the orbit (sets the units)
        a = semimajor axis
        e = eccentricity
        inc = inclination in radians
        node = long. of ascending node in radians
        argperi = argument of perihelion in radians
        ma = mean anomaly in radians
    outputs:
        flag (integer: 0 if failed, 1 if succeeded)
        x, y, z = cartesian positions (units set by GM)
        vx, vy, vz = cartesian velocities (units set by GM)
    """
    
    x = np.double(0.)
    y = np.double(0.)
    z = np.double(0.)
    vx = np.double(0.)
    vy = np.double(0.)
    vz = np.double(0.)
    
    # based on M. Duncan's routines in swift
    if(e >= 1. or e < 0. or a < 0.):
        print("in tools.aei_to_xv, the provided orbital eccentricity is")
        print("not between 0 and 1, so cannot proceed with conversion")
        return 0,  x, y, z, vx, vy, vz

    sp = np.sin(argperi)
    cp = np.cos(argperi)
    so = np.sin(node)
    co = np.cos(node)
    si = np.sin(inc)
    ci = np.cos(inc)
    
    d11 = cp*co - sp*so*ci
    d12 = cp*so + sp*co*ci
    d13 = sp*si
    d21 = -sp*co - cp*so*ci
    d22 = -sp*so + cp*co*ci
    d23 = cp*si

    cape = M_to_E_reb(M=ma, e=e)
    scap = np.sin(cape)
    ccap = np.cos(cape)
    sqe = np.sqrt(1.0 - e*e)
    sqgma = np.sqrt(GM*a)
    xfac1 = a*(ccap - e)
    xfac2 = a*sqe*scap
    ri = 1.0/(a*(1.0 - e*ccap))
    vfac1 = -ri*sqgma*scap
    vfac2 = ri*sqgma*sqe*ccap

    x = d11*xfac1 + d21*xfac2
    y = d12*xfac1 + d22*xfac2
    z = d13*xfac1 + d23*xfac2
    vx = d11*vfac1 + d21*vfac2
    vy = d12*vfac1 + d22*vfac2
    vz = d13*vfac1 + d23*vfac2
    
    return 1, x, y, z, vx, vy, vz




#################################################################
#################################################################
# Convert Mean anomaly M to Eccentric anomaly E
#################################################################
def M_to_E_reb(M=0., e=0.):
    """
    inputs:
        M = Mean anomaly in radians
        e = eccentricity
    returns:
        eccentric anomaly in radians
    """

    # borrowed from rebound tools.c
    M = np.mod(M,2.*np.pi)
    if(e < 1.):
        if(e < 0.8):
            E = M
        else:
            E = np.pi
        F = E - e*np.sin(E) - M
        for i in range(0, 100):
            E = E - F/(1. - e*np.cos(E))
            F = E - e*np.sin(E) - M
            if(np.abs(F) < 1.e-16):
                break
        E =  np.mod(E,2.*np.pi)
        return E
    else:
        E = M/np.abs(M)*np.log(2.*np.abs(M)/e + 1.8)
        F = E - e*np.sinh(E) + M
        for i in range(0, 100):
            E = E - F/(1.0 - e*np.cosh(E))
            F = E - e*np.sinh(E) + M
            if(np.abs(F) < 1.e-16):
                break
        E =  np.mod(E,2.*np.pi)
        return E
#################################################################



#################################################################
#################################################################
# apply mod2pi to a 1 or 2-d array, returning a new array in case
# you don't want to modify the original array for some reason
#################################################################
def arraymod2pi(x):
    """
    input:
        x, 1 or 2-d np array, angles in radians
    output:
        mx, 1 or 2-d np array of angles in radians re-centered from 0-2pi
    """
    mx = x.copy()
    oned = False
    #make it a 2-d array if it is 1-d
    if(len(mx.shape)<2):
        oned = True
        mx = np.array([mx])
    imax = mx.shape[0]
    for i in range (0,imax):
        mx[i] = np.mod(mx[i],2.*np.pi) 
    #if needed, turn it back into a 1-d array
    if(oned):
        mx = mx[0]
    return mx

#################################################################
#################################################################
# take a 1 or 2-d array of angles, returning a new array that is
# re-centered from -pi to pi
#################################################################
def arraymod2pi0(x):
    """
    input:
        x, 1 or 2-d np array, angles in radians
    output:
        mx, 1 or 2-d np array of angles in radians re-centered from 0-2pi
    """
    mx = x.copy()
    oned = False
    #make it a 2-d array if it is 1-d
    if(len(mx.shape)<2):
        oned = True
        mx = np.array([mx])
    imax = mx.shape[0]
    for i in range (0,imax):
        mx[i] = np.mod(mx[i]+np.pi,2.*np.pi) - np.pi 
    #if needed, turn it back into a 1-d array
    if(oned):
        mx = mx[0]
    return mx



##############################
# translate MPC designations to packed designations
#####################
def split(word):
    return [char for char in word]


def mpc_designation_translation(obj):
    """
    If an unpacked provisional MPC designation is input, this routine
    outputs a packed version of that designation. If a packed number
    is input, the unpacked number is returned. Otherwise it returns 
    the same string that was input. Also returns a designation type
    which is useful because comet queries sometimes require extra
    search terms. This makes the Horizons query function more robust. 
    input:
        obj: string, minor planet designation
    output: 
        des: string, obj, packed provisional MPC designation, or
             unpacked number
        type: string, type of designation "number" 
              or "provisional" or "other"
    """
    num = {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
           "6": "6", "7": "7", "8": "8", "9": "9", "A": "10", "B": "11",
           "C": "12", "D": "13", "E": "14", "F": "15", "G": "16", "H": "17",
           "I": "18", "J": "19", "K": "20", "L": "21", "M": "22", "N": "23",
           "O": "24", "P": "25", "Q": "26", "R": "27", "S": "28", "T": "29",
           "U": "30", "V": "31", "W": "32", "X": "33", "Y": "34", "Z": "35",
           "a": "36", "b": "37", "c": "38", "d": "39", "e": "40", "f": "41",
           "g": "42", "h": "43", "i": "44", "j": "45", "k": "46", "l": "47",
           "m": "48", "n": "49", "o": "50", "p": "51", "q": "52", "r": "53",
           "s": "54", "t": "55", "u": "56", "v": "57", "w": "58", "x": "59",
           "y": "60", "z": "61"
           }
    
    hex_list = list(num.keys())
    num_list = list(num.values())
    destype = 'other'

    #search for un-packed provisional designation with or without a space
    #after the year
    regex_provis = re.compile(r"\b(\d{4})([- _]?)([a-zA-Z]{2})(\d*)\b")
    #search for a packed provisional designation
    regex_packedprovis = re.compile(r"\b([a-zA-Z]{1})(\d{2})([a-zA-Z]{1})(\d{2}|[a-zA-Z]{1}\d{1})([a-zA-Z]{1})\b")
    #search for a packed number designation    
    regex_packednum = re.compile(r"\b([a-zA-Z]{1})(\d{4})\b")
    #search for an unpacked number designation        
    regex_num = re.compile(r"^[0-9]+$")    
    #search for a named object (no numbers in name)        
    regex_name = re.compile(r"^[a-zA-Z]+$")    

    #search for a comet that's described by both number and name (i.e., has a slash)       
    regex_cometname = re.compile(r"(^[0-9]+)([a-zA-Z])([/])([a-zA-Z]+)")    



    #in case the input was not specified as a string (e.g., numbered 
    #object not enclosed in ''), convert to string
    if(not(isinstance(obj, str))):
        obj =str(obj)
    
    provis = regex_provis.findall(obj)
    packedprovis = regex_packedprovis.findall(obj)    
    packednum = regex_packednum.findall(obj)
    num = regex_num.findall(obj)    
    name = regex_name.findall(obj)    
    cometname = regex_cometname.findall(obj)    
    if (provis):
        destype = 'provisional'
        if (len(provis[0]) == 4):
            year = provis[0][0]
            letters = provis[0][2]
            number = provis[0][3]
        else:
            year = provis[0][0]
            letters = provis[0][1]
            number = provis[0][2]
        ychars = split(year)
        j = num_list.index(ychars[0] + ychars[1])
        des = hex_list[j] + ychars[2] + ychars[3]
        nchars = split(number)
        lchars = split(letters)
        des += lchars[0]
        if (len(nchars) == 3):
            j = num_list.index(nchars[0] + nchars[1])
            des += hex_list[j] + nchars[2]
        elif (len(nchars) == 1):
            des += '0' + number
        elif (len(nchars) == 0):
            des += '00' + number
        else:
            des += number
        des += lchars[1]
    elif(packedprovis):
        destype = 'provisional'
        des = obj
    elif(packednum):
        destype = 'number'
        j = hex_list.index(packednum[0][0])
        des = str(num_list[j]) + str(packednum[0][1])
    elif(num):
        destype = 'number'
        des = obj
    elif(name):
        destype = 'named'
        des = obj
    elif(cometname):
        #reduce to the numbered version of the comet (e.g,, 29P instead of 29P/SW1)
        des = cometname[0][0] + cometname[0][1]
    else:
        #this should be essentially just comets, but leaving as 'other'
        des = obj

    return des, destype




#################################################################
#################################################################
# reads the simulation archive files into barycentric orbital 
# element arrays by hash
#################################################################
def read_sa_by_hash(obj_hash=None, archivefile=None, datadir='',
                    tmin=None,tmax=None,center='bary'):
    """
    Reads the simulation archive file produced by the run_reb
    routines
        obj_hash (str): hash of the simulation particle
        archivefile (str): path to rebound simulation archive
        datadir (optional): string, path for where files are stored,
            defaults to the current directory
        tmin (optional, float): minimum time to return (years)
        tmax (optional, float): maximum time to return (years)
            if not set, the entire time range is returned
        center (optional, string): 'bary' for barycentric orbits 
            (default) and 'helio' for heliocentric orbits
    output:
        flag (int): 1 if successful and 0 if there was a problem
        a (1-d float array): semimajor axis (au)
        e (1-d float array): eccentricity
        inc (1-d float array): inclination (rad)
        node (1-d float array): longitude of ascending node (rad)
        aperi (1-d float array): argument of perihelion (rad)
        MA (1-d float array): mean anomaly (rad)
        time (1-d float array): simulations time (years)
    """
    if(obj_hash==None):
        print("The name of an object must be provided as obj_hash")
        print("tools.read_sa_by_hash failed")
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)

    if(archivefile==None):
        print("A simulation archive file must be provided as archivefile")
        print("tools.read_sa_by_hash failed")
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)
    

    if(datadir):
        archivefile = datadir + "/" + archivefile

    #read the simulation archive and return the orbit of the
    #desired particle or planet with the provided hash
    try:
        sa = rebound.Simulationarchive(archivefile)
    except:
        print("tools.read_sa_by_hash failed")
        print("Problem reading the simulation archive file:")
        print(archivefile)
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)
    
    nout = len(sa)
    if(nout <2):
        print("tools.read_sa_by_hash failed")
        print("There are fewer than two snapshots in the archive file:")
        print(archivefile)
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)

    #check if we are reading for a planet and if we need
    #to make the planet's name lowercase
    planets=['mercury', 'venus', 'earth', 'mars',
                'jupiter', 'saturn', 'uranus', 'neptune']
    if( not(obj_hash in planets) and obj_hash.lower() in planets):
        obj_hash = obj_hash.lower()

    if(tmin == None and tmax == None):
        #user didn't set these, so we will read the whole thing
        tmin = -1e11
        tmax = 1e11
    elif(tmin==None):
        tmin = -1e11
    elif(tmax == None):
        tmax = 1e11

    #correct for backwards integrations
    if(tmax < tmin):
        temp = tmax
        tmax = tmin
        tmin = temp

    a = np.zeros(nout)
    e = np.zeros(nout)
    inc = np.zeros(nout)
    node = np.zeros(nout)
    aperi = np.zeros(nout)
    ma = np.zeros(nout)
    t = np.zeros(nout)
    
    it=0
    for i,sim in enumerate(sa):
        if(sim.t < tmin or sim.t > tmax):
            #skip this 
            continue
        #calculate the object's orbit relative to the barycenter
        try:
            p = sim.particles[obj_hash]
        except:
            print("tools.read_sa_by_hash failed")
            print("Problem finding a particle with that hash in the archive")
            return 0, a, e, inc, node, aperi, ma, t

        if(center == 'bary'):
            com = sim.com()
        elif(center == 'helio'):
            com = sim.particles[0]
        else:
            print("tools.read_sa_by_hash failed")
            print("center can only be 'bary' or 'helio'\n")
            return 0, a, e, inc, node, aperi, ma, t

        o = p.orbit(com)

        t[it] = sim.t

        a[it] = o.a
        e[it] = o.e
        inc[it] = o.inc
        node[it] = o.Omega
        aperi[it] = o.omega
        ma[it] = o.M
        it+=1

    if(it==0):
        print("tools.read_sa_by_hash failed")
        print("There were no simulation archives in the desired time range")
        return 0, [0.], [0.], [0.], [0.], [0.], [0.], [0.]
    else:
        t = t[0:it]
        a = a[0:it]
        e = e[0:it]
        inc = inc[0:it]
        node = node[0:it]
        aperi = aperi[0:it]
        ma = ma[0:it]
    
    return 1, a, e, inc, node, aperi, ma, t
#################################################################


#################################################################
#################################################################
# reads the simulation archive files into barycentric orbital 
# element arrays
#################################################################
def read_sa_for_sbody(des=None, archivefile=None, datadir='',
                      clones=None,tmin=None,tmax=None,center='bary'):
    """
    Reads the simulation archive file produced by the run_reb
    routines for the small body's orbital evolution
        des: string, user-provided small body designation
        archivefile (str; optional): name/path for the simulation
            archive file to be read. If not provided, the default
            file name will be tried
        datadir (optional): string, path for where files are stored,
            defaults to the current directory
        clones (optional): number of clones to read, 
            defaults to reading all clones in the simulation
        tmin (optional, float): minimum time to return (years)
        tmax (optional, float): maximum time to return (years)
            if not set, the entire time range is returned    
        center (optional, string): 'bary' for barycentric orbits 
            (default) and 'helio' for heliocentric orbits            
    output:
        ## if clones=0, all arrays are 1-d
        flag (int): 1 if successful and 0 if there was a problem
        a (1-d or 2-d float array): semimajor axis (au)
        e (1-d or 2-d float array): eccentricity
        inc (1-d or 2-d float array): inclination (rad)
        node (1-d or 2-d float array): longitude of ascending node (rad)
        aperi (1-d or 2-d float array): argument of perihelion (rad)
        MA (1-d or 2-d float array): mean anomaly (rad)
            above arrays are for the test particles in the 
            format [particle id number, output number]
            best-fit clone is id=0, clones are numbered
            starting at 1 
        time (1-d float array): simulations time (years)
    """

    if(des == None):
        print("The designation of a small body must be provided")
        print("tool.read_sa_for_sbody failed")
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)

    if(archivefile==None):
        archivefile = archive_file_name(des)
    if(datadir):
        archivefile = datadir + '/' +archivefile

    
    try:
        sa = rebound.Simulationarchive(archivefile)
    except:
        print("tools.read_sa_for_sbody failed")
        print("Problem reading the simulation archive file:")
        print(archivefile)
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)
    
    nout = len(sa)
    if(nout <2):
        print("tools.read_sa_for_sbody failed")
        print("There are fewer than two snapshots in the archive file:")
        print(archivefile)
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)

    if(tmin == None and tmax == None):
        #user didn't set these, so we will read the whole thing
        tmin = -1e11
        tmax = 1e11
    elif(tmin==None):
        tmin = -1e11
    elif(tmax == None):
        tmax = 1e11

    #correct for backwards integrations
    if(tmax < tmin):
        temp = tmax
        tmax = tmin
        tmin = temp

    ntp_max = sa[0].N - sa[0].N_active
    
    if(clones==None):
        ntp = ntp_max
    else:
        ntp = clones+1
        if(ntp > ntp_max):
            print("Warning! the number of clones in the simulation archive is smaller than")
            print("the number of clones specfied by the user! Resetting the number of clones.")
            clones = ntp_max - 1
            ntp = npt_max
            flag = 2
    
    a = np.zeros([ntp,nout])
    e = np.zeros([ntp,nout])
    inc = np.zeros([ntp,nout])
    node = np.zeros([ntp,nout])
    aperi = np.zeros([ntp,nout])
    ma = np.zeros([ntp,nout])
    t = np.zeros(nout)
        
    it=0
    for i,sim in enumerate(sa):
        if(sim.t < tmin or sim.t > tmax):
            #skip this 
            continue
        t[it] = sim.t
        if(center == 'bary'):
            com = sim.com()
        elif(center == 'helio'):
            com = sim.particles[0]
        else:
            print("tools.read_sa_for_sbody failed")
            print("center can only be 'bary' or 'helio'\n")
            return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)
        for j in range(0,ntp):
            #the hash format for clones
            tp_hash = str(des) + "_" + str(j)
            #except the best fit is just the designation:
            if(j==0):
                tp_hash = str(des)
            try:
                p = sim.particles[tp_hash]
            except:
                print("tools.read_sa_for_sbody failed")
                print("Problem finding a particle with that hash in the archive")
                return 0, a, e, inc, node, aperi, ma, t
            o = p.orbit(com)

            a[j,it] = o.a
            e[j,it] = o.e
            inc[j,it] = o.inc
            node[j,it] = o.Omega
            aperi[j,it] = o.omega
            ma[j,it] = o.M
        it+=1

    if(it==0):
        print("tools.read_sa_for_sbody failed")
        print("There were no simulation archives in the desired time range")
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)
    else:
        t = t[0:it]
        a = a[:,0:it]
        e = e[:,0:it]
        inc = inc[:,0:it]
        node = node[:,0:it]
        aperi = aperi[:,0:it]
        ma = ma[:,0:it]

    if(clones == 0):
        return 1, a[0,:], e[0,:], inc[0,:], node[0,:], aperi[0,:], ma[0,:], t
    else:
        return 1, a, e, inc, node, aperi, ma, t
#################################################################

#################################################################
#################################################################
# reads the simulation archive files into barycentric cartesian 
# position arrays for a small body
#################################################################
def read_sa_for_sbody_cartesian(des=None, archivefile=None, clones=None,
                                datadir='', center='bary',
                                tmin=None,tmax=None):
    """
    Reads the simulation archive file produced by the run_reb
    routines for the small body's orbital evolution
        des: string, user-provided small body designation
        archivefile (str; optional): name/path for the simulation
            archive file to be read. If not provided, the default
            file name will be tried
        clones (optional): number of clones to read, 
            defaults to reading all clones in the simulation            
        datadir (optional): string, path for where files are stored,
            defaults to the current directory
        center (optional, string): 'bary' for barycentric orbits 
            (default) and 'helio' for heliocentric orbits      
        tmin (optional, float): minimum time to return (years)
        tmax (optional, float): maximum time to return (years)
            if not set, the entire time range is returned        
    output:
        ## if clones=0, all arrays are 1-d
        flag (int): 1 if successful and 0 if there was a problem
        x (1-d or 2-d float array): barycentric x (au)
        y (1-d or 2-d float array): barycentric y (au)
        z (1-d or 2-d float array): barycentric z (au)
        vx (1-d or 2-d float array): barycentric vx (au/yr)
        vy (1-d or 2-d float array): barycentric vy (au/yr)
        vz (1-d or 2-d float array): barycentric vz (au/yr)
            above arrays are for the test particles in the 
            format [particle id number, output number]
            best-fit clone is id=0, clones are numbered
            starting at 1 
        time (1-d float array): simulations time (years)
    """

    if(des == None):
        print("The designation of a small body must be provided")
        print("tool.read_sa_for_sbody_cartesian failed")
        return flag, None, 0    
    
    if(archivefile==None):
        archivefile = archive_file_name(des)
    if(datadir):
        archivefile = datadir + '/' +archivefile

    try:
        sa = rebound.Simulationarchive(archivefile)
    except:
        print("tools.read_sa_for_sbody_cartesian failed")
        print("Problem reading the simulation archive file:")
        print(archivefile)
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)
    
    nout = len(sa)
    if(nout <2):
        print("tools.read_sa_for_sbody_cartesian failed")
        print("There are fewer than two snapshots in the archive file:")
        print(archivefile)
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)


    if(tmin == None and tmax == None):
        #user didn't set these, so we will read the whole thing
        tmin = -1e11
        tmax = 1e11
    elif(tmin==None):
        tmin = -1e11
    elif(tmax == None):
        tmax = 1e11

    #correct for backwards integrations if times were entered wrong
    if(tmax < tmin):
        temp = tmax
        tmax = tmin
        tmin = temp

    if(clones==None):
        ntp = sa[0].N - sa[0].N_active
    else:
        ntp = clones+1

    x = np.zeros([ntp,nout])
    y = np.zeros([ntp,nout])
    z = np.zeros([ntp,nout])
    vx = np.zeros([ntp,nout])
    vy = np.zeros([ntp,nout])
    vz = np.zeros([ntp,nout])
    t = np.zeros(nout)
        
    it=0
        
    for i,sim in enumerate(sa):
        if(sim.t < tmin or sim.t > tmax):
            #skip this 
            continue
        t[it] = sim.t
        if(center=='helio'):
            #need the sun's position 
            dx = sim.particles[0].x
            dy = sim.particles[0].y
            dz = sim.particles[0].z
            dvx = sim.particles[0].vx
            dvy = sim.particles[0].vy
            dvz = sim.particles[0].vz
        elif(center=='bary'):
            dx = 0.
            dy = 0.
            dz = 0.
            dvx = 0.
            dvy = 0.
            dvz = 0.
        else:
            print("tools.read_sa_for_sbody_cartesian failed")
            print("center can only be 'bary' or 'helio'\n")
            return 0, x, y, z, vx, vy, vz, t

        for j in range(0,ntp):
            #the hash format for clones
            tp_hash = str(des) + "_" + str(j)
            #except the best fit is just the designation:
            if(j==0):
                tp_hash = str(des)
            try:
                p = sim.particles[tp_hash]
            except:
                print("tools.read_sa_for_sbody_cartesian failed")
                print("Problem finding a particle with that hash in the archive")
                return 0, x, y, z, vx, vy, vz, t

            x[j,it] = p.x - dx
            y[j,it] = p.y - dy
            z[j,it] = p.z - dz
            vx[j,it] = p.vx - dvx
            vy[j,it] = p.vy - dvy
            vz[j,it] = p.vz - dvz
        it+=1

    if(it==0):
        print("tools.rread_sa_for_sbody_cartesian failed")
        print("There were no simulation archives in the desired time range")
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)
    else:
        t = t[0:it]
        x = x[:,0:it]
        y = y[:,0:it]
        z = z[:,0:it]
        vx = vx[:,0:it]
        vy = vy[:,0:it]
        vz = vz[:,0:it]

    if(clones == 0):
        return 1, x[0,:], y[0,:], z[0,:], vx[0,:], vy[0,:], vz[0,:], t
    else:
        return 1, x, y, z, vx, vy, vz, t
#################################################################


#################################################################
#################################################################
# reads the simulation archive files into barycentric cartesian 
# position arrays for a small body
#################################################################
def read_sa_by_hash_cartesian(obj_hash=None, archivefile=None, datadir='',
                              center='bary',
                              tmin=None,tmax=None):
    """
    Reads the simulation archive file produced by the run_reb
    routines
        obj_hash (str): hash of the simulation particle
        archivefile (str): path to rebound simulation archive
        datadir (optional): string, path for where files are stored,
            defaults to the current directory
        center (optional, string): 'bary' for barycentric orbits 
            (default) and 'helio' for heliocentric orbits      
        tmin (optional, float): minimum time to return (years)
        tmax (optional, float): maximum time to return (years)
            if not set, the entire time range is returned
    output:
        flag (int): 1 if successful and 0 if there was a problem
        x (float array): barycentric x (au)
        y (float array): barycentric y (au)
        z (float array): barycentric z (au)
        vx (float array): barycentric vx (au/yr)
        vy (float array): barycentric vy (au/yr)
        vz (float array): barycentric vz (au/yr)
        time (1-d float array): simulations time (years)
    """

    if(obj_hash==None):
        print("The name of an object must be provided as obj_hash")
        print("tools.read_sa_for_sbody_cartesian failed")
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)

    if(archivefile==None):
        print("A simulation archive file must be provided as archivefile")
        print("tools.read_sa_for_sbody_cartesian failed")
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)

    if(datadir):
        archivefile = datadir + "/" + archivefile

    try:
        sa = rebound.Simulationarchive(archivefile)
    except:
        print("tools.read_sa_for_sbody_cartesian failed")
        print("Problem reading the simulation archive file:")
        print(archivefile)
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)
    
    nout = len(sa)
    if(nout <2):
        print("tools.read_sa_for_sbody_cartesian failed")
        print("There are fewer than two snapshots in the archive file:")
        print(archivefile)
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)
        

    if(tmin == None and tmax == None):
        #user didn't set these, so we will read the whole thing
        tmin = -1e11
        tmax = 1e11
    elif(tmin==None):
        tmin = -1e11
    elif(tmax == None):
        tmax = 1e11

    #correct for backwards integrations
    if(tmax < tmin):
        temp = tmax
        tmax = tmin
        tmin = temp

    x = np.zeros(nout)
    y = np.zeros(nout)
    z = np.zeros(nout)
    vx = np.zeros(nout)
    vy = np.zeros(nout)
    vz = np.zeros(nout)
    t = np.zeros(nout)
        
    it=0
        
    for i,sim in enumerate(sa):
        if(sim.t < tmin or sim.t > tmax):
            #skip this 
            continue
        t[it] = sim.t
        if(center=='helio'):
            #need the sun's position 
            dx = sim.particles[0].x
            dy = sim.particles[0].y
            dz = sim.particles[0].z
            dvx = sim.particles[0].vx
            dvy = sim.particles[0].vy
            dvz = sim.particles[0].vz
        elif(center=='bary'):
            dx = 0.
            dy = 0.
            dz = 0.
            dvx = 0.
            dvy = 0.
            dvz = 0.
        else:
            print("tools.read_sa_by_hash_cartesian failed")
            print("center can only be 'bary' or 'helio'\n")
            return 0, x, y, z, vx, vy, vz, t

        try:
            p = sim.particles[obj_hash]
        except:
            print("tools.read_sa_by_hash_cartesian failed")
            print("Problem finding a particle with that hash in the archive")
            return 0, x, y, z, vx, vy, vz, t

        x[it] = p.x - dx
        y[it] = p.y - dy
        z[it] = p.z - dz
        vx[it] = p.vx - dvx
        vy[it] = p.vy - dvy
        vz[it] = p.vz - dvz
        it+=1

    if(it==0):
        print("tools.rread_sa_for_sbody_cartesian failed")
        print("There were no simulation archives in the desired time range")
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)
    else:
        t = t[0:it]
        x = x[0:it]
        y = y[0:it]
        z = z[0:it]
        vx = vx[0:it]
        vy = vy[0:it]
        vz = vz[0:it]
    
    return 1, x, y, z, vx, vy, vz, t
#################################################################


#################################################################
#################################################################
# Rotate cartesian position coordinates to a frame with an x-y
# plane matching a planet's plane and the planet located
# on the x-axis (useful for plotting resonant populations)
#################################################################
def rotating_frame_cartesian(ntp=1,x=0., y=0., z=0., vx=0., vy=0., vz=0.,
                             node=0., inc=0, argperi=0., ma=0., meanmotion=0.):
    """
    inputs:
        ntp (integer) number of particles in the arrays
        x, y, z cartesian position vector to be rotated
        vx, vy, vz cartesian velocity vector to be rotated 
        node = planet's long. of ascending node in radians 
        inc = planet's inclination in radians
        argperi = planet's argument of perihelion in radians
        ma = planet's mean anomaly in radians
        n = planet's mean motions in radians/time unit
    outputs:
        x, y, z = cartesian positions (units set by inputs)
                  in the rotating frame
        vx, vy, vz = cartesian velocities (units set by inputs)
                  in the rotating frame
    """

    if(ntp == 1 and np.isscalar(x)):
        #reshape the arrays since everything assumes 2-d
        x = np.array([x])
        y = np.array([y])
        z = np.array([z])
        vx = np.array([vx])
        vy = np.array([vy])
        vz = np.array([vz])

    # calculate the first rotation into the planet's plane
    r11 = np.cos(argperi)*np.cos(node) 
    r11 = r11 - np.cos(inc)*np.sin(argperi)*np.sin(node)
    r12 = np.cos(argperi)*np.sin(node)
    r12 = r12+np.cos(inc)*np.cos(node)*np.sin(argperi)
    r13 = np.sin(inc)*np.sin(argperi)

    r21 = -np.cos(inc)*np.sin(node)*np.cos(argperi)
    r21 = r21 - np.sin(argperi)*np.cos(node)
    r22 = np.cos(inc)*np.cos(argperi)*np.cos(node)
    r22 = r22 - np.sin(argperi)*np.sin(node)
    r23 = np.cos(argperi)*np.sin(inc)

    r31 = np.sin(inc)*np.sin(node)
    r32 = -np.cos(node)*np.sin(inc)
    r33 = np.cos(inc)

    # calculate the second rotation (about the new z-axis)
    pr11 = np.cos(ma)
    pr12 = np.sin(ma)
    pr21 = -np.sin(ma)
    pr22 = np.cos(ma)

    xr=np.zeros(ntp)
    yr=np.zeros(ntp)
    zr=np.zeros(ntp)
    vxr=np.zeros(ntp)
    vyr=np.zeros(ntp)
    vzr=np.zeros(ntp)
    
    for j in range(ntp):
        # apply the first rotation to the positions
        xt = r11*x[j] + r12*y[j] + r13*z[j]
        yt = r21*x[j] + r22*y[j] + r23*z[j]
        zt = r31*x[j] + r32*y[j] + r33*z[j]

        # apply the first rotation to the velocities
        vxt = r11*vx[j] + r12*vy[j] + r13*vz[j]
        vyt = r21*vx[j] + r22*vy[j] + r23*vz[j]
        vzt = r31*vx[j] + r32*vy[j] + r33*vz[j]

        # apply the second rotation to the positions
        xr[j] = pr11*xt + pr12*yt
        yr[j] = pr21*xt + pr22*yt
        zr[j] = zt 

        # apply the second rotation to the velocities
        # with the extra term for the rotating frame
        vxr[j] = pr11*vxt + pr12*vyt  + meanmotion*yr[j]
        vyr[j] = pr21*vxt + pr22*vyt  - meanmotion*xr[j] 
        vzr[j] = vzt 

    if(ntp==1):
        return xr[0], yr[0], zr[0], vxr[0], vyr[0], vzr[0]

    return xr, yr, zr, vxr, vyr, vzr
#################################################################



#################################################################
#################################################################
# Read in a simulation archive and return the small body's
# rotated cartesian position coordinates in a frame with an x-y
# plane matching a planet's plane and the planet located
# on the x-axis (useful for plotting resonant populations)
#################################################################
def calc_rotating_frame(des=None,planet=None, archivefile=None, 
                        datadir='', clones=None,
                        tmin=None, tmax=None):
    """
    Calculate the position of a small body in the rotating frame 
    input:
        des (str): name of the small body
        planet (str): name of the planet that sets the rotating frame
        archivefile (str): path to rebound simulation archive        
        clones (optional,int): number of clones of the best-fit orbit
        tmin (optional, float): minimum time (years)
        tmax (optional, float): maximum time (years)
            if not set, the entire time range is plotted           
    output:
        flag (int): 1 if successful and 0 if there was a problem
        xr (1-d or 2-d float array): barycentric x (au) in rotating frame
        yr (1-d or 2-d float array): barycentric y (au) in rotating frame
        zr (1-d or 2-d float array): barycentric z (au) in rotating frame
        vxr (1-d or 2-d float array): barycentric vx (au/yr) in rotating frame
        vyr (1-d or 2-d float array): barycentric vy (au/yr) in rotating frame
        vzr (1-d or 2-d float array): barycentric vz (au/yr) in rotating frame
            (the units could be different if your simulation has different units)
        t (1-d array): time (year)

    """

    if(des == None):
        print("The designation of a small body must be provided as des")
        print("tools.calc_rotating_frame failed")
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)

    if(planet == None):
        print("The name of a planet must be provided as planet")
        print("tools.calc_rotating_frame failed")
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)

    if(archivefile==None):
        archivefile = archive_file_name(des)
    if(datadir):
        archivefile = datadir + '/' +archivefile

    planets=['mercury', 'venus', 'earth', 'mars',
                'jupiter', 'saturn', 'uranus', 'neptune']
    if( not(planet in planets) and planet.lower() in planets):
        planet = planet.lower()
    elif not(planet in planets):
        print("tools.calc_rotating_frame failed")
        print("specified planet is not in the list of possible planets")
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)

    #read the planet orbit for the necessary rotation information
    plflag, apl,epl,ipl,nodepl,aperipl,mapl,t = \
        read_sa_by_hash(obj_hash=planet, archivefile=archivefile,
                        tmin=tmin,tmax=tmax)

    if(not plflag):
        print("tools.calc_rotating_frame failed")
        print("couldn't read in planet's orbital history")
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)

    tpflag, x, y, z, vx, vy, vz, t = \
        read_sa_for_sbody_cartesian(des=des, archivefile=archivefile,
                                    clones=clones,tmin=tmin,tmax=tmax)
    if(not tpflag):
        print("tools.calc_rotating_frame failed")
        print("couldn't read in small body's cartesian positions")
        return 0,np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)


    #read in the last simulation snapshot to get the value of GMsum and the number of clones
    tsim = rebound.Simulation(archivefile)
    com = tsim.com()
    GMcom = com.m*tsim.G
    if(clones == None):
        clones = tsim.N - tsim.N_active - 1
    tsim = None



    nout=len(t)
    ntp = clones+1
    xr = np.zeros([ntp,nout])
    yr = np.zeros([ntp,nout])
    zr = np.zeros([ntp,nout])
    vxr = np.zeros([ntp,nout])
    vyr = np.zeros([ntp,nout])
    vzr = np.zeros([ntp,nout])

    if(ntp == 1):
        #reshape the arrays since everything assumes 2-d
        x = np.array([x])
        y = np.array([y])
        z = np.array([z])
        vx = np.array([vx])
        vy = np.array([vy])
        vz = np.array([vz])


    for n in range (0,nout):
        mmotion = np.sqrt(GMcom/(apl[n]**3.0))
        xr[:,n], yr[:,n], zr[:,n], vxr[:,n], vyr[:,n], vzr[:,n] = \
            rotating_frame_cartesian(ntp=ntp, x=x[:,n], y=y[:,n], 
                                     z=z[:,n], vx=vx[:,n], vy=vy[:,n], 
                                     vz=vz[:,n],
                                     node=nodepl[n], inc=ipl[n], 
                                     argperi=aperipl[n], 
                                     ma=mapl[n], meanmotion=mmotion)


    if(ntp > 1):
        return 1, xr, yr, zr, vxr, vyr, vzr, t
    else:
        return 1, xr[0,:], yr[0,:], zr[0,:], vxr[0,:], vyr[0,:], vzr[0,:], t


