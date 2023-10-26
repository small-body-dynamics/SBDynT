import numpy as np
import re


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

    # based on M. Duncan's routines in swift
    if(e >= 1. or e < 0. or a < 0.):
        print("orbital eccentricity not between 0 and 1, cannot proceed")
        return 0, 0., 0., 0., 0., 0., 0.

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
# Rotate cartesian position coordinates to a frame with an x-y
# plane matching a planet's plane and the planet located
# on the x-axis (useful for plotting resonant populations)
#################################################################
def rotating_frame_xyz(x=0., y=0., z=0., node=0., inc=0, argperi=0., ma=0.):
    """
    inputs:
        x, y, z cartesian position vector
        node = planet's long. of ascending node in radians 
        inc = planet's inclination in radians
        argperi = planet's argument of perihelion in radians
        ma = planet's mean anomaly in radians
    outputs:
        x, y, z = cartesian positions (units set by inputs)
                  in the rotated frame
    """

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

    # apply the first rotation
    xt = r11*x + r12*y + r13*z
    yt = r21*x + r22*y + r23*z
    zt = r31*x + r32*y + r33*z

    # calculate the second rotation (about the new z-axis)
    pr11 = np.cos(ma)
    pr12 = np.sin(ma)
    pr21 = -np.sin(ma)
    pr22 = np.cos(ma)

    # apply the second rotation
    xr = pr11*xt + pr12*yt
    yr = pr21*xt + pr22*yt
    zr = zt 

    return xr, yr, zr
#################################################################


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
    M = mod2pi(M)
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
        E = mod2pi(E)
        return E
    else:
        E = M/np.abs(M)*np.log(2.*np.abs(M)/e + 1.8)
        F = E - e*np.sinh(E) + M
        for i in range(0, 100):
            E = E - F/(1.0 - e*np.cosh(E))
            F = E - e*np.sinh(E) + M
            if(np.abs(F) < 1.e-16):
                break
        return E
#################################################################


#################################################################
#################################################################
# returns an angle between 0 and 2pi
#################################################################
def mod2pi(x):
    """
    input:
        x = any angle in radians
    output:
        an angle in radians re-centered from 0-2pi
    """
    
    while(x > 2.*np.pi):
        x += -2.*np.pi
    while(x < 0.):
        x += 2.*np.pi
    return x


#################################################################
#################################################################
# returns an angle between 0 and 2pi
#################################################################
def arraymod2pi(x):
    """
    input:
        x = array of angles in radians
    output:
        array of angles in radians re-centered from 0-2pi
    """
    imax = len(x)
    for i in range (0,imax):
        while (x[i] > 2. * np.pi):
            x[i] += -2. * np.pi
        while (x[i] < 0.):
            x[i] += 2. * np.pi

    return x

#################################################################
#################################################################
# returns an angle between -pi and pi
#################################################################
def arraymod2pi0(x):
    """
    input:
        x = array of angles in radians
    output:
        array of angles in radians re-centered from -pi to pi
    """
    imax = len(x)
    for i in range (0,imax):
        while (x[i] > np.pi):
            x[i] += -2. * np.pi
        while (x[i] < -np.pi):
            x[i] += 2. * np.pi

    return x


def arraymod360(x):
    """
    input:
        x = array of angles in degrees
    output:
        array of angles in degrees recentered 0-360
    """
    imax = len(x)
    for i in range(0, imax):
        while (x[i] > 360):
            x[i] += -360
        while (x[i] < 0.):
            x[i] += 360

    return x


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

    #in case the input was not specified as a string (e.g., numbered 
    #object not enclosed in ''), convert to string
    if(not(isinstance(obj, str))):
        obj =str(obj)
    
    provis = regex_provis.findall(obj)
    packedprovis = regex_packedprovis.findall(obj)    
    packednum = regex_packednum.findall(obj)
    num = regex_num.findall(obj)    
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
    else:
        des = obj

    return des, destype
