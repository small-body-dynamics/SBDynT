import numpy as np
import re

#################################################################
#################################################################
# Convert orbital elements to cartesian coordinates
#################################################################
def aei_to_xv(GM=1.,a=1,e=0.,inc=0.,node=0.,argperi=0.,ma=0.):
    '''
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
    '''

    #based on M. Duncan's routines in swift
    if(e >= 1. or e<0. or a<0.):
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

    cape = M_to_E_reb(M=ma,e=e)
    scap = np.sin(cape)
    ccap = np.cos(cape)
    sqe = np.sqrt(1.0 -e*e)
    sqgma = np.sqrt(GM*a)
    xfac1 = a*(ccap - e)
    xfac2 = a*sqe*scap
    ri = 1.0/(a*(1.0 - e*ccap))
    vfac1 = -ri * sqgma * scap
    vfac2 = ri * sqgma * sqe * ccap

    x =  d11*xfac1 + d21*xfac2
    y =  d12*xfac1 + d22*xfac2
    z =  d13*xfac1 + d23*xfac2
    vx = d11*vfac1 + d21*vfac2
    vy = d12*vfac1 + d22*vfac2
    vz = d13*vfac1 + d23*vfac2
    
    return 1, x, y, z, vx, vy, vz




#################################################################
#################################################################
# Convert Mean anomaly M to Eccentric anomaly E
#################################################################
def M_to_E_reb(M=0.,e=0.):
    '''
    inputs:
        M = Mean anomaly in radians
        e = eccentricity
    returns:
        eccentric anomaly in radians
    '''

    #borrowed from rebound tools.c
    M = mod2pi(M)
    if(e < 1.):
        if(e<0.8):
            E = M
        else:
            E = np.pi
        F = E - e*np.sin(E) - M
        for i in range (0, 100): 
            E = E - F/(1.-e*np.cos(E))
            F = E - e*np.sin(E) - M
            if(np.abs(F) < 1.e-16):
                break
        E = mod2pi(E)
        return E
    else:
        E = M/np.abs(M)*np.log(2.*np.abs(M)/e + 1.8)
        F = E - e*np.sinh(E) + M
        for i in range (0, 100): 
            E = E - F/(1.0 - e*np.cosh(E))
            F = E - e*sinh(E) + M
            if(np.abs(F) < 1.e-16):
                break
        return E
#################################################################


#################################################################
#################################################################
# returns an angle between 0 and 2pi
#################################################################
def mod2pi(x):
    '''
    input:
        x = any angle in radians
    output
        an angle in radians re-centered from 0-2pi
    '''
    
    while(x>2.*np.pi):
        x+=-2.*np.pi
    while(x<0.):
        x+=2.*np.pi
    return x


##############################
# translate MPC designations to packed designations
#####################
def split(word):
    return [char for char in word]


def mpc_designation_translation(obj):
    """
    if an unpacked provisional MPC designation is input, this routine
    outputs a packed version of that designation to make the Horizons
    query function more robust. Otherwise it returns the same string
    that was input

    :param obj: string, minor planet designation
    :returns des: string, obj or packed provisional MPC designation
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


    regex_provis = re.compile(r"\b(\d{4})([- _]?)([a-zA-Z]{2})(\d*)\b")
    provis = regex_provis.findall(obj)
    if (provis):
        if (len(provis[0]) == 4):
            year = provis[0][0]
            letters = provis[0][2]
            number = provis[0][3]
        else:
            year = provis[0][0]
            let = provis[0][1]
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
    else:
        des = obj

    return des
