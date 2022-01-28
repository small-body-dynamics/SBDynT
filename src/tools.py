import numpy as np

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
