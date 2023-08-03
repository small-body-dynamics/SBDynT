# Written 30/5/14 by dh4gan
# Conversion of orbital state vectors into orbital elements
# And vice versa

from vector import Vector3D
import numpy as np

pi = 3.141592654
twopi = 2.0*pi
tiny = 1.0e-10

# G in various unit systems
Gsi = 6.67e-11
Gcgs = 6.67e-8
GmsolAUyr = 4.0*pi*pi
GmsolAUday = GmsolAUyr/(365.25*365.25)

class orbitalElements(object):
    """Set of orbital elements"""
    def __init__(self,a, e, i, longascend, argper, trueanom, position,velocity, G, totalMass):
        self.a = a
        self.e = e 
        self.rper = self.a*(1.0-e)
        self.i = i 
        self.longascend = longascend 
        self.argper = argper 
        self.trueanom = trueanom
        self.angmom = 0
        
        self.position = position
        self.velocity = velocity
        self.G = G
        self.totalMass = totalMass
        
    def __str__(self):
        s= 'a= %e \ne= %e \ni= %e \nlongascend= %e \nargper= %e \ntrueanom= %e ' % (self.a, self.e, self.i, self.longascend, self.argper, self.trueanom)
        return s
    
    def clone(self):
        return orbitalElements(self.a,self.e,self.i,self.longascend,self.argper,self.trueanom, self.position,self.velocity, self.G, self.totalMass)

    def calcOrbitFromVector(self):
        """Takes input state vectors and calculates orbital elements"""
        
        # Calculate orbital angular momentum

        angmomvec = self.position.cross(self.velocity)
        self.angmom = angmomvec.mag()
 
        # Calculate Eccentricity Vector
        
        gravparam = self.G * self.totalMass 
        magpos = self.position.mag() 
        magvel = self.velocity.mag() 
        vdotr = self.velocity.dot(self.position) 
        
        if (magpos == 0.0):
            eccentricityVector = Vector3D(0.0,0.0,0.0) 
        else:
            eccentricityVector = self.position.scalarmult(magvel*magvel).subtract(self.velocity.scalarmult(vdotr))
            eccentricityVector = eccentricityVector.scalarmult(1.0/gravparam).subtract(self.position.unitVector())

        self.e = eccentricityVector.mag() 
        
        # Calculate Semi-latus rectum
        
        self.semilat = self.angmom*self.angmom/(gravparam)
        
        etot = 0.5*magvel*magvel - gravparam/magpos
        
        # Semimajor axis
        try:
            self.a = self.semilat/(1.0-self.e*self.e)
            self.rper = self.a*(1.0-self.e)
        except ZeroDivisionError: # For parabolic orbits
            self.a = np.inf
            self.rper = self.semilat/2.0
            
        # Inclination

        if (self.angmom > 0.0):
            self.i = np.arccos(angmomvec.z/self.angmom)   
        else:
            self.i = 0.0 
            
        # Longitude of the ascending node
        
        zhat = Vector3D(0.0,0.0,1.0)
        
        nplane = zhat.cross(angmomvec)
        
        if(nplane.mag() < tiny):
            self.longascend = 0.0
        else:
            self.longascend = np.arccos(nplane.x/nplane.mag())
            if(nplane.y<0.0):
                self.longascend= twopi - self.longascend
        
        # True anomaly 
        if(self.e>tiny):
            edotR = eccentricityVector.dot(self.position) 
            edotR = edotR / (magpos * self.e) 
            rdotV = self.velocity.dot(self.position) 
            
            self.trueanom = np.arccos(edotR) 

            if (rdotV < tiny):
                self.trueanom = twopi - self.trueanom 
                    
        else:
            if(nplane.mag()>tiny):
                ndotR = nplane.unitVector().dot(self.position.unitVector()) 
                ndotV = nplane.unitVector().dot(self.velocity.unitVector()) 

                self.trueanom = np.arccos(ndotR) 

                if (ndotV > tiny):
                    self.trueanom = twopi - self.trueanom 
            else:
                self.trueanom = np.arccos(self.position.x/magpos)
                if(self.velocity.x>0.0):
                    self.trueanom = 2.0*pi-self.trueanom

        # Argument of periapsis 
        
        
        if(self.e>tiny):     
            if(nplane.mag()>tiny):       
                ndote = nplane.unitVector().dot(eccentricityVector.unitVector())
                self.argper = np.arccos(ndote)
                if(eccentricityVector.z<0.0):
                    self.argper =twopi - self.argper
            else:
                self.argper = np.arctan2(eccentricityVector.y, eccentricityVector.x)
                if(self.argper<0.0):
                    self.argper += 2.0*pi
                if(angmomvec.z>0.0):
                    self.argper = 2.0*pi-self.argper
        else:
            self.argper = 0.0

    
    def calcVectorFromOrbit(self):
        """Returns position and velocity vectors from orbital calculations"""
        # calculate distance from CoM using semimajor axis, eccentricity and true anomaly

        self.semilat = self.a*(1.0-self.e*self.e)
        magpos = self.semilat / (1.0+ self.e * np.cos(self.trueanom)) 

        self.position = Vector3D(0.0,0.0,0.0)
        self.velocity = Vector3D(0.0,0.0,0.0)
        
        # Calculate self.position vector in orbital plane
        self.position.x = magpos * np.cos(self.trueanom) 
        self.position.y = magpos * np.sin(self.trueanom) 
        self.position.z = 0.0 

        # Calculate self.velocity vector in orbital plane */
        gravparam = self.G * self.totalMass 

        try:
            magvel = np.sqrt(gravparam/self.semilat)
        except ZeroDivisionError:
            magvel = np.sqrt(2.0*gravparam/magpos)
            
        self.velocity.x = -magvel * np.sin(self.trueanom) 
        self.velocity.y = magvel * (np.cos(self.trueanom) + self.e) 
        self.velocity.z = 0.0 

        # Begin rotations:
        # Firstly, Rotation around z axis by -self.argper */

        if(self.argper>tiny):
            self.position.rotateZ(self.argper) 
            self.velocity.rotateZ(self.argper) 
     
        # Secondly, Rotate around x by -inclination */

        if(self.i >tiny):
            self.position.rotateX(self.i) 
            self.velocity.rotateX(self.i) 
     
        # Lastly, Rotate around z by self.longascend */

        if(self.longascend >tiny):
            self.position.rotateZ(self.longascend) 
            self.velocity.rotateZ(self.longascend) 
            
            
    def calcOrbitTrack(self, npoints):
        '''Given an input body's orbital parameters, 
        calculates x and y coordinates for
        its orbit over N points'''
        
        
        orbit = self.clone()
        orbit.semilat = orbit.a*(1.0-orbit.e*orbit.e)
        if(orbit.e <1.0):
            nu = np.linspace(0,2.0*np.pi, num=npoints)
        else:        
            nu = np.linspace(self.trueanom - np.arccos(1.0/self.e), self.trueanom + np.arccos(1.0/self.e), num=npoints)                
            
        x = np.zeros(npoints)
        y = np.zeros(npoints)
        z = np.zeros(npoints)
        
        for i in range(npoints):
            orbit.trueanom = nu[i]
            orbit.calcVectorFromOrbit()
            x[i] = orbit.position.x
            y[i] = orbit.position.y
            z[i] = orbit.position.z
    
        return x,y,z   