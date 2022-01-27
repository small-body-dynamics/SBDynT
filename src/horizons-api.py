from astroquery.jplsbdb import SBDB
from astroquery.jplhorizons import Horizons
import rebound
import numpy as np
import requests
import json
import base64




###################################################################################################

def query_horizons_api_planets(obj='Sun',epoch=2455000):
    #takes a planet and epoch and returns:
    # a status integer (0 if nothing was queried, 1 if horizons was sucessfully queried)
    # an object mass (in solar masses)
    # an object radius (in au)
    # an array of three heliocentric positions (in au)
    # an arrya of three heliocentric velocities (in au/time unit where 2pi time units is 1 year)

    #define the planet-id numbers used by Horizons for the barycenters of each
    #major planet in the solar system
    planet_id = {'Sun': 0, 'Mercury': 1, 'Venus': 2, 'Earth': 3, 'Mars': 4, 'Jupiter': 5,
                'Saturn': 6, 'Uranus': 7, 'Neptune': 8, 'sun': 0, 'mercury': 1, 'venus': 2, 
                'earth': 3, 'mars': 4, 'jupiter': 5, 'saturn': 6, 'uranus': 7, 'neptune': 8}
    
    mass = 0.
    rad = 0.
    x = np.zeros(3)
    v = np.zeros(3)
    
    #exit if the object isn't one of the major planets
    try:
        des = str(planet_id[obj])
    except:
        print("provided object is not one of the major planets")
        return 0, mass, rad, x, v
    
    #array of GM values queried January 2022
    #(there isn't a way to get this from Horizons, so we just have to hard code it)
    #values for giant planet systems are from Park et al. 2021 DE440 and DE441, 
    #https://doi.org/10.3847/1538-3881/abd414
    #all in km^3 kg^–1 s^–2
    #G = 6.6743015e-20 #in km^3 kg^–1 s^–2
    SS_GM = np.zeros(9)
    SS_GM[0] = 132712440041.93938 #Sun
    SS_GM[1] = 22031.86855 #Mercury
    SS_GM[2] = 324858.592 #Venus
    SS_GM[3] = 403503.235502 #Earth-Moon
    SS_GM[4] = 42828.375214 #Mars
    SS_GM[5] = 126712764.10 #Jupiter system
    SS_GM[6] = 37940584.8418 #Saturn system
    SS_GM[7] = 5794556.4 #Uranus system
    SS_GM[8] = 6836527.10058 #Neptune system

    #array of physical radius values queried January 2022
    #(not possible to pull directly via API)
    kmtoau = 6.68459e-9
    SS_r = np.zeros(9)
    SS_r[0] = 695700.*kmtoau #Sun
    SS_r[1] = 2440.53*kmtoau #Mercury
    SS_r[2] = 6051.8*kmtoau #Venus
    SS_r[3] = 66378.136*kmtoau #Earth
    SS_r[4] = 3396.19*kmtoau #Mars
    SS_r[5] = 71492.*kmtoau #Jupiter system
    SS_r[6] = 60268.*kmtoau #Saturn system
    SS_r[7] = 25559.*kmtoau #Uranus system
    SS_r[8] = 24764.*kmtoau #Neptune system


    #planet mass in solar masses
    mass = SS_GM[planet_id[obj]]/SS_GM[planet_id['Sun']]
    #planet radius
    rad = SS_r[planet_id[obj]]

    #we don't actually need to query for the Sun because
    #we are working in heliocentric coordinates
    if(obj == 'Sun' or obj == 'sun'):
        return 1, mass, rad, x, v    
    
    #build the url to query horizons
    start_time = 'JD'+str(epoch)
    stop_time = 'JD'+str(epoch+1)
    url = 'https://ssd.jpl.nasa.gov/api/horizons.api'
    spk_filename = 'spk_file.bsp'
    url += "?format=json&EPHEM_TYPE=Vectors&OBJ_DATA=YES&CENTER='@Sun'&OUT_UNITS='AU-D'"
    url += "&COMMAND=" + des + "&START_TIME=" +start_time + "&STOP_TIME=" + stop_time
    
    #run the query and exit if it fails
    response = requests.get(url)
    try:
        data = json.loads(response.text)
    except ValueError:
        print("Unable to decode JSON results from Horizons API request")
        return 0, mass, x, v


    #pull the lines we need from the resulting plain text return
    try:
        xvline = data["result"].split("X =")[1].split("\n")
    except:
        print("Unable to find \"X =\" in Horizons API request result")
        return 0, mass, rad, x, v

    try:
        #heliocentric positions:
        x[0] = float(xvline[0].split()[0])
        x[1] = float(xvline[0].split("Y =")[1].split()[0])
        x[2] = float(xvline[0].split("Z =")[1].split()[0])

        #heliocentric velocities converted to the units we want
        v[0] = float(xvline[1].split("VX=")[1].split()[0])*365.25/(2.*np.pi)
        v[1] = float(xvline[1].split("VY=")[1].split()[0])*365.25/(2.*np.pi)
        v[2] = float(xvline[1].split("VZ=")[1].split()[0])*365.25/(2.*np.pi)
    except:
        print("Unable to find Y,Y,Z, VX, VY, VZ in Horizons API request result")
        return 0, mass, rad, x, v

    return 1, mass, rad, x, v

###################################################################################################
