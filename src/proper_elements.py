import rebound
import numpy as np
from scipy import signal
# local
import PEdata
import tools
import run_reb
from pickle import dump
from pickle import load
from importlib import resources as impresources
from os import path



# files that contain pre-computed planet frequencies based on
# which planets are in the simulations
# these will be calculated and saved the first time this is
# called
default_all_planets_frequencies = 'all-planet-frequencies.pkl'
default_giant_planets_frequencies = 'giant-planet-frequencies.pkl'
default_all_planets_but_mercury_frequencies = 'all-planets-minus-mercury-frequencies.pkl'

tno_default_giant_planets_frequencies = 'tno-giant-planet-frequencies.pkl'


class proper_elements:
    def __init__(self,clones,timeseries=False,nout=0):
        self.clones = clones
        self.a = np.zeros(clones+1)
        self.e = np.zeros(clones+1)
        self.pomega = np.zeros(clones+1)
        self.sini = np.zeros(clones+1)
        self.node = np.zeros(clones+1)
        self.g = np.zeros(clones+1)
        self.s = np.zeros(clones+1)
        self.pg = np.zeros(clones+1)
        self.ps = np.zeros(clones+1)
        self.planet_freqs = planet_frequencies()
        self.filtered_g_frequencies = np.zeros([clones+1,nout])
        self.filtered_s_frequencies = np.zeros([clones+1,nout])
        if(timeseries==True):
            self.h = np.zeros([clones+1,nout])
            self.k = np.zeros([clones+1,nout])
            self.p = np.zeros([clones+1,nout])
            self.q = np.zeros([clones+1,nout])

            self.filtered_h = np.zeros([clones+1,nout])
            self.filtered_k = np.zeros([clones+1,nout])
            self.filtered_p = np.zeros([clones+1,nout])
            self.filtered_q = np.zeros([clones+1,nout])
            self.filtered_sini = np.zeros([clones+1,nout])
            self.filtered_ecc = np.zeros([clones+1,nout])
            self.original_sini = np.zeros([clones+1,nout])
            self.original_ecc = np.zeros([clones+1,nout])
            self.time = np.zeros(nout)
    def print_results(self):
        print("Clone number, proper a, proper e, proper sini:\n")
        for n in range(0,self.clones+1):
            print("%d, %e, %e, %e" % (n,self.a[n], self.e[n], self.sini[n])) 

class planet_frequencies:
    def __init__(self):
        self.s = np.zeros(1)
        self.g = np.zeros(1)
        self.freq = np.zeros(1)
        self.g5 = None
        self.g6 = None
        self.g7 = None
        self.g8 = None
        self.s6 = None
        self.s7 = None
        self.s8 = None
        


def calc_proper_elements(des=None, archivefile=None, datadir = '',
                         clones=None, tmin=None, tmax=None,
                         logfile=False, return_timeseries=False,
                         default_run=True):
    '''
    add documentation here....
    '''
    
    flag = 0

    if(des == None):
        print("The designation of the small body must be provided")
        print("failed at proper_elements.calc_proper_elements()")
        return flag, None

    if(archivefile==None):
        archivefile = tools.archive_file_name(des)
    if(datadir):
        archivefile = datadir + '/' +archivefile

    if(logfile==True):
        logf = tools.log_file_name(des=des)
    else:
        logf=logfile
    if(datadir and logf and logf!='screen'):        
        logf = datadir + '/' +logf


    #read in the small body data
    sflag, obj_type, t, clones, a, p, q, h, k = \
            read_sbody_for_proper_elements(des=des,archivefile=archivefile,
                default_run=default_run,tmax=tmax,tmin=tmin,clones=clones)

    if(sflag<1):
        print("proper_elements.calc_proper_elements failed when reading the small body data")
        return flag, None

    #initialize the proper elements class
    n = len(t)
    elements = proper_elements(clones,timeseries=return_timeseries,nout=n)
    if(n<1000):
        print("proper_elements.calc_proper_elements stopped because there are too few data points")
        return flag, elements

    #see which planets are in the simulation
    sa = rebound.Simulationarchive(archivefile)
    sim = sa[-1]
    all_planets = ['mercury', 'venus', 'earth', 'mars','jupiter', 'saturn', 'uranus', 'neptune']
    not_planets = []
    for pl in all_planets:
        try:
            tp = sim.particles[pl]
        except:
            not_planets.append(pl)
    #the list of planets in the simulation
    planets = np.setdiff1d(all_planets, not_planets,assume_unique=True).tolist() 

    #check to see if we can use a default set of pre-computed frequencies
    gp_only = ['jupiter', 'saturn', 'uranus', 'neptune']
    no_merc = ['venus', 'earth', 'mars','jupiter', 'saturn', 'uranus', 'neptune']
    freq_file = None
    if(planets == gp_only and default_run == True and obj_type == 'tno'):
        freq_file =  impresources.files(PEdata) / tno_default_giant_planets_frequencies    
    elif(planets == gp_only and default_run == True and obj_type == 'ast'):
        freq_file =  impresources.files(PEdata) / default_giant_planets_frequencies
    elif(planets == no_merc and default_run == True and obj_type == 'ast'):
        freq_file =  impresources.files(PEdata) / default_all_planets_but_mercury_frequencies
    elif(planets == all_planets and default_run == True and obj_type == 'ast'):
        freq_file =  impresources.files(PEdata) / default_all_planets_frequencies

    if(freq_file == None):
        #nope! we must calculate the frequencies from the simulation
        fflag, planet_freqs = calc_planet_frequencies(archivefile,planets,tmax=tmax,tmin=tmin)
        if(fflag < 1):
            print("proper_elements.calc_proper_elements failed when calculating planet frequencies")
            return flag, None
    elif(not path.exists(freq_file)):
        #yes! but the file doesn't exist yet
        #calculate the frequencies from the simulation and generate the save file
        fflag, planet_freqs = calc_planet_frequencies(archivefile,planets,obj_type,tmax=tmax,tmin=tmin)
        if(fflag):
            try:
                with open(freq_file, "wb") as f:
                    dump(planet_freqs, f, protocol=5)
            except:
                print("couldn't write the save-file")
        else:
            print("proper_elements.calc_proper_elements failed when calculating planet frequencies")
            return flag, None
    else:
        #yes! load the saved frequencies
        try:
            with open(freq_file, "rb") as f:
                planet_freqs = load(f)
        except:
            print("couldn't read the saved filter file")
            print("will calculate the frequencies instead")
            fflag, planet_freqs = calc_planet_frequencies(archivefile,planets,tmax=tmax,tmin=tmin)
            if(fflag < 1):
                print("proper_elements.calc_proper_elements failed when calculating planet frequencies")
                return flag, None

    elements.planet_freqs=planet_freqs

    dt = t[1:] - t[:-1] 
    dt_mean = np.mean(dt)

    
    #index of t=0
    nt0 = (np.abs(t)).argmin()
    
    #set up the FFT frequency array
    freq = np.fft.fftfreq(n,d=dt_mean)
    freqn = np.fft.rfftfreq(n,d=dt_mean)
    if(not(freq == planet_freqs.freq).all()):
        print("proper_elements.calc_proper_elements failed because the small body and planet data")
        print("aren't sampled at the same delta t")
        return flag, elements
    
    #disregard anything with a period shorter than 2000 years for asteroids or 5000 years for other objects
    if(obj_type == 'ast'):
        freqlim = 1./2000.
    else:
        freqlim = 1./5000.
    #make sure the frequency limit is bigger than sampling frequency
    if(1./dt_mean < freqlim):
        freqlim = 1./dt_mean


    #clone-by-clone proper element calculation
    for j in range(0,clones+1):
        Yhk= np.fft.fft(k[j,:]+1j*h[j,:])
        Ypq = np.fft.fft(q[j,:]+1j*p[j,:])
        Ya_f = np.fft.rfft(a[j,:])
        #the hamming window helps screen out secular resonant peaks
        window = signal.windows.hamming(n)
        Yhk_win = Yhk*window
        Ypq_win = Ypq*window
        Ypq[0]=0
      
        imax = len(Ypq)
           
        #make copies of the FFT outputs to filter on
        Ypq_f = Ypq.copy()
        Yhk_f = Yhk.copy()

        #find the peak natural frequencies of the e and i evolution:
        #this returns the indicies of the top-two powered frequencies
        [gind1, gind2] = np.argpartition(np.abs(Yhk_win[1:])**2., -2)[-2:] + 1
        [sind1, sind2] = np.argpartition(np.abs(Ypq_win[1:])**2., -2)[-2:] + 1

        #look at the sums near both frequencies to pick the better one
        summax1hk = np.sum(Yhk_win[gind1-3:gind1+4]**2.)
        summax2hk = np.sum(Yhk_win[gind2-3:gind2+4]**2.)
        summax1pq = np.sum(Ypq_win[sind1-3:sind1+4]**2.)
        summax2pq = np.sum(Ypq_win[sind2-3:sind2+4]**2.)
        if (summax2hk > summax1hk):
            gind = gind2
        else:
            gind = gind1
        if (summax2pq > summax1pq):
            sind = sind2
        else:
            sind = sind1

        g = freq[gind]  
        s = freq[sind]

        #check to see if one of the peaks picked up was the planet frequency
        #and switch to the other one if it was
        g1 = freq[gind1]  
        s1 = freq[sind1]
        g2 = freq[gind2]  
        s2 = freq[sind2]
        try:
            if( np.abs(g -planet_freqs.g5)/np.abs(planet_freqs.g5) <=0.01 ):
                if(np.abs(g1 -planet_freqs.g5)/np.abs(planet_freqs.g5) > 0.01):
                    g = g1
                    gind = gind1
                else:
                    g = g2
                    gind = gind2
        except:
            print("warning, Jupiter not in planet list")
        
        try:
            if( np.abs(g -planet_freqs.g6)/np.abs(planet_freqs.g6) <=0.01 ):
                if(np.abs(g1 -planet_freqs.g6)/np.abs(planet_freqs.g6) > 0.01):
                    g = g1
                    gind = gind1
                else:
                    g = g2
                    gind = gind2
            if( np.abs(s -planet_freqs.s6)/np.abs(planet_freqs.s6) <=0.01 ):
                if(np.abs(s1 -planet_freqs.s6)/np.abs(planet_freqs.s6) > 0.01):
                    s = s1
                    sind = sind1
                else:
                    s = s2
                    sind = sind2
        except:
            print("warning, Saturn not in planet list")

        try:
            if( np.abs(g -planet_freqs.g7)/np.abs(planet_freqs.g7) <=0.01 ):
                if(np.abs(g1 -planet_freqs.g7)/np.abs(planet_freqs.g7) > 0.01):
                    g = g1
                    gind = gind1
                else:
                    g = g2
                    gind = gind2
            if( np.abs(s -planet_freqs.s7)/np.abs(planet_freqs.s7) <=0.01 ):
                if(np.abs(s1 -planet_freqs.s7)/np.abs(planet_freqs.s7) > 0.01):
                    s = s1
                    sind = sind1
                else:
                    s = s2
                    sind = sind2
        except:
            print("warning, Uranus not in planet list")
        
        try:
            if( np.abs(g -planet_freqs.g8)/np.abs(planet_freqs.g8) <=0.01 ):
                if(np.abs(g1 -planet_freqs.g8)/np.abs(planet_freqs.g8) > 0.01):
                    g = g1
                    gind = gind1
                else:
                    g = g2
                    gind = gind2
            if( np.abs(s -planet_freqs.s8)/np.abs(planet_freqs.s8) <=0.01 ):
                if(np.abs(s1 -planet_freqs.s8)/np.abs(planet_freqs.s8) > 0.01):
                    s = s1
                    sind = sind1
                else:
                    s = s2
                    sind = sind2
        except:
            print("warning, Neptune not in planet list")




        elements.g[j] = g
        elements.s[j] = s

        pflag, g_filt_freq, s_filt_freq = calc_filter_frequencies(planet_freqs, g, s)


        if(j==0):
            #re-shape elements.filtered_g_frequencies and elements.filtered_s_frequencies
            n_sfreq = len(planet_freqs.s) + len(s_filt_freq)
            n_gfreq = len(planet_freqs.g) + len(g_filt_freq)
            elements.filtered_g_frequencies = np.zeros([clones+1,n_gfreq])
            elements.filtered_s_frequencies = np.zeros([clones+1,n_sfreq])
        
        g_filt_freq = np.concatenate((planet_freqs.g, g_filt_freq))
        s_filt_freq = np.concatenate((planet_freqs.s, s_filt_freq))

        elements.filtered_g_frequencies[j,:] = g_filt_freq
        elements.filtered_s_frequencies[j,:] = s_filt_freq

        gindicies = []
        for gf in g_filt_freq:
            #skip any short period terms we are zeroing out later
            if(np.abs(gf) >= freqlim):
                continue
            #find the index in the frequency array closest to that frequency
            try:
                gindicies.append((np.abs(freq - gf)).argmin())
            except:
                continue

        sindicies = []
        for sf in s_filt_freq:
            #skip any short period terms we are zeroing out later
            if(np.abs(sf) >= freqlim):
                continue
            #find the index in the frequency array closest to that frequency
            try:
                sindicies.append((np.abs(freq - sf)).argmin())
            except:
                continue

    

        #sort the frequencies from shortest period to longest
        temp_freq = 1./np.abs(freq[gindicies])
        sorted_i = np.argsort(temp_freq)
        gindicies=np.array(gindicies)
        gindicies.astype(int)
        gindicies = gindicies[sorted_i]

        temp_freq = 1./np.abs(freq[sindicies])
        sorted_i = np.argsort(temp_freq)
        sindicies=np.array(sindicies)
        sindicies.astype(int)
        sindicies = sindicies[sorted_i]


        spread_dist_g = 1.05
        spread_dist_s = 1.05
        freq_dist_g = 1.1
        freq_dist_s = 1.1
        spreads_g = np.zeros(len(gindicies))
        spreads_s = np.zeros(len(sindicies))
        spreads_g = spreads_g.astype(int)
        spreads_s = spreads_s.astype(int)
        freq_dist_lims_g = np.zeros(len(gindicies))
        freq_dist_lims_s = np.zeros(len(sindicies))
        freq_dist_lims_g = freq_dist_lims_g.astype(int)
        freq_dist_lims_s = freq_dist_lims_s.astype(int)


        #calculate the spread near each frequency we want to filter in
        #eccentricity
        i=-1
        for gi in gindicies: 
            i+=1
            if(freq[gi] == 0):
                #go to the next gi value, no spreads allowed at zero
                continue
            #check the ratio of frequencies around gi (ratio defined to be > 1)
            while( np.max([ (freq[gi+spreads_g[i]]/freq[gi-spreads_g[i]]),(freq[gi-spreads_g[i]]/freq[gi+spreads_g[i]])]) < spread_dist_g ):
                spreads_g[i]+=1   
                #go to the next gi value if we've reached the end of the array, a zero frequency, or switched signs in the frequency array
                if (gi-spreads_g[i] == 0 or freq[gi-spreads_g[i]] == 0):
                    break
                if (gi+spreads_g[i] >= len(freq) or freq[gi+spreads_g[i]] == 0):
                    break
                if (freq[gi+spreads_g[i]]*freq[gi-spreads_g[i]] < 0):
                    #go back a step
                    spreads_g[i] = spreads_g[i]-1
                    break

        #inclination
        i=-1
        for si in sindicies: 
            i+=1
            if(freq[si] == 0):
                #go to the next si value, no spreads allowed at zero
                continue
            #check the ratio of frequencies around si (ratio defined to be > 1)
            while( np.max([ (freq[si+spreads_s[i]]/freq[si-spreads_s[i]]),(freq[si-spreads_s[i]]/freq[si+spreads_s[i]])]) < spread_dist_s ):
                spreads_s[i]+=1   
                #go to the next si value if we've reached the end of the array, a zero frequency, or switched signs in the frequency array
                if (si-spreads_s[i] == 0 or freq[si-spreads_s[i]] == 0):
                    break
                if (si+spreads_s[i] >= len(freq) or freq[si+spreads_s[i]] == 0):
                    break
                if (freq[si+spreads_s[i]]*freq[si-spreads_s[i]] < 0):
                    #go back a step
                    spreads_s[i] = spreads_s[i]-1
                    break


        #do the actual filtering on the FFT results 
        
        #first, zero out the short period terms
        limit_ind = np.where(np.abs(freq) >= freqlim)[0]
        limit_indr = np.where(freqn >= freqlim)[0]

        Ypq_f[limit_ind] = 0.005+0.005j
        Ypq_f[0] = 0.
        Yhk_f[limit_ind] = 0.005+0.005j
        Ya_f[limit_indr] = 0.

        #eccentricity
        i=-1
        for gi in gindicies: 
            i+=1
            if ( gi == gind or  abs(gi - gind) < freq_dist_lims_g[i] ):
                #we don't want to filter out the natural frequency
                continue
            if (spreads_g[i] > 0):
                if ( (gi - 2*spreads_g[i] - 1) <= 0 or 
                    freq[max(gi - 2*spreads_g[i] - 1,1)]*freq[gi] < 0 ):
                    #filtered based on average of just the higher frequency side
                    tmpY = Yhk_f[gi+spreads_g[i]+1:gi+2*spreads_g[i]+1].copy()
                elif ( (gi + 2*spreads_g[i] + 1) >= len(freq) or 
                      freq[min(gi + 2*spreads_g[i] + 1,len(freq)-1)]*freq[gi]<0 ):
                    #filtered based on average of just the higher negative frequency side
                    tmpY = Yhk_f[gi-2*spreads_g[i]-1:gi-spreads_g[i]-1].copy()
                else:
                    tmpY = np.concatenate( (Yhk_f[gi-2*spreads_g[i]-1:gi-spreads_g[i]-1].copy(),
                                              Yhk_f[gi+spreads_g[i]+1:gi+2*spreads_g[i]+1].copy()) )
                #replace those values with the average across that selected range
                Yhk_f[gi-spreads_g[i]:gi+spreads_g[i]] = np.nanmean(tmpY)
            else:
                Yhk_f[gi] = 0.005+0.005j

        #inclination
        i=-1
        for si in sindicies: 
            i+=1
            if ( si == sind or  abs(si - sind) < freq_dist_lims_s[i] ):
                #we don't want to filter out the natural frequency
                continue
            if (spreads_s[i] > 0):
                if ( (si - 2*spreads_s[i] - 1) <= 0 or 
                    freq[max(si - 2*spreads_s[i] - 1,1)]*freq[si] < 0 ):
                    #filtered based on average of just the higher frequency side
                    tmpY = Ypq_f[si+spreads_s[i]+1:si+2*spreads_s[i]+1].copy()
                elif ( (si + 2*spreads_s[i] + 1) >= len(freq)  or 
                      freq[min(si + 2*spreads_s[i] + 1,len(freq)-1)]*freq[si]<0 ):
                    #filtered based on average of just the higher negative frequency side
                    tmpY = Ypq_f[si-2*spreads_s[i]-1:si-spreads_s[i]-1].copy()
                else:
                    tmpY = np.concatenate( (Ypq_f[si-2*spreads_s[i]-1:si-spreads_s[i]-1].copy(),
                                              Ypq_f[si+spreads_s[i]+1:si+2*spreads_s[i]+1].copy()) )
                #replace those values with the average across that selected range
                Ypq_f[si-spreads_s[i]:si+spreads_s[i]] = np.nanmean(tmpY)
            else:
                Ypq_f[si] = 0.005+0.005j

        #apply the short-period filtering again just in case they got 
        #modified in the above
        Ypq_f[limit_ind] = 0.005+0.005j
        Ypq_f[0] = 0.
        Yhk_f[limit_ind] = 0.005+0.005j

        #reconstruct the time-series
        pq_f = np.fft.ifft(Ypq_f,len(p[j,:]))
        hk_f = np.fft.ifft(Yhk_f,len(h[j,:]))
        a_f = np.fft.irfft(Ya_f,len(a[j,:]))


        #dallin's code
        #sini_f = np.abs(pq_f)
        #ecc_f = np.abs(hk_f) 
        q_f = np.real(pq_f)
        p_f = np.imag(pq_f)

        k_f = np.real(hk_f)
        h_f = np.imag(hk_f)

        sini_f = np.sqrt(p_f*p_f + q_f*q_f)
        ecc_f = np.sqrt(h_f*h_f + k_f*k_f) 
        node_f = np.arctan2(p_f/sini_f,q_f/sini_f)
        pomega_f = np.arctan2(h_f/ecc_f, k_f/ecc_f)

        #there are some iffy edge effects, so trim
        #5\% on each side before averaging
        nlim=int(np.floor(len(p_f)*0.05))

        #calculate the slope of the proper
        #node and longitude of perihelion
        #(alternative measure of the proper g/s)
        temp_node = np.unwrap(node_f[nlim:-nlim])
        temp_pomega = np.unwrap(pomega_f[nlim:-nlim])
        slope = np.polyfit(t[nlim:-nlim],temp_node,1)[0]
        elements.ps[j] = slope/(2*np.pi)
        slope = np.polyfit(t[nlim:-nlim],temp_pomega,1)[0]
        elements.pg[j] = slope/(2*np.pi)


        elements.e[j] = np.nanmean(ecc_f[nlim:-nlim])
        elements.sini[j] = np.nanmean(sini_f[nlim:-nlim])
        elements.a[j] = np.nanmean(a_f[nlim:-nlim])

        #find the proper node and longitude at t=0
        elements.pomega[j] = pomega_f[nt0]
        elements.node[j] = node_f[nt0]

        
        if(return_timeseries):
            elements.filtered_p[j] = p_f
            elements.filtered_q[j] = q_f
            elements.filtered_h[j] = h_f
            elements.filtered_k[j] = k_f
            elements.filtered_ecc[j] = ecc_f
            elements.filtered_sini[j] = sini_f
            elements.original_sini[j] = np.sqrt(p[j]*p[j] + q[j]*q[j])
            elements.original_ecc[j] = np.sqrt(h[j]*h[j] + k[j]*k[j])


    if(return_timeseries):
        elements.p = p
        elements.p = q
        elements.h = h
        elements.k = k
        elements.time = t


    return 1, elements

#return flag, time, clones, a, p, q, h, k
def read_sbody_for_proper_elements(des=None,archivefile=None,default_run=True,
                                   tmax=None,tmin=None,clones=None):
    '''
    add documentation here...
    '''
    flag = 0

    if(des == None):
        print("must provide designation")
        print("failed at proper_elements.read_sbody_for_proper_elements()")
        return flag,None,None,None,None,None,None,None,None

    if(archivefile == None):
        print("must provide archive file path")
        print("failed at proper_elements.read_sbody_for_proper_elements()")
        return flag,None,None,None,None,None,None,None,None


    try:
        sa = rebound.Simulationarchive(archivefile)
    except:
        print("couldn't load archive file")
        print("failed at proper_elements.read_sbody_for_proper_elements()")
        return flag,None,None,None,None,None,None,None,None


    try:
        tp = sa[0].particles[des]
    except:
        print("couldn't find des in the archive file")
        print("failed at proper_elements.read_sbody_for_proper_elements()")
        return flag,None,None,None,None,None,None,None,None

    com = sa[0].com()
    orbit = tp.orbit(com)
    a0 = orbit.a
    obj_type = 'other'
    if(a0<6.):
        obj_type = 'ast'
    elif(a0>25.):
        obj_type = 'tno'


    if(obj_type == 'tno' and default_run==True):
        #read first chunk of the forward integration and downsample it
        tmin = 0
        tmax = 0.5e6
        rflag, at, et, inct, nodet, aperit, mat, tt = \
                        tools.read_sa_for_sbody(des=des,
                        archivefile=archivefile,
                        tmax=tmax,tmin=tmin,clones=clones)
        dt = tt[2]-tt[1]
        nout = len(tt)
        if(dt > 55. or dt < 45. or nout < 9.9e3 or nout > 1.01e4):
            print("unexpected sampling in the first 0.5 Myr of the default")
            print("failed at proper_elements.read_sbody_for_proper_elements()")
            return flag,None,None,None,None,None,None,None,None

        #see how many particles and outputs we got back
        if(len(at.shape)<2):
            #there aren't any clones
            if(clones==None):
                clones=0
            if(clones > 0):
                print("warning! proper_elements.calc_proper_elements() was asked to ")
                print("use clones, but there are no clones in the archive file!")
                print("Only the best fit will be analyzed")
                clones = 0
                flag = 2
            ntp = 1
            #reshape the arrays since everything below assumes 2-d
            at = np.array([at])
            et = np.array([et])
            inct = np.array([inct])
            nodet = np.array([nodet])
            aperit = np.array([aperit])
        else:
            ntp = at.shape[0]
            if(clones==None):
                clones = ntp-1
            #if fewer clones were requested, adjust
            if(clones < ntp-1):
                ntp = clones+1


        tp1 = tt[::20]
        ap1 = at[0:ntp,::20]
        ep1 = et[0:ntp,::20]
        incp1 = inct[0:ntp,::20]
        nodep1 = nodet[0:ntp,::20]
        aperip1 = aperit[0:ntp,::20]
        #read rest of the forward integration
        tmin = 0.5001e6
        tmax = 50e6
        rflag, ap2, ep2, incp2, nodep2, aperip2, mat, tp2 = \
                        tools.read_sa_for_sbody(des=des,
                        archivefile=archivefile,
                        tmax=tmax,tmin=tmin,clones=clones)
        dt = tp2[2]-tp2[1]
        nout = len(tp2)
        if(dt > 1005. or dt < 995. or nout < 4.94e4 or nout > 4.96e4):
            print("unexpected sampling from 0.5-50 Myr of the default TNO integration")
            print("failed at proper_elements.read_sbody_for_proper_elements()")
            return flag,None,None,None,None,None,None,None,None

        if(clones==0):
            ap2 = np.array([ap2])
            ep2 = np.array([ep2])
            incp2 = np.array([incp2])
            nodep2 = np.array([nodep2])
            aperip2 = np.array([aperip2])

        #read the backward integration
        tmin = -50e6
        tmax = -0.0001e6
        flag, at, et, inct, nodet, aperit, mat, tt = \
                        tools.read_sa_for_sbody(des=des,
                        archivefile=archivefile,
                        tmax=tmax,tmin=tmin,clones=clones)
        if(clones==0):
            at = np.array([at])
            et = np.array([et])
            inct = np.array([inct])
            nodet = np.array([nodet])
            aperit = np.array([aperit])

        dt = np.abs(tt[2]-tt[1])
        nout = len(tt)
            
        if(dt > 1005. or dt < 995. or nout < 4.99e4 or nout > 5.01e4):
            print("unexpected sampling from negative 50 Myr of the default tno integration")
            print("failed at proper_elements.read_sbody_for_proper_elements()")
            return flag,None,None,None,None,None,None,None,None

        #re-order the arrays so time goes in a positive direction
        it = np.argsort(tt)
        tb = tt[it]
        ab = at[:,it]
        eb = et[:,it]
        incb = inct[:,it]
        nodeb = nodet[:,it]
        aperib = aperit[:,it]

        t = np.concatenate((tb,tp1,tp2))
        a = np.concatenate((ab,ap1,ap2),axis=1)
        ec = np.concatenate((eb,ep1,ep2),axis=1)
        inc = np.concatenate((incb,incp1,incp2),axis=1)
        node = np.concatenate((nodeb,nodep1,nodep2),axis=1)
        aperi = np.concatenate((aperib,aperip1,aperip2),axis=1)
        lperi = node+aperi

    elif(obj_type == 'ast' and default_run==True):
        #read the forward integration
        tmin = 0.
        tmax = 5e6
        flag, at, ep, incp, nodep, aperip, mat, tp = \
                        tools.read_sa_for_sbody(des=des,
                        archivefile=archivefile,
                        tmax=tmax,tmin=tmin,clones=clones)
        dt = tp[2]-tp[1]
        nout = len(tp)
        if(dt > 505. or dt < 495. or nout < 9.9e3 or nout > 1.01e4):
            print("unexpected sampling in the first 5 Myr of the default asteroid integration")
            print("failed at proper_elements.read_sbody_for_proper_elements()")
            return flag,None,None,None,None,None,None,None,None

        #see how many particles and outputs we got back
        if(len(at.shape)<2):
            #there aren't any clones
            if(clones==None):
                clones=0
            if(clones > 0):
                print("warning! proper_elements.calc_proper_elements() was asked to ")
                print("use clones, but there are no clones in the archive file!")
                print("Only the best fit will be analyzed")
                clones = 0
                flag = 2
            ntp = 1
            #reshape the arrays since everything below assumes 2-d
            at = np.array([at])
            et = np.array([et])
            inct = np.array([inct])
            nodet = np.array([nodet])
            aperit = np.array([aperit])
        else:
            ntp = at.shape[0]
            if(clones==None):
                clones = ntp-1
            #if fewer clones were requested, adjust
            if(clones < ntp-1):
                ntp = clones+1

        ap = at[0:ntp,:]
        ep = et[0:ntp,:]
        incp = inct[0:ntp,:]
        nodep = nodet[0:ntp,:]
        aperip = aperit[0:ntp,:]

        #read the backward integration
        tmin = -5e6
        tmax = -0.0001e6
        flag, at, et, inct, nodet, aperit, mat, tt = \
                        tools.read_sa_for_sbody(des=des,
                        archivefile=archivefile,
                        tmax=tmax,tmin=tmin,clones=clones)
        dt = np.abs(tt[2]-tt[1])
        nout = len(tt)
        if(dt > 505. or dt < 495. or nout < 9.9e3 or nout > 1.01e4):
            print("unexpected sampling in the backwards 5 Myr of the default asteroid integration")
            print("failed at proper_elements.read_sbody_for_proper_elements()")
            return flag,None,None,None,None,None,None,None,None

        if(clones==0):
            at = np.array([at])
            et = np.array([et])
            inct = np.array([inct])
            nodet = np.array([nodet])
            aperit = np.array([aperit])


        it = np.argsort(tt)
        tb = tt[it]
        ab = at[:,it]
        eb = et[:,it]
        incb = inct[:,it]
        nodeb = nodet[:,it]
        aperib = aperit[:,it]

        
        t = np.concatenate((tb,tp))
        a = np.concatenate((ab,ap),axis=1)
        ec = np.concatenate((eb,ep),axis=1)
        inc = np.concatenate((incb,incp),axis=1)
        node = np.concatenate((nodeb,nodep),axis=1)
        aperi = np.concatenate((aperib,aperip),axis=1)
        lperi = node+aperi
    else:
        flag, at, et, inct, nodet, aperit, mat, tt = \
                        tools.read_sa_for_sbody(des=des,
                        archivefile=archivefile,
                        tmax=tmax,tmin=tmin,clones=clones)
        #see how many particles and outputs we got back
        if(len(at.shape)<2):
            #there aren't any clones
            if(clones==None):
                clones=0
            if(clones > 0):
                print("warning! proper_elements.calc_proper_elements() was asked to ")
                print("use clones, but there are no clones in the archive file!")
                print("Only the best fit will be analyzed")
                clones = 0
                flag = 2
            ntp = 1
            #reshape the arrays since everything below assumes 2-d
            at = np.array([at])
            et = np.array([et])
            inct = np.array([inct])
            nodet = np.array([nodet])
            aperit = np.array([aperit])
        else:
            ntp = at.shape[0]
            if(clones==None):
                clones = ntp-1
            #if fewer clones were requested, adjust
            if(clones < ntp-1):
                ntp = clones+1
        it = np.argsort(tt)
        t = tt[it]
        a = at[:,it]
        e = et[:,it]
        inc = inct[:,it]
        node = nodet[:,it]
        aperi = aperit[:,it]
        lperi = node+aperi

            
    #check to make sure the time array is evenly sampled
    dt = t[1:] - t[:-1] 
    dt_std = np.std(dt)
    dt_mean = np.mean(dt)
    if(dt_std > 0.005*dt_mean):
        print("The time series provided is not appropriately evenly sampled")
        print("the time between outputs must be constant for FFT analysis")
        print("failed at proper_elements.read_sbody_for_proper_elements()")
        return flag,None,None,None,None,None,None,None,None

    h = ec*np.sin(lperi)
    k = ec*np.cos(lperi)
    p = np.sin(inc)*np.sin(node)
    q = np.sin(inc)*np.cos(node)

    flag=1

    return flag, obj_type, t, clones, a, p, q, h, k

        



def calc_planet_frequencies(archivefile, planets,obj_type=None,tmin=None,tmax=None):
    '''
    add documentation here...
    '''
    flag = 0

    #it matters what order we go through the planets in to make sure we get the 
    #correct mode for each planet
    ordered_planets = []
    if('jupiter' in planets):
        ordered_planets.append('jupiter')
    if('saturn' in planets):
        ordered_planets.append('saturn')
    if('uranus' in planets):
        ordered_planets.append('uranus')
    if('neptune' in planets):
        ordered_planets.append('neptune')
    if('earth' in planets):
        ordered_planets.append('earth')
    if('venus' in planets):
        ordered_planets.append('venus')
    if('mars' in planets):
        ordered_planets.append('mars')
    if('mercury' in planets):
        ordered_planets.append('mercury')

    g_freqs = []
    g_freqs_ind = []
    s_freqs = []
    s_freqs_ind = []

    #initialize the class
    planet_freqs = planet_frequencies()

    for pl in ordered_planets:
        if(obj_type == 'tno'):
            #read first chunk of the forward integration and downsample it
            tmin = 0
            tmax = 0.5e6
            flag, at, et, inct, nodet, aperit, mat, tt = \
                tools.read_sa_by_hash(obj_hash=pl,archivefile=archivefile,
                                      tmin=tmin,tmax=tmax)
            dt = tt[1]-tt[0]
            nout = len(tt)
            if(dt > 55. or dt < 45. or nout < 9.9e3 or nout > 1.01e4):
                print("unexpected sampling in the first 0.5 Myr of the default")
                print("tno integration. failed at proper_elements.calc_planet_frequencies())")
                return flag, None
            tp1 = tt[::20]
            ep1 = et[::20]
            incp1 = inct[::20]
            nodep1 = nodet[::20]
            aperip1 = aperit[::20]
            #read first rest of the forward integration and downsample it
            tmin = 0.5001e6
            tmax = 50e6
            flag, at, ep2, incp2, nodep2, aperip2, mat, tp2 = \
                tools.read_sa_by_hash(obj_hash=pl,archivefile=archivefile,
                                      tmin=tmin,tmax=tmax)
            dt = tp2[2]-tp2[1]
            nout = len(tp2)
            if(dt > 1005. or dt < 995. or nout < 4.94e4 or nout > 4.96e4):
                print("unexpected sampling from 0.5-50 Myr of the default")
                print("tno integration. failed at proper_elements.calc_planet_frequencies())")
                return flag, None
            #read the backward integration
            tmin = -50e6
            tmax = -0.0001e6
            flag, at, et, inct, nodet, aperit, mat, tt = \
                tools.read_sa_by_hash(obj_hash=pl,archivefile=archivefile,
                                      tmin=tmin,tmax=tmax)
            #re-order the arrays so time goes in a positive direction
            it = np.argsort(tt)
            tb = tt[it]
            eb = et[it]
            incb = inct[it]
            nodeb = nodet[it]
            aperib = aperit[it]

            t = np.concatenate((tb,tp1,tp2))
            ec = np.concatenate((eb,ep1,ep2))
            inc = np.concatenate((incb,incp1,incp2))
            node = np.concatenate((nodeb,nodep1,nodep2))
            aperi = np.concatenate((aperib,aperip1,aperip2))
            lperi = node+aperi
        elif(obj_type == 'ast'):
            #read the forward integration
            tmin = 0.
            tmax = 5e6
            flag, at, ep, incp, nodep, aperip, mat, tp = \
                tools.read_sa_by_hash(obj_hash=pl,archivefile=archivefile,
                                      tmin=tmin,tmax=tmax)
            dt = tp[1]-tp[0]
            nout = len(tp)
            if(dt > 505. or dt < 495. or nout < 9.9e3 or nout > 1.01e4):
                print("unexpected sampling in the first 5 Myr of the default")
                print("asteroid integration. failed at proper_elements.calc_planet_frequencies()")
                return flag, None
            #read the backward integration
            tmin = -5e6
            tmax = -0.0001e6
            flag, at, et, inct, nodet, aperit, mat, tt = \
                tools.read_sa_by_hash(obj_hash=pl,archivefile=archivefile,
                                      tmin=tmin,tmax=tmax)
            dt = np.abs(tt[2]-tt[1])
            nout = len(tt)
            if(dt > 505. or dt < 495. or nout < 9.9e3 or nout > 1.01e4):
                print("unexpected sampling in the backwards 5 Myr of the default")
                print("asteroid integration. failed at proper_elements.calc_planet_frequencies()")
                return flag, None
            it = np.argsort(tt)
            tb = tt[it]
            eb = et[it]
            incb = inct[it]
            nodeb = nodet[it]
            aperib = aperit[it]

            t = np.concatenate((tb,tp))
            ec = np.concatenate((eb,ep))
            inc = np.concatenate((incb,incp))
            node = np.concatenate((nodeb,nodep))
            aperi = np.concatenate((aperib,aperip))
            lperi = node+aperi
        else:
            flag, at, ep, incp, nodep, aperip, mat, tp = \
                tools.read_sa_by_hash(obj_hash=pl,archivefile=archivefile,
                                      tmin=tmin,tmax=tmax)
            #make sure time is going in a positive direction
            it = np.argsort(tt)
            t = tt[it]
            ec = et[it]
            inc = inct[it]
            node = nodet[it]
            aperi = aperit[it]
            lperi = node+aperi

        #check to make sure the time array is evenly sampled
        dt = t[1:] - t[:-1] 
        dt_std = np.std(dt)
        dt_mean = np.mean(dt)
        if(dt_std > 0.005*dt_mean):
            print("The time series provided is not appropriately evenly sampled")
            print("the time between outputs must be constant for FFT analysis")
            print("failed at proper_elements.calc_planet_frequencies()")
            return flag, None

        #run fft analyses to find the dominant frequencies
        h = ec*np.sin(lperi)
        k = ec*np.cos(lperi)
        p = np.sin(inc)*np.sin(node)
        q = np.sin(inc)*np.cos(node)

        Yh = np.fft.fft(k+1j*h)
        Yp = np.fft.fft(q+1j*p)

        n = len(h)
        dt = t[1] - t[0]
        freq = np.fft.fftfreq(n,d=dt_mean)
        
        #eccentricity frequencies
        #zero out any previously identified modes
        for i in g_freqs_ind:
            Yh[i] = 0.
        #index of the most powerful remaining eccentricity frequency
        ifmax = np.argmax(np.abs(Yh[1:])**2.) + 1
        g_freqs_ind.append(ifmax)
        g_freqs.append(freq[ifmax])
        if(pl=='jupiter'):
            planet_freqs.g5 = freq[ifmax]
        elif(pl=='saturn'):
            planet_freqs.g6 = freq[ifmax]
        elif(pl=='uranus'):
            planet_freqs.g7 = freq[ifmax]
        elif(pl=='neptune'):
            planet_freqs.g8 = freq[ifmax]

        #inclination frequencies
        if(pl != 'jupiter'):
            #zero out any previously identified modes
            for i in s_freqs_ind:
                Yp[i] = 0.
            #index of the most powerful remaining inclination frequency
            ifmax = np.argmax(np.abs(Yp[1:])**2.) + 1
            s_freqs_ind.append(ifmax)
            s_freqs.append(freq[ifmax])
            if(pl=='saturn'):
                planet_freqs.s6 = freq[ifmax]
            elif(pl=='uranus'):
                planet_freqs.s7 = freq[ifmax]
            elif(pl=='neptune'):
                planet_freqs.s8 = freq[ifmax]
    
    if(planet_freqs.g5 and planet_freqs.g6):
        z1 = 2.*planet_freqs.g6 - planet_freqs.g5
        z2 = 2.*planet_freqs.g5 - planet_freqs.g6
        z3 = planet_freqs.g5 - planet_freqs.g6
        g_freqs.append(z1)
        g_freqs.append(z2)
        g_freqs.append(z3)

    if(planet_freqs.s7 and planet_freqs.s6):
        z4 = planet_freqs.s7 - planet_freqs.s6
        z5 = 0.5*planet_freqs.s6
    #    z6 = 0.5*planet_freqs.s7
        s_freqs.append(z4)
        s_freqs.append(z5)
    #    s_freqs.append(z6)

    #if(planet_freqs.s8):
    #    z7 = 0.5*planet_freqs.s8
    #    s_freqs.append(z7)


    planet_freqs.s = np.array(s_freqs)
    planet_freqs.g = np.array(g_freqs)
    planet_freqs.freq = freq

    flag = 1
    return flag, planet_freqs
          
def calc_filter_frequencies(planet_freqs, g, s):

    flag = 0

    if( not(planet_freqs.g5 and planet_freqs.g6 and planet_freqs.g7 and planet_freqs.g8)):
        print("current proper elements implementation requires all four giant planets to")
        print("be in the simulation!")
        print("failed at proper_elements.calc_filter_frequencies()")
        return flag, None, None, None, None
    
    g5=planet_freqs.g5
    g6=planet_freqs.g6
    g7=planet_freqs.g7
    g8=planet_freqs.g8
    s6=planet_freqs.s6
    s7=planet_freqs.s7
    s8=planet_freqs.s8


    #in addition to the 8 modes associated with the planets,
    #we filter the following combinations of modes
    z = np.zeros(102)
    #z = np.zeros(75)
    
    #frequencies copied directly from findorb
    z[1]=g-g5 #gonly
    z[2]=g-g6 #gonly
    z[3]=s-s7 #sonly
    z[4]=s-s6 #sonly
    z[5]=g+s-s7-g5 
    z[6]=g+s-s7-g6 
    z[7]=g+s-s6-g5 
    z[8]=g+s-s6-g6 
    z[9]=2.*g-2.*s 
    z[10]=g-2.*g5+g6 
    z[11]=g+g5-2.*g6 
    z[12]=2.*g-g5-g6 #gonly
    z[13]=-g+s+g5-s7 
    z[14]=-g+s+g6-s7 
    z[15]=-g+s+g5-s6 
    z[16]=-g+s+g6-s6 
    z[17]=g-g5+s7-s6 
    z[18]=g-g5-s7+s6 
    z[19]=g-g6+s7-s6 
    z[20]=g-g6-s7+s6 
    z[21]=2.*g-s-s7 
    z[22]=2.*g-s-s6 
    z[23]=-g+2.*s-g5 
    z[24]=-g+2.*s-g6 
    z[25]=2.*g-2.*s7 
    z[26]=2.*g-2.*s6 
    z[27]=2.*g-s7-s6 
    z[28]=g-s+g5-s7 
    z[29]=g-s+g5-s6 
    z[30]=g-s+g6-s7 
    z[31]=g-s+g6-s6 
    z[32]=g+g5-2.*s7 
    z[33]=g+g6-2.*s7 
    z[34]=g+g5-2.*s6 
    z[35]=g+g6-2.*s6 
    z[36]=g+g5-s7-s6 
    z[37]=g+g6-s7-s6 
    z[38]=s-2.*s7+s6 #sonly
    z[39]=s+s7-2.*s6 #sonly
    z[40]=2.*s-s7-s6 #sonly
    z[41]=s+g5-g6-s7 
    z[42]=s-g5+g6-s7 
    z[43]=s+g5-g6-s6 
    z[44]=s-g5+g6-s6 
    z[45]=2.*s-2.*g5 
    z[46]=2.*s-2.*g6 
    z[47]=2.*s-g5-g6 
    z[48]=s-2.*g5+s7 
    z[49]=s-2.*g5+s6 
    z[50]=s-2.*g6+s7 
    z[51]=s-2.*g6+s6 
    z[52]=s-g5-g6+s7 
    z[53]=s-g5-g6+s6 
    z[54]=2.*g-2.*g5 #gonly
    z[55]=2.*g-2.*g6 #gonly
    z[56]=2.*s-2.*s7 #sonly
    z[57]=2.*s-2.*s6 #sonly
    z[58]=g-2.*g6+g7 #gonly
    z[59]=g-3.*g6+2.*g5 #gonly 
    z[60]=2.*(g-g6)+(s-s6) 
    z[61]=g+g5-g6-g7 #gonly
    z[62]=g-g5-g6+g7 #gonly
    z[63]=g+g5-2*g6-s6+s7 
    z[64]=3.*(g-g6) + (s-s6)

    #other modes added in based on examination of many 
    #example asteroids and tnos
    z[65]=g-(2.*(g-g6)+s-s6) 
    z[66]=g-(3.*(g-g6)+s-s6) 
    z[67]=g-((g-g6)+s-s6)
    z[68]=s-(2.*(g-g6)+s-s6) 
    z[69]=s-(3.*(g-g6)+s-s6) 
    z[70]=s-((g-g6)+s-s6)
    z[71]=g-2.*g7+g6 #gonly
    z[72]=-4.*g+4.*g7 #gonly
    z[73]=-2*s-s6 #sonly
    z[74]=-g+2*s-g5

    z[75] = g-g7 #gonly
    z[76] = g-g8 #gonly
    z[77] = s-s8 #sonly
    z[78]=g+s-s7-g8 
    z[79]=g+s-s6-g8 
    z[80]=-g+s+g5-s8 
    z[81]=-g+s+g6-s8 
    z[82]=-g+s+g7-s8 
    z[83]=2.*g-2.*g8 #gonly
    z[84]=2.*g-2.*g7 #gonly
    z[85]=2.*g-2.*s8 
    z[86]=2.*s-2.*g7 
    z[87]=g-2.*g7 #gonly
    z[88]=g-2.*g8 #gonly
    z[89]=s-2.*s7 #sonly
    z[90]=s-2.*s8 #sonly
    z[91]=3.*g-2.*g8 #gonly
    z[92]=2.*g-g7 #gonly
    z[93]=2.*g-g8 #gonly
    z[94]=2.*s-2.*s7 #sonly
    z[95]=2.*s-2.*s8 #sonly
    z[96]=3.*g-(g5-g7) #gonly
    z[97]=g5-g7 #gonly
    z[98]=2.*g-(g5-g7) #gonly
    z[99] = g -(g5-g7-g8)
    z[100] = -g +(g5-g7-g8)
    z[101] = -s +(g5-g7-g8)

    #eccentricity only z frequencies 
    eonly = [0,1,2,12,13,54,55,58,59,61,62,71,72,75,76,
             83,84,87,88,91,92,93,96,97,98,99,100]
    #inclination only z frequencies
    ionly = [0,3,4,38,39,40,56,57,73,77,89,90,94,95]

    z_e = np.delete(z,ionly)
    z_i = np.delete(z,eonly)

    flag = 1
    return flag, z_e, z_i

