import rebound
import numpy as np
# local
import tools
import run_reb

def calc_proper_elements(sbody='', archivefile='archive.bin',
                         nclones=0, tmin=None,tmax=None,
                         datadir='./'):
    

    prop_e = 0.
    prop_sini = 0.
    prop_a = 0.
    g = 0.
    s = 0.
    prop_g = 0.
    prop_s = 0.


    flag, a, e, inc, node, aperi, ma, t = tools.read_sa_for_sbody(sbody = des, 
                        archivefile=archivefile,datadir=datadir,
                        nclones=nclones,tmin=tmin,tmax=tmax)
    if(flag == 0):
        print("proper_elements.calc_proper_elements failed when reading in the data")
        return 0, prop_a, prop_e, prop_sini,g, s, prop_g, prop_s



    #see if the inner planets are in the simulation
    small_planets_flag = 0
    flag, at, et, inct, nodet, aperit, mat, tt =read_sa_by_hash(obj_hash = 'mercury',
                                                          archivefile=archivefile,datadir=datadir,
                                                          tmin=t[0],tmax=t[1])
    if(flag > 0):
        small_planets_flag = 1


    n = len(t)
    if(n<1000):
        print("proper_elements.calc_proper_elements stopped because there are too few data points")
        return 0, prop_a, prop_e, prop_sini

    #calculate the eccentricity and inclination vectors
    p = np.sin(inc)*np.sin(node)
    q = np.sin(inc_init)*np.cos(node)
    h = (e)*np.sin(node+aperi)
    k = (e)*np.cos(node+aperi)

    #set up the FFT frequency array
    dt = t[1]-t[0]
    freq = np.fft.rfftfreq(n,d=dt)
    
    #FFT of a and e/i vectors
    Ya = np.fft.rfft(a)
    Yh = np.fft.rfft(h)
    Yk = np.fft.rfft(k)
    Yp = np.fft.rfft(p)
    Yq = np.fft.rfft(q)

    #remove the dc term for the inclinations
    Yp[0] = 0.
    Yq[0] = 0.


    #find the peak frequencies in e and i for the particle
    gind = np.argmax(np.abs(Yh[1:]))+1    
    sind = np.argmax(np.abs(Yp[1:]))+1
    g = freq[gind]  
    s = freq[sind]


    #determine how wide we should consider that peak to be
    spread = 1
    #while ( int( 1. / ( freq[gind+spread] - freq[gind-spread]) / dt ) > 2500 ):
    while ( int( dt / ( freq[gind+spread] - freq[gind-spread])) > 2500 ):
        spread += 1
    
    freq_dist_lim = 1
    #while int(1/(freq[gind+freq_dist_lim]-freq[gind-freq_dist_lim])/dt) > 1250:
    while ( int(dt/(freq[gind+freq_dist_lim]-freq[gind-freq_dist_lim])) > 1250 ):
        freq_dist_lim += 1


    #calculate the power spectra
    #pYh = np.abs(Yh)**2.
    #pYk = np.abs(Yk)**2.
    #pYp = np.abs(Yp)**2.
    #pYq = np.abs(Yq)**2.


    import hard_coded_constants as const

    #in addition to the 8 modes associated with the planets,
    #we filter the following combinations of modes
    z1 = abs(g+s-const.g6-const.s6)
    z2 = abs(g+s-const.g5-const.s7)
    z3 = abs(g+s-const.g5-const.s6)
    z4 = abs(g-2.*const.g6+const.g5)
    z5 = abs(g-2.*const.g6+const.g7)
    z6 = abs(s-const.s6-const.g5+const.g6)
    z7 = abs(g-3.*const.g6+2.*const.g5)
    z8 = abs(2.*(g-const.g6)+s-const.s6)
    z9 = abs(3.*(g-const.g6)+s-const.s6)

    if small_planets_flag:
        freq1 = [const.g1,const.g2,const.g3,const.g4,const.g5,const.g6,const.g7,const.g8,z1,z2,z3,z4,z5,z7,z8,z9]
        freq2 = [const.s1,const.s2,const.s3,const.s4,const.s6,const.s7,const.s8,z1,z2,z3,z6,z8,z9]
    else:
        freq1 = [const.g5,const.g6,const.g7,const.g8,z1,z2,z3,z4,z5,z7,z8,z9]
        freq2 = [const.s6,const.s7,const.s8,z1,z2,z3,z6,z8,z9]

    #make copies of the FFT outputs to do the filtering
    Yp_f = Yp.copy()
    Yq_f = Yq.copy()
    Yh_f = Yh.copy()
    Yk_f = Yk.copy()

    #inclination and eccentricity analysis
    imax = len(Yp)
    
    #disregard anything with a period shorter than 2000 years
    freqlim = 1./2000.

    #find the indicies in the frequency array corresponding to the
    #secular frequencies
    secresind1 = []
    secresind2 = []
    for f in freq1:
        try:
            #return the first place in the array where the 
            #frequency is greater than the secular one
            secresind1.append(np.where(freq>=f)[0][0])
        except:
            continue
    for f in freq2:
        try:
            secresind2.append(np.where(freq>=f)[0][0])
        except:
            continue

    #filter out the power in the FFTs for those frequencies in e/i
    for i in range(len(secresind1)):
        if secresind1[i] == gind:
            continue
        if abs(secresind1[i] - gind) < freq_dist_lim:
            continue

        if spread > 0:
            Yh_f[secresind1[i]-spread:secresind1[i]+spread] = 0.
            Yk_f[secresind1[i]-spread:secresind1[i]+spread] = 0.
        else:
            Yh_f[secresind1[i]] = 0.
            Yk_f[secresind1[i]] = 0.
            
    for i in range(len(secresind2)):
        if secresind2[i] == sind:
            continue
        if abs(secresind2[i] - sind) < freq_dist_lim:
            continue

        if spread > 0:
            Yp_f[secresind2[i]-spread:secresind2[i]+spread] = 0.
            Yq_f[secresind2[i]-spread:secresind2[i]+spread] = 0.
        else:
            Yp_f[secresind2[i]] = 0.



    #filter out frequencies higher than the limit in e/i and a
    limit_ind = np.where(freq >= freqlim)[0]
    Yp_f[limit_ind] = 0.
    Yq_f[limit_ind] = 0.
    Yk_f[limit_ind] = 0.
    Yh_f[limit_ind] = 0.
    Ya_f[limit_ind] = 0.


    #reconstruct the filtered time series
    p_f = np.fft.irfft(Yp_f,len(p_init))
    q_f = np.fft.irfft(Yq_f,len(q_init))
    h_f = np.fft.irfft(Yh_f,len(h_init))
    k_f = np.fft.irfft(Yk_f,len(k_init))
    a_f = np.fft.irfft(Ya_f,len(a_init))

    sini_f = np.sqrt(p_f*p_f + q_f*q_f)
    ecc_f = np.sqrt(h_f*h_f + k_f*k_f) 

    #p = np.sin(inc)*np.sin(node)
    #q = np.sin(inc_init)*np.cos(node)
    #h = (e)*np.sin(node+aperi)
    #k = (e)*np.cos(node+aperi)
    
    node_f = np.atan2(p_f/sini_f,q_f/sini_f)
    lperi_f = np.atan2(h_f/ecc_f, k_f/ecc_f)

    Y_temp = np.fft.rfft(node_f)
    si_t = np.argmax(np.abs(Y_temp[1:]))+1
    prop_s = freq[si_t]  
    Y_temp = np.fft.rfft(lperi_f)
    gi_t = np.argmax(np.abs(Y_temp[1:]))+1
    prop_g= freq[gi_t]  

    prop_e = np.nanmean(ecc_f)
    prop_sini = np.nanmean(sini_f)
    prop_a = np.nanmean(a_f)

    return 1, prop_a, prop_e, prop_sini, g, s, prop_g, prop_s



