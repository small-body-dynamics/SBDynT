import rebound
import numpy as np
from scipy import signal
# local
import tools
import run_reb

class proper:
    def __init__(self,nclones):
        self.a = np.zeros(nclones+1)
        self.e = np.zeros(nclones+1)
        self.pomega = np.zeros(nclones+1)
        self.sini = np.zeros(nclones+1)
        self.node = np.zeros(nclones+1)
        self.g = np.zeros(nclones+1)
        self.s = np.zeros(nclones+1)
        self.pg = np.zeros(nclones+1)
        self.ps = np.zeros(nclones+1)

        


def calc_proper_elements(sbody='', archivefile='archive.bin',
                         nclones=0, tmin=None,tmax=None,
                         datadir='./'):
    

    #initialize the proper elements class
    elements = proper(nclones)

    flag, a, e, inc, node, aperi, ma, t = tools.read_sa_for_sbody(sbody = sbody, 
                        archivefile=archivefile,datadir=datadir,
                        nclones=nclones,tmin=tmin,tmax=tmax,center='bary')#'helio')
    if(flag == 0):
        print("proper_elements.calc_proper_elements failed when reading in the data")
        return 0, elements



    #see if the inner planets are in the simulation
    small_planets_flag = 0
    flag, at, et, inct, nodet, aperit, mat, tt = tools.read_sa_by_hash(obj_hash = 'mercury',
                                                          archivefile=archivefile,datadir=datadir,
                                                          tmin=t[0],tmax=t[1])
    if(flag > 0):
        small_planets_flag = 1


    n = len(t)
    if(n<1000):
        print("proper_elements.calc_proper_elements stopped because there are too few data points")
        return 0, elements

    #calculate the eccentricity and inclination vectors
    p = np.sin(inc)*np.sin(node)
    q = np.sin(inc)*np.cos(node)
    h = (e)*np.sin(node+aperi)
    k = (e)*np.cos(node+aperi)

    #set up the FFT frequency array
    dt = t[1]-t[0]
    freq = np.fft.fftfreq(n,d=dt)
    freqn = np.fft.rfftfreq(n,d=dt)
    
    
    #FFT of a and e/i vectors
    Ya = np.fft.rfft(a)
    Yhk = np.fft.fft(k + 1j*h)
    Ypq = np.fft.fft(q + p*1j)

    #remove the dc term for the inclinations
    Ypq[0] = 0.


    #find the peak frequencies in e and i for the particle
    gind = np.argmax(np.abs(Yhk[1:])**2.0)+1 
    sind = np.argmax(np.abs(Ypq[1:])**2.0)+1
    g = freq[gind]  
    s = freq[sind]

    
    elements.g = g
    elements.s = s

    #make copies of the FFT outputs to do the filtering
    Ypq_f = Ypq.copy()
    Yhk_f = Yhk.copy()
    Ya_f = Ya.copy()

    #inclination and eccentricity analysis
    imax = len(Ypq)
    
    #disregard anything with a period shorter than 5000 years
    freqlim = 1./5000.
    limit_ind = np.where(np.abs(freq) >= freqlim)[0]
    limit_indr = np.where(freqn >= freqlim)[0]


    freq1,freq2 = calc_frequencies(g,s,small_planets_flag)

    #find the indices in the frequency array corresponding to the
    #secular frequencies
    secresind1 = []
    secresind2 = []
    for i in freq1:
        try:
            secresind1.append((np.abs(freq - i)).argmin())
        except:
            continue
    for i in freq2:
        try:
            secresind2.append((np.abs(freq - i)).argmin())
        except:
            continue


    spread_dist = 1.10
    freq_dist = 1.15
    spreads1 = np.zeros(len(secresind1))
    spreads2 = np.zeros(len(secresind2))
    spreads1 = spreads1.astype(int)
    spreads2 = spreads2.astype(int)
    freq_dist_lims1 = np.zeros(len(secresind1))
    freq_dist_lims2 = np.zeros(len(secresind2))
    freq_dist_lims1 = freq_dist_lims1.astype(int)
    freq_dist_lims2 = freq_dist_lims2.astype(int)
    
    for i in range(len(secresind1)): 
        if freq[secresind1[i]] > 0:
            while ((freq[secresind1[i]+spreads1[i]]/freq[secresind1[i]-spreads1[i]])) < spread_dist:
                spreads1[i]+=1   
                if secresind1[i]+spreads1[i] >= len(freq):
                    break
            while ((freq[secresind1[i]+freq_dist_lims1[i]]/freq[secresind1[i]-freq_dist_lims1[i]])) < freq_dist:
                freq_dist_lims1[i]+=1
                if secresind1[i]+freq_dist_lims1[i] >= len(freq):
                    break
        else:
            while ((freq[secresind1[i]-spreads1[i]]/freq[secresind1[i]+spreads1[i]])) < spread_dist:
                spreads1[i]+=1   
                if secresind1[i]+spreads1[i] >= len(freq):
                    break               
            while ((freq[secresind1[i]-freq_dist_lims1[i]]/freq[secresind1[i]+freq_dist_lims1[i]])) < freq_dist:
                freq_dist_lims1[i]+=1
                if secresind1[i]+freq_dist_lims1[i] >= len(freq):
                     break
                
    for i in range(len(secresind2)):
        if freq[secresind2[i]] > 0:
            while ((freq[secresind2[i]+spreads2[i]]/freq[secresind2[i]-spreads2[i]])) < spread_dist:
                spreads2[i]+=1
                if secresind2[i]+spreads2[i] >= len(freq):
                    break
            while ((freq[secresind2[i]+freq_dist_lims2[i]]/freq[secresind2[i]-freq_dist_lims2[i]])) < freq_dist:
                freq_dist_lims2[i]+=1
                if secresind2[i]+freq_dist_lims2[i] >= len(freq):
                    break
        else:
            while ((freq[secresind2[i]-spreads2[i]]/freq[secresind2[i]+spreads2[i]])) < spread_dist:
                spreads2[i] +=1
                if secresind2[i]+spreads2[i] >= len(freq):
                    break
            while ((freq[secresind2[i]-freq_dist_lims2[i]]/freq[secresind2[i]+freq_dist_lims2[i]])) < freq_dist:
                freq_dist_lims2[i]+=1
                if secresind2[i]+freq_dist_lims2[i] >= len(freq):
                    break


    ##########################################################################################
    #Using different spreads for different values
    for i in range(len(secresind1)):
        if secresind1[i] == gind:
            continue
        if abs(secresind1[i] - gind) < freq_dist_lims1[i]:
            continue
    
        if spreads1[i] > 0:
            Yhk_f[secresind1[i]-spreads1[i]:secresind1[i]+spreads1[i]] = 0.
        else:
            Yhk_f[secresind1[i]] = 0.
                
    for i in range(len(secresind2)):
        if secresind2[i] == sind:
            continue
        if abs(secresind2[i] - sind) < freq_dist_lims2[i]:
            continue
    
        if spreads2[i] > 0:
            Ypq_f[secresind2[i]-spreads2[i]:secresind2[i]+spreads2[i]] = 0.
        else:
            Ypq_f[secresind2[i]] = 0.
    ##########################################################################################



    Ypq_f[limit_ind] = 0.
    Yhk_f[limit_ind] = 0.
    Ya_f[limit_indr] = 0.
    
    pq_f = np.fft.ifft(Ypq_f,len(p))
    hk_f = np.fft.ifft(Yhk_f,len(h))
    a_f = np.fft.irfft(Ya_f,len(a))


    #p = np.sin(inc)*np.sin(node)
    #q = np.sin(inc)*np.cos(node)
    #h = (e)*np.sin(node+aperi)
    #k = (e)*np.cos(node+aperi)
    #Yhk = np.fft.rfft(k + 1j*h)
    #Ypq = np.fft.rfft(q+1j*p)


    #dallin's code
    #sini_f = np.abs(pq_f)
    #ecc_f = np.abs(hk_f) 
    q_f = np.real(pq_f)
    p_f = np.imag(pq_f)

    k_f = np.real(hk_f)
    h_f = np.imag(hk_f)



    #reconstruct the filtered time series
    #p_f = np.fft.irfft(Yp_f,len(p))
    #q_f = np.fft.irfft(Yq_f,len(q))
    #h_f = np.fft.irfft(Yh_f,len(h))
    #k_f = np.fft.irfft(Yk_f,len(k))
    #a_f = np.fft.irfft(Ya_f,len(a))

    lim=int(np.floor(len(p_f)*0.05))
    

    sini_f = np.sqrt(p_f*p_f + q_f*q_f)
    ecc_f = np.sqrt(h_f*h_f + k_f*k_f) 

    node_f = np.arctan2(p_f/sini_f,q_f/sini_f)
    pomega_f = np.arctan2(h_f/ecc_f, k_f/ecc_f)


    temp_node = np.unwrap(node_f[lim:-lim])
    temp_pomega = np.unwrap(pomega_f[lim:-lim])

    slope = np.polyfit(t[lim:-lim],temp_node,1)[0]
    elements.ps = slope/(2*np.pi)

    slope = np.polyfit(t[lim:-lim],temp_pomega,1)[0]
    elements.pg = slope/(2*np.pi)

    elements.e = np.nanmean(ecc_f[2*lim:-2*lim])
    elements.sini = np.nanmean(sini_f[lim:-lim])
    elements.a = np.nanmean(a_f[lim:-lim])

    elements.pomega = pomega_f[0]
    elements.node = node_f[0]


    return 1, elements, p, q, p_f, q_f, pq_f, h, k, h_f, k_f, hk_f


def calc_frequencies(g,s,small_planets_flag):

    import hard_coded_constants as const

    g1 = const.g1
    g2 = const.g2
    g3 = const.g3
    g4 = const.g4
    g5 = const.g5
    g6 = const.g6
    g7 = const.g7
    g8 = const.g8
    g9 = const.g9
    g10 = const.g10

    s1 = const.s1
    s2 = const.s2
    s3 = const.s3
    s4 = const.s4
    s6 = const.s6
    s7 = const.s7
    s8 = const.s8

    #in addition to the 8 modes associated with the planets,
    #we filter the following combinations of modes
    z = np.zeros(66)
    #copied directly from findorb
    z[1]=g-g5 
    z[2]=g-g6 
    # this one cannot give rise to a resonance                              
    z[3]=g5-g6 
    z[4]=s-s7 
    z[5]=s-s6 
    #! this one cannot give rise to a resonance                              
    z[6]=s7-s6 
    z[7]=g+s-s7-g5 
    z[8]=g+s-s7-g6 
    z[9]=g+s-s6-g5 
    z[10]=g+s-s6-g6 
    z[11]=2.*g-2.*s 
    z[12]=g-2.*g5+g6 
    z[13]=g+g5-2.*g6 
    z[14]=2.*g-g5-g6 
    z[15]=-g+s+g5-s7 
    z[16]=-g+s+g6-s7 
    z[17]=-g+s+g5-s6 
    z[18]=-g+s+g6-s6 
    z[19]=g-g5+s7-s6 
    z[20]=g-g5-s7+s6 
    z[21]=g-g6+s7-s6 
    z[22]=g-g6-s7+s6 
    z[23]=2.*g-s-s7 
    z[24]=2.*g-s-s6 
    z[25]=-g+2.*s-g5 
    z[26]=-g+2.*s-g6 
    z[27]=2.*g-2.*s7 
    z[28]=2.*g-2.*s6 
    z[29]=2.*g-s7-s6 
    z[30]=g-s+g5-s7 
    z[31]=g-s+g5-s6 
    z[32]=g-s+g6-s7 
    z[33]=g-s+g6-s6 
    z[34]=g+g5-2.*s7 
    z[35]=g+g6-2.*s7 
    z[36]=g+g5-2.*s6 
    z[37]=g+g6-2.*s6 
    z[38]=g+g5-s7-s6 
    z[39]=g+g6-s7-s6 
    z[40]=s-2.*s7+s6 
    z[41]=s+s7-2.*s6 
    z[42]=2.*s-s7-s6 
    z[43]=s+g5-g6-s7 
    z[44]=s-g5+g6-s7 
    z[45]=s+g5-g6-s6 
    z[46]=s-g5+g6-s6 
    z[47]=2.*s-2.*g5 
    z[48]=2.*s-2.*g6 
    z[49]=2.*s-g5-g6 
    z[50]=s-2.*g5+s7 
    z[51]=s-2.*g5+s6 
    z[52]=s-2.*g6+s7 
    z[53]=s-2.*g6+s6 
    z[54]=s-g5-g6+s7 
    z[55]=s-g5-g6+s6 
    z[56]=2.*g-2.*g5 
    z[57]=2.*g-2.*g6 
    z[58]=2.*s-2.*s7 
    z[59]=2.*s-2.*s6 
    #  divisors appearing only in forced terms                              
    z[60]=g-2.*g6+g7 
    z[61]=g-3.*g6+2.*g5 
    #  degree six divisor z2                                                
    z[62]=2.*(g-g6)+(s-s6) 
    #!  other nonlinear forced terms                                         
    z[63]=g+g5-g6-g7 
    z[64]=g-g5-g6+g7 
    z[65]=g+g5-2*g6-s6+s7 
    #eccentricity only z frequencies 
    eonly = [0,1,2,3,12,13,14,56,57,60,61,63,64]
    #inclination only z frequencies
    ionly = [0,4,5,6,58,59]

    z_e = np.delete(z,ionly)
    z_i = np.delete(z,eonly)

    #z = z[1:65]

    if small_planets_flag:
        freq1 = [g1,g2,g3,g4,g5,g6,g7,g8,g9,g10]
        freq2 = [s1,s2,s3,s4,0.5*s6,s6,s7,s8]
    else:
        freq1 = [g5,g6,g7,g8,g9,g10]
        freq2 = [0.5*s6,s6,s7,s8]
    


    freq1.extend(z_e)
    freq2.extend(z_i)

    return freq1,freq2

