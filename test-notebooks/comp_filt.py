import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy


import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
#sys.path.insert(0, '/Users/kvolk/Documents/GitHub/SBDynT/src')
sys.path.insert(0, '../src')
import run_reb
import rebound
import numpy as np
import horizons_api
import tools
from matplotlib.backends.backend_pdf import PdfPages

import scipy.signal as signal
#(keep the space in sbody, it's a typo from when I ran it, but the hash has the space)\
#this will take a while, it's a 100 Myr integration with a 500 year output cadence
sbody = '2440'
objname = 'Educatio'
dt=500.
path = 'Asteroids/'+objname
#path = 'TNOs/'+objname


#these arrays are all a bit of a mess, at some point would want to 
#change it to be a multi-dimensional array for the planets so they 
#don't all have to be handled by hand
a = np.zeros(1);e = np.zeros(1);inc = np.zeros(1);
p = np.zeros(1);q = np.zeros(1);h = np.zeros(1);k = np.zeros(1);
pu = np.zeros(1);qu = np.zeros(1);hu = np.zeros(1);ku = np.zeros(1);
pj = np.zeros(1);qj = np.zeros(1);hj = np.zeros(1);kj = np.zeros(1);
ps = np.zeros(1);qs = np.zeros(1);hs = np.zeros(1);ks = np.zeros(1);
pn = np.zeros(1);qn = np.zeros(1);hn = np.zeros(1);kn = np.zeros(1);
omega = np.zeros(1);
Omega = np.zeros(1);
M = np.zeros(1)
t = np.zeros(1);
sa = rebound.SimulationArchive(path+"/archive.bin")
print("start time %f" % sa.tmin)
print("stop time %f" % sa.tmax)

sim=sa[-1]
print(sim)
print(sim.particles)
'''
planets = ['jupiter','saturn','uranus','neptune']
for i,sim in enumerate(sa):
    tp = sim.particles[sbody+"_bf"]
    nep = sim.particles["neptune"]
    ura = sim.particles["uranus"]
    sat = sim.particles["saturn"]
    jup = sim.particles["jupiter"]
    com = sim.calculate_com()
    o = tp.calculate_orbit(com)
    
    onep = nep.calculate_orbit(com)
    oura = ura.calculate_orbit(com)
    osat = sat.calculate_orbit(com)
    ojup = jup.calculate_orbit(com)

    t = np.append(t, sim.t)
    a = np.append(a, o.a)
    p = np.append(p, np.sin(o.inc)*np.sin(o.Omega))
    q = np.append(q, np.sin(o.inc)*np.cos(o.Omega))
    h = np.append(h, (o.e)*np.sin(o.Omega+o.omega))
    k = np.append(k, (o.e)*np.cos(o.Omega+o.omega))

    pj = np.append(pj, np.sin(ojup.inc)*np.sin(ojup.Omega))
    qj = np.append(qj, np.sin(ojup.inc)*np.cos(ojup.Omega))
    hj = np.append(hj, (ojup.e)*np.sin(ojup.Omega+ojup.omega))
    kj = np.append(kj, (ojup.e)*np.cos(ojup.Omega+ojup.omega))

    ps = np.append(ps, np.sin(osat.inc)*np.sin(osat.Omega))
    qs = np.append(qs, np.sin(osat.inc)*np.cos(osat.Omega))
    hs = np.append(hs, (osat.e)*np.sin(osat.Omega+osat.omega))
    ks = np.append(ks, (osat.e)*np.cos(osat.Omega+osat.omega))

    pu = np.append(pu, np.sin(oura.inc)*np.sin(oura.Omega))
    qu = np.append(qu, np.sin(oura.inc)*np.cos(oura.Omega))
    hu = np.append(hu, (oura.e)*np.sin(oura.Omega+oura.omega))
    ku = np.append(ku, (oura.e)*np.cos(oura.Omega+oura.omega))

    pn = np.append(pn, np.sin(onep.inc)*np.sin(onep.Omega))
    qn = np.append(qn, np.sin(onep.inc)*np.cos(onep.Omega))
    hn = np.append(hn, (onep.e)*np.sin(onep.Omega+onep.omega))
    kn = np.append(kn, (onep.e)*np.cos(onep.Omega+onep.omega))


    e = np.append(e, o.e)
    omega = np.append(omega, o.omega*180/np.pi)
    Omega = np.append(Omega, o.Omega*180/np.pi)
    M = np.append(M, o.M*180/np.pi)
    
    inc = np.append(inc, o.inc*180/np.pi)
'''
'''
series = pd.read_csv(path+'/series.csv')
t = series['t'].values
a = series['a'].values
e = series['e'].values
inc = series['inc'].values
omega = series['omega'].values
Omega = series['Omega'].values
M = series['M'].values
h = np.sin(inc)*np.sin(Omega)
k = np.sin(inc)*np.cos(Omega)
p = e*np.sin(Omega+omega)
q = e*np.cos(Omega+omega)

hj = np.delete(hj,0)
kj = np.delete(kj,0)
pj = np.sin
qj = np.delete(qj,0)

 pj = np.append(pj, np.sin(ojup.inc)*np.sin(ojup.Omega))
    qj = np.append(qj, np.sin(ojup.inc)*np.cos(ojup.Omega))
    hj = np.append(hj, (ojup.e)*np.sin(ojup.Omega+ojup.omega))
    kj = np.append(kj, (ojup.e)*np.cos(ojup.Omega+ojup.omega))
'''
series = pd.read_csv(path+'/series_new.csv')

t = series['t'].values
a = series['a'].values
e = series['e'].values
inc = series['inc'].values
omega = series['omega'].values
Omega = series['Omega'].values
M = series['M'].values
h = series['h'].values
k = series['k'].values
p = series['p'].values
q = series['q'].values

hj = series['hj'].values
hs = series['hs'].values
hu = series['hu'].values
hn = series['hn'].values
kj = series['kj'].values
ks = series['ks'].values
ku = series['ku'].values
kn = series['kn'].values
pj = series['pj'].values
ps = series['ps'].values
pu = series['pu'].values
pn = series['pn'].values
qj = series['qj'].values
qs = series['qs'].values
qu = series['qu'].values
qn = series['qn'].values
'''
t = np.delete(t,0)
a = np.delete(a,0)
e = np.delete(e,0)
inc = np.delete(inc,0)
omega = np.delete(omega,0)
Omega = np.delete(Omega,0)
M = np.delete(M,0)
h = np.delete(h,0)
k = np.delete(k,0)
p = np.delete(p,0)
q = np.delete(q,0)

hj = np.delete(hj,0)
kj = np.delete(kj,0)
pj = np.delete(pj,0)
qj = np.delete(qj,0)
hs = np.delete(hs,0)
ks = np.delete(ks,0)
ps = np.delete(ps,0)
qs = np.delete(qs,0)
hu = np.delete(hu,0)
ku = np.delete(ku,0)
pu = np.delete(pu,0)
qu = np.delete(qu,0)
hn = np.delete(hn,0)
kn = np.delete(kn,0)
pn = np.delete(pn,0)
qn = np.delete(qn,0)
'''

def butter_notch(sig, sigtype, sigmax):
    quality_factor = 3.
    nmax = len(freq)
    plt.figure(figsize = (9,9))
    fh1s = np.zeros((4,len(np.fft.rfft(sig))))
    fh2s = np.zeros((4,len(np.fft.rfft(sig))))
    ffreqs = np.zeros((4,len(np.fft.rfft(sig))))
    fs = 1/500.
    notch_sig = np.copy(sig)
    butter_sig = np.copy(sig)
    for i in range(len(sigmax)):
        filt = freq[sigmax[i]]
        print(1/freq[sigmax[i]])

    #Infinite Impulse Response Notch Filter
    #Give the function a frequnecy to filter out and it will filter that specific frequency
        b_f, a_f = signal.iirnotch(filt,quality_factor,fs=fs)
    
        notch_sig = signal.filtfilt(b_f, a_f, notch_sig)
                
        ffreq, fh = signal.freqz(b_f,a_f, fs=2.*np.pi,worN=nmax);
        #print('fh: ',fh)
        #Butterworth Bandstop Filter
        #A digital filter that 
        b, a = signal.butter(2, [freq[sigmax[i]-5],freq[sigmax[i]+5]], btype='bandstop',fs=fs)
        butter_sig = signal.filtfilt(b,a,butter_sig)
        
        ffreq2, fh2 = signal.freqz(b,a,fs=2*np.pi,worN=nmax)
        ffreqs[i] = ffreq
        fh1s[i] = fh
        fh2s[i] = fh2
    
    
    tY = np.fft.rfft(notch_sig)
    ptY = np.abs(tY)
    
    tY_2 = np.fft.rfft(butter_sig)
    ptY2 = np.abs(tY_2)
            
    '''
    for i in range(len(sigmax)):
        filt = freq[sigmax[i]]
        b_f, a_f = signal.iirnotch(filt,quality_factor,fs=fs)
        alt_f_p = signal.filtfilt(b_f, a_f, sig)
        tY = np.fft.rfft(alt_f_p)
        ptY = np.abs(tY)
        
        b, a = signal.butter(2, [freq[sigmax[i]-5],freq[sigmax[i]+5]], btype='bandstop',fs=fs)
        alt_f_p2 = signal.filtfilt(b,a,sig)
        
        tY_2 = np.fft.rfft(alt_f_p2)
        ptY2 = np.abs(tY_2)
        
        ffreq2, fh2 = signal.freqz(b,a,fs=2*np.pi,worN=nmax)
    '''
    #print(fh1s, np.abs(fh1s))
    fh1_total = np.abs(fh1s[0])*np.abs(fh1s[1])*np.abs(fh1s[2])*np.abs(fh1s[3])
    fh2_total = np.abs(fh2s[0])*np.abs(fh2s[1])*np.abs(fh2s[2])*np.abs(fh2s[3])
    #print(fh_total)
    plt.scatter(ffreqs[0]*fs/(2.*np.pi), np.abs(fh1_total),
                 c='r', label='Notch Bandpass filter',rasterized=True,s=1);
    plt.scatter(ffreqs[0]*fs/(2.*np.pi), np.abs(fh2_total)-0.75,
                 c='c', label='Butterworth Bandpass filter',rasterized=True,s=1);
    print(len(freq),len(np.abs(np.fft.rfft(sig))), len(ptY), len(ptY2))
    plt.scatter(freq, np.abs(np.fft.rfft(sig)),
             c='k', label='before filtering',rasterized=True,s=3,alpha=0.5)
    plt.scatter(freq, ptY,
             c='g', label='after Notch filtering',rasterized=True,s=3,alpha=0.5);
    plt.scatter(freq, ptY2,
             c='b', label='after Butter filtering',rasterized=True,s=3,alpha=0.5);
        
    plt.legend();
         
    plt.xlabel('Frequency [yr^{-1}]', fontsize=20);
    plt.ylabel('Magnitude [dB]', fontsize=20);
    plt.xscale('log');
    plt.yscale('log');
    #plt.xlim([1e-8,1e-4])
    #plt.ylim([1e-5,1e4])
    plt.title('Notch vs Butterworth Filters '+sigtype, fontsize=20);
    likelihoodspdf.attach_note(sigtype)
    likelihoodspdf.savefig()
    
    return ptY, ptY2

#set up all the FFT power spectra, etc
#the outputs are not exact, but it's about 

n = len(h)
freq = np.fft.rfftfreq(n,d=dt)

#particle eccentricity vectors
Yh= np.fft.rfft(k)
Yk = np.fft.rfft(h)

#giant planets
Yhu = np.fft.rfft(hu)
Yhj = np.fft.rfft(hj)
Yhn = np.fft.rfft(hn)
Yhs = np.fft.rfft(hs)
Yku = np.fft.rfft(ku)
Ykj = np.fft.rfft(kj)
Ykn = np.fft.rfft(kn)
Yks = np.fft.rfft(ks)

#convert to power
pYh = np.abs(Yh)
pYk = np.abs(Yk)
pYhu = np.abs(Yhu)
pYhn = np.abs(Yhn)
pYhj = np.abs(Yhj)
pYhs = np.abs(Yhs)
pYku = np.abs(Yku)
pYkn = np.abs(Ykn)
pYkj = np.abs(Ykj)
pYks = np.abs(Yks)


#find the max power and indexes of that max power
#(disregarding the frequency=0 terms)
kumax = pYku[1:].max()
knmax = pYkn[1:].max()
ksmax = pYks[1:].max()
kjmax = pYkj[1:].max()
humax = pYhu[1:].max()
hnmax = pYhn[1:].max()
hsmax = pYhs[1:].max()
hjmax = pYhj[1:].max()

#(these need the plus 1 to account for neglecting the f=0 term)
ihumax = np.argmax(pYhu[1:])+1
ihnmax = np.argmax(pYhn[1:])+1 
ihsmax = np.argmax(pYhs[1:])+1 
ihjmax = np.argmax(pYhj[1:])+1 
ikumax = np.argmax(pYku[1:])+1 
iknmax = np.argmax(pYkn[1:])+1
iksmax = np.argmax(pYks[1:])+1
ikjmax = np.argmax(pYkj[1:])+1 



#particle inclination vectors
Yp= np.fft.rfft(p)
Yq = np.fft.rfft(q)
#giant planets
Ypu = np.fft.rfft(pu)
Ypj = np.fft.rfft(pj)
Ypn = np.fft.rfft(pn)
Yps = np.fft.rfft(ps)
Yqu = np.fft.rfft(qu)
Yqj = np.fft.rfft(qj)
Yqn = np.fft.rfft(qn)
Yqs = np.fft.rfft(qs)

#convert to power
pYp = np.abs(Yp)
pYpu = np.abs(Ypu)
pYpn = np.abs(Ypn)
pYpj = np.abs(Ypj)
pYps = np.abs(Yps)
pYqu = np.abs(Yqu)
pYqn = np.abs(Yqn)
pYqj = np.abs(Yqj)
pYqs = np.abs(Yqs)

#find the max power and indexes of that max power
#(disregarding the frequency=0 terms)
pumax = pYpu[1:].max()
pnmax = pYpn[1:].max()
psmax = pYps[1:].max()
pjmax = pYpj[1:].max()
qumax = pYqu[1:].max()
qnmax = pYqn[1:].max()
qsmax = pYqs[1:].max()
qjmax = pYqj[1:].max()


ipumax = np.argmax(pYpu[1:])+1
ipnmax = np.argmax(pYpn[1:])+1 
ipsmax = np.argmax(pYps[1:])+1 
ipjmax = np.argmax(pYpj[1:])+1 
iqumax = np.argmax(pYqu[1:])+1 
iqnmax = np.argmax(pYqn[1:])+1
iqsmax = np.argmax(pYqs[1:])+1
iqjmax = np.argmax(pYqj[1:])+1 

pmax = np.array([ipjmax,ipsmax,ipumax,ipnmax])
qmax = np.array([iqjmax,iqsmax,iqumax,iqnmax])
hmax = np.array([ihjmax,ihsmax,ihumax,ihnmax])
kmax = np.array([ikjmax,iksmax,ikumax,iknmax])

signals = np.array([h,k,p,q])

print("peak planet eccentricity periods (years):")
print("Jupiter %f" % (1/freq[ikjmax]))
print("Saturn %f" % (1/freq[iksmax]))
print("Uranus %f" % (1/freq[ikumax]))
print("Neptune %f" % (1/freq[iknmax]))
likelihoodspdf = PdfPages(path+"/comp_butter_notch.pdf")
plt.figure(figsize = (9,9))


htY1, htY2 = butter_notch(h, "H vector", hmax)
ktY1, ktY2 = butter_notch(k, "K vector", kmax)
ptY1, ptY2 = butter_notch(p, "P vector", pmax)
qtY1, qtY2 = butter_notch(q, "Q vector", qmax)

   
#Plot figure

likelihoodspdf.close()
plt.close("all")

sigs = np.zeros((4,len(h)))
sigs[0] = h
sigs[1] = k
sigs[2] = p
sigs[3] = q

b_sigs = np.zeros((4,len(htY1)))
b_sigs[0] = htY1
b_sigs[1] = ktY1
b_sigs[2] = ptY1
b_sigs[3] = qtY1

n_sigs = np.zeros((4,len(htY1)))
n_sigs[0] = htY2
n_sigs[1] = ktY2
n_sigs[2] = ptY2
n_sigs[3] = qtY2

fftpdf = PdfPages(path+"/comp_butter_final.pdf")
#plt.figure(figsize = (9,9))
names = ['H vector','K vector','P vector','Q vector']
rt = np.fft.ifft(np.fft.rfft(t))
for i in range(4):
    plt.figure(figsize = (9,9))
    print(len(t),len(sigs[i]), len(n_sigs[i]),len(b_sigs[i]), len(rt))
    plt.scatter(rt, np.fft.ifft(np.fft.rfft(sigs[i])),c='k', label='before filtering',rasterized=True,s=3,alpha=0.5)
    plt.scatter(rt, np.fft.ifft(n_sigs[i]),c='g', label='after Notch filtering',rasterized=True,s=3,alpha=0.5)
    #plt.scatter(rt, np.fft.ifft(b_sigs[i]),c='b', label='after Butter filtering',rasterized=True,s=3,alpha=0.5);
        
    plt.legend();
         
    plt.xlabel('Time', fontsize=20);
    plt.ylabel('Signal: '+names[i], fontsize=20);
    #plt.xscale('log');
    #plt.yscale('log');
    #plt.xlim([1e-8,1e-4])
    #plt.ylim([1e-5,1e4])
    plt.title('Notch vs Butterworth Filtered Signal: '+names[i], fontsize=20);
    fftpdf.attach_note(names[i])
    fftpdf.savefig()
        
fftpdf.close()
plt.close("all")   

'''
p = sin(inc)*sin(Omega)
q = sin(inc)*cos(Omega)
h = e*sin(Omega+omega)
k = e*cos(Omega+omega)

sigs = [h,k,p,q]
Omega = arctan(p/q)
inc = arcsin(p/sin(Omega))
pomega = arctan(h/k)
ecc = h/sin(pomega)
'''

Omega_n = np.arctan2(n_sigs[2],n_sigs[3])
inc_n = np.arcsin(n_sigs[2]/np.sin(Omega_n))
print(inc_n)
omega_n = np.arctan2(n_sigs[0],n_sigs[1])-Omega_n
eccs_n = n_sigs[0]/np.sin(Omega_n+omega_n)

Omega_b = np.arctan2(b_sigs[2],b_sigs[3])
inc_b = np.arcsin(b_sigs[2]/np.sin(Omega_b))
print(inc_b)
omega_b = np.arctan2(b_sigs[0],b_sigs[1])-Omega_b
eccs_b = b_sigs[0]/np.sin(Omega_b+omega_b)

print('SMA: 46.5155993  Ecc: 0.1167592 I: 0.1937245')
print('Proper elements Notch Filter')
print('Omega Notch: ', np.mean(Omega_n))
print('omega Notch: ', np.mean(omega_n))
print('Sin(i) Notch: ', np.nanmean(np.sin(inc_n)))
print('Ecc Notch: ', np.mean(eccs_n))

print('Proper elements Butterworth Filter')
print('Omega Butter: ', np.mean(Omega_b))
print('omega Butter: ', np.mean(omega_b))
print('Sin(i) Butter: ', np.nanmean(np.sin(inc_b)))
print('Ecc Butter: ', np.mean(eccs_b))