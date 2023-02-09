import pandas as pd
import numpy as np
import rebound
import tools
import sys
import os

a = np.zeros(1);e = np.zeros(1);inc = np.zeros(1);phi = np.zeros(1)
t = np.zeros(1);

r2d = 180/np.pi

astdys = pd.read_csv('astdys_tnos.csv')

#for i in range(len(astdys)):
#for i in range(10):
objnum = int(sys.argv[1])
print(objnum)
objname = str(astdys['Name'].iloc[objnum])


#objname = 'Ceres'
sbody = objname

filename = "TNOs/"+objname
sa = rebound.SimulationArchive(filename+'/archive.bin')
print(sa.tmin)
print(sa.tmax)

planets = ['jupiter','saturn','uranus','neptune']

a = np.zeros(len(sa));
e = np.zeros(len(sa));
inc = np.zeros(len(sa));
qp = np.zeros(len(sa),dtype=complex);
ptp = np.zeros(len(sa));
qtp = np.zeros(len(sa));
kh = np.zeros(len(sa),dtype=complex);
#k = np.zeros(len(sa));
t = np.zeros(len(sa));
npl = 5

qppl = np.asfortranarray(np.zeros(shape=[len(sa),npl],dtype=complex));
#qpl = np.asfortranarray(np.zeros(shape=[len(sa),npl]));
khpl = np.asfortranarray(np.zeros(shape=[len(sa),npl],dtype=complex));
#kpl = np.asfortranarray(np.zeros(shape=[len(sa),npl]));

hpl = np.zeros((len(sa),npl))
kpl = np.zeros((len(sa),npl))
ppl = np.zeros((len(sa),npl))
qpl = np.zeros((len(sa),npl))

hpart = np.zeros(len(sa))
kpart = np.zeros(len(sa))
ppart = np.zeros(len(sa))
qpart = np.zeros(len(sa))

apl = np.zeros((len(sa),npl))
for j,sim in enumerate(sa):
    tp = sim.particles[sbody+"_bf"]
    com = sim.calculate_com()
    o = tp.calculate_orbit(com)
    t[j] = sim.t
    a[j] = o.a
    p = np.sin(o.inc)*np.sin(o.Omega)
    q = np.sin(o.inc)*np.cos(o.Omega)
    ptp[j] = p
    qtp[j] = q
    qp[j] = 1j*p + q
    h = (o.e)*np.sin(o.Omega+o.omega)
    k = (o.e)*np.cos(o.Omega+o.omega)
    
    hpart[j] = h
    kpart[j] = k
    ppart[j] = p
    qpart[j] = q
    
    kh[j] = 1j*k + h
    
    e[j] = o.e
    inc[j] = o.inc*180/np.pi
    #print(sim.particles[1].inc)
    for i in range (0,npl):
        #print(planets[i])
        
        pl = sim.particles[i]
        o = pl.calculate_orbit(com)
        
        apl[j,i] = o.a
        ptemp = np.sin(o.inc)*np.sin(o.Omega)
        qtemp = np.sin(o.inc)*np.cos(o.Omega)
        qppl[j,i] = 1j*ptemp + qtemp
        htemp = (o.e)*np.sin(o.Omega+o.omega)
        ktemp = (o.e)*np.cos(o.Omega+o.omega)
        khpl[j,i] = 1j*ktemp + htemp
        hpl[j,i] = htemp
        kpl[j,i] = ktemp
        ppl[j,i] = ptemp
        qpl[j,i] = qtemp

series = pd.DataFrame(columns=['t','a','ecc','inc','p','q','h','k','hj','kj','pj','qj','hs','ks','ps','qs','hu','ku','pu','qu','hn','kn','pn','qn','megno','lyapunov'])

print(len(hpl),len(hpl[0,:]))
series['t'] = t
series['a'] = a
series['ecc'] = e
series['inc'] = inc
series['p'] = ppart
series['q'] = qpart
series['h'] = hpart
series['k'] = kpart
'''
series['hmc'] = hpl[:,0]
series['kmc'] = kpl[:,0]
series['pmc'] = ppl[:,0]
series['qmc'] = qpl[:,0]
series['hv'] = hpl[:,1]
series['kv'] = kpl[:,1]
series['pv'] = ppl[:,1]
series['qv'] = qpl[:,1]
series['he'] = hpl[:,2]
series['ke'] = kpl[:,2]
series['pe'] = ppl[:,2]
series['qe'] = qpl[:,2]
series['hmr'] = hpl[:,3]
series['kmr'] = kpl[:,3]
series['pmr'] = ppl[:,3]
series['qmr'] = qpl[:,3]
'''
series['hj'] = hpl[:,0]
series['kj'] = kpl[:,0]
series['pj'] = ppl[:,0]
series['qj'] = qpl[:,0]
series['hs'] = hpl[:,1]
series['ks'] = kpl[:,1]
series['ps'] = ppl[:,1]
series['qs'] = qpl[:,1]
series['hu'] = hpl[:,2]
series['ku'] = kpl[:,2]
series['pu'] = ppl[:,2]
series['qu'] = qpl[:,2]
series['hn'] = hpl[:,3]
series['kn'] = kpl[:,3]
series['pn'] = ppl[:,3]
series['qn'] = qpl[:,3]
series['megno'] = np.ones(len(t))*sa[-1].calculate_megno()
series['lyapunov'] = np.ones(len(t))*sa[-1].calculate_lyapunov()
#print('Megno Calc: ',sa[-1].calculate_megno())
#print(sa)
#print(sa[0])
series.to_csv(filename+'/series.csv')

