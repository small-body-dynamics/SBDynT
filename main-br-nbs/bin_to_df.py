import numpy as np
import pandas as pd
import commentjson as json
import rebound
import tools
import sys
import os

class ReadJson(object):
    def __init__(self, filename):
        print('Read the runprops.txt file')
        self.data = json.load(open(filename))
    def outProps(self):
        return self.data
    
def bin_to_df(objname,archive):

    r2d = 180/np.pi
    
    astdys = pd.read_csv('data_files/astdys_tnos.csv')

    sbody = objname
    
    filename = "TNOs/"+objname
    archive = archive
    
    
    #print(archive.tmin)
    #print(archive.tmax)
    
    planets = ['jupiter','archiveturn','uranus','neptune']
    
    a = np.zeros(len(archive));
    e = np.zeros(len(archive));
    an = np.zeros(len(archive));
    en = np.zeros(len(archive));
    
    inc = np.zeros(len(archive));
    qp = np.zeros(len(archive),dtype=complex);
    ptp = np.zeros(len(archive));
    qtp = np.zeros(len(archive));
    kh = np.zeros(len(archive),dtype=complex);
    #k = np.zeros(len(archive));
    t = np.zeros(len(archive));
    npl = 5
    
    qppl = np.asfortranarray(np.zeros(shape=[len(archive),npl],dtype=complex));
    khpl = np.asfortranarray(np.zeros(shape=[len(archive),npl],dtype=complex));
    
    hpl = np.zeros((len(archive),npl))
    kpl = np.zeros((len(archive),npl))
    ppl = np.zeros((len(archive),npl))
    qpl = np.zeros((len(archive),npl))
    
    hpart = np.zeros(len(archive))
    kpart = np.zeros(len(archive))
    ppart = np.zeros(len(archive))
    qpart = np.zeros(len(archive))
    
    apl = np.zeros((len(archive),npl))
    for j,sim in enumerate(archive):
        
        #print("j ", j)
        try:
            tp = sim.particles[sbody+"_bf"]
        except:
            print('Object was ejected from simulation. Setting ejection to True in runprops.')
            series = pd.DataFrame(columns=['t','a','ecc','an','eccn','inc','p','q','h','k','hj','kj','pj','qj','hs','ks','ps','qs','hu','ku','pu','qu','hn','kn','pn','qn','megno','lyapunov'])
    
            print(len(hpl),len(hpl[0,:]))
            series['t'] = t
            series['a'] = a
            series['ecc'] = e
            series['an'] = an
            series['eccn'] = en
            series['inc'] = inc
            series['p'] = ppart
            series['q'] = qpart
            series['h'] = hpart
            series['k'] = kpart
    
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
    
            series.to_csv(filename+'/series.csv')
            runprops['Ejected'] = True
            
            break;
    
        #print(tp)
        tpn = sim.particles["neptune"]
        com = sim.calculate_com()
        o = tp.calculate_orbit(com)
        on = tpn.calculate_orbit(com)
        t[j] = sim.t
        a[j] = o.a
        an[j] = on.a
        
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
        en[j] = on.e
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
    
    series = pd.DataFrame(columns=['t','a','ecc','an','eccn','inc','p','q','h','k','hj','kj','pj','qj','hs','ks','ps','qs','hu','ku','pu','qu','hn','kn','pn','qn','megno','lyapunov'])
    
    series['t'] = t
    series['a'] = a
    series['ecc'] = e
    series['an'] = an
    series['eccn'] = en
    series['inc'] = inc
    series['p'] = ppart
    series['q'] = qpart
    series['h'] = hpart
    series['k'] = kpart
    
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

    return series

