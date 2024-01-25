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
    
def bin_to_df(folder,objname,archive, astdys,planet_set = '4planet', clonenum = 0):
    #print(folder,objname,archive,astdys)
    r2d = 180/np.pi
    sbody = str(astdys['Name'].iloc[int(objname)])
    #print(sbody)   #sbody = '2'
    
    filename = 'Sims/'+str(folder)+'/'+str(objname)
    #filename = '~/../../../hdd/haumea-data/djspenc/SBDynT_Sims/'+str(folder)+'/'+str(objname)
    #print(filename)
    #print(sbody)    
    
    #print(archive.tmin)
    #print(archive.tmax)
    if planet_set=='4planet':
        planets = ['jupiter','saturn','uranus','neptune']
        npl = 5
    else:
        planets = ['mercury','venus','earth','mars','jupiter','saturn','uranus','neptune']
        npl = 9
    
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
    #print('lenght:', len(archive))
    #print(planets)

    #print(npl)
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
    megno = np.zeros(len(archive))
    x = np.zeros(len(archive))
    y = np.zeros(len(archive))
    z = np.zeros(len(archive))
    vx = np.zeros(len(archive))
    vy = np.zeros(len(archive))
    vz = np.zeros(len(archive))
    ax = np.zeros(len(archive))
    ay = np.zeros(len(archive))
    az = np.zeros(len(archive))
    
    
    apl = np.zeros((len(archive),npl))
    for j,sim in enumerate(archive):
        
        #print("j ", j)
        try:
            #print('Looking for particle ', sbody + '_bf', ' in sim.')
            if clonenum == 0:
                tp = sim.particles[sbody+"_bf"]
            else:
                tp = sim.particles[sbody+"_"+str(clonenum)]
        
            #if j%10==0:
                #print(tp)
                #print(tp.x)
            #x[j] = tp.x
            #y[j] = tp.y
            #z[j] = tp.z
            #vx[j] = tp.vx
            #vy[j] = tp.vy
            #vz[j] = tp.vz
            #ax[j] = tp.ax
            #ay[j] = tp.ay
            #az[j] = tp.az
        
        except Exception as error:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(error) 
            print('Object was ejected from simulation. Setting ejection to True in runprops.')
            series = pd.DataFrame(columns=['t','a','ecc','an','eccn','inc','p','q','h','k','hmc','kmc','pcm','qmc','hv','kv','pv','qv','he','ke','pe','qe','hmr','kmr','pmr','qmr','hj','kj','pj','qj','hs','ks','ps','qs','hu','ku','pu','qu','hn','kn','pn','qn','megno','lyapunov'])
    
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
            series['x'] = x
            series['y'] = y
            series['z'] = z
            series['vx'] = vx
            series['vy'] = vy
            series['vz'] = vz
            series['ax'] = ax
            series['ay'] = ay
            series['az'] = az
            if planet_set == '4planet':
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
            else:
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
                
                series['hj'] = hpl[:,4]
                series['kj'] = kpl[:,4]
                series['pj'] = ppl[:,4]
                series['qj'] = qpl[:,4]
                series['hs'] = hpl[:,5]
                series['ks'] = kpl[:,5]
                series['ps'] = ppl[:,5]
                series['qs'] = qpl[:,5]
                series['hu'] = hpl[:,6]
                series['ku'] = kpl[:,6]
                series['pu'] = ppl[:,6]
                series['qu'] = qpl[:,6]
                series['hn'] = hpl[:,7]
                series['kn'] = kpl[:,7]
                series['pn'] = ppl[:,7]
                series['qn'] = qpl[:,7]
    
            series.to_csv(filename+'/series.csv')
            #runprops['Ejected'] = True
            
            break;
    
        #print(tp)
        tpn = sim.particles["neptune"]
        com = sim.calculate_com()
        o = tp.calculate_orbit(com)
        on = tpn.calculate_orbit(com)
        #print(sim.integrator)
        #if sim.integrator == 'whfast':
        #    megno[j] = sim.calculate_megno()
        #if sim.integrator == 'ias15':
        #    megno[j] = sim.calculate_megno()
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
        
        #print(sim.particles[1].in
        #print(npl)
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
    
    series = pd.DataFrame(columns=['t','a','ecc','an','eccn','inc','p','q','h','k','hmc','kmc','pcm','qmc','hv','kv','pv','qv','he','ke','pe','qe','hmr','kmr','pmr','qmr','hj','kj','pj','qj','hs','ks','ps','qs','hu','ku','pu','qu','hn','kn','pn','qn','megno','lyapunov','x','y','z'])
    
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
    series['megno'] = megno
    series['x'] = x
    series['y'] = y
    series['z'] = z
    series['vx'] = vx
    series['vy'] = vy
    series['vz'] = vz
    series['ax'] = ax
    series['ay'] = ay
    series['az'] = az
    print(planet_set)
    if planet_set == '4planet':
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
    else:
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
                
        series['hj'] = hpl[:,4]
        series['kj'] = kpl[:,4]
        series['pj'] = ppl[:,4]
        series['qj'] = qpl[:,4]
        series['hs'] = hpl[:,5]
        series['ks'] = kpl[:,5]
        series['ps'] = ppl[:,5]
        series['qs'] = qpl[:,5]
        series['hu'] = hpl[:,6]
        series['ku'] = kpl[:,6]
        series['pu'] = ppl[:,6]
        series['qu'] = qpl[:,6]
        series['hn'] = hpl[:,7]
        series['kn'] = kpl[:,7]
        series['pn'] = ppl[:,7]
        series['qn'] = qpl[:,7]

    
    return series

