import sys
import os
sys.path.insert(0, '../src/')
import run_reb
import rebound
import numpy as np
import horizons_api
import tools
import pandas as pd
import matplotlib.pyplot as plt

sbody = 'J99RP3Z'
objname = 'Borasisi'

import os

path = 'TNOs/'+objname

# Check whether the specified path exists or not
isExist = os.path.exists(path)

if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(path)
    
sim= rebound.Simulation()
flag, epoch, sim = run_reb.initialize_simulation(planets=['Jupiter','Saturn','Uranus','Neptune'],des=sbody,clones=105)
#sim.status()
#print(epoch)
com = sim.calculate_com()

p = sim.particles[sbody+"_bf"]
o = p.calculate_orbit(com)
r2d = 180./np.pi
print("%20.15E\t%20.15E\t%20.15E\t%20.15E\t%20.15E\t%20.15E\n" % (o.a,o.e,o.inc*r2d, o.Omega*r2d,o.omega*r2d,r2d*o.M))

tmax = 1e7
tout = 1e3

import datetime
start = datetime.datetime.now()
sim = run_reb.run_simulation(sim, tmax=tmax, tout=tout,filename="TNOs/"+objname+"/archive.bin",deletefile=True,mindist=20.)
print('Simulation took ', datetime.datetime.now() - start, ' seconds')

a = np.zeros(1);e = np.zeros(1);inc = np.zeros(1);phi = np.zeros(1);omega = np.zeros(1);Omega = np.zeros(1);M = np.zeros(1)
t = np.zeros(1);
sa = rebound.SimulationArchive("TNOs/"+objname+"/archive.bin")
print(sa.tmin)
print(sa.tmax)
for i,sim in enumerate(sa):
    p = sim.particles[sbody+"_5"]
    n = sim.particles['neptune']
    com = sim.calculate_com()
    o = p.calculate_orbit(com)
    on = n.calculate_orbit(com)

    t = np.append(t, sim.t)
    a = np.append(a, o.a)
    e = np.append(e, o.e)
    omega = np.append(omega, o.omega*180/np.pi)
    Omega = np.append(Omega, o.Omega*180/np.pi)
    M = np.append(M, o.M*180/np.pi)
    
    inc = np.append(inc, o.inc*180/np.pi)
    lamda = o.Omega+o.omega+o.M
    lamdan = on.Omega+on.omega+on.M
    pt = 3*lamda - 2*lamdan - (o.Omega+o.omega)
    pt = tools.mod2pi(pt)
    phi = np.append(phi,pt)
    
t = np.delete(t,0)
a= np.delete(a,0)
e = np.delete(e,0)
inc = np.delete(inc,0)
phi = np.delete(phi,0)
omega = np.delete(omega,0)
Omega = np.delete(Omega,0)
M = np.delete(M,0)

final = pd.DataFrame(columns=['t','a','e','inc','omega','Omega','M','phi'])
final['t'] = t
final['a'] = a
final['e'] = e
final['inc'] = inc
final['omega'] = omega
final['Omega'] = Omega
final['M'] = M
final['phi'] = phi

final.to_csv('TNOs/'+objname+'/series.csv')

new_Xph = 1
def filter_signal(th):
    f_s = fft_filter(th)
    return np.real(np.fft.ifft(f_s))
def fft_filter(perc, signal):
    fft_signal = np.fft.fft(signal)
    fft_abs = np.abs(fft_signal)
    th=perc*(2*fft_abs[0:int(len(signal)/2.)]/new_Xph).max()
    fft_tof=fft_signal.copy()
    fft_tof_abs=np.abs(fft_tof)
    fft_tof_abs=2*fft_tof_abs/new_Xph
    fft_tof[fft_tof_abs<=th]=0
    return fft_tof
def fft_filter_amp(th, signal):
    fft = np.fft.fft(signal)
    fft_tof=fft.copy()
    fft_tof_abs=np.abs(fft_tof)
    fft_tof_abs=fft_tof_abs/new_Xph
    fft_tof_abs[fft_tof_abs<=th]=0
    return fft_tof_abs

efreq = np.fft.fft(e)
plt.plot(efreq[0:int(len(e)/200.)],color='darkorange')
plt.xlabel('Frequency Space (1/days)')
plt.ylabel('Fourier Amplitude')
plt.show()


th_list = np.linspace(0,1,5)
th_list = th_list[0:len(th_list)-1]

finals = []
for i in range(0,4):
    plt.subplot(2,2,i+1)
    th_i=th_list[2].round(2)
    if i == 0:
        th_filter = fft_filter(th_i, a)
    if i == 1:
        th_filter = fft_filter(th_i, e)
    if i == 2:
        th_filter = fft_filter(th_i, inc)
    if i == 3:
        th_filter = fft_filter(th_i, phi)
    
    signal_filter = np.real(np.fft.ifft(th_filter))
    
    finals = np.append(finals,np.mean(th_filter))
    
    plt.plot(t,signal_filter,color='firebrick',label='Threshold = %.2f'%(th_list[i]))
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    print(signal_filter)
plt.tight_layout()
plt.show()

#th_filter = fft_filter_amp(0.5,e)
#inv_fft = np.fft.ifft(th_filter)
#plt.plot(inv_fft,color='darkorange')
#plt.xlabel('Time')
#plt.ylabel('Ecc')
#plt.savefig(objname+'/final_fft.png')

final = pd.DataFrame(columns=['sma','ecc','inc','phi'])                             
final['sma'] = finals[0]
final['ecc'] = finals[1]
final['inc'] = finals[2]
final['phi'] = finals[3]

final.to_csv('TNOs/'+objname+'/prop_elem.csv')

                            