import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import h5py
import random
import os

def MLE_Norm(parameters, vector):
        # extract parameters
        #const, phase, std_dev = parameters
    c1,c2,c3,c4, period, const, phase, offset = parameters
    freq = 1/period
    c1 = c1
    c2 = c2
    c3 = c3
    c4 = c4
    added = 0
    y = 0
    if vector == "h":
        added = c1*hj+c2*hs+c3*hu+c4*hn
        y = h
    elif vector == "k":
        added = c1*kj+c2*ks+c3*ku+c4*kn
        y = k
    elif vector == "p":
        added = c1*pj+c2*ps+c3*pu+c4*pn
        y = p
    elif vector == "q":
        added = c1*qj+c2*qs+c3*qu+c4*qn
        y = q
    # predict the output
       
    pred = const*np.sin(2*np.pi*freq*t+phase) + added + offset
    #print(y)
    chisq = np.sum((y-pred)**2)
    LL = -0.5*chisq
    neg_LL = -1*LL
    #print(neg_LL)
    return LL

    


if __name__ == '__main__':

   # sbody = '2004 KF19'
   # objname = '2004KF19'
    
    sbody = '2004 PY107'
    objname = '2004PY107'
    
            #path = 'Asteroids/'+objname
    path = 'TNOs/'+objname
    series = pd.read_csv(path+'/series.csv')
    
    t = series['t'].values
    a = series['a'].values
    e = series['e'].values
    inc = series['inc'].values/180*np.pi
    
    dt = t[1]        #omega = series['omega'].values/180*np.pi
            #Omega = series['Omega'].values/180*np.pi
            #M = series['M'].values/180*np.pi
    h = series['h'].values
    k = series['k'].values
            
            #h = np.sin(inc)*np.sin(Omega)
            #k = np.sin(inc)*np.cos(Omega)
    #print(h)
            #plt.plot(inc,c='r')
            #O_n = np.arctan2(h,k)
            #i_n = np.arcsin(h/np.sin(O_n))
            #plt.plot(i_n,c='b')
            #plt.show()
    p = series['p'].values
    q = series['q'].values
                
    #p = e*np.sin(Omega+omega)
    #q = e*np.cos(Omega+omega)
            
    hj = series['hj'].values
    kj = series['kj'].values
    pj = series['pj'].values
    qj = series['qj'].values
    
    hs = series['hs'].values
    ks = series['ks'].values
    ps = series['ps'].values
    qs = series['qs'].values
    
    hn = series['hn'].values
    kn = series['kn'].values
    pn = series['pn'].values
    qn = series['qn'].values
            
    hu = series['hu'].values
    ku = series['ku'].values
    pu = series['pu'].values
    qu = series['qu'].values
                    
    pYh = np.abs(np.fft.rfft(h))
    pYhj = np.abs(np.fft.rfft(hj))
    pYhs = np.abs(np.fft.rfft(hs))
    pYhu = np.abs(np.fft.rfft(hu))
    pYhn = np.abs(np.fft.rfft(hn))
            
    hmax = np.argmax(pYh[1:])+1
    ihumax = np.argmax(pYhu[1:])+1
    ihnmax = np.argmax(pYhn[1:])+1 
    ihsmax = np.argmax(pYhs[1:])+1 
    ihjmax = np.argmax(pYhj[1:])+1 
            
    pYk = np.abs(np.fft.rfft(k))
    pYkj = np.abs(np.fft.rfft(kj))
    pYks = np.abs(np.fft.rfft(ks))
    pYku = np.abs(np.fft.rfft(ku))
    pYkn = np.abs(np.fft.rfft(kn))
            
    kmax = np.argmax(pYk[1:])+1
    ikumax = np.argmax(pYku[1:])+1
    iknmax = np.argmax(pYkn[1:])+1 
    iksmax = np.argmax(pYks[1:])+1 
    ikjmax = np.argmax(pYkj[1:])+1 
            
    pYp = np.abs(np.fft.rfft(p))
    pYpj = np.abs(np.fft.rfft(pj))
    pYps = np.abs(np.fft.rfft(ps))
    pYpu = np.abs(np.fft.rfft(pu))
    pYpn = np.abs(np.fft.rfft(pn))
            
    pmax = np.argmax(pYp[1:])+1
    ipumax = np.argmax(pYpu[1:])+1
    ipnmax = np.argmax(pYpn[1:])+1 
    ipsmax = np.argmax(pYps[1:])+1 
    ipjmax = np.argmax(pYpj[1:])+1 
    #print(pmax)
            
    pYq = np.abs(np.fft.rfft(q))
    pYqj = np.abs(np.fft.rfft(qj))
    pYqs = np.abs(np.fft.rfft(qs))
    pYqu = np.abs(np.fft.rfft(qu))
    pYqn = np.abs(np.fft.rfft(qn))
            
    qmax = np.argmax(pYq[1:])+1
    iqumax = np.argmax(pYqu[1:])+1
    iqnmax = np.argmax(pYqn[1:])+1 
    iqsmax = np.argmax(pYqs[1:])+1 
    iqjmax = np.argmax(pYqj[1:])+1 
        
    n = len(h)
    freqs = np.fft.rfftfreq(n,d=dt)
    #print(freqs[pmax])
    #plt.figure(figsize=(12,8))
    xs = [1,10000]
    import tqdm
    from schwimmbad import MPIPool
    #print('schwimmbad in')
    with MPIPool() as pool:
        #print('mpipool pooled')    
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        
        
        '''
    plt.vlines(x = [1/freqs[ihumax], 1/freqs[ihnmax], 1/freqs[ihsmax], 1/freqs[ihjmax],1/freqs[hmax],1/8.018234063349899e-05], ymin = 0, ymax = max(xs),
           colors = 'purple',
           label = 'vline_multiple - full height')
plt.scatter(1/freqs[1:],pYh[1:],label='particle',c='k')
plt.scatter(1/freqs[1:],pYhj[1:],label='particle',c='b')
plt.scatter(1/freqs[1:],pYhs[1:],label='particle',c='r')
plt.scatter(1/freqs[1:],pYhu[1:],label='particle',c='y')
plt.scatter(1/freqs[1:],pYhn[1:],label='particle',c='g')
plt.xscale('log')
plt.yscale('log')
        '''

        from scipy.optimize import minimize, differential_evolution, brute
        from scipy import stats
#x = np.linspace(-10, 30, 100)

        print(len(pYhj),ihjmax)
        print(freqs[hmax],freqs[kmax],freqs[pmax],freqs[qmax])
        newfreq = 8.01e-5
        best = -100
    
        str1 = "h"
        str2 = "k"
        str3 = "p"
        str4 = "q"
    
        nwalkers = 80
        nburnin = 800
        nsteps = 1000
        run = [True,True,True,True]
        mean1 = np.array([0,0,0,0,1/freqs[hmax],0,0,0])
        stdev1 = np.array([0.1,0.1,0.1,0.1,100,0.1,np.pi,0.1])
        ndim = len(mean1)
        
        for i in range(ndim):
            if i == 0:
                dist_arr = np.random.normal(mean1[i],stdev1[i],nwalkers)
            else:
                dist_arr = np.vstack((dist_arr,np.random.normal(mean1[i],stdev1[i],nwalkers)))
        p0 = np.transpose(dist_arr)
#===================================================H vector==============================================================
        if run[0]:
            ch_filename = path + '/chain_h.h5'
            sampler_h = 0
            if os.path.exists(ch_filename):
                sampler_h = emcee.backends.HDFBackend(path + '/chain_h.h5')
            else:
                backend = emcee.backends.HDFBackend(ch_filename)
                backend.reset(nwalkers, ndim)
                moveset = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),]
                moveset = [(emcee.moves.StretchMove(), 1.0),]
                        
                sampler_h = emcee.EnsembleSampler(nwalkers, ndim, MLE_Norm, backend=backend, pool=pool, args = (str1) ,moves = moveset)
            
            #state = sampler_h.run_mcmc(p0, nburnin, progress = True, store = True)
            state2 = sampler_h.run_mcmc(p0, nsteps, progress = True, store = True)

#===================================================K vector==============================================================
        if run[1]:
            ch_filename = path + '/chain_k.h5'
            
            sampler_k = 0
            if os.path.exists(ch_filename):
                sampler_k = emcee.backends.HDFBackend(path + '/chain_k.h5')
            else:
                backend = emcee.backends.HDFBackend(ch_filename)
                backend.reset(nwalkers, ndim)
                moveset = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),]
                moveset = [(emcee.moves.StretchMove(), 1.0),]
                        
                sampler_k = emcee.EnsembleSampler(nwalkers, ndim, MLE_Norm, backend=backend, pool=pool, args = (str2) ,moves = moveset)
            
            #state = sampler_k.run_mcmc(p0, nburnin, progress = True, store = True)
            state2 = sampler_k.run_mcmc(p0, nsteps, progress = True, store = True)

#===================================================P vector==============================================================
        if run[2]:
            ch_filename = path + '/chain_p.h5'
            sampler_p = 0
            if os.path.exists(ch_filename):
                sampler_p = emcee.backends.HDFBackend(path + '/chain_p.h5')
            else:
                backend = emcee.backends.HDFBackend(ch_filename)
                backend.reset(nwalkers, ndim)
                moveset = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),]
                moveset = [(emcee.moves.StretchMove(), 1.0),]
                        
                sampler_p = emcee.EnsembleSampler(nwalkers, ndim, MLE_Norm, backend=backend, pool=pool, args = (str3) ,moves = moveset)
            
            #state = sampler_p.run_mcmc(p0, nburnin, progress = True, store = True)
            state2 = sampler_p.run_mcmc(p0, nsteps, progress = True, store = True)

#===================================================Q vector==============================================================        
        if run[3]:
            ch_filename = path + '/chain_q.h5'
            sampler_q = 0
            if os.path.exists(ch_filename):
                sampler_q = emcee.backends.HDFBackend(path + '/chain_q.h5')
            else:
                backend = emcee.backends.HDFBackend(ch_filename)
                backend.reset(nwalkers, ndim)
                moveset = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),]
                moveset = [(emcee.moves.StretchMove(), 1.0),]
                        
                sampler_q = emcee.EnsembleSampler(nwalkers, ndim, MLE_Norm, backend=backend, pool=pool, args = (str4) ,moves = moveset)
            
            #state = sampler_q.run_mcmc(p0, nburnin, progress = True, store = True)
            state2 = sampler_q.run_mcmc(p0, nsteps, progress = True, store = True)

        import plot_fit


#mle_model_h = minimize(MLE_Norm, np.array([0,0,0,0,freqs[hmax]*0.99,0.1,0,0]), method='Powell', args=(str1), bounds=((-1,1),(-1,1),(-1,1),(-2,2),(freqs[hmax-10],freqs[hmax+10]),(-1,1),(-np.pi,np.pi),(-1,1)))
#mle_model_k = minimize(MLE_Norm, np.array([0, 0, 0, 0, freqs[kmax], 0.06, 0, 0]), method='Powell',args=(str2),  bounds=((-1,1),(-1,1),(-1,1),(-2,2),(freqs[kmax-10],freqs[kmax+10]),(-1,1),(-np.pi,np.pi),(-1,1)))
#mle_model_p = minimize(MLE_Norm, np.array([0.1, 0.1, 0.1, 1, freqs[pmax], 0.06, 0.2, 0]), method='Powell', args=(str3),  bounds=((-1,1),(-1,1),(-2,2),(-2,2),(freqs[pmax-10],freqs[pmax+10]),(-1,1),(-np.pi,np.pi),(-1,1)))
#mle_model_q = minimize(MLE_Norm, np.array([0,0,0,-1, freqs[qmax], -0.01, -0.01, 0]), method='Powell',args=(str4),  bounds=((-1,1),(-1,1),(-1,1),(-2,2),(freqs[qmax-10],freqs[pmax+10]),(-1,1),(-np.pi,np.pi),(-1,1)))

#mle_model_h = minimize(MLE_Norm, np.array([0,0,0,0,freqs[hmax],0.06,0,0]), method='Powell', args=(str1))
#mle_model_k = minimize(MLE_Norm, np.array([0, 0, 0, 0, freqs[kmax], 0.06, 0, 0]), method='Powell',args=(str2))
#mle_model_p = minimize(MLE_Norm, np.array([0.1, 0.1, 0.1, 0.1, freqs[pmax], 0.06, 0.2, 0]), method='Powell', args=(str3))
#mle_model_q = minimize(MLE_Norm, np.array([7.34045094e-01, 2.88185257e-01, -4.57326319e-02, -8.05082929e-01, freqs[qmax]/0.99, -0.04, -0.04, 0]), method='Powell',args=(str4))


'''
c1_h = mle_model_h.x[0]
c2_h = mle_model_h.x[1]
c3_h = mle_model_h.x[2]
c4_h = mle_model_h.x[3]
freq_h = mle_model_h.x[4]
const_h = mle_model_h.x[5]
phase_h = mle_model_h.x[6]
offset_h = mle_model_h.x[7]
guess_h = const_h*np.sin(2*np.pi*freq_h*t+phase_h)
sec_h = c1_h*hj+c2_h*hs+c3_h*hu+c4_h*hn
pred_h = sec_h + guess_h + offset_h

h_rem = h - sec_h - offset_h

Omega_n = np.arctan2(h,k)
inc_n = np.arcsin(h/np.sin(Omega_n))

Omega_new = np.arctan2(h_rem,k_rem)
inc_new = np.arcsin(h_rem/np.sin(Omega_new))

Omega_new2 = np.arctan2(guess_h,guess_k)
inc_new2 = np.arcsin(guess_h/np.sin(Omega_new2))
print(guess_h,np.sin(Omega_new2),inc_new2)

#q = e*np.cos(Omega+omega)

plt.plot(inc_n,'b')
plt.show()
plt.plot(inc_new,c='g')
plt.show()

plt.plot(ecc_n,'b')
plt.show()
plt.plot(ecc_new,c='g')
plt.show()

print('Inc')
print('Calc:',np.mean(inc_new))
print('Guess Calc:',np.mean(inc_new2))
print('AstDys:',np.arcsin(0.0891079))
#print('AstDys:',np.arcsin(0.0578858))
print( 'Unfiltered:',np.mean(inc))
print()


print('Ecc')
print('Calc:',np.mean(ecc_new))
print('Guess Calc:',np.mean(ecc_new2))
print('AstDys', 0.0112290)
#print('AstDys', 0.0344540)
print('Unfiltered:', np.mean(e))
'''
