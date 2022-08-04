import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import h5py
import corner

def plot_data():
    objname = 'Ceres'
    burnin = 800
    
    
    path = 'Asteroids/' + objname
    series = pd.read_csv(path + '/series.csv')
    t = series['t'].values
    e = series['e'].values
    inc = series['inc'].values
    h = series['h'].values
    k = series['k'].values
    p = series['p'].values
    q = series['q'].values
    
    hmc = series['hmc'].values
    hv = series['hv'].values
    he = series['he'].values
    hmr = series['hmr'].values
    hj = series['hj'].values
    hs = series['hs'].values
    hu = series['hu'].values
    hn = series['hn'].values
    
    kmc = series['kmc'].values
    kv = series['kv'].values
    ke = series['ke'].values
    kmr = series['kmr'].values
    kj = series['kj'].values
    ks = series['ks'].values
    ku = series['ku'].values
    kn = series['kn'].values
    
    pmc = series['pmc'].values
    pv = series['pv'].values
    pe = series['pe'].values
    pmr = series['pmr'].values
    pj = series['pj'].values
    ps = series['ps'].values
    pu = series['pu'].values
    pn = series['pn'].values
    
    qmc = series['qmc'].values
    qv = series['qv'].values
    qe = series['qe'].values
    qmr = series['qmr'].values
    qj = series['qj'].values
    qs = series['qs'].values
    qu = series['qu'].values
    qn = series['qn'].values
    
    sampler_h = emcee.backends.HDFBackend(path + '/chain_h.h5')
    sampler_k = emcee.backends.HDFBackend(path + '/chain_k.h5')
    sampler_p = emcee.backends.HDFBackend(path + '/chain_p.h5')
    sampler_q = emcee.backends.HDFBackend(path + '/chain_q.h5')
    
    flatchain_h = sampler_h.get_chain(flat = True)
    flatchain_k = sampler_k.get_chain(flat = True)
    flatchain_p = sampler_p.get_chain(flat = True)
    flatchain_q = sampler_q.get_chain(flat = True)
    
    llhoods_h = sampler_h.get_log_prob(flat = True)
    llhoods_k = sampler_k.get_log_prob(flat = True)
    llhoods_p = sampler_p.get_log_prob(flat = True)
    llhoods_q = sampler_q.get_log_prob(flat = True)
    
    ind_h = np.argmax(llhoods_h)
    ind_k = np.argmax(llhoods_k)
    ind_p = np.argmax(llhoods_p)
    ind_q = np.argmax(llhoods_q)
    
    params_h = flatchain_h[ind_h,:].flatten()
    params_k = flatchain_k[ind_k,:].flatten()
    params_p = flatchain_p[ind_p,:].flatten()
    params_q = flatchain_q[ind_q,:].flatten()
    
    print(params_h)
    
    c1_h = params_h[0]
    c2_h = params_h[1]
    c3_h = params_h[2]
    c4_h = params_h[3]
    c5_h = params_h[4]
    c6_h = params_h[5]
    c7_h = params_h[6]
    c8_h = params_h[7]
    freq_h = params_h[8]
    const_h = params_h[9]
    phase_h = params_h[10]
    offset_h = params_h[11]
    guess_h = const_h*np.sin(2*np.pi/freq_h*t+phase_h)
    sec_h = c1_h*hmc+c2_h*hv+c3_h*he+c4_h*hmr+c5_h*hj+c6_h*hs+c7_h*hu+c8_h*hn
    pred_h = sec_h + guess_h + offset_h
    
    h_rem = h - sec_h - offset_h
    
    
    c1_k = params_k[0]
    c2_k = params_k[1]
    c3_k = params_k[2]
    c4_k = params_k[3]
    c5_k = params_k[4]
    c6_k = params_k[5]
    c7_k = params_k[6]
    c8_k = params_k[7]
    freq_k = params_k[8]
    const_k = params_k[9]
    phase_k = params_k[10]
    offset_k = params_k[11]
    guess_k = const_k*np.sin(2*np.pi/freq_k*t+phase_k)
    sec_k = c1_k*kmc+c2_k*kv+c3_k*ke+c4_k*kmr+c5_k*kj+c6_k*ks+c7_k*ku+c8_k*kn
    pred_k = sec_k + guess_k + offset_k
    
    k_rem = k - sec_k - offset_k
    
    c1_p = params_p[0]
    c2_p = params_p[1]
    c3_p = params_p[2]
    c4_p = params_p[3]
    c5_p = params_p[4]
    c6_p = params_p[5]
    c7_p = params_p[6]
    c8_p = params_p[7]
    freq_p = params_p[8]
    const_p = params_p[9]
    phase_p = params_p[10]
    offset_p = params_p[11]
    '''
    c1_p = 0
    c2_p = 0
    c3_p = 0
    c4_p = 0
    freq_p = freqs[pmax]/0.98
    const_p = -0.12
    phase_p = 0
    offset_p = 0
    '''
    guess_p = const_p*np.sin(2*np.pi/freq_p*t+phase_p)
    sec_p = c1_p*pmc+c2_p*pv+c3_p*pe+c4_p*pmr+c5_p*pj+c6_p*ps+c7_p*pu+c8_p*pn
    pred_p = sec_p + guess_p + offset_p
    
    p_rem = p - sec_p - offset_p
    
    c1_q = params_q[0]
    c2_q = params_q[1]
    c3_q = params_q[2]
    c4_q = params_q[3]
    c5_q = params_q[4]
    c6_q = params_q[5]
    c7_q = params_q[6]
    c8_q = params_q[7]
    freq_q = params_q[8]
    const_q = params_q[9]
    phase_q = params_q[10]
    offset_q = params_q[11]
    guess_q = const_q*np.sin(2*np.pi/freq_q*t+phase_q)
    sec_q = c1_q*qmc+c2_q*qv+c3_q*qe+c4_q*qmr+c5_q*qj+c6_q*qs+c7_q*qu+c8_q*qn
    pred_q = sec_q + guess_q + offset_q
    
    q_rem = q - sec_q - offset_q
    plt.figure()
    plt.plot(h)
    plt.plot(pred_h)
    plt.savefig(path+ '/model_h.png')
    chisq = (h-pred_h)**2
    llh = -0.5*np.sum(chisq)
    print('Log-likelihood:',llh)

    plt.figure()
    plt.plot(k)
    plt.plot(pred_k)
    plt.savefig(path+ '/model_k.png')
    chisq = (k-pred_k)**2
    llh = -0.5*np.sum(chisq)
    print('Log-likelihood:',llh)
    
    plt.figure()
    plt.plot(p)
    plt.plot(pred_p)
    plt.savefig(path+ '/model_p.png')
    chisq = (p-pred_p)**2
    llh = -0.5*np.sum(chisq)
    print('Log-likelihood:',llh)
    
    plt.figure()
    plt.plot(q)
    plt.plot(pred_q)
    plt.savefig(path+ '/model_q.png')
    chisq = (q-pred_q)**2
    llh = -0.5*np.sum(chisq)
    print('Log-likelihood:',llh)
    
    
    #p = np.sin(o.inc)*np.sin(o.Omega)
    #h = (o.e)*np.sin(o.Omega+o.omega)
    
    
    Omega_n = np.arctan2(p,q)
    inc_n = np.arcsin(p/np.sin(Omega_n))
    
    Omega_new = np.arctan2(p_rem,q_rem)
    inc_new = np.arcsin(p_rem/np.sin(Omega_new))
    
    Omega_new2 = np.arctan2(guess_p,guess_q)
    inc_new2 = np.arcsin(guess_p/np.sin(Omega_new2))
    print(guess_p,np.sin(Omega_new2),inc_new2)
    
    #q = e*np.cos(Omega+omega)
    
    pomega_n = np.arctan2(h,k)
    omega_n = pomega_n - Omega_n
    ecc_n = np.arcsin(h/np.sin(pomega_n))
    
    pomega_new = np.arctan2(h_rem,k_rem)
    omega_new = pomega_new - Omega_new
    ecc_new = np.arcsin(h_rem/np.sin(pomega_new))
    
    pomega_new2 = np.arctan2(guess_h,guess_k)
    omega_new2 = pomega_new2 - Omega_new2
    ecc_new2 = np.arcsin(guess_h/np.sin(pomega_new2))
    
    fig = corner.corner(flatchain_h, bins = 40, show_titles = True, 
    plot_datapoints = False, color = "blue", fill_contours = True,
    title_fmt = ".3f", truths = params_h, label_kwargs=dict(fontsize=20))
    fname = path+"/corner_h.pdf"       
    fig.savefig(fname, format = 'pdf')
    plt.close("all")
    
    fig = corner.corner(flatchain_k, bins = 40, show_titles = True, 
    plot_datapoints = False, color = "blue", fill_contours = True,
    title_fmt = ".3f", truths = params_h, label_kwargs=dict(fontsize=20))
    fname = path+"/corner_k.pdf"       
    fig.savefig(fname, format = 'pdf')
    plt.close("all")
    
    fig = corner.corner(flatchain_p, bins = 40, show_titles = True, 
    plot_datapoints = False, color = "blue", fill_contours = True,
    title_fmt = ".3f", truths = params_h, label_kwargs=dict(fontsize=20))
    fname = path+"/corner_p.pdf"       
    fig.savefig(fname, format = 'pdf')
    plt.close("all")
    
    fig = corner.corner(flatchain_q, bins = 40, show_titles = True, 
    plot_datapoints = False, color = "blue", fill_contours = True,
    title_fmt = ".3f", truths = params_q, label_kwargs=dict(fontsize=20))
    fname = path+"/corner_q.pdf"       
    fig.savefig(fname, format = 'pdf')
    plt.close("all")

    print('Inc')
    print('Calc:',np.mean(inc_new))
    #print('Guess Calc:',np.mean(inc_new2))
    #print('AstDys:',np.arcsin(0.0891079))
    print('AstDys:',np.arcsin(0.1675846))
    print( 'Unfiltered:',np.mean(inc))
    print()
    
    
    print('Ecc')
    print('Calc:',np.mean(ecc_new))
    #print('Guess Calc:',np.mean(ecc_new2))
    #print('AstDys', 0.0112290)
    print('AstDys', 0.1161977)
    print('Unfiltered:', np.mean(e))

    
plot_data()