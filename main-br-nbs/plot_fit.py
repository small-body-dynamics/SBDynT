import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import h5py
import corner

def plot_data(objfile, objname):
    #objname = '2004PY107'
    burnin = 800
    
    path = 'TNOs/' + objfile
    series = pd.read_csv(path + '/series.csv')
    t = series['t'].values
    e = series['e'].values
    inc = series['inc'].values
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
    print(params_k)
    print(params_p)
    print(params_q)
    
    c1_h = params_h[0]
    c2_h = params_h[1]
    c3_h = params_h[2]
    c4_h = params_h[3]
    freq_h = params_h[4]
    const_h = params_h[5]
    phase_h = params_h[6]
    offset_h = params_h[7]
    guess_h = const_h*np.sin(2*np.pi/freq_h*t+phase_h)
    sec_h = c1_h*hj+c2_h*hs+c3_h*hu+c4_h*hn
    pred_h = sec_h + guess_h + offset_h
    
    h_rem = h - sec_h - offset_h
    
    
    c1_k = params_k[0]
    c2_k = params_k[1]
    c3_k = params_k[2]
    c4_k = params_k[3]
    freq_k = params_k[4]
    const_k = params_k[5]
    phase_k = params_k[6]
    offset_k = params_k[7]
    guess_k = const_k*np.sin(2*np.pi/freq_k*t+phase_k)
    sec_k = c1_k*kj+c2_k*ks+c3_k*ku+c4_k*kn
    pred_k = sec_k + guess_k + offset_k
    
    k_rem = k - sec_k - offset_k
    
    c1_p = params_p[0]
    c2_p = params_p[1]
    c3_p = params_p[2]
    c4_p = params_p[3]
    freq_p = params_p[4]
    const_p = params_p[5]
    phase_p = params_p[6]
    offset_p = params_p[7]
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
    sec_p = c1_p*pj+c2_p*ps+c3_p*pu+c4_p*pn
    pred_p = sec_p + guess_p + offset_p
    
    p_rem = p - sec_p - offset_p
    
    c1_q = params_q[0]
    c2_q = params_q[1]
    c3_q = params_q[2]
    c4_q = params_q[3]
    freq_q = params_q[4]
    const_q = params_q[5]
    phase_q = params_q[6]
    offset_q = params_q[7]
    guess_q = const_q*np.sin(2*np.pi/freq_q*t+phase_q)
    sec_q = c1_q*qj+c2_q*qs+c3_q*qu+c4_q*qn 
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
    
    
    plt.figure()
    plt.plot(inc_n,'b')
    plt.plot(inc_new,c='g')
    fname = path+"/inc_new_mcmc.pdf"       
    plt.savefig(fname, format = 'pdf')
    plt.close("all")
    
    plt.figure()
    plt.plot(h,'b')
    plt.plot(h_rem,c='g')
    fname = path+"/h_rem_mcmc.pdf"       
    plt.savefig(fname, format = 'pdf')
    plt.close("all")
    
    plt.figure()
    plt.plot(k,'b')
    plt.plot(k_rem,c='g')
    fname = path+"/k_rem_mcmc.pdf"       
    plt.savefig(fname, format = 'pdf')
    plt.close("all")
    
    plt.figure()
    plt.plot(ecc_n,'b')
    plt.plot(ecc_new,c='g')
    fname = path+"/ecc_new_mcmc.pdf"       
    plt.savefig(fname, format = 'pdf')
    plt.close("all")
    
    plt.figure()
    plt.axes().set_aspect('equal')
    plt.scatter(h,k)
    plt.scatter(h_rem,k_rem, s=0.01)
    plt.xlim([-0.12,0.12]);
    plt.ylim([-0.12,0.12]);
    
    fname = path+"/hk_mcmc.png"       
    plt.savefig(fname, format = 'png')
    plt.close("all")
    
    plt.figure()
    plt.axes().set_aspect('equal')
    plt.scatter(p,q)
    plt.scatter(p_rem,q_rem, s=0.01)
    plt.xlim([-0.12,0.12]);
    plt.ylim([-0.12,0.12]);
    
    fname = path+"/pq_mcmc.png"       
    plt.savefig(fname, format = 'png')
    plt.close("all")
    
    
    plt.figure()
    plt.plot(p,'b')
    plt.plot(p_rem,c='g')
    fname = path+"/p_rem_mcmc.pdf"       
    plt.savefig(fname, format = 'pdf')
    plt.close("all")
    
    plt.figure()
    plt.plot(q,'b')
    plt.plot(q_rem,c='g')
    fname = path+"/q_rem_mcmc.pdf"       
    plt.savefig(fname, format = 'pdf')
    plt.close("all")
    

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
    if objname == '2004PY107':
        print('AstDys:', np.arcsin(0.0112290))
    #print('AstDys:',np.arcsin(0.0891079))
    if objname == '2004KF19':
        print('AstDys:',np.arcsin(0.0344540))
    print( 'Unfiltered:',np.mean(inc))
    print()

    
    print('Ecc')
    print('Calc:',np.mean(ecc_new))

    if objname == '2004PY107':
        print('AstDys:', 0.0891079)
    if objname == '2004KF19':
        print('AstDys:', 0.0578858)

    print('Unfiltered:', np.mean(e))


objfile = '2004PY107_2'
objname = '2004PY107'
plot_data(objfile, objname)
