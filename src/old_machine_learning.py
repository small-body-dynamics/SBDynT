import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '../src')
import tools
import run_reb

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from datetime import date



def calc_ML_features(time,a,ec,inc,node,argperi,pomega,q,rh,phirf,tn):
    """
    calculate data features from a time-series
    """

    ########################################################
    ########################################################
    #
    # Very basic time-series data features
    #
    ########################################################
    ########################################################
    
    #calculate mean motions
    mm = 2*np.pi/np.power(a,1.5)


    mm_min=np.amin(mm)
    a_min=np.amin(a)
    e_min=np.amin(ec)
    i_min=np.amin(inc)
    q_min=np.amin(q)
    tn_min=np.amin(tn)

    mm_max=np.amax(mm)
    a_max=np.amax(a)
    e_max=np.amax(ec)
    i_max=np.amax(inc)
    q_max=np.amax(q)
    tn_max=np.amax(tn)


    mm_del = mm_max - mm_min
    a_del = a_max - a_min
    e_del = e_max - e_min
    i_del = i_max - i_min
    q_del = q_max - q_min
    tn_del = tn_max - tn_min


    mm_mean = np.mean(mm)
    a_mean = np.mean(a)
    e_mean = np.mean(ec)
    i_mean = np.mean(inc)
    q_mean = np.mean(q)
    tn_mean = np.mean(tn)

    mm_std = np.std(mm)
    a_std = np.std(a)
    e_std = np.std(ec)
    i_std = np.std(inc)
    q_std = np.std(q)
    tn_std = np.std(tn)

    mm_std_norm = mm_std/mm_mean
    a_std_norm = a_std/a_mean
    e_std_norm = e_std/e_mean
    i_std_norm = i_std/i_mean
    q_std_norm = q_std/q_mean

    mm_del_norm = mm_del/mm_mean
    a_del_norm = a_del/a_mean
    e_del_norm = e_del/e_mean
    i_del_norm = i_del/i_mean
    q_del_norm = q_del/q_mean



    #arg peri 0-2pi

    argperi = tools.arraymod2pi(argperi)
    node = tools.arraymod2pi(node)
    pomega = tools.arraymod2pi(pomega)

    argperi_min = np.amin(argperi)
    argperi_max = np.amax(argperi)
    argperi_del = argperi_max - argperi_min
    argperi_mean = np.mean(argperi)
    argperi_std = np.std(argperi)

    #recenter arg peri around 0 and repeat
    argperi_zero = tools.arraymod2pi0(argperi)
    argperi_min2 = np.amin(argperi_zero)
    argperi_max2 = np.amax(argperi_zero)
    #typo here before
    argperi2_del = argperi_max2 - argperi_min2
    argperi_mean2 = np.mean(argperi_zero)
    argperi_std2 = np.std(argperi_zero)

    #take the better values for delta, mean, and standard deviation:
    if(argperi2_del < argperi_del):
        argperi_del = argperi2_del
        argperi_mean = tools.mod2pi(argperi_mean2)
        argperi_std = argperi_std2

    #calculate time derivatives
    dt = time[1:] - time[:-1] 

    dmm_dt = mm[1:] - mm[:-1]
    dmm_dt = dmm_dt/dt
    da_dt = a[1:] - a[:-1]
    da_dt = da_dt/dt
    de_dt = ec[1:] - ec[:-1]
    de_dt = de_dt/dt
    di_dt = inc[1:] - inc[:-1]
    #was a typo before!
    di_dt = di_dt/dt
    dq_dt = q[1:] - q[:-1]
    dq_dt = dq_dt/dt
    dtn_dt = tn[1:] - tn[:-1]
    dtn_dt = dtn_dt/dt

    #unwrap the angles first to be sure we get proper differences
    temp = np.unwrap(argperi)
    dargperi_dt = temp[1:] - temp[:-1]
    dargperi_dt = dargperi_dt/dt

    temp = np.unwrap(node)
    dnode_dt = temp[1:] - temp[:-1]
    dnode_dt = dnode_dt/dt

    temp = np.unwrap(pomega)
    dpomega_dt = temp[1:] - temp[:-1]
    dpomega_dt = dpomega_dt/dt

    
    mmdot_min = np.amin(dmm_dt)
    mmdot_max = np.amax(dmm_dt)
    mmdot_mean = np.mean(dmm_dt)
    mmdot_std = np.std(dmm_dt)
    mmdot_std_norm = mmdot_std/mmdot_mean
    mmdot_del = mmdot_max - mmdot_min
    mmdot_del_norm = mmdot_del/mmdot_mean
    
    adot_min = np.amin(da_dt)
    adot_max = np.amax(da_dt)
    adot_mean = np.mean(da_dt)
    adot_std = np.std(da_dt)
    adot_std_norm = adot_std/adot_mean
    adot_del = adot_max - adot_min
    adot_del_norm = adot_del/adot_mean

    edot_min = np.amin(de_dt)
    edot_max = np.amax(de_dt)
    edot_mean = np.mean(de_dt)
    edot_std = np.std(de_dt)
    edot_std_norm = edot_std/edot_mean
    edot_del = edot_max - edot_min
    edot_del_norm = edot_del/edot_mean


    idot_min = np.amin(di_dt)
    idot_max = np.amax(di_dt)
    idot_mean = np.mean(di_dt)
    idot_std = np.std(di_dt)
    idot_std_norm = idot_std/idot_mean
    idot_del = idot_max - idot_min
    idot_del_norm = idot_del/idot_mean


    nodedot_min = np.amin(dnode_dt)
    nodedot_max = np.amax(dnode_dt)
    nodedot_mean = np.mean(dnode_dt)
    nodedot_std = np.std(dnode_dt)
    nodedot_std_norm = nodedot_std/nodedot_mean
    nodedot_del = nodedot_max - nodedot_min
    nodedot_del_norm = nodedot_del/nodedot_mean


    argperidot_min = np.amin(dargperi_dt)
    argperidot_max = np.amax(dargperi_dt)
    argperidot_mean = np.mean(dargperi_dt)
    argperidot_std = np.std(dargperi_dt)
    argperidot_std_norm = argperidot_std/argperidot_mean
    argperidot_del = argperidot_max - argperidot_min
    argperidot_del_norm = argperidot_del/argperidot_mean


    pomegadot_min = np.amin(dpomega_dt)
    pomegadot_max = np.amax(dpomega_dt)
    pomegadot_mean = np.mean(dpomega_dt)
    pomegadot_std = np.std(dpomega_dt)
    pomegadot_std_norm = pomegadot_std/pomegadot_mean
    pomegadot_del = pomegadot_max - pomegadot_min
    pomegadot_del_norm = pomegadot_del/pomegadot_mean

    qdot_min = np.amin(dq_dt)
    qdot_max = np.amax(dq_dt)
    qdot_mean = np.mean(dq_dt)
    qdot_std = np.std(dq_dt)
    qdot_std_norm = qdot_std/qdot_mean
    qdot_del = qdot_max - qdot_min
    qdot_del_norm = qdot_del/qdot_mean

    tndot_min = np.amin(dtn_dt)
    tndot_max = np.amax(dtn_dt)
    tndot_mean = np.mean(dtn_dt)
    tndot_std = np.std(dtn_dt)
    tndot_std_norm = tndot_std/tndot_mean
    tndot_del = tndot_max - tndot_min
    tndot_del_norm = tndot_del/tndot_mean


    ########################################################
    ########################################################
    #
    # Rotating Frame data features
    #
    ########################################################
    ########################################################

    phirf = tools.arraymod2pi(phirf)
    
    #divide heliocentric distance into 10 bins and theta_n
    #into 20 bins
    qmin = np.amin(rh) - 0.01
    Qmax = np.amax(rh) + 0.01
    nrbin = 10.
    nphbin = 20.
    dr = (Qmax - qmin) / nrbin
    dph = (2. * np.pi) / nphbin
    # center on the planet in phi (so when we bin, we will
    # add the max and min bins together since they're really
    # half-bins
    phmin = -dph / 2.

    # radial plus aziumthal binning
    # indexing is radial bin, phi bin: rph_count[rbin,phibin]
    rph_count = np.zeros((int(nrbin), int(nphbin)))
    # radial only binning
    r_count = np.zeros(int(nrbin))

    # for calculating the average sin(ph) and cos(ph)
    # indexing is   sinphbar[rbin,resorder]
    resorder_max = 10
    sinphbar = np.zeros((int(nrbin), resorder_max+1))
    cosphbar = np.zeros((int(nrbin), resorder_max+1))

    # divide into radial and azimuthal bins
    nmax = len(rh)
    for n in range(0, nmax):
        rbin = int(np.floor((rh[n] - qmin) / dr))
        for resorder in range(1,resorder_max+1):
            tcos = np.cos(float(resorder)*phirf[n])
            tsin = np.sin(float(resorder)*phirf[n])
            sinphbar[rbin,resorder]+=tsin
            cosphbar[rbin,resorder]+=tcos
        r_count[rbin]+=1.
        phbin = int(np.floor((phirf[n] - phmin) / dph))
        if (phbin == int(nphbin)):
            phbin = 0
        rph_count[rbin, phbin] += 1

    # perihelion distance bin stats
    nempty = np.zeros(int(nrbin))
    nadjempty = np.zeros(int(nrbin))
    rbinmin = np.zeros(int(nrbin))
    rbinmax = np.zeros(int(nrbin))
    rbinavg = np.zeros(int(nrbin))
    rbinstd = np.zeros(int(nrbin))

    for nr in range(0, int(nrbin)):
        rbinmin[nr] = 1e9
        for resorder in range(1,resorder_max+1):
            sinphbar[nr,resorder] = sinphbar[nr,resorder]/r_count[nr]
            cosphbar[nr,resorder] = cosphbar[nr,resorder]/r_count[nr]
        for n in range(0, int(nphbin)):
            if (rph_count[nr, n] == 0):
                nempty[nr] += 1
            if (rph_count[nr, n] < rbinmin[nr]):
                rbinmin[nr] = rph_count[nr, n]
            if (rph_count[nr, n] > rbinmax[nr]):
                rbinmax[nr] = rph_count[nr, n]
            rbinavg[nr] += rph_count[nr, n]
        rbinavg[nr] = rbinavg[nr] / nphbin

        for n in range(0, int(nphbin)):
            rbinstd[nr] += (rph_count[nr, n] - rbinavg[nr]) * (
                        rph_count[nr, n] - rbinavg[nr])
        if (not (rbinavg[nr] == 0)):
            rbinstd[nr] = np.sqrt(rbinstd[nr] / nphbin) #/ rbinavg[nr]
        else:
            rbinstd[nr] = 0.

        if (rph_count[nr, 0] == 0):
            nadjempty[nr] = 1
            for n in range(1, int(np.floor(nphbin / 2.)) + 1):
                if (rph_count[nr, n] == 0):
                    nadjempty[nr] += 1
                if (rph_count[nr, n] != 0):
                    break
            for n in range(int(nphbin) - 1, int(np.floor(nphbin / 2.)), -1):
                if (rph_count[nr, n] == 0):
                    nadjempty[nr] += 1
                if (rph_count[nr, n] != 0):
                    break


    n_peri_empty = nempty[0]
    n_apo_empty = nempty[-1]
    nadj_peri_empty = nadjempty[-1]
    nadj_apo_empty = nadjempty[-1]

    navg_peri = rbinavg[0]
    nstd_peri = rbinstd[0]
    ndel_peri = rbinmax[0] - rbinmin[0]
    if(navg_peri>0):
        ndel_peri_norm = ndel_peri/navg_peri
        nstd_peri_norm = nstd_peri/navg_peri
    else:
        ndel_peri_norm = 0.
        nstd_peri_norm = 0.

    navg_apo = rbinavg[-1]
    nstd_apo = rbinstd[-1]
    ndel_apo = rbinmax[-1] - rbinmin[-1]
    if(navg_apo>0):
        ndel_apo_norm = ndel_apo/navg_apo
        nstd_apo_norm = nstd_apo/navg_apo
    else:
        ndel_apo_norm = 0.
        nstd_apo_norm = 0.
    #    n = -2
 
    #add the rayleigh z-test statistics at perihelion and aphelion
    rz_peri = np.zeros(resorder_max+1)
    rz_apo = np.zeros(resorder_max+1)
    for resorder in range(1, resorder_max+1):
        rz_peri[resorder] = np.sqrt(sinphbar[0,resorder]*sinphbar[0,resorder] +
                       cosphbar[0,resorder]*cosphbar[0,resorder])
        rz_apo[resorder] = np.sqrt(sinphbar[-1,resorder]*sinphbar[-1,resorder] +
                       cosphbar[-1,resorder]*cosphbar[-1,resorder])


    rzperi_max = np.amax(rz_peri[1:resorder_max])
    rzapo_max = np.amax(rz_apo[1:resorder_max])


    spatial_counts = rph_count.flatten()
    grid_nz_minval = np.min(spatial_counts[np.nonzero(spatial_counts)])
    grid_nz_avg = np.mean(spatial_counts[np.nonzero(spatial_counts)])
    grid_nz_std = np.std(spatial_counts[np.nonzero(spatial_counts)])
    grid_avg =  np.mean(spatial_counts)
    grid_std =  np.std(spatial_counts)
    grid_deltaavg = grid_nz_avg - grid_avg
    grid_deltastd = grid_std - grid_nz_std
    
    n_empty=0
    n_almost_empty=0
    for n in range(0,len(spatial_counts)):
        if(spatial_counts[n]==0):
            n_empty += 1
        if(spatial_counts[n]<7):
            n_almost_empty += 1

    ########################################################
    ########################################################
    #
    # FFT data features
    #
    ########################################################
    ########################################################

    #calculate the correlations between a and e, a and i, and e and i
    aecorr =  max_corelation(a,ec)
    aicorr =  max_corelation(a,inc)
    eicorr =  max_corelation(ec,inc)

    #calculate spectral fractions
    deltat = time[2] - time[1]
    #a
    asf, amaxpower, amaxpower3, af1, af2, af3 = spectral_characteristics(a,deltat)
    # eccentricity, via e*sin(varpi)
    hec = ec*np.sin(pomega)
    esf, emaxpower, emaxpower3, ef1, ef2, ef3 = spectral_characteristics(hec,deltat)
    # inclination, via sin(i)sin(Omega)
    pinc = np.sin(inc)*np.sin(node)
    isf, imaxpower, imaxpower3, if1, if2, if3 = spectral_characteristics(pinc,deltat)
    #amd
    amd = 1. - np.sqrt(1.- ec*ec)*np.cos(inc)
    amd = amd*np.sqrt(a)
    amdsf, amdmaxpower, amdmaxpower3, amdf1, amdf2, amdf3 = spectral_characteristics(amd,deltat)


    ########################################################
    ########################################################
    #
    # additional time-series based features
    #
    ########################################################
    ########################################################

    #Do some binning in the a, e, and i-distributions
    #compare visit distributions


    em_a, lh_a, min_em_a, max_em_a, delta_em_a, delta_em_a_norm, min_lh_a, max_lh_a, delta_lh_a, delta_lh_a_norm =  histogram_features(a,a_min,a_max,a_mean,a_std)
    em_e, lh_e, min_em_e, max_em_e, delta_em_e, delta_em_e_norm, min_lh_e, max_lh_e, delta_lh_e, delta_lh_e_norm =  histogram_features(ec,e_min,e_max,e_mean,e_std)
    em_i, lh_i, min_em_i, max_em_i, delta_em_i, delta_em_i_norm, min_lh_i, max_lh_i, delta_lh_i, delta_lh_i_norm =  histogram_features(inc,i_min,i_max,i_mean,i_std)
    
    em_a2, lh_a2, min_em_a2, max_em_a2, delta_em_a2, delta_em_a_norm2, min_lh_a2, max_lh_a2, delta_lh_a2, delta_lh_a_norm2 =  alt_histogram_features(a,a_min,a_max,a_mean,a_std)
    em_e2, lh_e2, min_em_e2, max_em_e2, delta_em_e2, delta_em_e_norm2, min_lh_e2, max_lh_e2, delta_lh_e2, delta_lh_e_norm2 =  alt_histogram_features(ec,e_min,e_max,e_mean,e_std)
    em_i2, lh_i2, min_em_i2, max_em_i2, delta_em_i2, delta_em_i_norm2, min_lh_i2, max_lh_i2, delta_lh_i2, delta_lh_i_norm2 =  alt_histogram_features(inc,i_min,i_max,i_mean,i_std)

    da1 = (a_mean-a_min)/(a_max-a_mean)
    da2 = (a_max-a_mean)/(a_mean-a_min)
    da_symmetry = np.amax([da1,da2])
   

    features = [
        mm_min,mm_mean,mm_max,mm_std,mm_std_norm,mm_del,mm_del_norm,
        mmdot_min,mmdot_mean,mmdot_max,mmdot_std,mmdot_std_norm,mmdot_del,mmdot_del_norm,
        a_min,a_mean,a_max,a_std,a_std_norm,a_del,a_del_norm,
        adot_min,adot_mean,adot_max,adot_std,adot_std_norm,adot_del,adot_del_norm,
        e_min,e_mean,e_max,e_std,e_std_norm,e_del,e_del_norm,
        edot_min,edot_mean,edot_max,edot_std,edot_std_norm,edot_del,edot_del_norm,
        i_min,i_mean,i_max,i_std,i_std_norm,i_del,i_del_norm,
        idot_min,idot_mean,idot_max,idot_std,idot_std_norm,idot_del,idot_del_norm,
        nodedot_min,nodedot_mean,nodedot_max,nodedot_std,nodedot_std_norm,nodedot_del,nodedot_del_norm,
        argperi_min,argperi_mean,argperi_max,argperi_std,argperi_del,
        argperidot_min,argperidot_mean,argperidot_max,argperidot_std,argperidot_std_norm,argperidot_del,argperidot_del_norm,
        pomegadot_min,pomegadot_mean,pomegadot_max,pomegadot_std,pomegadot_std_norm,pomegadot_del,pomegadot_del_norm,
        q_min,q_mean,q_max,q_std,q_std_norm,q_del,q_del_norm,
        qdot_min,qdot_mean,qdot_max,qdot_std,qdot_std_norm,qdot_del,qdot_del_norm,
        tn_min,tn_mean,tn_max,tn_std,tn_del,
        tndot_min,tndot_mean,tndot_max,tndot_std,tndot_std_norm,tndot_del,tndot_del_norm,
        n_peri_empty,nadj_peri_empty,navg_peri,nstd_peri,nstd_peri_norm,ndel_peri,ndel_peri_norm,rzperi_max,
        rz_peri[1],rz_peri[2],rz_peri[3],rz_peri[4],rz_peri[5],rz_peri[6],rz_peri[7],rz_peri[8],rz_peri[9],rz_peri[10],
        n_apo_empty,nadj_apo_empty,navg_apo,nstd_apo,nstd_apo_norm,ndel_apo,ndel_apo_norm,rzapo_max,
        rz_apo[1],rz_apo[2],rz_apo[3],rz_apo[4],rz_apo[5],rz_apo[6],rz_apo[7],rz_apo[8],rz_apo[9],rz_apo[10],
        grid_nz_minval,grid_nz_avg,grid_nz_std,grid_avg,grid_std,grid_deltaavg,grid_deltastd,n_empty,n_almost_empty,
        aecorr,aicorr,eicorr, 
        asf,amaxpower,amaxpower3,af1,af2,af3,esf,emaxpower,emaxpower3,ef1,ef2,ef3,isf,imaxpower,imaxpower3,if1,if2,if3,
        amdsf,amdmaxpower,amdmaxpower3,amdf1,amdf2,amdf3,
        da_symmetry,
        em_a,lh_a,min_em_a,max_em_a,delta_em_a,delta_em_a_norm,min_lh_a,max_lh_a,delta_lh_a,delta_lh_a_norm,
        em_e,lh_e,em_i,lh_i,
        em_a2,lh_a2,min_em_a2,max_em_a2,delta_em_a2,delta_em_a_norm2,min_lh_a2,max_lh_a2,delta_lh_a2,delta_lh_a_norm2,
        em_e2,lh_e2,em_i2,lh_i2,
        ]
   
    return np.array(features)  # make sure features is a numpy array





def compute_ML_features_from_dbase_file(fname,kbo_id=1):
    '''
    Load data from file and calculate features
    fname=file name that contains simulation data; 
        MUST contain outputs only every 1000 years, only contain lines from expected object
        Will only take first lines lines from fname, which should be times [0,1E3,2E3,...,99E3,100E3]
    col_order=indexes for columns: time, a (semi-major axis), eccentriciy, 
        inclination, Omega (longitude of ascending node), omega (argument of pericenter)
    Returns features for ML classification
    '''

    #################################
    # define the id for Neptune
    ######################################
    pl_id = -5

    ########################################################
    # read in the data from the follow file
    ########################################################

    data_t = np.genfromtxt(fname,
                    names=['id', 't', 'a', 'e', 'inc', 'node', 'peri','MA'])

    data_pl = data_t[data_t['id'] == pl_id]
    data_sb = data_t[data_t['id'] == kbo_id]


    lines = len(data_sb)

    pomega_sb = np.zeros(lines)
    a_sb = np.zeros(lines)
    time = np.zeros(lines)
    e_sb = np.zeros(lines)
    q_sb = np.zeros(lines)
    i_sb = np.zeros(lines)
    t_sb = np.zeros(lines)
    tiss_sb = np.zeros(lines)
    node_sb = np.zeros(lines)
    peri_sb = np.zeros(lines)
    rrf = np.zeros(lines)
    phirf = np.zeros(lines)

    j = 0
    for i in range(0, lines):
        pomega_sb[i] = (data_sb['node'][j] + data_sb['peri'][j])*np.pi/180.
        time[i] = data_sb['t'][j]
        a_sb[i] = data_sb['a'][j]
        e_sb[i] = data_sb['e'][j]
        q_sb[i] = data_sb['a'][j]*(1.-data_sb['e'][j])
        i_sb[i] = data_sb['inc'][j]*np.pi/180.
        t_sb[i] = data_sb['t'][j]
        node_sb[i] = data_sb['node'][j]*np.pi/180.
        peri_sb[i] = data_sb['peri'][j]*np.pi/180.
        tiss_sb[i] = data_pl['a'][j]/data_sb['a'][j]
        tiss_sb[i] += 2.*np.cos(i_sb[i])*np.sqrt(data_sb['a'][j]/data_pl['a'][j]*(1.-e_sb[i]*e_sb[i]))

        [flag, x, y, z, vx, vy, vz] = tools.aei_to_xv(
            GM=1., a=data_sb['a'][j],e=data_sb['e'][j],
            inc=data_sb['inc'][j] * np.pi / 180.,
            node=data_sb['node'][j] * np.pi / 180.,
            argperi=data_sb['peri'][j] * np.pi / 180.,
            ma=data_sb['MA'][j] * np.pi / 180.)

        [xrf, yrf, zrf, vxrf, vyrf, vzrf] = tools.rotating_frame_cartesian(x=x, y=y, z=z,
            node=data_pl['node'][j] * np.pi / 180.,
            inc=data_pl['inc'][j] * np.pi / 180.,
            argperi=data_pl['peri'][j] * np.pi / 180.,
            ma=data_pl['MA'][j] * np.pi / 180.)

        rrf[i] = np.sqrt(xrf*xrf + yrf*yrf + zrf*zrf)
        phirf[i] = np.arctan2(yrf, xrf)

        j += 1

    peri_sb = tools.arraymod2pi(peri_sb)
    node_sb = tools.arraymod2pi(node_sb)
    pomega_sb = tools.arraymod2pi(pomega_sb)
    phirf = tools.arraymod2pi(phirf)

    data = np.array(t_sb)[:, np.newaxis]

    data = np.concatenate((data, a_sb[:, np.newaxis]), axis=1)
    data = np.concatenate((data, e_sb[:, np.newaxis]), axis=1)
    data = np.concatenate((data, i_sb[:, np.newaxis]), axis=1)
    data = np.concatenate((data, node_sb[:, np.newaxis]), axis=1)
    data = np.concatenate((data, peri_sb[:, np.newaxis]), axis=1)
    data = np.concatenate((data, pomega_sb[:, np.newaxis]), axis=1)
    data = np.concatenate((data, q_sb[:, np.newaxis]), axis=1)
    data = np.concatenate((data, rrf[:, np.newaxis]), axis=1)
    data = np.concatenate((data, phirf[:, np.newaxis]), axis=1)
    data = np.concatenate((data, tiss_sb[:, np.newaxis]), axis=1)

    # Compute features
    #features = ML_parse_features(data)
    # features = parse(data)

    features = calc_ML_features(time,a_sb,e_sb,i_sb,node_sb,peri_sb,pomega_sb,q_sb,rrf,phirf,tiss_sb)

    return features

def max_corelation(d1, d2):
    d1 = (d1 - np.mean(d1)) / (np.std(d1))
    d2 = (d2 - np.mean(d2)) / (np.std(d2))  
    cmax = (np.correlate(d1, d2, 'full')/len(d1)).max()
    return cmax


def spectral_characteristics(data,dt):
    Y = np.fft.rfft(data)
    n = len(data)
    freq = np.fft.rfftfreq(n,d=dt)
    jmax = len(Y)
    Y = Y[1:jmax]
    Y = np.abs(Y)**2.
    arr1 = Y.argsort()    
    sorted_Y = Y[arr1[::-1]]
    sorted_freq = freq[arr1[::-1]]
    f1 = sorted_freq[0]
    f2 = sorted_freq[1]
    f3 = sorted_freq[2]
    ytot = 0.
    for Y in (sorted_Y):
        ytot+=Y
    norm_Y = sorted_Y/ytot
    count=0
    maxnorm_Y = sorted_Y/sorted_Y[0]
    for j in range(0,jmax-1):
        if(maxnorm_Y[j] > 0.05):
            count+=1
    sf = 1.0*count/(jmax-1.)
    maxpower = sorted_Y[0]
    max3 = sorted_Y[0] + sorted_Y[1] + sorted_Y[2]
    return sf, maxpower, max3, f1, f2, f3

def histogram_features(x,xmin,xmax,xmean,xstd):
    x1 = xmin
    x2 = xmean-0.75*xstd
    x3 = xmean-0.375*xstd
    x4 = xmean+0.375*xstd
    x5 = xmean+0.75*xstd
    x6 = xmax
    if(x1 < x2 and x2 < x3 and x4 < x5 and x5 < x6):
        xbins = [x1,x2,x3,x4,x5,x6]
    else:
        dx = (xmax-xmin)/8.
        x2 = xmin + 2.*dx
        x3 = x2 + dx
        x4 = x3 + 2.*dx
        x5 = x4 + dx
        xbins = [x1,x2,x3,x4,x5,x6]
    xcounts, tbins = np.histogram(x,bins=xbins)

    #average ratio of extreme-x density to middle-x density
    if(xcounts[2] == 0):
        xcounts[2] = 1 #avoid a nan
    em_x = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
    #ratio of extreme-low-x density to extreme high-x density
    if(xcounts[4] == 0):
        xcounts[4] = 1
    lh_x = (xcounts[0]/xcounts[4])

    #repeat across a couple time bins 
    dj = x.size//4
    xcounts, tbins = np.histogram(x[0:dj],bins=xbins)
    if(xcounts[2] == 0):
        xcounts[2] = 1
    em1 = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
    if(xcounts[4] == 0):
        xcounts[4] = 1
    lh1 = (xcounts[0]/xcounts[4])
 
    xcounts, tbins = np.histogram(x[dj:2*dj],bins=xbins)
    if(xcounts[2] == 0):
        xcounts[2] = 1
    em2 = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
    if(xcounts[4] == 0):
        xcounts[4] = 1
    lh2 = (xcounts[0]/xcounts[4])
    
    xcounts, tbins = np.histogram(x[2*dj:3*dj],bins=xbins)
    if(xcounts[2] == 0):
        xcounts[2] = 1
    em3 = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
    if(xcounts[4] == 0):
        xcounts[4] = 1
    lh3 = (xcounts[0]/xcounts[4])


    xcounts, tbins = np.histogram(x[3*dj:4*dj],bins=xbins)
    if(xcounts[2] == 0):
        xcounts[2] = 1
    em4 = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
    if(xcounts[4] == 0):
        xcounts[4] = 1
    lh4 = (xcounts[0]/xcounts[4])

    min_em_x = min(em1,em2,em3,em4)
    max_em_x = max(em1,em2,em3,em4)

    delta_em_x = max_em_x - min_em_x
    delta_em_x_norm = delta_em_x/em_x    

    min_lh_x = min(lh1,lh2,lh3,lh4)
    max_lh_x = max(lh1,lh2,lh3,lh4)

    delta_lh_x = max_lh_x - min_lh_x
    delta_lh_x_norm = delta_lh_x/lh_x

    return  em_x, lh_x, min_em_x, max_em_x, delta_em_x, delta_em_x_norm, min_lh_x, max_lh_x, delta_lh_x, delta_lh_x_norm


def alt_histogram_features(x,xmin,xmax,xmean,xstd):
    dx = (xmax-xmin)/8.
    x1 = xmin
    x2 = xmin + 2.*dx
    x3 = x2 + dx
    x4 = x3 + 2.*dx
    x5 = x4 + dx
    x6 = xmax
    xbins = [x1,x2,x3,x4,x5,x6]
    xcounts, tbins = np.histogram(x,bins=xbins)

    #average ratio of extreme-x density to middle-x density
    if(xcounts[2] == 0):
        xcounts[2] = 1 #avoid a nan
    em_x = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
    #ratio of extreme-low-x density to extreme high-x density
    if(xcounts[4] == 0):
        xcounts[4] = 1
    lh_x = (xcounts[0]/xcounts[4])

    #repeat across a couple time bins 
    dj = x.size//4
    xcounts, tbins = np.histogram(x[0:dj],bins=xbins)
    if(xcounts[2] == 0):
        xcounts[2] = 1
    em1 = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
    if(xcounts[4] == 0):
        xcounts[4] = 1
    lh1 = (xcounts[0]/xcounts[4])
 
    xcounts, tbins = np.histogram(x[dj:2*dj],bins=xbins)
    if(xcounts[2] == 0):
        xcounts[2] = 1
    em2 = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
    if(xcounts[4] == 0):
        xcounts[4] = 1
    lh2 = (xcounts[0]/xcounts[4])
    
    xcounts, tbins = np.histogram(x[2*dj:3*dj],bins=xbins)
    if(xcounts[2] == 0):
        xcounts[2] = 1
    em3 = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
    if(xcounts[4] == 0):
        xcounts[4] = 1
    lh3 = (xcounts[0]/xcounts[4])


    xcounts, tbins = np.histogram(x[3*dj:4*dj],bins=xbins)
    if(xcounts[2] == 0):
        xcounts[2] = 1
    em4 = (xcounts[0] + xcounts[4])/(2.*xcounts[2])
    if(xcounts[4] == 0):
        xcounts[4] = 1
    lh4 = (xcounts[0]/xcounts[4])

    min_em_x = min(em1,em2,em3,em4)
    max_em_x = max(em1,em2,em3,em4)

    delta_em_x = max_em_x - min_em_x
    delta_em_x_norm = delta_em_x/em_x    

    min_lh_x = min(lh1,lh2,lh3,lh4)
    max_lh_x = max(lh1,lh2,lh3,lh4)

    delta_lh_x = max_lh_x - min_lh_x
    delta_lh_x_norm = delta_lh_x/lh_x

    return  em_x, lh_x, min_em_x, max_em_x, delta_em_x, delta_em_x_norm, min_lh_x, max_lh_x, delta_lh_x, delta_lh_x_norm


def read_TNO_training_data(training_file):
    all_TNOs = pd.read_csv(training_file, skipinitialspace=True, index_col=False, low_memory=False)
    all_TNOs = all_TNOs.drop(columns='junk')
    
    filtered1_TNOs = all_TNOs[all_TNOs['a_max']<1000.0].copy()
    filtered2_TNOs = filtered1_TNOs[filtered1_TNOs['a_delta_short']<30.0].copy()
    filtered_TNOs = filtered2_TNOs[filtered2_TNOs['a_delta']<100.0].copy()

    filtered_TNOs['l5'] = filtered_TNOs.apply(label5_particle, axis=1)
    filtered_TNOs['l6'] = filtered_TNOs.apply(label6_particle, axis=1)
    filtered_TNOs['l7'] = filtered_TNOs.apply(label_particle_7, axis=1)


    return filtered_TNOs


def train_and_test_TNO_classifier(training_file):
    dataset = read_TNO_training_data(training_file)
    perm_drop_features = ['r_o_a','des', 'particle', 'clone', 'cloneclass', 'l1', 'l2', 'l3', 'l4','l5','l6','l7',
                      'p', 'q', 'mio', 'res_char', 'm', 'n',
                      "avg_grid","avg_grid_short",
                      'i_delta_normed', 'i_delta_normed_short',
                      'e_delta_normed', 'e_delta_normed_short',
                      'e_sigma_normed', 'e_sigma_normed_short',
                      'i_sigma_normed','i_sigma_normed_short',
                      'adot_delta_normed', 'edot_delta_normed', 'idot_delta_normed', 'qdot_delta_normed',
                      'adot_std_normed', 'edot_std_normed', 'idot_std_normed', 'qdot_std_normed',
                      'adot_delta_normed_short', 'edot_delta_normed_short', 
                      'idot_delta_normed_short', 'qdot_delta_normed_short',
                      'adot_std_normed_short', 'edot_std_normed_short',
                      'idot_std_normed_short', 'qdot_std_normed_short',               
                      "rzperi_1","rzperi_2","rzperi_3","rzperi_4","rzperi_5",
                      "rzperi_6","rzperi_7","rzperi_8","rzperi_9","rzperi_10",
                      "rzperi_1_short","rzperi_2_short","rzperi_3_short","rzperi_4_short","rzperi_5_short",
                      "rzperi_6_short","rzperi_7_short","rzperi_8_short","rzperi_9_short","rzperi_10_short",
                      "rzapo_1","rzapo_2","rzapo_3","rzapo_4","rzapo_5",
                      "rzapo_6","rzapo_7","rzapo_8","rzapo_9","rzapo_10",
                      "rzapo_1_short","rzapo_2_short","rzapo_3_short","rzapo_4_short","rzapo_5_short",
                      "rzapo_6_short","rzapo_7_short","rzapo_8_short","rzapo_9_short","rzapo_10_short",]

    perm_drop_features = ['r_o_a','des', 'clone', 'particle', 'cloneclass', 'l1', 'l2', 'l3', 'l4','l5','l6','l7',
                'p', 'q',
                'mio', 'res_char', 'm', 'n',
                "avg_grid","avg_grid_short",
                                "mm_min", "mm_mean", "mm_max", "mm_sigma", "mm_sigma_normed", "mm_delta","mm_delta_normed", 
                "mmdot_min", "mmdot_mean", "mmdot_max","mmdot_std","mmdot_std_normed","mmdot_delta","mmdot_delta_normed",
                'a_min','a_max',
                'i_delta_normed', 'i_delta_normed_short',
                'e_delta_normed', 'e_delta_normed_short',
                'e_sigma_normed', 'e_sigma_normed_short',
                'i_sigma_normed','i_sigma_normed_short',
                'adot_delta_normed', 'edot_delta_normed', 'idot_delta_normed', 'qdot_delta_normed',
                'adot_std_normed', 'edot_std_normed', 'idot_std_normed', 'qdot_std_normed',
                'adot_delta_normed_short', 'edot_delta_normed_short', 
                'idot_delta_normed_short', 'qdot_delta_normed_short',
                'adot_std_normed_short', 'edot_std_normed_short',
                'idot_std_normed_short', 'qdot_std_normed_short',               
                "rzperi_1","rzperi_2","rzperi_3","rzperi_4","rzperi_5",
                "rzperi_6","rzperi_7","rzperi_8","rzperi_9","rzperi_10",
                "rzperi_1_short","rzperi_2_short","rzperi_3_short","rzperi_4_short","rzperi_5_short",
                "rzperi_6_short","rzperi_7_short","rzperi_8_short","rzperi_9_short","rzperi_10_short",
                "rzapo_1","rzapo_2","rzapo_3","rzapo_4","rzapo_5",
                "rzapo_6","rzapo_7","rzapo_8","rzapo_9","rzapo_10",
                "rzapo_1_short","rzapo_2_short","rzapo_3_short","rzapo_4_short","rzapo_5_short",
                "rzapo_6_short","rzapo_7_short","rzapo_8_short","rzapo_9_short","rzapo_10_short",  
                
                "adj_empty_Qa_sec", "adj_empty_Qa_sec_short",
                "nstd_q_sec_normed","ndel_q_sec_normed",
                "nstd_q_sec_normed_short","ndel_q_sec_normed_short",
                "nstd_Qa_sec_normed","ndel_Qa_sec_normed",
                "nstd_Qa_sec_normed_short","ndel_Qa_sec_normed_short",
                
                'navg_Qa_sec_short', 'navg_Qa_sec', 
                'navg_q_sec_short', 'navg_q_sec',
                
                
                'min_aminmax_to_mean_density_ratio_short', 'max_aminmax_to_mean_density_ratio_short',
                'min_aminmax_to_mean_density_ratio', 'max_aminmax_to_mean_density_ratio',
                'min_amin_to_amax_density_ratio_short', 'max_amin_to_amax_density_ratio_short',
                'min_amin_to_amax_density_ratio', 'max_amin_to_amax_density_ratio',
                
                #'Omdot_min_short', 'Omdot_mean_short', 'Omdot_max_short', 
                'Omdot_std_short', 
                #'Omdot_std_normed_short', 
                'Omdot_delta_short', 
                #'Omdot_delta_normed_short', 
                'o_min_short', 
                'o_mean_short', 'o_max_short', 'o_sigma_short', 'o_delta_short', 
                #'odot_min_short', 
                #'odot_mean_short', 'odot_max_short', 
                'odot_sigma_short', 
                #'odot_sigma_normed_short', 
                'odot_delta_short', 
                #'odot_delta_normed_short', 
                #'podot_min_short', 'podot_mean_short', 
                #'podot_max_short', 
                'podot_std_short', 
                #'podot_std_normed_short', 
                'podot_delta_short', 
                #'podot_delta_normed_short',
                
                
                #'tn_sigma_short', 'tn_delta_short', 
                'tndot_min_short', 'tndot_mean_short', 'tndot_max_short', 
                'tndot_std_short', 'tndot_delta_short', 
                'tndot_std_normed_short','tndot_delta_normed_short',
                'tndot_min', 'tndot_mean', 'tndot_max', 
                'tndot_std', 'tndot_delta', 
                'tndot_std_normed','tndot_delta_normed',
                
                #'delta_aminmax_to_mean_density_ratio',
                'delta_aminmax_to_mean_density_ratio_normed', 
                #'delta_amin_to_amax_density_ratio',
                'delta_amin_to_amax_density_ratio_normed',
                
                
                
                
                #"asf", "amaxpower","amaxpower3",
                #"af1",
                #"af2","af3",
                #"af1_short",
                #"af2_short","af3_short",

                "esf_short", "emaxpower_short","emaxpower3_short",
                #"ef1","ef2","ef3",
                "ef1_short","ef2_short","ef3_short",

                "isf_short", "imaxpower_short","imaxpower3_short",
                #"if1","if2","if3",
                "if1_short","if2_short","if3_short",
                
                #"amdsf", "amdmaxpower","amdmaxpower3",
                #"amdf1_short","amdf2_short","amdf3_short",
                #"amdf1","amdf2","amdf3",
                
                
                "n_almost_empty_short", "n_almost_empty",
                
                'grid_deltaavg_short', 'grid_deltaavg',
                
                
               ]



    feature_names = dataset.columns.to_list()
    features_remaining = []
    for i in range(0,len(dataset.columns)):
        if(dataset.columns[i] not in (perm_drop_features)):
            features_remaining.append(dataset.columns[i])
    
    clasfeat = 'l5'
    all_types1 = list( set(dataset[clasfeat]) )
    types1_dict = { all_types1[i] : i for i in range( len(all_types1) ) }
    int1_dict = { i : all_types1[i] for i in range( len(all_types1) ) }
    classes1 = dataset[clasfeat].map(types1_dict)
    rs=283
    features_train, features_test, classes_train, classes_test = train_test_split(
                        dataset, classes1, test_size=0.333, random_state=rs)

    ids_train = features_train['particle'].to_numpy()
    ids_test = features_test['particle'].to_numpy()

    features_train.drop(perm_drop_features, axis=1, inplace=True)
    features_train = features_train.to_numpy()

    features_test.drop(perm_drop_features, axis=1, inplace=True)
    features_test = features_test.to_numpy()

    rs = 42
    clf = GradientBoostingClassifier(max_leaf_nodes = None, min_impurity_decrease=0.0, min_weight_fraction_leaf = 0.0, 
                                     min_samples_leaf = 1, min_samples_split=3, 
                                     criterion = 'friedman_mse',subsample = 0.9, learning_rate=0.15,
                                     max_depth=8, max_features='log2', 
                                     n_estimators=300, random_state=rs)
    clf.fit(features_train, classes_train)

    classes_predict = clf.predict(features_test)
    score = accuracy_score(classes_test, classes_predict)

    return clf, score, classes_train, classes_test, features_train, features_test, features_remaining, int1_dict

def label5_particle(row):
    newlabel = 'none'
    if (row['cloneclass'] == 'detached' or row['cloneclass'] == 'classical'):
        newlabel = 'class-det'
    else:
        newlabel= row['cloneclass']
    return newlabel

def label6_particle(row):
    newlabel = 'none'
    if (row['cloneclass'] == 'detached' or row['cloneclass'] == 'classical'):
        newlabel = 'class-det'
    elif (row['cloneclass'] == 'Nresonant'):
        if (row['res_char']== 'im' or row['res_char']== 'rm'):
            newlabel = 'mixed-res'
        else:
            newlabel = 'res'
    else:
        newlabel= row['cloneclass']
    return newlabel

def label_particle_7(row):
    newlabel = 'none'
    if ( row['res_char']== 'am' or row['res_char']== 'a'):
        newlabel = 'res-interacting'
    elif (row['res_char']== 'i' or row['res_char']== 'im' or row['res_char']== 'r' or row['res_char']== 'rm'):
        newlabel = 'res'
    else:
        newlabel = 'nonres'
    if(row['cloneclass']=='scattering'):
        newlabel= row['cloneclass']
    return newlabel





def run_TNO_integration_for_ML(tno='',clones=2):
    '''
    '''

    if(clones==2):
        find_3_sigma=True
    else:
        find_3_sigma=False

    flag, epoch, sim = run_reb.initialize_simulation(planets=['jupiter', 'saturn', 'uranus', 'neptune'],
                          des=tno, clones=clones, find_3_sigma=find_3_sigma)


    features, shortarchive, longarchive = run_sim_for_TNO_ML(sim,des=tno,clones=clones,date=True)

    return features, shortarchive, longarchive


def run_sim_for_TNO_ML(sim, des='',clones=0,date=False):
    '''
    '''
    sim_2 = sim.copy()

    if(date):
        today = date.today()
        datestring = today.strftime("%b-%d-%Y")
        shortarchive = datestring + "-" + des + "-short-archive.bin"
        longarchive = datestring + "-" + des + "-long-archive.bin"
    else:
        shortarchive = des + "-short-archive.bin"
        longarchive = des + "-long-archive.bin"

        
    flag, sim = run_reb.run_simulation(sim,tmax=0.5e6,tout=50.,filename=shortarchive,deletefile=True)
    
    flag, sim_2 = run_reb.run_simulation(sim_2,tmax=10e6,tout=1000.,filename=longarchive,deletefile=True)


    flag, a, ec, inc, node, peri, ma, t = tools.read_sa_for_sbody(sbody=des,archivefile=shortarchive,nclones=clones)
    pomega = peri+ node 
    flag, apl, ecpl, incpl, nodepl, peripl, mapl, tpl = tools.read_sa_by_hash(obj_hash='neptune',archivefile=shortarchive)
    q = a*(1.-ec)
    flag, xr, yr, zr, vxr, vyr, vzr, tr = tools.calc_rotating_frame(sbody=des, planet='neptune', 
                                                                    archivefile=shortarchive, nclones=clones)
    rrf = np.sqrt(xr*xr + yr*yr + zr*zr)
    phirf = np.arctan2(yr, xr)
    tiss = apl/a + 2.*np.cos(inc)*np.sqrt(a/apl*(1.-ec*ec))

    flag, l_a, l_ec, l_inc, l_node, l_peri, l_ma, l_t = tools.read_sa_for_sbody(sbody=des,archivefile=longarchive,nclones=clones)
    l_pomega = l_peri+ l_node 
    flag, apl, ecpl, incpl, nodepl, peripl, mapl, tpl = tools.read_sa_by_hash(obj_hash='neptune',archivefile=longarchive)
    l_q = l_a*(1.-l_ec)
    flag, xr, yr, zr, vxr, vyr, vzr, tr = tools.calc_rotating_frame(sbody=des, planet='neptune', 
                                                                    archivefile=longarchive, nclones=clones)
    l_rrf = np.sqrt(xr*xr + yr*yr + zr*zr)
    l_phirf = np.arctan2(yr, xr)
    l_tiss = apl/l_a + 2.*np.cos(l_inc)*np.sqrt(l_a/apl*(1.-l_ec*l_ec))

    #first list is the always removed set
    index_remove = [25, 27, 32, 34, 39, 41, 46, 48, 53, 55, 93, 95, 116, 117, 118, 119, 120, 121, 122, 
                    123, 124, 125, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 147, 234, 236, 241, 
                    243, 248, 250, 255, 257, 262, 264, 302, 304, 325, 326, 327, 328, 329, 330, 331, 332, 
                    333, 334, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 356]

    #set to remove for current best classifier (subject to change)
    index_remove = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 25, 27, 32, 34, 39, 41, 46, 48, 
                    53, 55, 93, 95, 101, 102, 103, 104, 105, 106, 107, 110, 112, 114, 116, 117, 118, 119, 
                    120, 121, 122, 123, 124, 125, 127, 128, 130, 132, 134, 135, 136, 137, 138, 139, 140, 
                    141, 142, 143, 147, 149, 152, 183, 184, 186, 187, 188, 190, 234, 236, 241, 243, 248, 
                    250, 255, 257, 262, 264, 268, 270, 272, 273, 274, 275, 276, 280, 282, 287, 289, 302, 
                    304, 310, 311, 312, 313, 314, 315, 316, 319, 321, 323, 325, 326, 327, 328, 329, 330, 
                    331, 332, 333, 334, 336, 337, 339, 341, 343, 344, 345, 346, 347, 348, 349, 350, 351, 
                    352, 356, 358, 361, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 392, 
                    393, 396, 397]


    n=0
    short_features = calc_ML_features(t,a[n],ec[n],inc[n],node[n],peri[n],
                                        pomega[n],q[n],rrf[n],phirf[n],tiss[n])
    long_features = calc_ML_features(l_t,l_a[n],l_ec[n],l_inc[n],l_node[n],l_peri[n],
                                        l_pomega[n],l_q[n],l_rrf[n],l_phirf[n],l_tiss[n])

    all_features = np.concatenate((long_features, short_features),axis=0)
    temp_features = np.delete(all_features,index_remove)
    features = np.array([temp_features])

    for n in range(1,clones+1):
        short_features = calc_ML_features(t,a[n],ec[n],inc[n],node[n],peri[n],
                                        pomega[n],q[n],rrf[n],phirf[n],tiss[n])
        long_features = calc_ML_features(l_t,l_a[n],l_ec[n],l_inc[n],l_node[n],l_peri[n],
                                        l_pomega[n],l_q[n],l_rrf[n],l_phirf[n],l_tiss[n])
        all_features = np.concatenate((long_features, short_features),axis=0)
        temp_features = np.delete(all_features,index_remove)
        features = np.append(features,[temp_features],axis=0)
   
    return features, shortarchive, longarchive

def read_TNO_integration_for_ML(sim=None,sim_init=None,clones=0,tno=''):
    '''
    '''
    
    if(clones==2):
        find_3_sigma=True
    else:
        find_3_sigma=False

    today = date.today()
    datestring = today.strftime("%b-%d-%Y")

    shortarchive = '../data/ML_archives/'+datestring + "-" + tno + "-short-archive.bin"
    flag, sim = run_reb.run_simulation(sim_init,tmax=0.5e6,tout=50.,filename=shortarchive,deletefile=True)
    
    longarchive = '../data/ML_archives/'+datestring + "-" + tno + "-long-archive.bin"
    sim_2.save(longarchive)
    
    flag, a, ec, inc, node, peri, ma, t = tools.read_sa_for_sbody(sbody=tno,archivefile=shortarchive,nclones=clones)
    pomega = peri+ node 
    flag, apl, ecpl, incpl, nodepl, peripl, mapl, tpl = tools.read_sa_by_hash(obj_hash='neptune',archivefile=shortarchive)
    q = a*(1.-ec)
    flag, xr, yr, zr, vxr, vyr, vzr, tr = tools.calc_rotating_frame(sbody=tno, planet='neptune', 
                                                                    archivefile=shortarchive, nclones=clones)
    rrf = np.sqrt(xr*xr + yr*yr + zr*zr)
    phirf = np.arctan2(yr, xr)
    tiss = apl/a + 2.*np.cos(inc)*np.sqrt(a/apl*(1.-ec*ec))

    flag, l_a, l_ec, l_inc, l_node, l_peri, l_ma, l_t = tools.read_sa_for_sbody(sbody=tno,archivefile=longarchive,nclones=clones)
    l_pomega = l_peri+ l_node 
    flag, apl, ecpl, incpl, nodepl, peripl, mapl, tpl = tools.read_sa_by_hash(obj_hash='neptune',archivefile=longarchive)
    l_q = l_a*(1.-l_ec)
    flag, xr, yr, zr, vxr, vyr, vzr, tr = tools.calc_rotating_frame(sbody=tno, planet='neptune', 
                                                                    archivefile=longarchive, nclones=clones)
    l_rrf = np.sqrt(xr*xr + yr*yr + zr*zr)
    l_phirf = np.arctan2(yr, xr)
    l_tiss = apl/l_a + 2.*np.cos(l_inc)*np.sqrt(l_a/apl*(1.-l_ec*l_ec))

    #first list is the always removed set
    index_remove = [25, 27, 32, 34, 39, 41, 46, 48, 53, 55, 93, 95, 116, 117, 118, 119, 120, 121, 122, 
                    123, 124, 125, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 147, 234, 236, 241, 
                    243, 248, 250, 255, 257, 262, 264, 302, 304, 325, 326, 327, 328, 329, 330, 331, 332, 
                    333, 334, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 356]

    #set to remove for current best classifier (subject to change)
    index_remove = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 25, 27, 32, 34, 39, 41, 46, 48, 
                    53, 55, 93, 95, 101, 102, 103, 104, 105, 106, 107, 110, 112, 114, 116, 117, 118, 119, 
                    120, 121, 122, 123, 124, 125, 127, 128, 130, 132, 134, 135, 136, 137, 138, 139, 140, 
                    141, 142, 143, 147, 149, 152, 183, 184, 186, 187, 188, 190, 234, 236, 241, 243, 248, 
                    250, 255, 257, 262, 264, 268, 270, 272, 273, 274, 275, 276, 280, 282, 287, 289, 302, 
                    304, 310, 311, 312, 313, 314, 315, 316, 319, 321, 323, 325, 326, 327, 328, 329, 330, 
                    331, 332, 333, 334, 336, 337, 339, 341, 343, 344, 345, 346, 347, 348, 349, 350, 351, 
                    352, 356, 358, 361, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 392, 
                    393, 396, 397]


    n=0
    short_features = calc_ML_features(t,a[n],ec[n],inc[n],node[n],peri[n],
                                        pomega[n],q[n],rrf[n],phirf[n],tiss[n])
    long_features = calc_ML_features(l_t,l_a[n],l_ec[n],l_inc[n],l_node[n],l_peri[n],
                                        l_pomega[n],l_q[n],l_rrf[n],l_phirf[n],l_tiss[n])

    all_features = np.concatenate((long_features, short_features),axis=0)
    temp_features = np.delete(all_features,index_remove)
    features = np.array([temp_features])

    for n in range(1,clones+1):
        short_features = calc_ML_features(t,a[n],ec[n],inc[n],node[n],peri[n],
                                        pomega[n],q[n],rrf[n],phirf[n],tiss[n])
        long_features = calc_ML_features(l_t,l_a[n],l_ec[n],l_inc[n],l_node[n],l_peri[n],
                                        l_pomega[n],l_q[n],l_rrf[n],l_phirf[n],l_tiss[n])
        all_features = np.concatenate((long_features, short_features),axis=0)
        temp_features = np.delete(all_features,index_remove)
        features = np.append(features,[temp_features],axis=0)
   
    return features, shortarchive, longarchive



def print_TNO_ML_results(pred_class,classes_dictionary,class_probs,clones=2):
    nclas = len(classes_dictionary)
    print("Clone number, most probable class, probability of most probable class, ",end ="")
    for n in range(nclas):
        print("probability of %s," % classes_dictionary[n],end ="")
    print("\n",end ="")
    format_string = "%d, %s, "
    for n in range(nclas-1):
        format_string+="%e, "
    format_string+="%e,\n"
    for n in range(0,clones+1):
        print("%d, %s, %e, " % (n,classes_dictionary[pred_class[n]], class_probs[n][pred_class[n]]),end ="")
        for j in range(nclas):
            print("%e, " % class_probs[n][j] ,end ="")
        print("\n",end ="")

def print_TNO_ML_results_to_file(des,pred_class,classes_dictionary,class_probs,clones=2):
    outfile = des + "-classes.txt"
    out = open(outfile,"w")
    nclas = len(classes_dictionary)
    line = "Designation, clone number, most probable class, probability of most probable class, "
    for n in range(nclas):
        line+= ("probability of %s, " % classes_dictionary[n])
    line+="\n"
    out.write(line)
    format_string = "%d, %s, "
    for n in range(nclas-1):
        format_string+="%e, "
    format_string+="%e,\n"
    for n in range(0,clones+1):
        line = des + ', '
        line+=("%d, %s, %e, " % (n,classes_dictionary[pred_class[n]], class_probs[n][pred_class[n]]))
        for j in range(nclas):
            line+=("%e, " % class_probs[n][j])
        line+="\n"
        out.write(line)

    out.close()


def farey_tree(num, denom, prmin, prmax):
    order_max = 20
    # Initialize fractions
    flag = 0
    oldnum = num.copy()  
    olddenom = denom.copy()
    nfractions = len(oldnum)
    if nfractions == 1:
        # Only one fraction left, can't keep building the tree
        return flag, num, denom,np.array([0.]),np.array([0.]), 0
        
    # the next layer in the farey tree will have nfraction-1 new fractions
    newnum = np.zeros(nfractions-1)  
    newdenom = np.zeros(nfractions-1)

    # the full set of numbers will have 2*nfractions -1 entries
    num = np.zeros(2*nfractions-1)  
    denom = np.zeros(2*nfractions-1)
    
    nn = 0
    new_n = 0
    for n in range(0,nfractions-1):
        num[nn] = oldnum[n]
        denom[nn] = olddenom[n] 
        nn+=1
        num[nn] = oldnum[n] + oldnum[n + 1]
        denom[nn] = olddenom[n] + olddenom[n + 1]
        if(num[nn]<=order_max):
            nn+=1
            newnum[new_n] = oldnum[n] + oldnum[n + 1]
            newdenom[new_n] = olddenom[n] + olddenom[n + 1]
            new_n+=1


    num[nn] = oldnum[nfractions-1]
    denom[nn] = olddenom[nfractions-1]

    if(new_n <1):
        flag = 0
    else:
        flag = 1
    
    newnum = newnum[0:new_n]
    newdenom = newdenom[0:new_n]
    num = num[0:nn+1]
    denom = denom[0:nn+1]
    
    
    left = 0
    right = nn+1
    for n in range(0,nn):
        if(prmin > float(num[n]/denom[n])):# and left ==0):
            left = n

    for n in range(nn,0,-1):
        if(prmax < float(num[n]/denom[n])):# and right==nn):
            right = n+1
    num = num[left:right]
    denom = denom[left:right]

    new_check_q = np.empty(0)
    new_check_p = np.empty(0)

    for n in range(0,new_n):
        pr = float(newnum[n]/newdenom[n])
        if(pr>=prmin and pr<=prmax):
            new_check_q = np.append(new_check_q,int(newnum[n]))
            new_check_p = np.append(new_check_p,int(newdenom[n]))
            

    n_check = len(new_check_p)
    return flag, num, denom,new_check_q, new_check_p, n_check 


