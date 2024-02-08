import sys
import numpy as np
sys.path.insert(0, '../src')
import tools


def ML_parse_features(data):
    """
    ML_parse_features(data) computes the necessary features to classify
    Returns features for classification
    """

    # column to stop at for the standard analysis (column +1)
    # 0    1 2  3        4          5
    # time,a,e,inc(rad),node(rad),argperi(rad),
    #    6          7          8                9
    # longperi(rad),q,rotating-frame r, rotating-frame phi(rad)
    # tisserand parameter (10)
    cmax = 8

    # Take stats of simulations

    mins = np.amin(data[:, 1:cmax], axis=0)
    maxes = np.amax(data[:, 1:cmax], axis=0)
    dels = maxes - mins

    means = np.mean(data[:, 1:cmax], axis=0)
    stdev = np.std(data[:, 1:cmax], axis=0)

    amin = mins[0]
    amax = maxes[0]
    amean = means[0]
    astd = stdev[0]

    emin = mins[1]
    emax = maxes[1]
    emean = means[1]
    estd = stdev[1]

    imin = mins[2]
    imax = maxes[2]
    imean = means[2]
    istd = stdev[2]    


    # Take time derivatives
    diffs = data[1:, :] - data[:-1, :]
    #correct the angular means, dels, and diffs for wrapping
    # correct the differencs
    for j in range(4,7):
        for k in range(1,len(diffs[:,j])):
            dif1 = np.abs(data[k,j] - data[k-1,j])
            dif2 = np.abs(data[k,j] + 2.*np.pi - data[k-1,j])
            dif3 = np.abs(data[k, j] - (data[k-1,j] + 2.*np.pi))
            if(dif2 < dif1 and dif2 < dif3):
                diffs[k-1,j] = data[k,j] + 2.*np.pi - data[k-1,j]
            elif(dif3 < dif1 and dif3 < dif2):
                diffs[k-1,j] = data[k,j] - (data[k-1,j] + 2.*np.pi)
            #otherwise it keeps the standard difference from above

    #correct the deltas, means for wrapping
    for j in range(3,6):
        #original, 0-2pi diff
        del1 = maxes[j] - mins[j]
        #re-center -pi to pi (index j+1 because we removed the time axis)
        tempangle = np.copy(data[:,j+1])
        tempangle = tools.arraymod2pi0(tempangle)
        del2 = np.amax(tempangle) - np.amin(tempangle)
        if(del2 < del1):
            dels[j] = del2
            mean2 = np.mean(tempangle)
            if(mean2<0):
                mean2+=2.*np.pi
            means[j] = mean2
            stdev[j] = np.std(tempangle)
        #otherwise keep original 0-2pi values

    # divide by time difference and
    # add on new axis to time to give same dimensionality as the numerator
    dxdt = diffs[:, 1:cmax] / diffs[:, 0,np.newaxis]

    mindxdt = np.amin(dxdt, axis=0)
    absmindxdt = np.amin(np.abs(dxdt), axis=0)
    meandxdt = np.mean(dxdt, axis=0)
    absmaxdxdt = np.amax(np.abs(dxdt), axis=0)
    maxdxdt = np.amax(dxdt, axis=0)
    deldxdt = maxdxdt - mindxdt

    stdev_norm = stdev/means
    dels_norm = dels/means
    deldxdt_norm = deldxdt/meandxdt

    # rearrange data into the order I want
    # arrs = [mins,means,maxes,absmaxes,stdev,dels,absmindxdt,mindxdt,meandxdt,maxdxdt,absmaxdxdt,deldxdt]
    arrs = [mins, means, maxes, stdev, stdev_norm, dels, dels_norm,
            absmindxdt, mindxdt, meandxdt, maxdxdt, absmaxdxdt, deldxdt, deldxdt_norm]
    inds = [0, 1, 2, 3, 4, 5, 6]  # a, e, i, Omega, omega, pomega, q
    features = []

    for i in inds:
        for a in arrs:
            features += [a[i]]

    # add in the features based on the rotating frame positions
    r = data[:, 8]
    ph = data[:, 9]

    qmin = np.amin(r)
    Qmax = np.amax(r) + 0.01

    nrbin = 10.
    nphbin = 20.
    dr = (Qmax - qmin) / nrbin
    dph = (2. * np.pi) / nphbin
    # center on the planet in phi
    phmin = -dph / 2.

    # radial plus aziumthal binning
    # indexing is radial bin, phi bin: rph_count[rbin,phibin]
    rph_count = np.zeros((int(nrbin), int(nphbin)))
    # radial only binning
    r_count = np.zeros(int(nrbin))

    # for calculating the average sin(ph) and cos(ph)
    # indexing is   sinphbar[rbin,resorder]
    sinphbar = np.zeros((int(nrbin), 5))
    cosphbar = np.zeros((int(nrbin), 5))


    # divide into radial and azimuthal bins
    nmax = len(r)
    for n in range(0, nmax):
        rbin = int(np.floor((r[n] - qmin) / dr))
        for resorder in range(1,5):
            tcos = np.cos(float(resorder)*ph[n])
            tsin = np.sin(float(resorder)*ph[n])
            sinphbar[rbin,resorder]+=tsin
            cosphbar[rbin,resorder]+=tcos
        r_count[rbin]+=1.
        phbin = int(np.floor((ph[n] - phmin) / dph))
        if (phbin == int(nphbin)):
            phbin = 0
        rph_count[rbin, phbin] += 1

    # perihelion distance bin stats
    nqempty = np.zeros(int(nrbin))
    nadjempty = np.zeros(int(nrbin))
    rbinmin = np.zeros(int(nrbin))
    rbinmax = np.zeros(int(nrbin))
    rbinavg = np.zeros(int(nrbin))
    rbinstd = np.zeros(int(nrbin))

    for nr in range(0, int(nrbin)):
        rbinmin[nr] = 1e9
        for resorder in range(1,5):
            sinphbar[nr,resorder] = sinphbar[nr,resorder]/r_count[nr]
            cosphbar[nr,resorder] = cosphbar[nr,resorder]/r_count[nr]
        for n in range(0, int(nphbin)):
            if (rph_count[nr, n] == 0):
                nqempty[nr] += 1
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
            rbinstd[nr] = np.sqrt(rbinstd[nr] / nphbin) / rbinavg[nr]
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


    nqempty = nqempty / nphbin
    nadjempty = nadjempty / nphbin

    features += [nqempty[0]]
    features += [nadjempty[0]]
    if (not (rbinavg[0] == 0)):
        temp = (rbinmax[0] - rbinmin[0]) / rbinavg[0]
    else:
        temp = 0.
    features += [temp]
    features += [rbinstd[0]]

    n = 5
    features += [nqempty[n]]
    features += [nadjempty[n]]
    if (not (rbinavg[n] == 0)):
        temp = (rbinmax[n] - rbinmin[n]) / rbinavg[n]
    else:
        temp = 0.
    features += [temp]
    features += [rbinstd[n]]

    n = -2
    features += [nqempty[n]]
    features += [nadjempty[n]]
    if (not (rbinavg[n] == 0)):
        temp = (rbinmax[n] - rbinmin[n]) / rbinavg[n]
    else:
        temp = 0.
    features += [temp]
    features += [rbinstd[n]]

    #add the rayleigh z-test statistics at perihelion and aphelion
    for resorder in range(1, 5):
        temp = np.sqrt(sinphbar[0,resorder]*sinphbar[0,resorder] +
                       cosphbar[0,resorder]*cosphbar[0,resorder])
        features += [temp]
        temp = np.sqrt(sinphbar[-1,resorder]*sinphbar[-1,resorder] +
                       cosphbar[-1,resorder]*cosphbar[-1,resorder])
        features += [temp]



    # rph_count[rbin,phibin]

    spatial_counts = rph_count.flatten()
    nz_minval = np.min(spatial_counts[np.nonzero(spatial_counts)])
    nz_avg = np.mean(spatial_counts[np.nonzero(spatial_counts)])
    nz_std = np.std(spatial_counts[np.nonzero(spatial_counts)])
    avg =  np.mean(spatial_counts)
    std =  np.std(spatial_counts)
    features += [avg]
    features += [nz_avg]
    deltaavg = nz_avg-avg
    features += [deltaavg]
    features += [std]
    features += [nz_std]
    deltastd = std - nz_std
    features += [deltastd]
    n_empty=0
    n_almost_empty=0
    
    for n in range(0,len(spatial_counts)):
        if(spatial_counts[n]==0):
            n_empty += 1
        if(spatial_counts[n]<7):
            n_almost_empty += 1
    features += [n_empty]
    features += [n_almost_empty]





    #calculate the correlations between a and e, a and i, and e and i
    temp =  max_corelation(data[:,1],data[:,2])
    features += [temp]
    temp =  max_corelation(data[:,1],data[:,3])
    features += [temp]
    temp =  max_corelation(data[:,2],data[:,3])
    features += [temp]

    #calculate spectral fractions
    deltat = data[2,0] - data[1,0]
    #a
    temp1, temp2, temp3, f1, f2, f3 = spectral_characteristics(data[:,1],deltat)
    features += [temp1]
    features += [temp2]
    features += [temp3]
    features += [f1]
    features += [f2]
    features += [f3]
    

    # eccentricity, via e*sin(varpi)
    hec = data[:,2]*np.sin(data[:,6])
    temp1, temp2, temp3, f1, f2, f3 = spectral_characteristics(hec,deltat)
    features += [temp1]
    features += [temp2]
    features += [temp3]
    features += [f1]
    features += [f2]
    features += [f3]


    # inclination, via sin(i)sin(Omega)
    pinc = np.sin(data[:,3])*np.sin(data[:,4])
    temp1, temp2, temp3, f1, f2, f3 = spectral_characteristics(pinc,deltat)
    features += [temp1]
    features += [temp2]
    features += [temp3]
    features += [f1]
    features += [f2]
    features += [f3]


    #amd
    amd = 1. - np.sqrt(1.- data[:,2]*data[:,2])*np.cos(data[:,3])
    amd = amd*np.sqrt(data[:,1])
    temp1, temp2, temp3, f1, f2, f3 = spectral_characteristics(amd,deltat)
    features += [temp1]
    features += [temp2]
    features += [temp3]
    features += [f1]
    features += [f2]
    features += [f3]



    #Do some binning in the a, e, and i-distributions

    a1 = amin
    a2 = amean-0.75*astd
    a3 = amean-0.375*astd
    a4 = amean+0.375*astd
    a5 = amean+0.75*astd
    a6 = amax

    if(a1 < a2 and a2 < a3 and a4 < a5 and a5 < a6):
        abins = [a1,a2,a3,a4,a5,a6]
    else:
        da = (amax-amin)/8.
        a2 = amin + 2.*da
        a3 = a2 + da
        a4 = a3 + 2.*da
        a5 = a4 + da
        abins = [a1,a2,a3,a4,a5,a6]
        
    acounts, tbins = np.histogram(data[:,1],bins=abins)

    #average ratio of extreme-a density to middle-a density
    if(acounts[2] == 0):
        acounts[2] = 1
    em_a = (acounts[0] + acounts[4])/(2.*acounts[2])
    features += [em_a]
    #ratio of extreme-low-a density to extreme high-a density
    if(acounts[4] == 0):
        acounts[4] = 1
    lh_a = (acounts[0]/acounts[4])
    features += [lh_a]

    #repeat but across a couple time bins
    dj = data[:,1].size//4

    
    acounts, tbins = np.histogram(data[0:dj,1],bins=abins)
    if(acounts[2] == 0):
        acounts[2] = 1
    em1 = (acounts[0] + acounts[4])/(2.*acounts[2])
    #ratio of extreme-low-a density to extreme high-a density
    if(acounts[4] == 0):
        acounts[4] = 1
    lh1 = (acounts[0]/acounts[4])
 
    acounts, tbins = np.histogram(data[dj:2*dj,1],bins=abins)
    if(acounts[2] == 0):
        acounts[2] = 1
    em2 = (acounts[0] + acounts[4])/(2.*acounts[2])
    #ratio of extreme-low-a density to extreme high-a density
    if(acounts[4] == 0):
        acounts[4] = 1
    lh2 = (acounts[0]/acounts[4])
    
    acounts, tbins = np.histogram(data[2*dj:3*dj,1],bins=abins)
    if(acounts[2] == 0):
        acounts[2] = 1
    em3 = (acounts[0] + acounts[4])/(2.*acounts[2])
    #ratio of extreme-low-a density to extreme high-a density
    if(acounts[4] == 0):
        acounts[4] = 1
    lh3 = (acounts[0]/acounts[4])


    acounts, tbins = np.histogram(data[3*dj:4*dj,1],bins=abins)
    if(acounts[2] == 0):
        acounts[2] = 1
    em4 = (acounts[0] + acounts[4])/(2.*acounts[2])
    #ratio of extreme-low-a density to extreme high-a density
    if(acounts[4] == 0):
        acounts[4] = 1
    lh4 = (acounts[0]/acounts[4])

    min_em = min(em1,em2,em3,em4)
    max_em = max(em1,em2,em3,em4)

    temp = max_em - min_em
    features += [temp]
    temp = temp/em_a
    features += [temp]
    features += [min_em]
    features += [max_em]
    

    min_lh = min(lh1,lh2,lh3,lh4)
    max_lh = max(lh1,lh2,lh3,lh4)

    temp = max_lh - min_lh
    features += [temp]
    temp = temp/lh_a
    features += [temp]
    features += [min_lh]
    features += [max_lh]


    #eccentricity distribution

    a1 = emin
    a2 = emean-0.75*estd
    a3 = emean-0.375*estd
    a4 = emean+0.375*estd
    a5 = emean+0.75*estd
    a6 = emax

    if(a1 < a2 and a2 < a3 and a4 < a5 and a5 < a6):
        ebins = [a1,a2,a3,a4,a5,a6]
    else:
        de = (emax-emin)/8.
        a2 =emin + 2.*de
        a3 = a2 + de
        a4 = a3 + 2.*de
        a5 = a4 + de
        ebins = [a1,a2,a3,a4,a5,a6]

    ecounts, tbins = np.histogram(data[:,2],bins=ebins)

    #average ratio of extreme-a density to middle-a density
    if(ecounts[2] == 0):
        ecounts[2] = 1
    temp = (ecounts[0] + ecounts[4])/(2.*ecounts[2])
    features += [temp]
    
    if(ecounts[4] == 0):
        ecounts[4] = 1
    #ratio of extreme-low-a density to extreme high-a density
    temp = (ecounts[0]/ecounts[4])
    features += [temp]


    a1 = imin
    a2 = imean-0.75*istd
    a3 = imean-0.375*istd
    a4 = imean+0.375*istd
    a5 = imean+0.75*istd
    a6 = imax

    if(a1 < a2 and a2 < a3 and a4 < a5 and a5 < a6):
        ibins = [a1,a2,a3,a4,a5,a6]
    else:
        di = (imax-imin)/8.
        a2 =imin + 2.*di
        a3 = a2 + di
        a4 = a3 + 2.*di
        a5 = a4 + di
        ibins = [a1,a2,a3,a4,a5,a6]

    icounts, tbins = np.histogram(data[:,3],bins=ibins)

    #average ratio of extreme-a density to middle-a density
    if(icounts[2] == 0):
        icounts[2] = 1    
    temp = (icounts[0] + icounts[4])/(2.*icounts[2])
    features += [temp]

    #ratio of extreme-low-a density to extreme high-a density
    if(ecounts[4] == 0):
        ecounts[4] = 1
    temp = (icounts[0]/icounts[4])
    features += [temp]

    #some data features based on the tisserand paramter:
    mintn = np.amin(data[:, 10])
    features+=[mintn]
    meantn = np.mean(data[:, 10])
    features+=[meantn]
    maxtn = np.amax(data[:, 10])
    features+=[maxtn]
    
    stdevtn = np.std(data[:, 10])
    features+=[stdevtn]
    stdev_norm_tn = stdevtn/meantn
    features+=[stdev_norm_tn]
    
    deltn = maxtn - mintn
    features+=[deltn]
    dels_norm_tn = deltn/meantn
    features+=[dels_norm_tn]

    difftn = data[1:,10] - data[:-1,10]
    dtndt = difftn[:] / diffs[:, 0]

    mindtndt = np.amin(dtndt)
    features+=[mindtndt]
    meandtndt = np.mean(dtndt)
    features+=[meandtndt]
    maxdtndt = np.amax(dtndt)
    features+=[maxdtndt]
    deldtndt = maxdtndt - mindtndt
    features+=[deldtndt]





    return np.array(features).reshape(1,-1)  # make sure features is a 2d array














def calc_ML_features(time,a,ec,inc,node,argperi,pomega,q,rh,phirf,tn):
    """
    ML_parse_features(data) computes the necessary features to classify
    Returns features for classification
    """

    ########################################################
    ########################################################
    #
    # Very basic time-series data features
    #
    ########################################################
    ########################################################
    a_min=np.amin(a)
    e_min=np.amin(ec)
    i_min=np.amin(inc)
    q_min=np.amin(q)
    tn_min=np.amin(tn)

    a_max=np.amax(a)
    e_max=np.amax(ec)
    i_max=np.amax(inc)
    q_max=np.amax(q)
    tn_max=np.amax(tn)


    a_del = a_max - a_min
    e_del = e_max - e_min
    i_del = i_max - i_min
    q_del = q_max - q_min
    tn_del = tn_max - tn_min


    a_mean = np.mean(q)
    e_mean = np.mean(ec)
    i_mean = np.mean(inc)
    q_mean = np.mean(q)
    tn_mean = np.mean(tn)

    a_std = np.std(q)
    e_std = np.std(ec)
    i_std = np.std(inc)
    q_std = np.std(q)
    tn_std = np.std(tn)

    a_std_norm = a_std/a_mean
    e_std_norm = e_std/e_mean
    i_std_norm = i_std/i_mean
    q_std_norm = q_std/q_mean

    a_del_norm = a_del/a_mean
    e_del_norm = e_del/e_mean
    i_del_norm = i_del/i_mean
    q_del_norm = q_del/q_mean



    #arg peri 0-2pi
    argperi_min=np.amin(argperi)
    argperi_max=np.amax(argperi)
    argperi_del = argperi_max - argperi_min
    argperi_mean = np.mean(argperi)
    argperi_std = np.std(argperi)

    #recenter arg peri around 0 and repeat
    argperi_zero = tools.arraymod2pi0(argperi)
    argperi_min2=np.amin(argperi_zero)
    argperi_max2=np.amax(argperi_zero)
    argperi2_del = argperi_max - argperi_min
    argperi_mean2 = np.mean(argperi_zero)
    argperi_std2 = np.std(argperi_zero)

    #take the better values for delta, mean, and standard deviation:
    if(argperi2_del < argperi_del):
        argperi_del = argperi2_del
        argperi_mean = tools.mod2pi(argperi_mean2)
        argperi_std = argperi_std2

    #calculate time derivatives
    dt = time[1:] - time[:-1] 

    da_dt = a[1:] - a[:-1]
    da_dt = da_dt/dt
    de_dt = ec[1:] - ec[:-1]
    de_dt = de_dt/dt
    di_dt = inc[1:] - inc[:-1]
    de_dt = de_dt/dt
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


    #absmindxdt = np.amin(np.abs(dxdt), axis=0)
    #absmaxdxdt = np.amax(np.abs(dxdt), axis=0)


    ########################################################
    ########################################################
    #
    # Rotating Frame data features
    #
    ########################################################
    ########################################################


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


    a1 = a_min
    a2 = a_mean-0.75*a_std
    a3 = a_mean-0.375*a_std
    a4 = a_mean+0.375*a_std
    a5 = a_mean+0.75*a_std
    a6 = a_max
    if(a1 < a2 and a2 < a3 and a4 < a5 and a5 < a6):
        abins = [a1,a2,a3,a4,a5,a6]
    else:
        da = (a_max-a_min)/8.
        a2 = a_min + 2.*da
        a3 = a2 + da
        a4 = a3 + 2.*da
        a5 = a4 + da
        abins = [a1,a2,a3,a4,a5,a6]
    acounts, tbins = np.histogram(a,bins=abins)

    #average ratio of extreme-a density to middle-a density
    if(acounts[2] == 0):
        acounts[2] = 1 #avoid a nan
    em_a = (acounts[0] + acounts[4])/(2.*acounts[2])
    #ratio of extreme-low-a density to extreme high-a density
    if(acounts[4] == 0):
        acounts[4] = 1
    lh_a = (acounts[0]/acounts[4])

    #repeat across a couple time bins (might help identify intermittent resonant libration)
    dj = a.size//4
    acounts, tbins = np.histogram(a[0:dj],bins=abins)
    if(acounts[2] == 0):
        acounts[2] = 1
    em1 = (acounts[0] + acounts[4])/(2.*acounts[2])
    #ratio of extreme-low-a density to extreme high-a density
    if(acounts[4] == 0):
        acounts[4] = 1
    lh1 = (acounts[0]/acounts[4])
 
    acounts, tbins = np.histogram(a[dj:2*dj],bins=abins)
    if(acounts[2] == 0):
        acounts[2] = 1
    em2 = (acounts[0] + acounts[4])/(2.*acounts[2])
    #ratio of extreme-low-a density to extreme high-a density
    if(acounts[4] == 0):
        acounts[4] = 1
    lh2 = (acounts[0]/acounts[4])
    
    acounts, tbins = np.histogram(a[2*dj:3*dj],bins=abins)
    if(acounts[2] == 0):
        acounts[2] = 1
    em3 = (acounts[0] + acounts[4])/(2.*acounts[2])
    #ratio of extreme-low-a density to extreme high-a density
    if(acounts[4] == 0):
        acounts[4] = 1
    lh3 = (acounts[0]/acounts[4])


    acounts, tbins = np.histogram(a[3*dj:4*dj],bins=abins)
    if(acounts[2] == 0):
        acounts[2] = 1
    em4 = (acounts[0] + acounts[4])/(2.*acounts[2])
    #ratio of extreme-low-a density to extreme high-a density
    if(acounts[4] == 0):
        acounts[4] = 1
    lh4 = (acounts[0]/acounts[4])

    min_em_a = min(em1,em2,em3,em4)
    max_em_a = max(em1,em2,em3,em4)

    delta_em_a = max_em_a - min_em_a
    delta_em_a_norm = delta_em_a/em_a    

    min_lh_a = min(lh1,lh2,lh3,lh4)
    max_lh_a = max(lh1,lh2,lh3,lh4)

    delta_lh_a = max_lh_a - min_lh_a
    delta_lh_a_norm = delta_lh_a/lh_a


    #eccentricity distribution
    a1 = e_min
    a2 = e_mean-0.75*e_std
    a3 = e_mean-0.375*e_std
    a4 = e_mean+0.375*e_std
    a5 = e_mean+0.75*e_std
    a6 = e_max
    if(a1 < a2 and a2 < a3 and a4 < a5 and a5 < a6):
        ebins = [a1,a2,a3,a4,a5,a6]
    else:
        de = (e_max-e_min)/8.
        a2 = e_min + 2.*de
        a3 = a2 + de
        a4 = a3 + 2.*de
        a5 = a4 + de
        ebins = [a1,a2,a3,a4,a5,a6]

    ecounts, tbins = np.histogram(ec,bins=ebins)

    #average ratio of extreme-a density to middle-a density
    if(ecounts[2] == 0):
        ecounts[2] = 1
    em_e = (ecounts[0] + ecounts[4])/(2.*ecounts[2])
    if(ecounts[4] == 0):
        ecounts[4] = 1
    #ratio of extreme-low-a density to extreme high-a density
    lh_e = (ecounts[0]/ecounts[4])

    #inclination distribution
    a1 = i_min
    a2 = i_mean-0.75*i_std
    a3 = i_mean-0.375*i_std
    a4 = i_mean+0.375*i_std
    a5 = i_mean+0.75*i_std
    a6 = i_max
    if(a1 < a2 and a2 < a3 and a4 < a5 and a5 < a6):
        ibins = [a1,a2,a3,a4,a5,a6]
    else:
        di = (i_max-i_min)/8.
        a2 =i_min + 2.*di
        a3 = a2 + di
        a4 = a3 + 2.*di
        a5 = a4 + di
        ibins = [a1,a2,a3,a4,a5,a6]

    icounts, tbins = np.histogram(inc,bins=ibins)
    #average ratio of extreme-a density to middle-a density
    if(icounts[2] == 0):
        icounts[2] = 1    
    em_i = (icounts[0] + icounts[4])/(2.*icounts[2])
    #ratio of extreme-low-a density to extreme high-a density
    if(ecounts[4] == 0):
        ecounts[4] = 1
    lh_i = (icounts[0]/icounts[4])


    features = [
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
        em_a,lh_a,min_em_a,max_em_a,delta_em_a,delta_em_a_norm,min_lh_a,max_lh_a,delta_lh_a,delta_lh_a_norm,
        em_e,lh_e,em_i,lh_i,
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
