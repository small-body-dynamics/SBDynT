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
    meandxdt = np.mean(dxdt, axis=0)
    maxdxdt = np.amax(dxdt, axis=0)
    deldxdt = maxdxdt - mindxdt

    stdev_norm = stdev/means
    dels_norm = dels/means
    deldxdt_norm = deldxdt/meandxdt

    # rearrange data into the order I want
    # arrs = [initials,finals,mins,means,maxes,stdev,dels,mindxdt,meandxdt,maxdxdt,deldxdt]
    arrs = [mins, means, maxes, stdev, stdev_norm, dels, dels_norm,
            mindxdt, meandxdt, maxdxdt, deldxdt, deldxdt_norm]
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
        temp = 0
    features += [temp]
    features += [rbinstd[0]]

    n = 5
    features += [nqempty[n]]
    features += [nadjempty[n]]
    if (not (rbinavg[n] == 0)):
        temp = (rbinmax[n] - rbinmin[n]) / rbinavg[n]
    else:
        temp = 0
    features += [temp]
    features += [rbinstd[n]]

    n = -2
    features += [nqempty[n]]
    features += [nadjempty[n]]
    if (not (rbinavg[n] == 0)):
        temp = (rbinmax[n] - rbinmin[n]) / rbinavg[n]
    else:
        temp = 0
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

    #calculate the correlations between a and e, a and i, and e and i

    temp =  max_corelation(data[:,1],data[:,2])
    features += [temp]
    temp =  max_corelation(data[:,1],data[:,3])
    features += [temp]
    temp =  max_corelation(data[:,2],data[:,3])
    features += [temp]

    #calculate spectral fractions for a, e, i
    temp1, temp2, temp3 = spectral_characteristics(data[:,1])
    features += [temp1]
    features += [temp2]
    features += [temp3]
    temp1, temp2, temp3 = spectral_characteristics(data[:,2])
    features += [temp1]
    features += [temp2]
    features += [temp3]
    temp1, temp2, temp3 = spectral_characteristics(data[:,3])
    features += [temp1]
    features += [temp2]
    features += [temp3]


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
        a2 =amin + 2.*da
        a3 = a2 + da
        a4 = a3 + 2.*da
        a5 = a4 + da
        abins = [a1,a2,a3,a4,a5,a6]
        
    acounts, tbins = np.histogram(data[:,1],bins=abins)

    #average ratio of extreme-a density to middle-a density
    temp = (acounts[0] + acounts[4])/(2.*acounts[2])
    features += [temp]

    #ratio of extreme-low-a density to extreme high-a density
    temp = (acounts[0]/acounts[4])
    features += [temp]


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
    temp = (ecounts[0] + ecounts[4])/(2.*ecounts[2])
    features += [temp]

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
    temp = (icounts[0] + icounts[4])/(2.*icounts[2])
    features += [temp]

    #ratio of extreme-low-a density to extreme high-a density
    temp = (icounts[0]/icounts[4])
    features += [temp]



    return np.array(features).reshape(1,-1)  # make sure features is a 2d array


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
    e_sb = np.zeros(lines)
    q_sb = np.zeros(lines)
    i_sb = np.zeros(lines)
    t_sb = np.zeros(lines)
    node_sb = np.zeros(lines)
    peri_sb = np.zeros(lines)
    rrf = np.zeros(lines)
    phirf = np.zeros(lines)

    j = 0
    for i in range(0, lines):
        pomega_sb[i] = (data_sb['node'][j] + data_sb['peri'][j])*np.pi/180.
        a_sb[i] = data_sb['a'][j]
        e_sb[i] = data_sb['e'][j]
        q_sb[i] = data_sb['a'][j]*(1.-data_sb['e'][j])
        i_sb[i] = data_sb['inc'][j]*np.pi/180.
        t_sb[i] = data_sb['t'][j]
        node_sb[i] = data_sb['node'][j]*np.pi/180.
        peri_sb[i] = data_sb['peri'][j]*np.pi/180.

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

    # Compute features
    features = ML_parse_features(data)
    # features = parse(data)

    return features

def max_corelation(d1, d2):
    d1 = (d1 - np.mean(d1)) / (np.std(d1))
    d2 = (d2 - np.mean(d2)) / (np.std(d2))  
    cmax = (np.correlate(d1, d2, 'full')/len(d1)).max()
    return cmax


def spectral_characteristics(data):
    Y = np.fft.rfft(data)
    jmax = len(Y)
    Y = Y[1:jmax]
    Y = np.abs(Y)**2.
    arr1 = Y.argsort()    
    sorted_Y = Y[arr1[::-1]]
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
    return sf, maxpower, max3
