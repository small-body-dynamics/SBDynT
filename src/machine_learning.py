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

    return np.array(features).reshape(1,-1)  # make sure features is a 2d array


def compute_ML_features_from_dbase_file(fname,kbo_id=1):
    '''
    Load data from file and calculate features
    fname=file name that contains simulation data; MUST contain outputs only every 1000 years, only contain lines from expected object
      Will only take first lines lines from fname, which should be times [0,1E3,2E3,...,99E3,100E3]
    col_order=indexes for columns: time, a (semi-major axis), eccentriciy, inclination, Omega (longitude of ascending node), omega (argument of pericenter)
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

        [xrf, yrf, zrf] = tools.rotating_frame_xyz(x=x, y=y, z=z,
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