import rebound
import numpy as np
import matplotlib.pyplot as plt
# local
import tools



#################################################################
#################################################################
# reads the simulation archive files into orbital element
# arrays necessary to produce a resonant plot
#################################################################
def read_sa_for_resonance(sbody = '', archivefile='',planet='',p=0,q=0,m=0,n=0,r=0,s=0,nclones=0):
    """
    Reads the simulation archive file produced by the run_reb
    routines
    If only p and q are input for the resonance integers, we 
    will assume an eccentricity-type resonance
    input:
        sbody (str): name of the small body
        archivefile (str): path to rebound simulation archive
        planet (str): hash for the planet the resonance is with
        p,q,m,n,r,s (int): the integers for the p:q resonance 
            with the planet where the resonance angle will be 
            phi = p*lambda_tp - q*lambda_planet - m*varpi_tp
                  -n*node_tp - r*varpi_planet - s*node_planet
        nclones (int): number of clones of the best-fit orbit
    output:
        ## if nclones=0, all arrays are 1-d    
        flag (int): 1 if successful and 0 if there was a problem
        a (1-d or 2-d float array): semimajor axis (au)
        e (1-d or 2-d float array): eccentricity
        inc (1-d or2-d float array): inclination (rad)
        node (1-d or 2-d float array): longitude of ascending node (rad)
        aperi (1-d or 2-d float array): argument of perihelion (rad)
        MA (1-d or 2-d float array): mean anomaly (rad)
        phi (1-d or 2-d float array): resonant angle (rad)
            above arrays are for the test particles in the 
            format [particle id number, output number]
            best-fit clone is id=0, clones are numbered
            starting at 1 
        time (1-d float array): simulations time (years)
        res_string (str): latex formatted string describing
            the resonance angle calculated
    """

    #check if any integers besides p and q were set. If not, 
    #calculate m for the eccentricity-type resonance. If they
    #are set, check for legal resonant argument
    if(m==0 and n==0 and r==0 and s==0):
        m = p-q
    else:
        checkint = p - q - m - n - r - s
        if(checkint != 0):
            print("The specified resonant integers are not allowed! They do not sum to zero!")
            return 0,[[0.],],[[0.],],[[0.],],[[0.],],[0.]
        node_sum = s+n
        if(node_sum % 2):
            print("The specified resonant integers are not allowed! The sum of nodes is odd!")
            return 0,[[0.],],[[0.],],[[0.],],[[0.],],[0.]


    #read the simulation archive and calculate resonant angles
    sa = rebound.SimulationArchive(archivefile)

    nout = len(sa)

    if(nout <1):
        print("resonances.read_sa_for_resonance failed")
        print("Problem reading the simulation archive file")
        return 0, [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], ''

    ntp = nclones+1


    a = np.zeros([ntp,nout])
    e = np.zeros([ntp,nout])
    inc = np.zeros([ntp,nout])
    node = np.zeros([ntp,nout])
    aperi = np.zeros([ntp,nout])
    ma = np.zeros([ntp,nout])    
    phi = np.zeros([ntp,nout])
    t = np.zeros(nout)

    for i,sim in enumerate(sa):
        #calculate the planet's orbit relative to the barycenter
        pl = sim.particles[planet]
        com = sim.calculate_com()
        o_pl = pl.calculate_orbit(com)
        #planet's mean longitude for the resonant angles
        lamda_pl = o_pl.Omega+o_pl.omega+o_pl.M

        t[i] = sim.t

        for j in range(0,ntp):
            #the hash format for clones
            tp_hash = sbody + "_" + str(j)
            #except the best fit is different:
            if(j==0):
                tp_hash = sbody + "_bf"
            #grab the particle and calculate its barycentric orbit
            try:
                tp = sim.particles[tp_hash]
            except:
                print("resonances.read_sa_for_resonance failed")
                print("Problem finding a particle with that hash in the archive")
                return 0, a, e, inc, node, aperi, ma, phi, t, ''
            o = tp.calculate_orbit(com)
            a[j,i] = o.a
            e[j,i] = o.e
            inc[j,i] = o.inc
            node[j,i] = o.Omega
            aperi[j,i] = o.omega
            ma[j,i] = o.M
            
            lamda = o.Omega+o.omega+o.M
            #calculate the resonant angle
            pt = float(p)*lamda - float(q)*lamda_pl - float(m)*(o.Omega+o.omega)
            if(n!=0 or r!=0 or s!=0):
                pt = pt - float(n)*o.Omega - float(r)*(o_pl.Omega+o_pl.omega) - float(s)*o_pl.Omega
            pt = tools.mod2pi(pt)
            phi[j,i] = pt
    
    flag = 1

    res_string = '$' + "\phi = " + str(p) + "\lambda - " + str(q) + "\lambda_{" + planet + "}"
    if(m!=0):
        res_string+= " - " + str(m) + "\\varpi "
    if(n!=0):
        res_string+= " - " + str(n) + "\Omega "
    if(r!=0):
        res_string+= " - " + str(r) + "\\varpi_{" + planet + "} "
    if(s!=0):
        res_string+= " - " + str(s) + "\Omega_{" + planet + "} "
    res_string+='$'


    if(nclones == 0):
        return flag, a[0,:], e[0,:], inc[0,:], node[0,:], aperi[0,:], ma[0,:], phi[0,:], t, res_string
    else:
        return flag, a, e, inc, node, aperi, ma, phi, t, res_string
#################################################################



#################################################################
#################################################################
# plots the resonance angle and a, e, i
#################################################################
def plot_resonance(sbody = '', res_string='',a=[[0.],], e=[[0.],], inc=[[0.],], phi=[[0.],], t=[0.],nclones=0,figfile=''):
    """
    Makes a plot of a, e, inc, phi for a resonance based on having
    read in the simulation archive using read_sa_for_resonance
    input:
        sbody (str): name of the small body
        res_string (str): the string that prints the resonant angle
        a (2-d float array): semimajor axis (au)        
        e (2-d float array): eccentricity
        inc (2-d float array): inclination (rad)
        phi (2-d float array): resonant angle (rad)
            all arrays are for the test particles in the 
            format [particle id number, output number]
            best-fit clone is id=0, clones are numbered
            starting at 1 
        time (1-d float array): simulations time (years)

        nclones (int): number of clones of the best-fit orbit
        figfile (str): path to save the figure to; if not set, the
            figure will not be saved but just displayed
    output:
        flag (int): 1 if successful and 0 if there was a problem
    """

    ntp = nclones+1

    if(nclones == 0 and len(a.shape)<2):
        #reshape the arrays since everything assumes 2-d
        a = np.array([a])
        e = np.array([e])
        inc = np.array([inc])
        phi = np.array([phi])
    #we will plot in degrees, so convert inc and phi
    rad_to_deg = 180./np.pi
    
    inc=inc*rad_to_deg
    phi=phi*rad_to_deg

    fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=0.92, wspace=0.35, hspace=0.25)
    
    plt.suptitle('object ' + sbody+ ' plotting '+ res_string)


    a_ax1=plt.subplot2grid((4,2),(0,0))
    a_ax1.set_ylabel('a (au)')
    a_ax2=plt.subplot2grid((4,2),(0,1))
    a_ax2.set_ylabel('a (au)')
    a_ax2.set_title(str(nclones) + ' clones')
    a_ax1.set_title('best-fit orbit')


    e_ax1=plt.subplot2grid((4,2),(1,0))
    e_ax1.set_ylabel('e')
    e_ax2=plt.subplot2grid((4,2),(1,1))
    e_ax2.set_ylabel('e')


    i_ax1=plt.subplot2grid((4,2),(2,0))
    i_ax1.set_ylabel('inc (deg)')
    i_ax2=plt.subplot2grid((4,2),(2,1))
    i_ax2.set_ylabel('inc (deg)')

    p_ax1=plt.subplot2grid((4,2),(3,0))
    p_ax1.set_ylabel('res angle (deg)')
    p_ax2=plt.subplot2grid((4,2),(3,1))
    p_ax2.set_ylabel('res angle (deg)')


    p_ax1.set_ylim([0,360])
    p_ax2.set_ylim([0,360])
    p_ax1.set_yticks(np.arange(0, 361, 60))
    p_ax2.set_yticks(np.arange(0, 361, 60))
    p_ax1.set_xlabel('time (Myr)')
    p_ax2.set_xlabel('time (Myr)')

    tmax = t[-1]

    a_ax1.set_xlim([0,tmax/1e6])
    a_ax2.set_xlim([0,tmax/1e6])
    e_ax1.set_xlim([0,tmax/1e6])
    e_ax2.set_xlim([0,tmax/1e6])
    i_ax1.set_xlim([0,tmax/1e6])
    i_ax2.set_xlim([0,tmax/1e6])
    p_ax1.set_xlim([0,tmax/1e6])
    p_ax2.set_xlim([0,tmax/1e6])


    #just the best fit on the left panels:
    a_ax1.scatter(t/1e6,a[0,:],s=0.01,c='k')
    e_ax1.scatter(t/1e6,e[0,:],s=0.01,c='k')
    i_ax1.scatter(t/1e6,inc[0,:],s=0.01,c='k')
    p_ax1.scatter(t/1e6,phi[0,:],s=0.01,c='k')

    #all the clones on the right panels
    for tp in range (1,ntp):
        a_ax2.scatter(t/1e6,a[tp,:],s=0.01,alpha=0.5)
        e_ax2.scatter(t/1e6,e[tp,:],s=0.01,alpha=0.5)
        i_ax2.scatter(t/1e6,inc[tp,:],s=0.01,alpha=0.5)
        p_ax2.scatter(t/1e6,phi[tp,:],s=0.01,alpha=0.5)

    if(figfile != ''):
        plt.savefig(figfile)

    flag = 1
    return flag



def analyze_res(tmin=0.,tmax=0.,dtwindow = 5e5,a=[[0.],], e=[[0.],], inc=[[0.],], phi=[[0.],], t=[0.],nclones=0):
    """
    Analyzes the time-series data for resonant libration
    input:
        tmin (float): start time for the analysis
        tmax (float): end time for the analysis
        dtwindow (float): length of the window to check for phi libration
                          should be several libration timescales
        a (2-d float array): semimajor axis (au)        
        e (2-d float array): eccentricity
        inc (2-d float array): inclination (rad)
        phi (2-d float array): resonant angle (rad)
            all arrays are for the test particles in the 
            format [particle id number, output number]
            best-fit clone is id=0, clones are numbered
            starting at 1 
        time (1-d float array): simulations time (years)
        nclones (int): number of clones of the best-fit orbit
    output:
        flag (int): 1 if successful and 0 if there was a problem
                    2 if there were non-ideal conditions for the analysis
                    (mixed time output intervals, relatively few points/window)
        a_stats (2-d float array): semimajor axis statistics
        e_stats (2-d float array): eccentricity statistics
        i_stats (2-d float array): inclination statistics
            (a,e,i)_stats are 2-d arrays. The first index is the 
            particle number, the second indicies give:
                0: average value from tmin to tmax
                1: maximum difference in the average value within all windows
                2: average of the standard deviation within all windows
                3: maximum difference in the standard deviation within all windows
        phi_stats (2-d float array): resonant angle statistics. The first index
            is the particle number, the second index gives:
                0: average phi value from tmin to tmax (centered 0-360)
                1: maximum difference in the average value within all windows
                2: average of (max_phi - min_phi) within all windows
                3: standard deviation of (max_phi - min_phi) within all windows

    """
    
    flag = 1
    ntp = nclones+1
    deg_to_rad = np.pi/180.

    if(nclones == 0 and len(a.shape)<2):
        #reshape the arrays since everything assumes 2-d
        a = np.array([a])
        e = np.array([e])
        inc = np.array([inc])
        phi = np.array([phi])

    
    #(a,e,i)_stats is a 2-d array containing the analysis
    #of (a,e,i) over the windows. The first index is the 
    #particle number, the second indicies are as follows:
    # 0: average value from tmin to tmax
    # 1: maximum difference in the average value within all windows
    # 2: average of the standard deviation within all windows
    # 3: maximum difference in the standard deviation within all windows
    a_stats = np.zeros([ntp,4]) 
    e_stats = np.zeros([ntp,4]) 
    i_stats = np.zeros([ntp,4]) 
    
    #for phi_stats, the first index is the particle id, 
    #and the second indicies are:
    # 0: average value from tmin to tmax
    # 1: maximum difference in the average value within all windows
    # 2: average of (max_phi - min_phi) within all windows
    # 3: standard deviation of (max_phi - min_phi) within all windows
    phi_stats = np.zeros([ntp,4]) 

    #fraction of windows with (max_phi - min_phi)<dpmax
    fwindows = np.zeros(ntp)
    dpmax = 350.*deg_to_rad #can be lowered to 340 degrees if time sampling isn't high enough
    
    #do a bunch of checks to make sure tmin, tmax, and dtwindow are sensible
    nout = len(t)
    if(nout < 1000):
        print("not enough outputs to do a resonance analysis (need >1000)")
        return 0,a_stats, e_stats, i_stats, phi_stats, fwindows
    
    i_t0 = -1
    i_tf = -1
    if(tmin == t[0]):
        i_t0=0
    elif(t[0] > tmin):
        print("tmin is less than t[0], starting at tmin=%f instead" % t[0])
        i_t0=0
    
    if(np.abs(tmax-t[-1])<1):
        i_tf = nout-1
    elif(t[-1] < tmax):
        print("tmax is greater than t[-1], starting at tmax=%f instead" % t[-1])
        i_tf = nout-1
        
    for i in range(1,nout):
        if(i_t0<0 and t[i]==tmin):
            i_t0 = i
            dt0 = t[i+1]-t[i]
        elif(i_t0<0 and t[i]>tmin and t[i-1]<tmin):
            i_t0 = i
        if(i_tf<0 and t[i]==tmax):
            i_tf = i
        elif(i_tf<0 and t[i]>tmax and t[i-1]<tmax):
            i_tf = i
    
    nout = len(t[i_t0:i_tf])
    if(nout < 300):
        print("not enough outputs between tmin and tmax to do a resonance analysis (need >300)")
        return 0, a_stats, e_stats, i_stats, phi_stats, fwindows
    
    dt0 = t[i_t0+1] - t[i_t0]
    #start and end of analysis window
    i_win1 = i_t0
    n_windows = 0
    sampling_warning = 0
    nwindows_warning = 0
    for i in range(i_t0,i_tf+1):
        dt = t[i]-t[i-1]
        deltadt = np.abs(dt-dt0)/dt0
        if(deltadt > 0.015 and i>i_t0 and sampling_warning==0):
            print("caution: uneven time sampling for the windows")
            flag = 2
            sampling_warning = 1
        twindow = t[i] - t[i_win1]
        if(twindow >= dtwindow or np.abs(twindow-dtwindow)<5):
            n_windows+=1
            n_points = i-i_win1
            if(n_points<150 and nwindows_warning==0):
                print("caution: at least one window has <150 points")
                print("lowering libration threshold to 340 degrees as a result")
                dpmax = 340.*deg_to_rad
                flag = 2
                nwindows_warning=1
            if(n_points<80):
                print("window time is too short for the dataset (one window has <80 points)")
                return 0, a_stats, e_stats, i_stats, phi_stats, fwindows   
            i_win1 = i
    if(n_windows < 10):
        print("caution: fewer than 10 distinct windows across the time sample")
        flag = 2
    if(n_windows < 3):
        print("not enough windows to do a meaningful analysis")
        return 0.,a_stats, e_stats, i_stats, phi_stats, fwindows 
    #end checks of the time sampling        
        
    #start the window analysis
    for tp in range (0,ntp):
        abar = np.mean(a[tp,i_t0:i_tf])
        ebar = np.mean(e[tp,i_t0:i_tf])
        ibar = np.mean(inc[tp,i_t0:i_tf])
        i_win1 = i_t0
        nwin = 0
        libwin = 0
        wabar = np.zeros(n_windows)
        webar = np.zeros(n_windows)
        wibar = np.zeros(n_windows)
        wastd = np.zeros(n_windows)
        westd = np.zeros(n_windows)
        wistd = np.zeros(n_windows)
        wphibar = np.zeros(n_windows)
        wdphi = np.zeros(n_windows)

        for i in range(i_t0,i_tf+1):
            twindow = t[i] - t[i_win1]
            if(twindow >= dtwindow or np.abs(twindow-dtwindow)<5):
                #calculate average and standard deviations
                #for a, e, inc
                wabar[nwin] = np.mean(a[tp,i_win1:i])
                webar[nwin] = np.mean(e[tp,i_win1:i])
                wibar[nwin] = np.mean(inc[tp,i_win1:i])
                wastd[nwin] = np.std(a[tp,i_win1:i])
                westd[nwin] = np.std(e[tp,i_win1:i])
                wistd[nwin] = np.std(inc[tp,i_win1:i])
                
                
                #make a temporary new array with the phi values from the window
                wphi = phi[tp,i_win1:i].copy()
                #sort the phi values
                wphi.sort()
                #find the differences between all adjacent phi values
                dphi = np.zeros(len(wphi))
                dphi[1::] = np.diff(wphi)
                #grab the difference between the first and last entry
                #corrected for the wrapping
                dphi[0] = wphi[0] - wphi[-1] + 2.*np.pi 
                j = np.argmax(dphi)
                if(dphi[j]>=(2.*np.pi-dpmax)):
                    #librating
                    libwin+=1
                    if(j>0): 
                        #libration not centered within 0-360
                        #recenter sorted phi to account for that
                        for k in range (0,j):
                            wphi[k] = wphi[k] + 2.*np.pi
                        wphi.sort()
                    #average phi in the window
                    wphibar[nwin] = np.mean(wphi)
                    if(wphibar[nwin] >2.*np.pi):
                        wphibar[nwin]= wphibar[nwin]-2.*np.pi
                    #delta phi in the window (slightly weighted to account for outliers)
                    wdphi[nwin] = (2.*wphi[-1] + wphi[-2] - 2.*wphi[0] - wphi[1])/3.

                nwin+=1
                i_win1 = i
            #end window analysis
        #end time loop for a test particle
        fwindows[tp] = float(libwin/n_windows)
        if(libwin>0):
            #trim out the non-librating windows to 
            #get the average and standard deviations
            wphibar.sort()
            wphibar = np.trim_zeros(wphibar)
            phi_stats[tp,0] = np.mean(wphibar)
            phi_stats[tp,1] = wphibar[-1] - wphibar[0]
            wdphi.sort()
            wdphi = np.trim_zeros(wdphi)
            phi_stats[tp,2] = np.mean(wdphi)
            phi_stats[tp,3] = np.std(wdphi)
        else:
            #set the values to -5 to signify no libration stats
            phi_stats[tp] = [-5.,-5.,-5.,-5]
        
        a_stats[tp,0] = abar
        wabar.sort()
        a_stats[tp,1] = wabar[-1] - wabar[0]
        wastd.sort()
        a_stats[tp,2] = np.mean(wastd)
        a_stats[tp,3] = wastd[-1] - wastd[0]

        e_stats[tp,0] = ebar
        webar.sort()
        e_stats[tp,1] = webar[-1] - webar[0]
        westd.sort()
        e_stats[tp,2] = np.mean(westd)
        e_stats[tp,3] = westd[-1] - westd[0]    
    
        i_stats[tp,0] = ibar
        wibar.sort()
        i_stats[tp,1] = wibar[-1] - wibar[0]
        wistd.sort()
        i_stats[tp,2] = np.mean(wistd)
        i_stats[tp,3] = wistd[-1] - wistd[0]    
    
    #end test particle loop
             
    
    return flag, a_stats, e_stats, i_stats, phi_stats, fwindows
