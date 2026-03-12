import matplotlib.pyplot as plt
import numpy as np
import stability_indicators as si


def plot_aei(des=None, datadir='', archivefile=None, 
             a=None, e=None, inc=None, t=None, clones=None,
             figfile=None,bfps=1.0,cps=0.5,calpha=0.5,
             tmin=None,tmax=None):
    """
    Makes a plot of a, e, inc over time
    input:
        des (str; optional if a,e,i,t provided): string, user-provided 
            small body designation
        archivefile (str; optional): name/path for the simulation
            archive file to be read if a,e,i,t provided. 
            If not provided, the default file name will be tried
        clones (optional): number of clones to read, 
            defaults to reading all clones in the simulation            
        datadir (optional): string, path for where files are stored,
            defaults to the current directory
        a (1-d or 2-d float array; optional): semimajor axis (au)        
        e (1-d or 2-d float array; optional): eccentricity
        inc (1-d or 2-d float array; optional): inclination (rad)
            all arrays are for the test particles in the 
            format [particle id number, output number]
            best-fit clone is id=0, clones are numbered
            starting at 1 
            if there are no clones, a 1-d array is used
            if these are not provided, they will be read from 
            archivefile
        time (1-d float array): simulations time (years)
            if this is not provided, it will be read from 
            archivefile
        figfile (optional,str): path to save the figure to; if not set, the
            figure will not be saved but just displayed
        bfps (optional,float): matplotlib point size argument for best-fit orbit
        cps (optional,float): matplotlib point size argument for clone orbits
        calpha (optional,float): matplotlib alpha argument for clone orbits
        tmin (optional, float): minimum time for x-axis (years)
        tmax (optional, float): maximum time for x-axis (years)
    output:
        flag (int): 1 if successful and 0 if there was a problem
        fig: matplotlib figure instance
    """
    flag = 0

    if(des == None):
        #see if orbital element arrays were provided. If so, we don't need des
        try:
            temp = len(a.shape)
        except:
            print("You must either pass orbital element arrays (a,e,i and time)")
            print("or a designation to this routine to generate plots")
            print("failed at plotting_scripts.plot_aei()")
            return flag, None
    else:
        try:
            temp = len(a.shape)
        except:
            #read in the orbital elements since they weren't provided
            rflag, a, e, inc, node, aperi, ma, t = tools.read_sa_for_sbody(des=des,
                                                datadir=datadir,archivefile=archivefile,
                                                tmax=tmax,tmin=tmin,clones=clones)
            if(rflag < 1):
                print("Could not generate arrays (a,e,i and time) for the provided")
                print("designation and/or archivefile")
                print("failed at plotting_scripts.plot_aei()")
                return flag, None
    
    if(len(a.shape)<2):
        #there aren't any clones
        if(clones==None):
            clones=0
        if(clones > 0):
            print("warning! plotting_scripts.plot_aei() was asked to plot")
            print("clones, but there are no clones in the archive file or")
            print("the provided arrays. Only the best fit will be plotted")
            clones = 0
            flag = 2
        ntp = 1
        #reshape the arrays since everything below assumes 2-d
        a = np.array([a])
        e = np.array([e])
        inc = np.array([inc])
    else:
        ntp = a.shape[0]
        if(clones==None):
            clones = ntp-1
        #if fewer clones were requested, adjust
        if(clones < ntp-1):
            ntp = clones+1


    rad_to_deg = 180./np.pi
    inc=inc*rad_to_deg


    nrows = 3

    if(clones > 0):
        ncol = 2
        xwidth= 10
    else:
        ncol = 1
        xwidth= 5


    fig = plt.figure(figsize=(xwidth, 8))
    plt.subplots_adjust(left=None, bottom=None, right=None, 
                        top=0.92, wspace=0.35, hspace=0.25)
    
    if(des != None):
        plt.suptitle('object ' + str(des))
        

    if(tmin == None):
        tmin = np.amin(t)
    if(tmax == None):
        tmax = np.amax(t)

    deltat = tmax-tmin
    timelabel = "time (yr)"
    tscale =1.
    #choose reasonable units for the time-axis
    if(deltat >= 1e9):
        tscale = 1e9
        timelabel = "time (Gyr)"
    elif(deltat >= 1e6):
        tscale = 1e6
        timelabel = "time (Myr)"
    elif(deltat >= 1e4):
        tscale = 1e3
        timelabel = "time (kyr)"


    a_ax1=plt.subplot2grid((nrows,ncol),(0,0))
    a_ax1.set_ylabel('a (au)')
    a_ax1.set_title('best-fit orbit')
    e_ax1=plt.subplot2grid((nrows,ncol),(1,0))
    e_ax1.set_ylabel('e')
    i_ax1=plt.subplot2grid((nrows,ncol),(2,0))
    i_ax1.set_ylabel('inc (deg)')
    try:
        i_ax1.set_xlabel(timelabel)
    except:
        print(tmax, tmin)
    a_ax1.set_xlim([tmin/tscale,tmax/tscale])
    e_ax1.set_xlim([tmin/tscale,tmax/tscale])
    i_ax1.set_xlim([tmin/tscale,tmax/tscale])


    #just the best fit on the left panels:
    a_ax1.scatter(t/tscale,a[0,:],s=bfps,c='k')
    e_ax1.scatter(t/tscale,e[0,:],s=bfps,c='k')
    i_ax1.scatter(t/tscale,inc[0,:],s=bfps,c='k')

    if(clones > 0):

        a_ax2=plt.subplot2grid((nrows,ncol),(0,1))
        a_ax2.set_ylabel('a (au)')
        a_ax2.set_title(str(clones) + ' clones')
        e_ax2=plt.subplot2grid((nrows,ncol),(1,1))
        e_ax2.set_ylabel('e')
        i_ax2=plt.subplot2grid((nrows,ncol),(2,1))
        i_ax2.set_ylabel('inc (deg)')
        i_ax2.set_xlabel(timelabel)
        a_ax2.set_xlim([tmin/tscale,tmax/tscale])
        e_ax2.set_xlim([tmin/tscale,tmax/tscale])
        i_ax2.set_xlim([tmin/tscale,tmax/tscale])

        #all the clones on the right panels
        for tp in range (1,ntp):
            a_ax2.scatter(t/tscale,a[tp,:],s=cps,alpha=calpha)
            e_ax2.scatter(t/tscale,e[tp,:],s=cps,alpha=calpha)
            i_ax2.scatter(t/tscale,inc[tp,:],s=cps,alpha=calpha)

    if(figfile != None):
        if(datadir):
            figfile = datadir + "/" + figfile
        plt.savefig(figfile)

    flag = 1
    return flag, fig




def calc_and_plot_rotating_frame(des=None, planet=None, archivefile=None, clones=None,
                                 datadir = '', figfile=None,
                                 bfps=0.1, cps=0.5,calpha=0.5,
                                 tmin=None, tmax=None):
    """
    Makes a plot of a small body in the rotating frame 
    input:
        des: string, user-provided small body designation
        planet (str): name of the planet that sets the rotating frame
        archivefile (str; optional): name/path for the simulation
            archive file to be read. If not provided, the default
            file name will be tried
        clones (optional): number of clones to read, 
            defaults to reading all clones in the simulation            
        datadir (optional): string, path for where files are stored,
            defaults to the current directory
        figfile (optional,str): path to save the figure to; if not set, the
            figure will not be saved but just displayed
        tmin (optional, float): minimum time (years)
        tmax (optional, float): maximum time (years)
            if not set, the entire time range is plotted           
        bfps (optional,float): matplotlib point size argument for best-fit orbit
        cps (optional,float): matplotlib point size argument for clone orbits
        calpha (optional,float): matplotlib alpha argument for clone orbits
    output:
        flag (int): 1 if successful and 0 if there was a problem
    """

    flag = 0

    if(des == None):
        print("You must pass a designation to this function")
        print("plotting_scripts.calc_and_plot_rotating_frame failed")
        return flag, None

    if(planet == None):
        print("You must pass a planet name to this function")
        print("plotting_scripts.calc_and_plot_rotating_frame failed")
        return flag, None
  

    if(archivefile==None):
        archivefile = tools.archive_file_name(des)
    if(datadir):
        archivefile = datadir + '/' +archivefile

    #calculate the rotating frame values for the small body
    rflag, xr, yr, zr, vxr, vyr, vzr, t = \
            tools.calc_rotating_frame(des=des, archivefile=archivefile,
                                      planet=planet, clones=clones,
                                      tmin=tmin,tmax=tmax)

    if(rflag<1):
        print("plotting_scripts.calc_and_plot_rotating_frame failed")
        print("couldn't get the rotating frame positions for the small body")
        return flag,None
    #calculate the rotating frame values for the planet
    rflag, pxr, pyr, pzr, pvxr, pvyr, pvzr, t = \
            tools.calc_rotating_frame(des=planet, archivefile=archivefile,
                                      planet=planet, clones=0,
                                      tmin=tmin,tmax=tmax)

    if(rflag<1):
        print("plotting_scripts.calc_and_plot_rotating_frame failed")
        print("couldn't get the rotating frame positions for the planet")
        return flag,None



    if(len(xr.shape)<2):
        #there aren't any clones
        if(clones==None):
            clones=0
        if(clones > 0):
            print("warning! plotting_scripts.plot_aei() was asked to plot")
            print("clones, but there are no clones in the archive file or")
            print("the provided arrays. Only the best fit will be plotted")
            clones = 0
            flag = 2
        ntp = 1
    else:
        ntp = xr.shape[0]
        if(clones==None):
            clones = ntp-1
        #if fewer clones were requested, adjust
        if(clones < ntp-1):
            ntp = clones+1
    
    nrows = 3
    ncol = 3

    fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(left=None, bottom=None, right=None, 
                        top=0.92, wspace=0.35, hspace=0.35)
    
    if(tmin == None):
        tmin = np.amin(t)
    if(tmax == None):
        tmax = np.amax(t)

    deltat = tmax-tmin

    timelabel = "time (yr)"
    tscale =1.

    if(tmax >=1e4 and deltat>1e3):
        tscale = 1e3
        timelabel = "time (kyr)"
    if(tmax >=1e6 and deltat>1e5):
        tscale = 1e6
        timelabel = "time (Myr)"
    elif(tmax >=1e6 and deltat>1e4):
        tscale = 1e3
        timelabel = "time (kyr)"
    if(tmax >1e9 and deltat > 1e8):
        tscale = 1e9
        timelabel = "time (Gyr)"
    elif(tmax >1e9 and deltat > 1e6):
        tscale = 1e6
        timelabel = "time (Myr)"
    elif(tmax >1e9 and deltat > 1e4):
        tscale = 1e3
        timelabel = "time (kyr)"

    time1 = tmin/tscale
    time2 = tmax/tscale
    timestring = " from %1.4f to %1.4f " % (time1,time2)
    
    plt.suptitle('object ' + str(des) + timestring + timelabel + "in "+planet+"'s rotating frame")


    ax1=plt.subplot2grid((nrows,ncol),(0,0))
    ax1.set_ylabel('y (au)')
    ax1.set_xlabel('x (au)')

    ax2=plt.subplot2grid((nrows,ncol),(0,1))
    ax2.set_ylabel('z (au)')
    ax2.set_xlabel('x (au)')

    ax3=plt.subplot2grid((nrows,ncol),(0,2))
    ax3.set_ylabel('z (au)')
    ax3.set_xlabel('y (au)')


    ax4=plt.subplot2grid((nrows,ncol),(1,0))
    ax4.set_ylabel('vy (au/year)')
    ax4.set_xlabel('vx (au/year)')

    ax5=plt.subplot2grid((nrows,ncol),(1,1))
    ax5.set_ylabel('vz (au/year)')
    ax5.set_xlabel('vx (au/year)')

    ax6=plt.subplot2grid((nrows,ncol),(1,2))
    ax6.set_ylabel('vz (au/year)')
    ax6.set_xlabel('vy (au/year)')


    if(clones > 0):
        for tp in range (1,ntp):
            ax1.scatter(xr[tp,:],yr[tp,:],s=cps,alpha=calpha)
            ax2.scatter(xr[tp,:],zr[tp,:],s=cps,alpha=calpha)
            ax3.scatter(yr[tp,:],zr[tp,:],s=cps,alpha=calpha)

            ax4.scatter(vxr[tp,:],vyr[tp,:],s=cps,alpha=calpha)
            ax5.scatter(vxr[tp,:],vzr[tp,:],s=cps,alpha=calpha)
            ax6.scatter(vyr[tp,:],vzr[tp,:],s=cps,alpha=calpha)

        ax1.scatter(xr[0,:],yr[tp,:],s=bfps,c='k')
        ax2.scatter(xr[0,:],zr[tp,:],s=bfps,c='k')
        ax3.scatter(yr[0,:],zr[tp,:],s=bfps,c='k')

        ax4.scatter(vxr[0,:],vyr[0,:],s=bfps,c='k')
        ax5.scatter(vxr[0,:],vzr[0,:],s=bfps,c='k')
        ax6.scatter(vyr[0,:],vzr[0,:],s=bfps,c='k')
    else:
        ax1.scatter(xr,yr,s=bfps,c='k',alpha=0.2)
        ax2.scatter(xr,zr,s=bfps,c='k',alpha=0.2)
        ax3.scatter(yr,zr,s=bfps,c='k',alpha=0.2)

        ax4.scatter(vxr,vyr,s=bfps,c='k',alpha=0.2)
        ax5.scatter(vxr,vzr,s=bfps,c='k',alpha=0.2)
        ax6.scatter(vyr,vzr,s=bfps,c='k',alpha=0.2)


    ax1.scatter(pxr,pyr,s=bfps,c='darkgrey')
    ax2.scatter(pxr,pzr,s=bfps,c='darkgrey')
    ax3.scatter(pyr,pzr,s=bfps,c='darkgrey')

    ax4.scatter(pvxr,pvyr,s=bfps,c='darkgrey')
    ax5.scatter(pvxr,pvzr,s=bfps,c='darkgrey')
    ax6.scatter(pvyr,pvzr,s=bfps,c='darkgrey')



    if(figfile != None):
        if(datadir):
            figfile = datadir + "/" + figfile        
        plt.savefig(figfile)

    flag = 1
    return flag, fig




###########################################################################################################################
# Proper Element Plotting Scripts
###########################################################################################################################

def plot_osc_and_prop(prop_elem):
    objname = prop_elem.des

    a = prop_elem.a_original
    an = prop_elem.a_filtered
    
    hk = prop_elem.hk_original
    pq = prop_elem.pq_original

    hkn = prop_elem.hk_filtered
    pqn = prop_elem.pq_filtered


    Ypq = np.fft.fft(pq)
    Ypqn = np.fft.fft(pqn)

    #Ypqn[0] = np.mean((Ypq[1],Ypq[-1]))

    #pqn = np.fft.ifft(Ypqn)
    #pq_new = pqn

    t = prop_elem.time/1e6
    

    fig,ax=plt.subplots(1,3,figsize=(12,4),sharex=True)
    fig.subplots_adjust(hspace=0,wspace=0.025)

    
    
    ax[0].plot(t[::1],a,alpha=0.55, label='Unfiltered Array')
    ax[0].plot(t[::1],an,alpha=0.55, label='Filtered Array')
    y0,y1 = ax[0].get_ylim()


    
    ax[1].plot(t[::1],np.abs(hk)[::1],alpha=0.55)
    ax[1].plot(t[::1],np.abs(hkn)[::1],alpha=0.55)
    y0,y1 = ax[1].get_ylim()

    c1 = 'darkgreen'
    c2 = 'darkgreen'
    c1 = 'darkred'
    c2 = 'darkred'

    ax[1].set_ylim(y0,y1)
    
    fig.supxlabel('Time (Myr)',y=0.025,fontsize=14)


    ax[2].plot(t,np.arcsin(np.abs(pq))*180/np.pi%90,alpha=0.55)
    ax[2].plot(t,np.arcsin(np.abs(pqn))*180/np.pi%90,alpha=0.55)


    y0,y1 = ax[1].get_ylim()
    y2,y3 = ax[2].get_ylim()

    
    ax[0].hlines(np.mean(an),xmin=t[0],xmax=t[-1],colors='k',ls='--')
    ax[1].hlines(np.mean(np.abs(hkn)),xmin=t[0],xmax=t[-1],colors='k',label='Proper Element',ls='--')
    ax[2].hlines(np.mean(np.arcsin(np.abs(pqn)))*180/np.pi,xmin=t[0],xmax=t[-1],colors='k',ls='--')
    
    ax[0].hlines(np.mean(a),xmin=t[0],xmax=t[-1],colors='r',ls='--',alpha=0.75)
    ax[1].hlines(np.mean(np.abs(hk)),xmin=t[0],xmax=t[-1],colors='r',label='Mean Element',ls='--',alpha=0.75)
    ax[2].hlines(np.mean(np.arcsin(np.abs(pq)))*180/np.pi,xmin=t[0],xmax=t[-1],colors='r',ls='--',alpha=0.75)

    
    


    #ax[0].legend(loc = 'lower right')
    ax[0].legend(loc = 'upper right')
    ax[1].legend(loc = 'upper left')

    y2,y3 = ax[2].get_ylim()
    ax[2].set_ylim(y2,y3)
    
    ax[2].set_ylabel(r'Inc ($^{\circ}$)',fontsize=14)

    ax[1].set_ylabel('Ecc',fontsize=14)
    ax[0].set_ylabel('SMA',fontsize=14)
    fig.suptitle('Small Body: '+str(objname),fontsize=16,x=0.52,y=0.94)
    import matplotlib.ticker as mticker
    formatter = mticker.StrMethodFormatter('{x:.3f}')
    ax[0].yaxis.set_major_formatter(formatter)
    ax[1].yaxis.set_major_formatter(formatter)
    ax[2].yaxis.set_major_formatter(formatter)
    ax[1].tick_params(axis='x', labelsize=11)
    ax[0].tick_params(axis='y', labelsize=11)
    ax[1].tick_params(axis='y', labelsize=11)
    ax[2].tick_params(axis='y', labelsize=11)

    fig.tight_layout()
    #plt.savefig('../data/results/'+str(objname)+'_eccinc_bf.pdf',transparent=True,bbox_inches='tight')
    plt.show()

    return 0

def plot_clone_osc(ci):
    t = ci.t / 1e6
    sb_elems = ci.sb_elems
    clone_elems = ci.clone_elems

    fig,ax = plt.subplots(2,3,figsize=(12,7),sharex=True)
    ax = ax.flatten()

    ax[0].plot(t, ci.sb_elems[0])
    ax[1].plot(t, ci.sb_elems[1])
    ax[2].plot(t, ci.sb_elems[2]*180/np.pi)

    for i in range(len(clone_elems)):
        ax[3].plot(t, ci.clone_elems[i,0], alpha=0.3)
        ax[4].plot(t, ci.clone_elems[i,1], alpha=0.3)
        ax[5].plot(t, ci.clone_elems[i,2]*180/np.pi, alpha=0.3)

    ax[0].set_title('SMA (AU)', fontsize=14)
    ax[1].set_title('Ecc', fontsize=14)
    ax[2].set_title('Inc ($^{\circ}$)', fontsize=14)

    ax[0].set_ylabel('Best-fit Orbit')
    ax[3].set_ylabel('Clone Orbits')

    fig.supxlabel('Time (Myr)', fontsize=14)
    fig.suptitle('Small Body - ' + str(ci.des), fontsize=16)

    fig.tight_layout()
    plt.show()
    
def plot_entropy(ci):
    a = ci.sb_elems[0]
    e = ci.sb_elems[1]
    t = ci.t
    
    hs = np.sqrt(a*(1-e**2))

    bins = int(len(a)/10)
    hs_nonan = hs[~np.isnan(hs)]
    t_nonan = t[~np.isnan(hs)]
    baseline = np.log10(bins)
    
    fig, ax = plt.subplots(1,2, figsize=(8,3))
    fig.subplots_adjust(wspace=0)
    ax[1].hist(hs_nonan, bins = bins, orientation='horizontal', weights=np.ones(len(hs_nonan))/len(hs_nonan))
    ax[0].plot(t_nonan, hs_nonan)

    fig.suptitle(str(ci.des) + ' Entropy=' + str(round(ci.Entropy, 2)))
    ax[0].set_ylabel(r'Specific angular momentum $h_s$')

    ax[1].set_yticks([])
    ax[1].set_yticklabels([])

    ax[0].set_xlabel('Time (Myr)')
    ax[1].set_xlabel('Density')
    plt.show()

def plot_ACFI(ci):

    
    fig, ax = plt.subplots(1,2, figsize=(3,8))
    
    return 

def plot_power(ci, pe_obj = None):

    e = ci.sb_elems[1]; I = ci.sb_elems[2]
    omega = ci.sb_elems[3]; Omega = ci.sb_elems[4]
    varpi = omega+Omega

    t = ci.t

    power, g, s, gs_dict = si.power_prop_calc(t, e, I, omega, Omega, size = 5, pe_obj = pe_obj)
    
    freq = np.fft.fftfreq(len(t), t[1]-t[0]); freqr = np.fft.rfftfreq(len(t), t[1]-t[0])

    gind = np.argmin(abs(freq - g))
    sind = np.argmin(abs(freq - s))

    hk = e*np.cos(varpi) + 1j*e*np.sin(varpi)
    pq = np.sin(I)*np.cos(Omega) + 1j*np.sin(I)*np.sin(Omega)
    
    

    Yhk = np.abs(np.fft.fft(hk))**2; Ypq = np.abs(np.fft.fft(pq))**2
    
    Ye = np.abs(np.fft.rfft(np.abs(hk)))**2; YI = np.abs(np.fft.rfft(np.abs(pq)))**2
    Yv = np.abs(np.fft.rfft(np.cos(np.angle(hk))))**2; YO = np.abs(np.fft.rfft(np.cos(np.angle(pq))))**2

    #Yhk_sorted = np.argsort(Yhk[1:])[::-1]; Ypq_sorted = np.argsort(Ypq[1:])[::-1]
    #top3_hk = np.sum(Yhk[Yhk_sorted[:3]+1]); top3_pq = np.sum(Ypq[Ypq_sorted[:3]+1])

    top_hk = np.sum(Yhk[gind-5:gind+6]); top_pq = np.sum(Ypq[sind-5:sind+6])

    total_hk = np.sum(Yhk); total_pq = np.sum(Ypq)
    
    
    alp = 0.3
    fig,ax = plt.subplots(1,2,figsize=(9,4), sharex = True)

    ax[0].scatter(1/freq, Yhk, s=1)
    ax[1].scatter(1/freq, Ypq, s=1)
    
    ax[0].scatter(1/freq[gind-5:gind+6], Yhk[gind-5:gind+6], s=3, c='tab:orange')
    ax[1].scatter(1/freq[sind-5:sind+6], Ypq[sind-5:sind+6], s=3, c='tab:orange')



    dt = abs(t[1]-t[0])
    ax[0].set_xscale('symlog', linthresh=dt, linscale=1e-2)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

        
    #ax[0].axhline(Yhk[0], ls='--', alpha=0.2,c='tab:blue', label=r'Power($\nu=0$)')
    #ax[1].axhline(Ypq[0], ls='--', alpha=0.2,c='tab:blue')
    
    ax[0].axhline(total_hk, ls='--', alpha=0.8,c='k', label=r'$\sum{Power}$')
    ax[1].axhline(total_pq, ls='--', alpha=0.8,c='k')
    
    ax[0].axhline(top_hk, ls='--', alpha=0.8,c='tab:orange', label=r'$\sum{Power_{ Proper}}$')
    ax[1].axhline(top_pq, ls='--', alpha=0.8,c='tab:orange')
    
    import matplotlib.ticker as ticker
    import math

    tsort = np.sort(np.abs(t))
    xmax = round(np.log10(dt*len(t)))+1
    xmin = round(np.log10(dt))

    xrange = 10**np.arange(xmin, xmax)
    
    xticks = np.concatenate((-xrange[::-1], xrange))
    ax[0].set_xticks(xticks)

    fig.supxlabel('Period (yrs)')

    ax[0].set_ylabel('Power')

    ax[0].set_title('hk Power')
    ax[1].set_title('pq Power')

    ax[0].legend()
    #ax[1].legend()
    fig.suptitle('Small Body: '+str(ci.des) + ', Proper Power=' + str(round(power, 2)*100) + '% of the Total Power',fontsize=16,x=0.52,y=0.94)
    fig.tight_layout()

    plt.show()

    
    return 


def plot_angles(prop_elem, plot_cos=False, ifreqs={}):
    objname = prop_elem.des

    a = prop_elem.a_original
    an = prop_elem.a_filtered
    
    hk = prop_elem.hk_original
    pq = prop_elem.pq_original

    hkn = prop_elem.hk_filtered
    pqn = prop_elem.pq_filtered


    Ypq = np.fft.fft(pq)
    Ypqn = np.fft.fft(pqn)

    t = prop_elem.time  
    dt = abs(t[1]-t[0])

    fig,ax=plt.subplots(2,2,figsize=(8,7))
    ax = ax.flatten()
    fig.subplots_adjust(hspace=0,wspace=0.025)

    varpi = np.angle(hk) % (2*np.pi)
    Omega = np.angle(pq) % (2*np.pi)
    omega = (varpi - Omega)%(2*np.pi)
    phi = (varpi + Omega)%(2*np.pi)
    
    varpin = np.angle(hkn) % (2*np.pi)
    Omegan = np.angle(pqn) % (2*np.pi)
    omegan = (varpin - Omegan)%(2*np.pi)
    phin = (varpin + Omegan)%(2*np.pi)

    g = prop_elem.proper_elements['g(rev/yr)']
    s = prop_elem.proper_elements['s(rev/yr)']
    
    g_s = prop_elem.proper_elements['g(rev/yr)'] - prop_elem.proper_elements['s(rev/yr)']
    gps = prop_elem.proper_elements['g(rev/yr)'] + prop_elem.proper_elements['s(rev/yr)']

    ind0 = np.argmin(abs(t))
    #ind0 = 0
    gt = g*t*2*np.pi % (2*np.pi) + varpin[ind0]
    st = s*t*2*np.pi % (2*np.pi) + Omegan[ind0]
    g_st = g_s*t*2*np.pi % (2*np.pi) + omegan[ind0]
    gpst = gps*t*2*np.pi % (2*np.pi) + phin[ind0]

    

    if plot_cos:
        ax[3].plot(t,np.cos(phi),alpha=0.55)
        ax[3].plot(t,np.cos(phin),alpha=0.55)
        ax[3].plot(t,np.cos(gpst % (2*np.pi)),alpha=0.35,ls='--')
    
        ax[2].plot(t,np.cos(omega),alpha=0.55)
        ax[2].plot(t,np.cos(omegan),alpha=0.55)
        ax[2].plot(t,np.cos(g_st % (2*np.pi)),alpha=0.35,ls='--')
    
        ax[1].plot(t,np.cos(Omega),alpha=0.55)
        ax[1].plot(t,np.cos(Omegan),alpha=0.55)
        ax[1].plot(t,np.cos(st % (2*np.pi)),alpha=0.35,ls='--')

        ax[0].plot(t,np.cos(varpi),alpha=0.55, label='Unfiltered Array')
        ax[0].plot(t,np.cos(varpin),alpha=0.55, label='Filtered Array')
        ax[0].plot(t,np.cos(gt % (2*np.pi)),alpha=0.35,ls='--', label='Reported precession rate')

        ax[3].set_title(r'$\cos(\phi) = \cos(\varpi + \Omega)$',fontsize=14)
        ax[2].set_title(r'$\cos(\omega)$',fontsize=14)

        ax[1].set_title(r'$\cos(\Omega)$',fontsize=14)
        ax[0].set_title(r'$\cos(\varpi)$',fontsize=14)

        for num, vals in ifreqs.items():
            freq = vals[1]
            label = vals[0]
            if num == 0:
                line = 2*np.pi*freq*t + varpin[ind0]
                ax[0].plot(t, np.cos(line), alpha=0.35,ls='--', label=label)
            if num == 1:
                line = 2*np.pi*freq*t + Omegan[ind0]
                ax[1].plot(t, np.cos(line), alpha=0.35,ls='--', label=label)
                ax[1].legend()
            if num == 2:
                line = 2*np.pi*freq*t + omegan[ind0]
                ax[2].plot(t, np.cos(line), alpha=0.35,ls='--', label=label)
                ax[2].legend()
            if num == 3:
                line = 2*np.pi*freq*t + phin[ind0]
                ax[3].plot(t, np.cos(line), alpha=0.35,ls='--', label=label)
                ax[3].legend()
                
                
    else:
        ax[3].plot(t,(phi),alpha=0.55)
        ax[3].plot(t,(phin),alpha=0.55)
        ax[3].plot(t,(gpst % (2*np.pi)),alpha=0.35,ls='--')
    
        ax[2].plot(t,(omega),alpha=0.55)
        ax[2].plot(t,(omegan),alpha=0.55)
        ax[2].plot(t,(g_st % (2*np.pi)),alpha=0.35,ls='--')
    
        ax[1].plot(t,(Omega),alpha=0.55)
        ax[1].plot(t,(Omegan),alpha=0.55)
        ax[1].plot(t,(st % (2*np.pi)),alpha=0.35,ls='--')

        ax[0].plot(t,(varpi),alpha=0.55, label='Unfiltered Array')
        ax[0].plot(t,(varpin),alpha=0.55, label='Filtered Array')
        ax[0].plot(t,(gt % (2*np.pi)),alpha=0.35,ls='--', label='Reported precession rate')

        ax[3].set_title(r'$\phi = \varpi + \Omega$',fontsize=14)
        ax[2].set_title(r'$\omega$',fontsize=14)

        ax[1].set_title(r'$\Omega$',fontsize=14)
        ax[0].set_title(r'$\varpi$',fontsize=14)
        
        for num, vals in ifreqs.items():
            freq = vals[0]
            label = vals[1]
            if num == 0:
                line = 2*np.pi*freq*t + varpin[ind0]
                ax[0].plot(t, line % (2*np.pi), alpha=0.35,ls='--', label=label)
            if num == 1:
                line = 2*np.pi*freq*t + Omegan[ind0]
                ax[1].plot(t, line % (2*np.pi), alpha=0.35,ls='--', label=label)
                ax[1].legend()
            if num == 2:
                line = 2*np.pi*freq*t + omegan[ind0]
                ax[2].plot(t, line % (2*np.pi), alpha=0.35,ls='--', label=label)
                ax[2].legend()
            if num == 3:
                line = 2*np.pi*freq*t + phin[ind0]
                ax[3].plot(t, line % (2*np.pi), alpha=0.35,ls='--', label=label)
                ax[3].legend()
        

    if abs(1/g) < len(t)*dt/10:
        glim1, glim2 = ax[0].get_xlim()
        if abs(1/g) < len(t)*dt/100:
            glim1 = glim1/20
            glim2 = glim2/20
        else:
            glim1 = glim1/2
            glim2 = glim2/2
        ax[0].set_xlim(glim1, glim2)
        
    if abs(1/s) < len(t)*dt/10:
        slim1, slim2 = ax[1].get_xlim()
        if abs(1/s) < len(t)*dt/100:
            slim1 = slim1/20
            slim2 = slim2/20
        else:
            slim1 = slim1/2
            slim2 = slim2/2
        ax[1].set_xlim(slim1, slim2)

    g_slim11, g_slim21 = ax[0].get_xlim()
    g_slim12, g_slim22 = ax[1].get_xlim()

    g_slim1 = np.mean((g_slim11, g_slim12))
    g_slim2 = np.mean((g_slim21, g_slim22))

    ax[2].set_xlim(g_slim1, g_slim2)
    
    fig.supxlabel('Time (yr)',y=0.025,fontsize=14)

    ax[0].legend(loc = 'upper right')
    #ax[1].legend(loc = 'upper left')

    fig.suptitle('Small Body: '+str(objname),fontsize=16)
    import matplotlib.ticker as mticker
    formatter = mticker.StrMethodFormatter('{x:.3f}')
    ax[0].yaxis.set_major_formatter(formatter)
    ax[1].yaxis.set_major_formatter(formatter)
    ax[2].yaxis.set_major_formatter(formatter)
    ax[1].tick_params(axis='x', labelsize=11)
    ax[0].tick_params(axis='y', labelsize=11)
    ax[1].tick_params(axis='y', labelsize=11)
    ax[2].tick_params(axis='y', labelsize=11)

    fig.tight_layout()
    #plt.savefig('../data/results/'+str(objname)+'_eccinc_bf.pdf',transparent=True,bbox_inches='tight')
    plt.show()

    return 0


def plot_freq_space(prop_elem, ifreqs={}):
    
    objname = prop_elem.des
    
    hk = prop_elem.hk_original
    pq = prop_elem.pq_original

    hkn = prop_elem.hk_filtered
    pqn = prop_elem.pq_filtered

    t = prop_elem.time

    freq = np.fft.fftfreq(len(t), t[1]-t[0])
    freqr = np.fft.rfftfreq(len(t), t[1]-t[0])

    Yhk = np.abs(np.fft.fft(hk))**2
    Ypq = np.abs(np.fft.fft(pq))**2
    Ye = np.abs(np.fft.rfft(np.abs(hk)))**2
    YI = np.abs(np.fft.rfft(np.abs(pq)))**2
    Yv = np.abs(np.fft.rfft(np.cos(np.angle(hk))))**2
    YO = np.abs(np.fft.rfft(np.cos(np.angle(pq))))**2

    
    Yhkn = np.abs(np.fft.fft(hkn))**2
    Ypqn = np.abs(np.fft.fft(pqn))**2
    Yen = np.abs(np.fft.rfft(np.abs(hkn)))**2
    YIn = np.abs(np.fft.rfft(np.abs(pqn)))**2
    Yvn = np.abs(np.fft.rfft(np.cos(np.angle(hkn))))**2
    YOn = np.abs(np.fft.rfft(np.cos(np.angle(pqn))))**2

    pf = prop_elem.planet_freqs

    #colors = rcParams['axes.prop_cycle'].by_key()['color']
    
    alp = 0.3
    if prop_elem.p_hkpq:

        fig,ax = plt.subplots(1,2,figsize=(9,4), sharex = True)

        ax[0].scatter(1/freq, Yhk, s=1, label='Unfiltered')
        ax[0].scatter(1/freq, Yhkn, s=1, label='Filtered')
    
        ax[1].scatter(1/freq, Ypq, s=1)
        ax[1].scatter(1/freq, Ypqn, s=1)

        ax[0].axvline(1/prop_elem.proper_elements['g(rev/yr)'], c='blue', ls='--', alpha=alp, label='g')
        ax[0].axvline(1/pf['g5'], c='r', ls='--', alpha=alp, label='g5')
        ax[0].axvline(1/pf['g6'], c='goldenrod', ls='--', alpha=alp, label='g6')
        ax[0].axvline(1/pf['g7'], c='g', ls='--', alpha=alp, label='g7')
        ax[0].axvline(1/pf['g8'], c='purple', ls='--', alpha=alp, label='g8')
        
        ax[1].axvline(1/prop_elem.proper_elements['s(rev/yr)'], c='blue', ls='--', alpha=alp, label='s')
        ax[1].axvline(1/pf['s6'], c='goldenrod', ls='--', alpha=alp, label='s6')
        ax[1].axvline(1/pf['s7'], c='g', ls='--', alpha=alp, label='s7')
        ax[1].axvline(1/pf['s8'], c='purple', ls='--', alpha=alp, label='s8')

        dt = abs(t[1]-t[0])
        ax[0].set_xscale('symlog', linthresh=dt, linscale=1e-2)
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')

        
        ax[0].axhline(Yhk[0], ls='--', alpha=0.2,c='tab:blue')
        ax[0].axhline(Yhkn[0], ls='--', alpha=0.2,c='tab:orange')
        ax[1].axhline(Ypq[0], ls='--', alpha=0.2,c='tab:blue')
        ax[1].axhline(Ypqn[0], ls='--', alpha=0.2,c='tab:orange')
        #ax[1].axhline(Ypqn[0], ls='--', alpha=0.2,c='k')

        
        for num, vals in ifreqs.items():
            freq = vals[1]
            label = vals[0]
            
            if num == 1:
                ax[0].axvline(1/freq, alpha=0.35,ls='-.', label=label)
                ax[0].legend()
                ax[1].axvline(1/freq, alpha=0.35,ls='-.', label=label)
                ax[1].legend()
    
        import matplotlib.ticker as ticker
        import math

        tsort = np.sort(np.abs(t))
        xmax = round(np.log10(dt*len(t)))+1
        xmin = round(np.log10(dt))

        xrange = 10**np.arange(xmin, xmax)
        #xticks = np.concatenate((-xrange[::-1], np.array([0])))
        #xticks = np.concatenate((xticks, xrange))

        xticks = np.concatenate((-xrange[::-1], xrange))
        
        #print(xticks)
        
        #thresh = 10**(math.ceil(np.log10(dt)))
        #locmin = ticker.SymmetricalLogLocator(linthresh=thresh, base=10)
        ax[0].set_xticks(xticks)

        fig.supxlabel('Period (yrs)')

        ax[0].set_ylabel('Power')

        ax[0].set_title('HK Power')
        ax[1].set_title('PQ Power')

        ax[0].legend()
        ax[1].legend()
        fig.suptitle('Small Body: '+str(objname),fontsize=16,x=0.52,y=0.94)
        fig.tight_layout()

        plt.show()

    
    if prop_elem.p_eI:
        fig,ax = plt.subplots(1,2,figsize=(9,4), sharex = True)
        ax = ax.flatten()

        ax[0].scatter(1/freqr, Ye, s=1, label='Unfiltered')
        ax[0].scatter(1/freqr, Yen, s=1, label='Filtered')
    
        ax[1].scatter(1/freqr, YI, s=1, label='Unfiltered')
        ax[1].scatter(1/freqr, YIn, s=1, label='Filtered')

        ax[0].axvline(abs(1/(prop_elem.proper_elements['g(rev/yr)'] - pf['g5'])), c='r', ls='--', alpha=alp, label='g-g5')
        ax[0].axvline(abs(1/(prop_elem.proper_elements['g(rev/yr)'] - pf['g6'])), c='goldenrod', ls='--', alpha=alp, label='g-g6')
        ax[0].axvline(abs(1/(prop_elem.proper_elements['g(rev/yr)'] - pf['g7'])), c='g', ls='--', alpha=alp, label='g-g7')
        ax[0].axvline(abs(1/(prop_elem.proper_elements['g(rev/yr)'] - pf['g8'])), c='purple', ls='--', alpha=alp, label='g-g8')
        
        ax[1].axvline(abs(1/(prop_elem.proper_elements['s(rev/yr)'] - pf['s6'])), c='goldenrod', ls='--', alpha=alp, label='s-s6')
        ax[1].axvline(abs(1/(prop_elem.proper_elements['s(rev/yr)'] - pf['s7'])), c='g', ls='--', alpha=alp, label='s-s7')
        ax[1].axvline(abs(1/(prop_elem.proper_elements['s(rev/yr)'] - pf['s8'])), c='purple', ls='--', alpha=alp, label='s-s8')
        
        ax[0].axhline(Ye[0], ls='--', alpha=0.2,c='tab:blue')
        ax[0].axhline(Yen[0], ls='--', alpha=0.2,c='tab:orange')
        ax[1].axhline(YI[0], ls='--', alpha=0.2,c='tab:blue')
        ax[1].axhline(YIn[0], ls='--', alpha=0.2,c='tab:orange')

        for num, vals in ifreqs.items():
            freq = vals[1]
            label = vals[0]
            
            if num == 2:
                ax[0].axvline(abs(1/freq), alpha=0.35,ls='-.', label=label)
                ax[0].legend()
                ax[1].axvline(abs(1/freq), alpha=0.35,ls='-.', label=label)
                ax[1].legend()
        
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        fig.supxlabel('Period (yrs)')
        fig.supylabel('Power')
        ax[0].set_title('Ecc Power')
        ax[1].set_title('Inc Power')
        ax[0].legend()
        ax[1].legend()
        fig.suptitle('Small Body: '+str(objname),fontsize=16,x=0.52,y=0.94)
        fig.tight_layout()
        plt.show()

    
    if prop_elem.p_vO:
        
        fig,ax = plt.subplots(1,2,figsize=(9,4), sharex = True)
        ax[0].scatter(1/freqr, Yv, s=1, label='Unfiltered')
        ax[0].scatter(1/freqr, Yvn, s=1, label='Filtered')
    
        ax[1].scatter(1/freqr, YO, s=1, label='Unfiltered')
        ax[1].scatter(1/freqr, YOn, s=1, label='Filtered')

        ax[0].axvline(1/prop_elem.proper_elements['g(rev/yr)'], c='blue', ls='--', alpha=alp, label='g')
        ax[0].axvline(1/pf['g5'], c='r', ls='--', alpha=alp, label='g5')
        ax[0].axvline(1/pf['g6'], c='goldenrod', ls='--', alpha=alp, label='g6')
        ax[0].axvline(1/pf['g7'], c='g', ls='--', alpha=alp, label='g7')
        ax[0].axvline(1/pf['g8'], c='purple', ls='--', alpha=alp, label='g8')
        
        ax[1].axvline(abs(1/prop_elem.proper_elements['s(rev/yr)']), c='blue', ls='--', alpha=alp, label='s')
        ax[1].axvline(abs(1/pf['s6']), c='goldenrod', ls='--', alpha=alp, label='s6')
        ax[1].axvline(abs(1/pf['s7']), c='g', ls='--', alpha=alp, label='s7')
        ax[1].axvline(abs(1/pf['s8']), c='purple', ls='--', alpha=alp, label='s8')
        
        
        ax[0].axhline(Yv[0], ls='--', alpha=0.2,c='tab:blue')
        ax[0].axhline(Yvn[0], ls='--', alpha=0.2,c='tab:orange')
        ax[1].axhline(YO[0], ls='--', alpha=0.2,c='tab:blue')
        ax[1].axhline(YOn[0], ls='--', alpha=0.2,c='tab:orange')
        
        for num, vals in ifreqs.items():
            freq = vals[1]
            label = vals[0]
            
            if num == 3:
                ax[0].axvline(abs(1/freq), alpha=0.35,ls='-.', label=label)
                ax[0].legend()
                ax[1].axvline(abs(1/freq), alpha=0.35,ls='-.', label=label)
                ax[1].legend()

        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')

        fig.supxlabel('Period (yrs)')
        fig.supylabel('Power')

        ax[0].set_title(r'$\varpi$ Power')
        ax[1].set_title(r'$\Omega$ Power')
        ax[0].legend()
        ax[1].legend()
        fig.suptitle('Small Body: '+str(objname),fontsize=16,x=0.52,y=0.94)
        fig.tight_layout()

        plt.show()
    
    return 0




def plot_hkpq(prop_elem):

    objname = prop_elem.des

    hk = prop_elem.hk_original
    pq = prop_elem.pq_original

    hkn = prop_elem.hk_filtered
    pqn = prop_elem.pq_filtered

    t = prop_elem.time

    fig,ax = plt.subplots(1,2, figsize=(10,4))

    ax[0].scatter(np.imag(hk), np.real(hk), s=1, alpha=0.2, label='Unfiltered')
    ax[0].scatter(np.imag(hkn), np.real(hkn), s=1, alpha=0.2, label='Filtered')
    
    ax[1].scatter(np.imag(pq), np.real(pq), s=1, alpha=0.2, label='Unfiltered')
    ax[1].scatter(np.imag(pqn), np.real(pqn), s=1, alpha=0.2, label='Filtered')

    ax[0].set_xlabel('h',fontsize=13)
    ax[0].set_ylabel('k',fontsize=13)
    
    ax[1].set_xlabel('p',fontsize=13)
    ax[1].set_ylabel('q',fontsize=13)

    ax[0].axvline(0,ls='--', c='grey', alpha=0.35)
    ax[0].axhline(0,ls='--', c='grey', alpha=0.35)
    ax[1].axvline(0,ls='--', c='grey', alpha=0.35)
    ax[1].axhline(0,ls='--', c='grey', alpha=0.35)


    per5 = int(len(hk)/20)
    ax[0].scatter(np.nanmedian(np.imag(hk[per5:-per5])),np.nanmedian(np.real(hk[per5:-per5])), marker='x', c='tab:blue')
    ax[0].scatter(np.nanmedian(np.imag(hkn[per5:-per5])),np.nanmedian(np.real(hkn[per5:-per5])), marker='x', c='tab:orange')
    
    ax[1].scatter(np.nanmedian(np.imag(pq[per5:-per5])),np.nanmedian(np.real(pq[per5:-per5])), marker='x', c='tab:blue')
    ax[1].scatter(np.nanmedian(np.imag(pqn[per5:-per5])),np.nanmedian(np.real(pqn[per5:-per5])), marker='x', c='tab:orange')
    fig.suptitle('Small Body: '+str(objname),fontsize=16,x=0.52,y=0.94)
    
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')

    plt.show()
    
    return 0


