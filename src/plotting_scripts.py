import matplotlib.pyplot as plt
import numpy as np
import tools


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
        if(True):
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
        tmin = t[0]
    if(tmax == None):
        tmax = t[-1]
    #correct for backwards integrations
    if(tmax < tmin):
        temp = tmax
        tmax = tmin
        tmin = temp


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
    i_ax1.set_xlabel(timelabel)
    
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
    
    
    plt.suptitle('object ' + str(des) + " in " + planet + "'s rotating frame")


    

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
        ax1.scatter(xr,yr,s=bfps,c='k')
        ax2.scatter(xr,zr,s=bfps,c='k')
        ax3.scatter(yr,zr,s=bfps,c='k')

        ax4.scatter(vxr,vyr,s=bfps,c='k')
        ax5.scatter(vxr,vzr,s=bfps,c='k')
        ax6.scatter(vyr,vzr,s=bfps,c='k')


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
