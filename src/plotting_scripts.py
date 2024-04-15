import matplotlib.pyplot as plt
import numpy as np
import tools


def plot_aei(sbody = '',a=[[0.],], e=[[0.],], inc=[[0.]], t=[0.],nclones=0,
             figfile=None,bfps=1.0,cps=0.5,calpha=0.5,tmin=None,tmax=None):
    """
    Makes a plot of a, e, inc over time
    input:
        sbody (str): name of the small body
        a (1-d or 2-d float array): semimajor axis (au)        
        e (1-d or 2-d float array): eccentricity
        inc (1-d or 2-d float array): inclination (rad)
            all arrays are for the test particles in the 
            format [particle id number, output number]
            best-fit clone is id=0, clones are numbered
            starting at 1 
            if there are no clones, a 1-d array is fine
        time (1-d float array): simulations time (years)
        nclones (optional,int): number of clones of the best-fit orbit
        figfile (optional,str): path to save the figure to; if not set, the
            figure will not be saved but just displayed
        bfps (optional,float): matplotlib point size argument for best-fit orbit
        cps (optional,float): matplotlib point size argument for clone orbits
        calpha (optional,float): matplotlib alpha argument for clone orbits
        tmin (optional, float): minimum time for x-axis (years)
        tmax (optional, float): maximum time for x-axis (years)
    output:
        flag (int): 1 if successful and 0 if there was a problem
    """

    ntp = nclones+1

    if(nclones == 0 and len(a.shape)<2):
        #reshape the arrays since everything assumes 2-d
        a = np.array([a])
        e = np.array([e])
        inc = np.array([inc])
    #we will plot in degrees, so convert inc 
    rad_to_deg = 180./np.pi
    inc=inc*rad_to_deg


    nrows = 3

    if(nclones > 0):
        ncol = 2
        xwidth= 10
    else:
        ncol = 1
        xwidth= 5


    fig = plt.figure(figsize=(xwidth, 8))
    plt.subplots_adjust(left=None, bottom=None, right=None, 
                        top=0.92, wspace=0.35, hspace=0.25)
    
    plt.suptitle('object ' + sbody)
        
    if(tmin == None):
        tmin = t[0]
    if(tmax == None):
        tmax = t[-1]

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

    if(nclones > 0):

        a_ax2=plt.subplot2grid((nrows,ncol),(0,1))
        a_ax2.set_ylabel('a (au)')
        a_ax2.set_title(str(nclones) + ' clones')
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
        plt.savefig(figfile)

    flag = 1
    return flag, fig




def calc_and_plot_rotating_frame(sbody='',planet = '', archivefile='', nclones=0,
                                     figfile=None,bfps=1.0,cps=0.5,calpha=0.5,
                                     tmin=None, tmax=None):
    """
    Makes a plot of a small body in the rotating frame 
    input:
        sbody (str): name of the small body
        planet (str): name of the planet that sets the rotating frame
        archivefile (str): path to rebound simulation archive        
        nclones (optional,int): number of clones of the best-fit orbit
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


   

    #calculate the rotating frame values for the small body
    flag, xr, yr, zr, vxr, vyr, vzr, t = \
            tools.calc_rotating_frame(sbody=sbody, archivefile=archivefile,
                                      planet=planet, nclones=nclones,
                                      tmin=tmin,tmax=tmax)

    if(not flag):
        print("plotting_scripts.calc_and_plot_rotating_frame failed")
        print("couldn't get the rotating frame positions for the small body")
        return 0,None
    #calculate the rotating frame values for the planet
    flag, pxr, pyr, pzr, pvxr, pvyr, pvzr, t = \
            tools.calc_rotating_frame(sbody=planet, archivefile=archivefile,
                                      planet=planet, nclones=0,
                                      tmin=tmin,tmax=tmax)

    if(not flag):
        print("plotting_scripts.calc_and_plot_rotating_frame failed")
        print("couldn't get the rotating frame positions for the planet")
        return 0,None


    ntp = nclones+1

    
    nrows = 3
    ncol = 3

    fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(left=None, bottom=None, right=None, 
                        top=0.92, wspace=0.35, hspace=0.35)
    
        
    if(tmin == None):
        tmin = t[0]
    if(tmax == None):
        tmax = t[-1]

    deltat = tmax-tmin
    timelabel = "time (yr)"
    tscale =1.

    if(tmax >=1e4 and deltat>1e3):
        tscale = 1e3
        timelabel = "kyr "
    if(tmax >=1e6 and deltat>1e5):
        tscale = 1e6
        timelabel = "Myr "
    elif(tmax >=1e6 and deltat>1e4):
        tscale = 1e3
        timelabel = "kyr "
    if(tmax >1e9 and deltat > 1e8):
        tscale = 1e9
        timelabel = "Gyr "
    elif(tmax >1e9 and deltat > 1e6):
        tscale = 1e6
        timelabel = "Myr "
    elif(tmax >1e9 and deltat > 1e4):
        tscale = 1e3
        timelabel = "kyr "


    time1 = tmin/tscale
    time2 = tmax/tscale
    timestring = " from %1.4f to %1.4f " % (time1,time2)
    
    plt.suptitle('object ' + sbody + timestring + timelabel + "in "+planet+"'s rotating frame")


    

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


    if(nclones > 0):
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
        plt.savefig(figfile)

    flag = 1
    return flag, fig
