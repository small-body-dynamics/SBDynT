import matplotlib.pyplot as plt
import numpy as np


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


    ncol = 3

    if(nclones > 0):
        nrows = 2
        xwidth= 10
    else:
        nrows = 1
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


    

    a_ax1=plt.subplot2grid((ncol,nrows),(0,0))
    a_ax1.set_ylabel('a (au)')
    a_ax1.set_title('best-fit orbit')
    e_ax1=plt.subplot2grid((ncol,nrows),(1,0))
    e_ax1.set_ylabel('e')
    i_ax1=plt.subplot2grid((ncol,nrows),(2,0))
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

        a_ax2=plt.subplot2grid((ncol,nrows),(0,1))
        a_ax2.set_ylabel('a (au)')
        a_ax2.set_title(str(nclones) + ' clones')
        e_ax2=plt.subplot2grid((ncol,nrows),(1,1))
        e_ax2.set_ylabel('e')
        i_ax2=plt.subplot2grid((ncol,nrows),(2,1))
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
    return flag
