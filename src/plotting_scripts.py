import matplotlib.pyplot as plt
import numpy as np


def plot_aei(sbody = '',a=[[0.],], e=[[0.],], inc=[[0.]], t=[0.],nclones=0,figfile='',bfps=0.1,cps=0.05):
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
        bfps (optional,floag): matplotlib point size argument for best-fit orbit
        cps (optional,floag): matplotlib point size argument for clone orbit
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
    plt.subplots_adjust(left=None, bottom=None, right=None, top=0.92, wspace=0.35, hspace=0.25)
    
    plt.suptitle('object ' + sbody)
        
    tmax = t[-1]

    a_ax1=plt.subplot2grid((ncol,nrows),(0,0))
    a_ax1.set_ylabel('a (au)')
    a_ax1.set_title('best-fit orbit')
    e_ax1=plt.subplot2grid((ncol,nrows),(1,0))
    e_ax1.set_ylabel('e')
    i_ax1=plt.subplot2grid((ncol,nrows),(2,0))
    i_ax1.set_ylabel('inc (deg)')
    i_ax1.set_xlabel('time (Myr)')
    
    a_ax1.set_xlim([0,tmax/1e6])
    e_ax1.set_xlim([0,tmax/1e6])
    i_ax1.set_xlim([0,tmax/1e6])


    #just the best fit on the left panels:
    a_ax1.scatter(t/1e6,a[0,:],s=bfps,c='k')
    e_ax1.scatter(t/1e6,e[0,:],s=bfps,c='k')
    i_ax1.scatter(t/1e6,inc[0,:],s=bfps,c='k')

    if(nclones > 0):

        a_ax2=plt.subplot2grid((ncol,nrows),(0,1))
        a_ax2.set_ylabel('a (au)')
        a_ax2.set_title(str(nclones) + ' clones')
        e_ax2=plt.subplot2grid((ncol,nrows),(1,1))
        e_ax2.set_ylabel('e')
        i_ax2=plt.subplot2grid((ncol,nrows),(2,1))
        i_ax2.set_ylabel('inc (deg)')
        i_ax2.set_xlabel('time (Myr)')
        a_ax2.set_xlim([0,tmax/1e6])
        e_ax2.set_xlim([0,tmax/1e6])
        i_ax2.set_xlim([0,tmax/1e6])

        #all the clones on the right panels
        for tp in range (1,ntp):
            a_ax2.scatter(t/1e6,a[tp,:],s=cps,alpha=0.5)
            e_ax2.scatter(t/1e6,e[tp,:],s=cps,alpha=0.5)
            i_ax2.scatter(t/1e6,inc[tp,:],s=cps,alpha=0.5)

    if(figfile != ''):
        plt.savefig(figfile)

    flag = 1
    return flag
