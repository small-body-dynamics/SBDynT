import rebound
import numpy as np
# local
import tools
import run_reb

def calc_proper_elements(sbody='', archivefile='archive.bin', 
                         nclones=0, tmin=None,tmax=None,
                         datadir='./'):
    

    prop_e = 0.
    prop_sini = 0.
    prop_a = 0.

    flag, a, e, inc, node, aperi, ma, t = tools.read_sa_for_sbody(sbody = des, 
                        archivefile=archivefile,datadir=datadir,
                        nclones=nclones,tmin=tmin,tmax=tmax)
    if(flag == 0):
        print("proper_elements.calc_proper_elements failed when reading in the dada")
        return 0, prop_a, prop_e, prop_sini






    import hard_coded_constants as const
        # g1s1 -> g4s4 taken from Murray and Dermott SSD Table 7.1
    g1 = 5.46326/rev
    s1 = -5.20154/rev
    g2 = 7.34474/rev
    s2 = -6.57080/rev
    g3 = 17.32832/rev
    s3 = -18.74359/rev
    g4 = 18.00233/rev
    s4 = -17.63331/rev
    
    # g5s6 -> g8s8 taken directly from OrbFit software.
    g5 = 4.25749319/rev
    g6 = 28.24552984/rev
    g7 = 3.08675577/rev
    g8 = 0.67255084/rev
    s6 = -26.34496354/rev
    s7 = -2.99266093/rev
    s8 = -0.69251386/rev

    #print(g,s,g6,s6)
    z1 = abs(g+s-g6-s6)
    z2 = abs(g+s-g5-s7)
    z3 = abs(g+s-g5-s6)
    z4 = abs(g-2*g6+g5)
    z5 = abs(g-2*g6+g7)
    z6 = abs(s-s6-g5+g6)
    z7 = abs(g-3*g6+2*g5)
    z8 = abs(2*(g-g6)+s-s6)
    z9 = abs(3*(g-g6)+s-s6)


