def initialize_simulation_from_sv(planets=['mercury', 'venus', 'earth', 'mars',
                                   'jupiter', 'saturn', 'uranus', 'neptune'],
                          des='', clones=0, sb=[0,0,0,0,0,0,0]):
    """
    inputs:
        planets (optional): string list, list of planet names - defaults to all
        des: string, small body designation
        clones (optional): integer, number of clones - defaults to 0
        sb: list of floats, [epoch,x,y,z,vx,vy,vz] of the small body object. This is 
        effectively a way to start a simulation using data outside of Horizons.
    outputs:
        flag: integer, 0 if failed, 1 if successful
        epoch: float, date of the simulation start (JD)
        sim: rebound simulation instance with planets and test particles added
             with adjustments for missing major perturbers
    """
    
    # make all planet names lowercase
    planets = [pl.lower() for pl in planets]
    
    # initialize simulation variable
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')

    # set up small body variables
    ntp = 1 + clones
    sbx = np.zeros(ntp)
    sby = np.zeros(ntp)
    sbz = np.zeros(ntp)
    sbvx = np.zeros(ntp)
    sbvy = np.zeros(ntp)
    sbvz = np.zeros(ntp)

    # get the small body's position and velocity
    flag, epoch, sbx[:], sby[:], sbz[:], sbvx[:], sbvy[:], sbvz[:] = 1,sb[0],sb[1],sb[2],sb[3],sb[4],sb[5],sb[6]
    
    if(flag < 1):
        print("run_reb.initialize_simulation failed at horizons_api.query_sb_from_jpl")
        return 0, 0., sim
    
    # add the planets and return the position/velocity corrections for
    # missing planets
    #print(planets)
    apflag, sim, sx, sy, sz, svx, svy, svz = add_planets(sim, planets=planets,
                epoch=epoch)
    if(apflag < 1):
        print("run_reb.initialize_simulation failed at run_reb.add_planets")
        return 0, 0., sim
    
    maxpos = np.sqrt(sb[1]**2+sb[2]**2+sb[3]**2)
    maxvel = np.sqrt(sb[4]**2+sb[5]**2+sb[6]**2)
    
    if(clones > 0):    
        for i in range(1,ntp):
            sbx[i] += np.random.normal(0,maxpos*0.0001);
            sby[i] += np.random.normal(0,maxpos*0.0001);
            sbz[i] += np.random.normal(0,maxpos*0.0001);
            sbvx[i] += np.random.normal(0,maxvel*0.001);
            sbvy[i] += np.random.normal(0,maxvel*0.001);
            sbvz[i] += np.random.normal(0,maxvel*0.001);
            
            
        for i in range(0, ntp):
            if(i == 0):
                #first clone is always just the best-fit orbit
                #and the hash is not numbered
                sbhash = str(des)
            else:
                sbhash = str(des) + '_' + str(i)
            # correct for the missing planets
            #print(sbx,sby,sbz,sbvx,sbvy,sbz)
            sbx[i] += sx; sby[i] += sy; sbz[i] += sz
            sbvx[i] += svx; sbvy[i] += svy; sbvz[i] += svz
            sim.add(m=0., x=sbx[i], y=sby[i], z=sbz[i],
                    vx=sbvx[i], vy=sbvy[i], vz=sbvz[i], hash=sbhash)
    else:
        sbx += sx; sby += sy; sbz += sz
        sbvx += svx; sbvy += svy; sbvz += svz
        sbhash = str(des) 
        sim.add(m=0., x=sbx, y=sby, z=sbz,
                vx=sbvx, vy=sbvy, vz=sbvz, hash=sbhash)

    sim.move_to_com()

    return 1, epoch, sim

