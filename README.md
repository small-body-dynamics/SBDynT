# SBDynT
We are developing a well-documented, open-source Python package, the Small Body Dynamics Tool (SBDynT), that can be used to calculate a variety of dynamical parameters and determine dynamical characterizations and classifications from integrations of a solar system small body’s orbital evolution.

### Current state of development
We are in the very early stages of development, primarily focusing on adding individual functions that will later be incoporated into a larger work-flow. Functions currently available are:
- query_horizons_api_planets(obj=<planet>,epoch=<JD date>) -> returns flag, mass, radius, heliocentric x, y, z (au), heliocentricvx, vy, vz (au/time)
- query_sb_from_jpl(des=<designation>,clones=0) -> returns flag, epoch, arrays of heliocentric x, y, z (au), heliocentricvx, vy, vz (au/time) where index 0 is the best fit and higher indices are clones (if clones>0) sampled from the covariance matrix in JPL's small body database browser


### Funding Acknowledgements 
This work is supported in part by the Preparing for Astrophysics with LSST Program, funded by the Heising Simons Foundation through grant 2021-2975, and administered by Las Cumbres Observatory.
