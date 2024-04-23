# SBDynT
We are developing a well-documented, open-source Python package, the Small Body Dynamics Tool (SBDynT), that can be used to calculate a variety of dynamical parameters and determine dynamical characterizations and classifications from integrations of a solar system small body’s orbital evolution.

### Current state of development
We are in the very early stages of development, primarily focusing on adding individual functions that will later be incoporated into a larger work-flow. Functions currently available are shown in the demonstration notebook in the example-notebooks directory. These include already useful tools to generate clones of an observed small body's orbit over its orbit uncertainties using JPL's orbit fit and associated covariance matrix and to initialize and run rebound simulations for those clones.

### Requirements and installation
Currently, to use SBDynT, download the repository and add the following to your python code:<br>
import sys<br>
sys.path.insert(0,'path-to-where-you-downloaded-the-repository/SBDynT-main/src')<br>

The following packages must be installed:<br>
rebound -- Version 4+<br>
numpy<br>
astroquery

### Contact information
Kat Volk, kat.volk@gmail.com

### Contributors
Code has been directly contributed to this repository by:
- [Kat Volk](https://github.com/katvolk)
- [Dallin Spencer](https://github.com/dallinspencer)

We have also modified code originally written by [Rachel Smullen](https://github.com/rsmullen) from the [KBO_Classifier](https://github.com/rsmullen/KBO_Classifier) repository

Development of these tools also include contributions from Renu Malhotra, Darin Ragozzine, Federica Spoto, Severance Graham, Henry Hsieh, and Marco Micheli.

### Funding Acknowledgements 
This work is supported by NASA grant 80NSSC23K0886 (formerly grant 80NSSC22K0512). Early development was supported by the Preparing for Astrophysics with LSST Program, funded by the Heising Simons Foundation through grant 2021-2975, and administered by Las Cumbres Observatory.
