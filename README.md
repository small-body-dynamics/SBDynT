# SBDynT
The Small Body Dynamics Tool (SBDynT) is an open-source python tool that can be used to easily investigate a solar system small body’s orbital evolution.

### SBDynT V1.0
SBDynT has reached its first release. Available functions are shown in the demonstration notebook in the example-notebooks directory. These include routines to easily run a standard set of dynamical analyses for main belt asteroids and transneptunian objects (TNOs). These routines initialize Rebound simualtions based on any observed small body, including producing clones that sample the small body's orbit-fit uncertainties as given by JPL's Small Body Database and automatically querying JPL Horizons for planet positions at the orbit-fit epoch. The Rebound simulations are then run over dynamically relevant timescales and the outputs are analyzed. The available analyses include synthetic proper orbital elements, stability indicators, and, for TNOs, detailed machine learning dynamical classifications including identification of mean motion resonant angles. The analyses can be run in an automated way with default choices, but they are also highly customizeable. Many of the available functions within SBDynT are highlighted in the demonstration notebooks in the example-notebooks directory. 

### Requirements and installation
Download the repository and either place it in your path or add the following to your python code where you will run SBDynT:<br>
import sys<br>
sys.path.insert(0,'path-to-where-you-downloaded-the-repository/SBDynT-main/src')<br>

The following packages must be installed (note that the required rebound package has its own dependencies):<br>
python -- Version 3.9+<br>
rebound -- Version 4+<br>
numpy<br>
scipy<br>
pandas<br>
matplotlib<br>
astroquery<br>
scikit-learn<br>
scikit-image<br>
pickle<br>
importlib<br>
os<br>
datetime<br>
pytest<br>

### Contact information
Kat Volk, kat.volk@gmail.com <br>
Dallin Spencer, dallinspencer@gmail.com

### Publications
When using any aspect of SBDynT, please cite Spencer et al., under review (will update with link upon paper acceptance)

The main TNO machine learning classifier is described in [Volk and Malhotra 2025](https://ui.adsabs.harvard.edu/abs/2025mlsm.book..173V/abstract). A follow-up paper is in the works to fully describe the resonant angle identification classifier.

The proper elements and chaos indicators are described in Spencer et al., under review

### Contributors
Code has been directly contributed to this repository by:
- [Kat Volk](https://github.com/katvolk)
- [Dallin Spencer](https://github.com/dallinspencer)

We have also adapted some code originally written by [Rachel Smullen](https://github.com/rsmullen) from the [KBO_Classifier](https://github.com/rsmullen/KBO_Classifier) repository

Development of these tools also include contributions from Renu Malhotra, Darin Ragozzine, Federica Spoto, Severance Graham, Henry Hsieh, and Marco Micheli.

### Funding Acknowledgements 
This work is supported by NASA grant 80NSSC23K0886 (formerly grant 80NSSC22K0512). Early development was supported by the Preparing for Astrophysics with LSST Program, funded by the Heising Simons Foundation through grant 2021-2975, and administered by Las Cumbres Observatory.
