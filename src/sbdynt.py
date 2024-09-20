from horizons_api import *
from tools import *
from resonances import *
from run_reb import *
from plotting_scripts import *
from hard_coded_constants import *
from add_orbits import *
from machine_learning import *

# Globally disable two rebound-specific warnings that come up a lot regarding the 
# use of simulation archive functions
import warnings
warnings.filterwarnings("ignore", message="You have to reset function pointers after creating a reb_simulation struct with a binary file.")
warnings.filterwarnings("ignore", message="File in use for Simulationarchive already exists. Snapshots will be appended.")

# Globally disable a syntaxwarning that has to do with generating latex-friendly plot labele
#warnings.filterwarnings("ignore", category=SyntaxWarning)


class small_body:
    # class that is returned by the main sbdynt function
    # it contains all of the calculated dynamical parameters and 
    # other pertinent information about the simulations
    def __init__(self, designation, object_type=None, clones=0):
        self.designation = designation
        self.object_type = object_type
        self.clones = clones
        self.cloning_method = 'Gaussian'
        self.clone_weights = np.ones(clones+1)
        self.planets = None
        self.archivefile = None
        self.tno_ml_outputs = TNO_ML_outputs(self.clones)


            


    

