
.. sbdynt documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to sbdynt's documentation!
========================================================================================

The Small Body Dynamics Tool (SBDynT) is an open-source python tool that can be used to easily investigate a solar system small body’s orbital evolution.

Installation
=================

Setup Your Conda Environment 
------------------------------

**Step 1** Create a conda or mamba environment.

If using conda::

   conda create -n sbdynt -c conda-forge "rebound<5" numpy numba pandas astropy scipy pandas matplotlib astroquery scikit-learn scikit-image ipykernel tqdm h5py importlib_resources python=3.11

If using mamba::

   mamba create -n sbdynt -c conda-forge "rebound<5" numpy numba pandas astropy scipy pandas matplotlib astroquery scikit-learn scikit-image ipykernel tqdm h5py importlib_resources python=3.11

.. tip::
   We recommend using python version 3.9 or higher with  ``SBDynT``. The conda/mamba install command uses python 3.11.

**Step 2** Activate your conda/mamba environment

On conda::

   conda activate sbdynt

On mamba::

   mamba activate sbdynt



Installing SBDynT
---------------------------------------------------------------------

**This is the installation method for adding/edit SBDynT's codebase, running unit tests, or working on/updating SBdynT's documentation.**


**Step 1** Download the ``SBDynT`` source code via::

   git clone https://github.com/small-body-dynamics/SBDynT

**Step 2** Navigate to the  ``SBDynT`` repository directory::

   cd SBDynT
  
**Step 3** Install an editable (in-place) development version of ``SBDynT``. This will allow you to run the code from the source directory.

If you just want the source code installed so edits in the source code are automatically installed::

   pip install -e .

If you are going to be doing significant software development, editing documentation, running unit tests, modifying unit tests, or manually running all of the example demo notebooks, you will need to install the full development version::

   pip install -e '.[dev]'

**Step 4 (Optional unless working on documentation):** You will need to install the pandoc package (either via conda/pip or `direct download <https://pandoc.org/installing.html>`


Dev Guide - Getting Started
---------------------------

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

.. code-block:: console

   >> conda create env -n <env_name> python=3.10
   >> conda activate <env_name>


Once you have created a new environment, you can install this project for local
development using the following commands:

.. code-block:: console

   >> pip install -e .'[dev]'
   >> pre-commit install
   >> conda install pandoc


Notes:

1) The single quotes around ``'[dev]'`` may not be required for your operating system.
2) ``pre-commit install`` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on
   `pre-commit <https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html>`_.
3) Installing ``pandoc`` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   `Sphinx and Python Notebooks <https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks>`_.


.. toctree::
   :hidden:

   Home page <self>
   API Reference <autoapi/index>
   Notebooks <notebooks>
