# dockstring

![CI Tests](https://github.com/mgarort/dockstring/workflows/Install%20conda%20env%20and%20run%20pytest./badge.svg?branch=main)
![Code Style: yapf](https://img.shields.io/badge/code%20style-yapf-orange.svg)

This is a forked repo from the original [dockstring code](https://github.com/dockstring/dockstring), a python package for easy molecular docking and docking benchmarking. This code is modfied to incorporate automated docking of protein bining ligands from LLMs.

What's changed:
    - Incorporated timeout block (The code spends 15 minutes to try to find the docking site, after 15 minutes it restarts the docking process). The motivation behind this code block is to avoid the code getting stuck at a certain site and to accelerate docking process.
    - Prints out time taken to dock a single ligand to a protein target.


For details, see [paper](https://pubs.acs.org/doi/full/10.1021/acs.jcim.1c01334)
and [website](https://dockstring.github.io/):

> García-Ortegón, Miguel, et al. "DOCKSTRING: easy molecular docking yields better benchmarks for ligand design." Journal of Chemical Information and Modeling (2021).

## Installation
**Supported platforms:**
To install run `pip install .`

You might have to do `chmod +x dockstring/build/lib/dockstring/resources/bin/{VINA_FILE}` if you get `Permission_denied` error.
**Package versions:**

When installing dockstring, please be mindful of which package versions you install.
The dockstring dataset was created using:

- `rdkit=2021.03.3`
- `openbabel=3.1.1`

For rdkit installation: `conda install -c conda-forge rdkit`
For openbabel installation: `pip install openbabel-wheel`.


However, this will *not* install the dependencies because `openbabel` currently cannot be installed with pip.


1. Check whether the installation was successful by running a test script.
   Running without error indicates a successful install.
   ```bash
   python tutorials/simple_example.py
   ```
1. *(optional)* Install [PyMol](https://pymol.org/) for target, search box and ligand visualization:
   ```bash
   conda install -c conda-forge pymol-open-source
   ```
1. *(optional)* Check whether your local version of dockstring matches the dockstring dataset.
   This is only necessary if you plan to mix pre-computed docking scores from the dockstring dataset
   with locally-computed scores, or if you want to compare results with the dockstring paper.

   We have created a `pytest` test which randomly docks `N` molecules from the dockstring dataset
   and checks whether they match. The value of `N` can be changed by setting the environment variable
   `num_dockstring_test_molecules`. We recommend starting with `N=50`, then progressing to `N=1000`
   to do a full test. The test can be run with the following commands:
   ```bash
   conda install -c conda-forge pytest  # only if not installed already
   num_dockstring_test_molecules=1000 python -m pytest tests/test_dataset_matching.py  # change "1000" to the number you wish to dock
   ```
   If the test passes then your local version of docktring matches the dataset exactly! 🥳
   If the test does not pass, we encourage you to look how the error rate (this will be displayed in the error messages).
   If 99%+ of scores match then it is probably ok to use dockstring in the benchmarks, but there will of course be some error
   and this should be noted.



## Tutorials

- See dockstring's basic usage [here](tutorials/1_docking_risperidone_against_DRD2.ipynb).
- Learn how to visualize docking poses [here](tutorials/2_visualizing_dataset_poses.ipynb)

See [our website](https://dockstring.github.io/) for links to tutorials for
our dataset and benchmarks.
