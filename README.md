# Graph Learning meets regularized factor models

This repository contains code used to produce results showcased in the paper:
```
Graph Learning meets regularized factor models,
Florent Bouchard, Arnaud Breloy, Alexandre Hippert-Ferrer, Ammar Mian, Titouan Vayer
```

The draft of the article is available on arXiv:
[XXXX](XXXX).


## Repository organisation
```
.
├── conda/
│   └── environment.yml
├── data/
│   ├── gps_up.npy
|   ├── gps_up_2014_2022.npy
│   └── animal.mat
├── src/
│   ├── utils.py
│   ├── elliptical_estimation.py
│   ├── estimators.py
│   ├── manifolds.py
│   ├── models.py
│   ├── NGL.py
│   ├── sparse_penalties.py
│   ├── studentGL.py
│   └── visualization.py
├── simulations/
│   ├── ROC_curves.py
│   ├── benchmark_animal.py
│   ├── benchmark_models.py
│   ├── benchmark_concepts.py
│   └── graphical_models.py
├── install.sh
├── LICENCE
└── README.md
```

Most of the development is done as functions and classes available in the **./src/** directory. Simulations presented in the paper are available in the **./simulations/** directory. Data is sotred in the **./data/** directory.

## Pre-requesites

The simulationw were developped on linux system with python 3.10 with various libraries. In order to help with reproductibility matters, a conda environment was used. To obtain the same running environment, the best way is to install the same environment. 

It is then necessary to have either miniconda or Anaconda available:
* Miniconda: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
* Anaconda: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

1. Installing conda environment

    To install the environment, a bash script **install.sh** is given in the root of this directory. Simply run:
    ```
    sh install.sh
    ```

2. Activating conda environment and running a simulation

    To activate environment, run:
    ```
    conda activate graphfactormodels
    ```

    Then running a simulation is easily done using python. For example, run:
    ```
    python simulations/graphical_models.py
    ```
    to generate and visualize the considered graph structures.

## Simulations and paper results match

The following correspondance table shows the files (all situated in folder **simulations/**) used to produce the figures presented in the paper:
| File                            | Produces Figure             |
|---------------------------------|-----------------------------|
| simulations/graphical_models.py | (1) in Suppl. material      |
| simulations/ROC_curves.py       | (2)--(7) in Suppl. material |
| simulations/benchmark_animal.py | (1) in main paper           |
| simulations/benchmark_gps.py    | (2) in main paper           |
| simulations/benchmark_concepts  | (8), (9** in Suppl. material |

**Please see the documentation to know how to use the above scripts.**

## Authors

For any issue, please contact: [mail a remplir](mail a remplir)

* Florent Bouchard
* Arnaud Breloy
* Alexandre Hippert-Ferrer
* Ammar Mian
* Titouan Vayer
