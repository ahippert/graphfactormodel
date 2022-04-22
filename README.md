# Graph Learning meets regularized factor models

This repository contains code used to produce results showcased in the paper:
```
Graph Learning meets regularized factor models,
Florent Bouchard, Arnaud Breloy, Alexandre Hippert-Ferrer, Ammar Mian, Titouan Vayer
```
submitted at the conference NEURIPS 2022. The draft of the article is available on arXiv:
[url a remplir](url a remplir).


## Repository organisation
```
.
├── conda/
│   └── environment.yml
├── data/
│   ├── genomics
│   ├── networks
│   └── animal.mat
├── results/
│   ├── numerical/
│   │   └── benchmark_toydata.py
│   └── applications/
│       └── benchmark_animaldata.py
├── src/
│   ├── utils.py
│   ├── models.py
│   ├── optimization.py
│   ├── data.py
│   ├── pca.py
│   └── visualization.py
├── simulations/
│   ├── numerical
│   └── applications
├── install.sh
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
    python simulations/numerical/gaussian_model.py
    ```
    to run a numerical simulation with Gaussian distributed data.

## Simulations and paper results match

The following correspondance table shows the files (all situated in folder **simulations/**) used to produce the tables and figures presented in the paper:
| File                         | Produces Table | Produces Figure |
|------------------------------|----------------|-----------------|
| numerical/gaussian_model.py  |                | (1)             |
| blabla                       |                | (2), (3)        |
| blabla 2                     | (1), (2)       |                 |
| applications/genomic_data.py | (3)            |                 |

## Authors

For any issue, please contact: [mail a remplir](mail a remplir)

* Florent Bouchard
* Arnaud Breloy
* Alexandre Hippert-Ferrer
* Ammar Mian
* Titouan Vayer