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
|
│── ROC_curves.py
│── benchmark_animal.py
│── benchmark_gps.py
│── benchmark_concepts.py
│── graphical_models.py
|
├── install.sh
├── LICENCE
└── README.md
```

Most of the development is done as functions and classes available in the **./src/** directory. Simulations presented in the paper can be reproduced using the scripts **ROC_curves.py, benchmark_animal.py, benchmark_gps.py, benchmark_concepts.py**. Data is stored in the **./data/** directory.

## Pre-requesites

The simulationw were developped on linux system with python 3.10 with various libraries. In order to help with reproductibility matters, a conda environment was used. To obtain the same running environment, the best way is to install the same environment. 

It is then necessary to have either miniconda or Anaconda available:
* Miniconda: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
* Anaconda: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

1. Installing conda environment

    To install the environment, a bash script **install.sh** is given in the root of this directory. Simply run:
 
    ```
    ./install.sh
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
    to generate and visualize the considered synthetic graph structures in the paper.

## Simulations and paper results match

The following correspondance table shows the files used to produce figures presented in the paper:

| File                  | Produces figure             |
|:----------------------|:----------------------------|
| ```graphical_model.py```    | (1) in Suppl. material      |
|  ```ROC_curves.py ```         | (2)--(7) in Suppl. material |
|  ```benchmark_animal.py ```   | (1) in main paper           |
|  ```benchmark_gps.py ```      | (2) in main paper           |
|  ```benchmark_concepts.py ``` | (8), (9** in Suppl. material |
## How to use the scripts

### ```ROC_curves.py```

#### Description

Compute and visualize receiver operating caracteristics (ROC) curves. Three ROC curves are computed to perform the following:

* Sensitivity of the proposed approaches to the regularization parameter;
* Sensitivity of the proposed approaches to the rank of the factor model;
* Comparison of the proposed approaches to state-of-the art methods.

#### Parameters

| Parameter                  | Type             | Possible values | Description |
|:----------------------|:----------------------------|:---------------------------|:--------------------|
| method    |   ```str```    | 'GGM', 'EGM', 'GGFM', 'EGFM', 'all' (default: 'GGM') | Method(s) to use. 'all' performs a comparison of SOTA methods, which can only be used with roc='compare'. Other values are to used with roc='lambda' or 'roc=rank'.|
| graph         | ```str``` | 'BA', 'ER', 'WS', 'RG' (default: 'BA') | Graph structure to be considered. |
| roc   | ```str```           | 'lambda', 'rank', 'compare' (default: 'lambda')| Type of ROC curve to be computed. 'compare' can only be used with a value of method='all'. 'lambda' and 'rank' modes are to be used with method='GGM', 'EGM', 'GGFM' or 'EGFM'.|
| lambda_val      | ```float```           | Real positive value (default: 0.05) | Regularization parameter. |
| rank | ```int``` | Positive integer (default: 20) | Rank of the factor model. |
| n | ```int``` | Positive integer (default: 105) | Number of samples. |
| multi | ```bool``` | ```True``` or ```False``` (default: ```True```) | Whether to use multi-threading or not. |
| save | ```bool``` | ```True``` or ```False``` (default: ```False```) | Save plot in pdf format in a **/results** folder. |

#### Output
ROC curve in pdf format.

#### Usage

```python ROC_curves.py --method <method> --graph <graph> --roc <roc_type> --lambda_val <lambda_value>  
--rank <rank> -n <number_of_samples> --multi <bool> --save <bool>
```

#### Examples
```python ROC_curves --method EGM --graph ER -n 200 --roc rank```  
```python ROC_curves --method EGM --graph ER --roc lambda```  
```python ROC_curves --method all --graph ER -n 200 --roc compare```

### ```benchmark_animal.py```; ```benchmark_gps.py```; ```benchmark_concepts.py```

#### Description

Estimate and visualize estimated graphs for three data sets:

* Comparison of the proposed approaches to state-of-the art methods on the animal data set;
* Comparison of the proposed approaches to state-of-the art methods on the GPS data set;
* Comparison of the proposed approaches to state-of-the art methods on the concepts data set. **Note: authors are not authorized to make the concepts data set public. This data set can be shared on demand.**

#### Parameters
There are no parameters for these scripts.

#### Output
Estimated graphs in networkx format.

#### Usage

```python benchmark_animal.py```  
```python benchmark_gps.py```


## Authors

For any issue, please contact: [mail a remplir](mail a remplir)

* Florent Bouchard
* Arnaud Breloy
* Alexandre Hippert-Ferrer
* Ammar Mian
* Titouan Vayer
