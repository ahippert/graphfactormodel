#!/usr/bin/env bash

# Path variables
CWD=$(pwd)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CWD=$(printf %q "$CWD")
SCRIPT_DIR=$(printf %q "$SCRIPT_DIR")

# Moving to conda directory to install everything
echo "moving to directory: $SCRIPT_DIR"
eval cd $SCRIPT_DIR

# create conda environment
echo "Creating conda environment graphfactormodels"
conda env create -f conda/environment.yml

# Activating environment for isntalling further stuff
eval "$(conda shell.bash hook)"  # bug fix: https://github.com/conda/conda/issues/7980#issuecomment-492784093 
conda activate graphfactormodels

# Installing pymanopt outside because it fails with pip whatsoever
echo "Installing pymanopt"

# The version of pymanopt used in the simulaitons is not available anymore on github so we put it in the repository
cd pymanopt
python setup.py install
cd ..
#rm -r pymanopt

# Installing other librairies
echo "Installing StructuredGraphLearning"
pip install StructuredGraphLearning

# Finishing install
echo "\n ------------------------------------------------------------"
echo "Done installing. Moving back to directory: $CWD"
eval cd $CWD
echo "To use then enviroment run command:"
echo "conda activate graphfactormodels"
