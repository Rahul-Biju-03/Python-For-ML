#!/bin/bash

ENV_NAME=training
deactivate
rm -rf $ENV_NAME
python -m venv $ENV_NAME
source $ENV_NAME/bin/activate
pip install --upgrade pip 
pip install transformers matplotlib ipykernel
pip install pandas
pip install scikit-learn-intelex
pip install numpy
pip install seaborn tqdm
python -m ipykernel install --user --name $ENV_NAME  #Register the env as a kernal for using it with jupyter notebook
