
```
# create a conda environment
conda create --name yourenvname python=3.8

# activate conda environment
conda activate yourenvname

# install pycaret
pip install pycaret

# create notebook kernel
python -m ipykernel install --user --name yourenvname --display-name "display-name"
```

conda create --name pycaret_env python=3.8

conda activate pycaret_env

conda deactivate

pip install pycaret

python -m ipykernel install --user --name pycaret_env --display-name "pycaret environment"
