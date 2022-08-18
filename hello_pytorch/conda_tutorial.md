
conda create -n my_python_env python=3.4

conda activate dacon_env

conda deactivate




conda install jupyter notebook

conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge



conda create -n pytorch_env

conda activate pytorch_env

conda install pytorch torchvision torchaudio cpuonly -c pytorch