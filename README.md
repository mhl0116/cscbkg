# cscbkg

### Installation of environment

Clone the repository and work inside it:
```bash
git clone git@github.com:mhl0116/cscbkg.git 
cd cscbkg 
```

Install conda and get all the dependencies:
```bash
curl -O -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b 

# add conda to the end of ~/.bashrc, so relogin after executing this line
~/miniconda3/bin/conda init

# stop conda from activating the base environment on login
conda config --set auto_activate_base false
conda config --add channels conda-forge

# install package to tarball environments
conda install --name base conda-pack -y

ANALYSISENVNAME="analysisenv"

# create environments with as much stuff from anaconda
# ipython==7.10.1 because of https://stackoverflow.com/questions/63413807/deprecation-warning-from-jupyter-should-run-async-will-not-call-transform-c
packages="uproot dask dask-jobqueue matplotlib pandas jupyter pyarrow fastparquet numba numexpr bottleneck ipython<=7.10.1"
conda create --name $WORKERENVNAME $packages -y
conda create --name $ANALYSISENVNAME $packages -y

# and then install residual packages with pip
conda run --name $ANALYSISENVNAME pip install yahist jupyter-server-proxy coffea jupyter_nbextensions_configurator awkward awkward0 uproot3 pdroot

```

### To activate environment 

conda activate $ANALYSISENVNAME

### To analysis csc background

1. Perform pre-selection (preselection.py), select chambers satisfying certain criteria for analysis, take root ntuple as input and save output into panda dataframe
2. Make performance/diagnostic plots (plotter.py) related to CSC background study (rate vs lumi, rate vs BXID etc) 
3. In case one needs to make a fit for rate vs BXID in the large LHC empty bunches, use this script: fitbkg.py 
