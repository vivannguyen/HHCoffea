# HH Coffea 
adapted from SUEP Coffea. HH analysis using [Coffea](https://coffeateam.github.io/coffea/)

## Quick start
```bash
cmsrel CMSSW_10_6_4
cd CMSSW_10_6_4/src
cmsenv

git clone git@github.com:vivannguyen/HHCoffea.git
```

## Merge files for coffea
edit paths for input ntuples and output
```bash
python merger.py
```

## To run the producer
histograms are defined in HH_Producer.py
```bash
python3 condor_HH_WS.py --isMC=0/1 --era=201X --infile=XXX.root
```

If you do not have the requirements set up then you can also run this through the docker container that the coffea team provides. This is simple and easy to do. You just need to enter the Singularity and then issue the command above. To do this use:
```bash
singularity shell -B ${PWD} -B /afs -B /eos /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest
```

Inside the singularity shell, can run a shell script over all files. This gives output root files from coffea. There is a runner for each year data and MC (TODO update this)
```bash
./runner_2016mc.sh
./runner_2016data.sh
```

## Plotter
With Miniconda, use the configuration file to create the virtual environment 'plotting'
```bash
conda env create -f plotting_env.yml
```

Activate the environment
```bash
conda activate plotting
```

Alternatively, you can pip install the packages listed inside the yml file.

To plot run HHplotter.py. Options (python HHplotter.py --help) for input are histogram directory of files from coffea made above, input samples directory, input xsection yaml, year, output directory, option to run without background normalizations ('--nonorm'), and option to run in series (default runs in parallel). Example command:
```bash
python HHplotter.py --sample_dir /eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v2/2017/ --hist_dir 2017-v2/ --xfile /afs/cern.ch/work/v/vinguyen/private/CMSSW_10_6_4/src/PhysicsTools/MonoZ/data/xsections_2017.yaml --year 2017 --outdir plots_2017-v2
```

## Requirements

- Python 3
- uproot
- coffea
- HTCondor cluster

Alternatively, everything can be run through the docker container provided by the coffea team:
/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest


