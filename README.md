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

## to run the producer

histograms are defined in HH_Producer.py

```bash
python3 condor_HH_WS.py --isMC=0/1 --era=201X --infile=XXX.root
```

If you do not have the requirements set up then you can also run this through the docker container that the coffea team provides. This is simple and easy to do. You just need to enter the Singularity and then issue the command above. To do this use:

```bash
singularity shell -B ${PWD} -B /afs -B /eos /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest
```

Inside the singularity shell, can run a shell script over all files (TODO update this)
```bash
./runner_2016mc.sh
./runner_2016data.sh
```

## Plotter
To plot run HHplotter.py. Options for input are histogram directory from coffea, input samples directory, xsection yaml, and option to parallelize. Example command:
```bash
python HHplotter.py --parallel --hist_dir=qcdtestv2_coffea --sample_dir=/eos/user/v/vinguyen/qcdtestv2/merged/
```

## Requirements

- Python 3
- uproot
- coffea
- HTCondor cluster

Alternatively, everything can be run through the docker container provided by the coffea team:
/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest


