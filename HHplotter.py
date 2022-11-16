#!/usr/bin/env/ python
#Plotter for HH variables
#%config InlineBackend.figure_format = 'retina'
import sys

import glob
import json
import logging
import os
from pathlib import Path
import re
import yaml

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import matplotlib.patches as mpatch
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

try:
    import uproot3 as uproot
    import uproot3_methods as uproot_methods
except:
    import uproot
    import uproot_methods
logging.getLogger().setLevel(logging.INFO)

#COLORS=list(sns.color_palette('Set2').as_hex())
COLORS=('#E33719','#2FDBCA','#F59D2D','#A9F274','#5CA4F5','#FF47EB')
#COLORS=('#E63719','#2FDBCA','#F59D2D','#ABF575','#5CA4F5','#FF47EB')
#COLORS=('#B82721','#98D17C','#E06E2D','#4954B0','#56A3CD','#FF47EB')
#COLORS=('#B82721','#7CAD51','#EDDCDA','#4954B0','#56A3CD','#FF47EB')
#COLORS=('#FF4E48','#F58A55','#FFEDEB','#436BE6','#52DBA9','#FF47EB')
#plt.style.use('physics.mplstyle')

LUMI = {'2016':36, '2017':41.5, '2018': 59.8}

def cleanup(samples_directory, hist_dir, xsections):
    sample_files = set([Path(f).stem
                        for f in glob.glob(os.path.join(samples_directory, '*.root'))])
    histogram_files = set([Path(f).stem.replace('_WS_selections','')
                           for f in glob.glob(os.path.join(hist_dir, '*.root'))
                           ])
    overlap = sample_files.intersection(histogram_files)
    data_kwds = ['DoubleEG', 'DoubleMuon', 'SingleElectron', 'SingleMuon', 'EGamma']
    to_keep = []
    for f in overlap:
        if any([kwd in f for kwd in data_kwds]):
            to_keep.append(f)
        else:
            if f in xsections:
                to_keep.append(f)
    sample_paths = [os.path.join(samples_directory, f'{f}.root')
                    for f in to_keep]
    hist_paths = [os.path.join(hist_dir, f'{f}_WS_selections.root')
                  for f in to_keep]
    return sample_paths, hist_paths

def get_histograms(hist_paths, year, channel):
    histograms={}

    logging.info('Loading histograms into memory.')
    for filename in tqdm(hist_paths):

        if channel == 'muon': skip = ['SingleElectron','DoubleEG','EGamma']
        elif channel == 'electron': skip = ['SingleMuon','DoubleMuon']
        else:
            raise ValueError(f'channel must be \'muon\' or \'electron\', not {channel}')

        if any([s in filename for s in skip]): continue
        if year == '2016':
            pass
        elif year == '2017':
            if channel == 'muon':
                if 'TTTo' in filename: continue
            elif channel == 'electron':
                if 'TTTo' in filename: continue
                #if 'TTJets' in filename: continue
        elif year == '2018':
            if 'ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-madgraph-pythia8' in filename: continue
            if channel == 'muon':
                if 'TTTo' in filename: continue
            elif channel == 'electron':
                if 'TTTo' in filename: continue
                #if 'TTJet' in filename: continue

        samplename = Path(filename).stem.split('_WS')[0]

        f_sample = uproot.open(filename)
        histogram_sample = f_sample.allitems( filterclass=lambda cls: issubclass(cls, uproot_methods.classes.TH1.Methods))
        if not histogram_sample:
            continue
        histograms[samplename] = histogram_sample

    logging.info('Finished loading histograms.')
    logging.info(f'Number of samples remaining: {len(histograms)}')
    return histograms

def xs_scale(xsections, year, ufile, proc):
    xsec  = xsections[proc]["xsec"]
    xsec *= xsections[proc]["kr"]
    xsec *= xsections[proc]["br"]
    xsec *= 1000.0
    #print (proc, xsec)
    assert xsec > 0, "{} has a null cross section!".format(proc)
    scale = LUMI[year]*1000/ufile["Runs"].array("genEventSumw").sum()
    return scale

def get_xsections(infile):
    with open(infile) as f:
        try:
            xsections = yaml.safe_load(f.read())
        except yaml.YAMLError as exc:
            print(exc)
    return xsections

def get_normalizations(sample_paths, xsections, histogram_names, year):
    norm_dict = {}

    logging.info('Obtaining normalizations.')

    for fn in tqdm(sample_paths):
#        if 'TTTo' in fn : continue
        if 'ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-madgraph-pythia8' in fn : continue
        if os.path.isfile(fn):
            _proc = os.path.basename(fn).replace(".root","")
            _file = uproot.open(fn)
            if ("DoubleEG") in fn:
                _scale = 1
            elif ("DoubleMuon") in fn:
                _scale = 1
            elif ("SingleElectron") in fn:
                _scale = 1
            elif ("SingleMuon") in fn:
                _scale = 1
            elif ("EGamma") in fn:
                _scale = 1
            else:
                _scale  = xs_scale(xsections, year, ufile=_file, proc=_proc)
#            if 'GluGluToHHTo2B2ZTo2L2J_node_cHHH1' in fn:
#                print("hello", _scale)
#                _scale = _scale * .031047 *0.004
#                print("goodbye", _scale)
            for idx, name in enumerate(histogram_names):
                if name == Path(fn).stem:
                    norm_dict[name] = _scale
                    break
    logging.info('Finished obtaining normalizations.')
    logging.info(f'Number normalizations in memory: {len(norm_dict)}.')
# These normalizations scale by lumi/genweight. The xsec is already applied in the plots from coffea
    return(norm_dict)

def rebin( a, newshape ):
        '''Rebin an array to a new shape.
        '''
        assert len(a.shape) == len(newshape)

        slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
        coordinates = np.mgrid[slices]
        indices = coordinates.astype('i')   #choose the biggest smaller integer index
        return a[tuple(indices)]

def rebin_factor( a, newshape ):
        '''Rebin an array to a new shape.
        newshape must be a factor of a.shape.
        '''
        assert len(a.shape) == len(newshape)
        assert not sometrue(mod( a.shape, newshape ))

        slices = [ slice(None,None, old/new) for old,new in zip(a.shape,newshape) ]
        return a[slices]

def normalize_event_yields(event_yields, normalizations, file_to_category, var=False):
    categorized_yields = {}

    for sample in event_yields:
        category = file_to_category[sample]
        if category not in categorized_yields:
            try:
                if var:
                    categorized_yields[category] = (normalizations[sample]**2) * event_yields[sample]
                else:
                    categorized_yields[category] = normalizations[sample] * event_yields[sample]
            except:
                if sample not in normalizations and sample not in event_yields:
                    print(f'{sample} not in both')
                elif sample not in event_yields:
                    print(f'{sample} not in event_yields')
                elif sample not in normalizations:
                    print(f'{sample} not in normalizations')
        else:
            if sample not in normalizations and sample not in event_yields:
                print(f'{sample} not in both')
            elif sample not in event_yields:
                print(f'{sample} not in event_yields')
            elif sample not in normalizations:
                print(f'{sample} not in normalizations')
            if var:
                categorized_yields[category] += (normalizations[sample]**2) * event_yields[sample]
            else:
                categorized_yields[category] += normalizations[sample] * event_yields[sample]

    return categorized_yields

def get_bins_and_event_yields(histograms, normalizations, year, filter_categories=False, print_yields=False):
    with open(f'{year}_sample_reference_VBF.json', 'r') as infile:
        file_to_category = json.load(infile)

    categories = set(file_to_category.values())
    if filter_categories:
        for category in ['QCD', 'Radion', 'Graviton', 'NonRes']:#, 'NonResVBF']:
        #for category in ['QCD', 'NonResVBF', 'Radion', 'Graviton', 'NonRes', 'NonResSM']:
            categories.remove(category)

    df_dict = {}
    df_dict['sample_name'] = []
    df_dict['bins'] = []
    df_dict['var'] = []
    for category in categories:
        df_dict[category] = []
        df_dict['var_'+category] = []
    df_dict['Other'] = []
    df_dict['var_Other'] = []
    df_dict['up'] = []
    df_dict['down'] = []

    logging.info('Getting bins and event yields.')

    arb_key = next(iter(histograms))

    for idx, (name, roothist) in enumerate(tqdm(histograms[arb_key])):
        name = name.decode("utf-8")
        name = name.replace(";1", "")

        # TODO make this more robust
        if "genEventSumw" == name:
            continue
        if "sys" in name:
            continue

#        for idx, (name, roothist) in enumerate(tqdm(histograms[arb_key])):
#            name = name.decode("utf-8")
#            name = name.replace(";1", "")

        event_yields = {}
        event_variances = {}
        event_sys_up = {}
        event_sys_down = {}

        # TODO Only one for loop here.
        for key, value in histograms.items():
             # THIS IS WHERE YOU PRINT THINGS WHEN THINGS GO WRONG
#            print(key)
#            print(value)
#            print(name)
            sys_up = np.zeros(len(value[idx][1].numpy()[0]))
            sys_down = np.zeros(len(value[idx][1].numpy()[0]))
            for histogram in value:
                hist_name = histogram[0].decode("utf-8").lower()
                if "qcd_c" in hist_name: continue
                if "up" in hist_name:
#                    print("each hist", hist_name)
#                    print("up hist", (np.sum(histogram[1].numpy()[0])-np.sum(value[idx][1].numpy()[0]))/ np.sum(value[idx][1].numpy()[0])  )
#                    print("nom hist", value[idx][1].numpy()[0])
                    sys_up = sys_up + np.sqrt((histogram[1].numpy()[0] - value[idx][1].numpy()[0])**2)
#                    print("sys up", sys_up)
                    #print(histogram[1].numpy()[0]-value[idx][1].numpy()[0])
                if "down" in hist_name:
                    sys_down = sys_down + np.sqrt((value[idx][1].numpy()[0]-histogram[1].numpy()[0])**2)

            event_sys_up[key] = sys_up
            event_sys_down[key] = sys_down

            # Getting the idx from the top for loop, this loop goes over each sample, each key is the sample
            event_yields[key] = value[idx][1].numpy()[0]
            event_variances[key] = value[idx][1].variances
        output = normalize_event_yields(event_yields, normalizations, file_to_category)
        output_var = normalize_event_yields(event_variances, normalizations, file_to_category, var=True)
        output_sys_up = normalize_event_yields(event_sys_up, normalizations, file_to_category)
        output_sys_down = normalize_event_yields(event_sys_down, normalizations, file_to_category)
        output['Other'] = output['VV'] + output['SingleTop'] + output['Wjets'] + output['ttV']
        output_var['Other'] = output_var['VV'] + output_var['SingleTop'] + output_var['Wjets'] + output_var['ttV']

#        output['Other'] = output['VV'] + output['SingleTop'] + output['ttV']

#        output_var['Other'] = output_var['VV'] + output_var['SingleTop'] + output_var['ttV']
        output_sys_up['Other'] = output_sys_up['VV'] + output_sys_up['SingleTop'] + output_sys_up['ttV']
        output_sys_down['Other'] = output_sys_down['VV'] + output_sys_down['SingleTop'] + output_sys_down['ttV']

        total_var = output_var['Other'] + output_var['SMHiggs'] + output_var['DY'] + output_var['TT']
        total_sys_up = output_sys_up['Other'] + output_sys_up['SMHiggs'] + output_sys_up['DY'] + output_sys_up['TT']
        total_sys_down = output_sys_down['Other'] + output_sys_down['SMHiggs'] + output_sys_down['DY'] + output_sys_down['TT']
        if print_yields:
            if name == 'BDTscore':
                print("yooo cHHH1", output['cHHH1'].sum())

        if year == '2016':
            output['cHHH1'] = output['cHHH1']*1000 #scale 2016 to 1pb
            output['VBF1'] = output['VBF1']*1000 #scale 2016 to 1pb

        for category in output:
            df_dict[category].append(output[category])
            df_dict['var_'+category].append(output_var[category])
        df_dict['var'].append(total_var)
        df_dict['bins'].append(roothist.numpy()[1])
        df_dict['sample_name'].append(name)
        df_dict['up'].append(total_sys_up)
        df_dict['down'].append(total_sys_down)

        if print_yields:
            if name == 'BDTscore':
                if year == '2016':
                    y = ['GluGluToHHTo2B2ZTo2L2J_node_cHHH1_TuneCUETP8M1_PSWeights_13TeV-powheg-pythia8',
                         'VBFHHTo2B2ZTo2L2J_CV_1_C2V_1_C3_1_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8'
                         #'GluGluToHHTo2B2ZTo2L2J_node_SM_TuneCUETP8M1_PSWeights_13TeV-madgraph-pythia8',
                         #'GluGluToRadionToHHTo2B2ZTo2L2J_M-260_narrow_13TeV-madgraph-v2',
                         #'GluGluToRadionToHHTo2B2ZTo2L2J_M-600_narrow_13TeV-madgraph-v2',
                         #'GluGluToRadionToHHTo2B2ZTo2L2J_M-1000_narrow_13TeV-madgraph-v2',
                         #'GluGluToRadionToHHTo2B2ZTo2L2J_M-3000_narrow_TuneCUETP8M1_PSWeights_13TeV-madgraph-pythia8',
                         #'TT_TuneCUETP8M2T4_13TeV-powheg-pythia8',
                         #'DYToLL_0J_13TeV-amcatnloFXFX-pythia8',
                         #'DYToLL_1J_13TeV-amcatnloFXFX-pythia8',
                         #'DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8',
                         #'DYToLL_2J_13TeV-amcatnloFXFX-pythia8'
                        ]
                if year == '2017':
                    y = ['GluGluToHHTo2B2ZTo2L2J_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8',
                         'VBFHHTo2B2ZTo2L2J_CV_1_C2V_1_C3_1_dipoleRecoilOff-TuneCP5_PSweights_13TeV-madgraph-pythia8',
                         #'GluGluToHHTo2B2ZTo2L2J_node_SM_13TeV-madgraph_correctedcfg',
                         #'GluGluToRadionToHHTo2B2ZTo2L2J_M-260_narrow_13TeV-madgraph_correctedcfg',
                         #'GluGluToRadionToHHTo2B2ZTo2L2J_M-600_narrow_13TeV-madgraph_correctedcfg',
                         #'GluGluToRadionToHHTo2B2ZTo2L2J_M-1000_narrow_13TeV-madgraph_correctedcfg',
                         #'GluGluToRadionToHHTo2B2ZTo2L2J_M-3000_narrow_TuneCP5_PSWeights_13TeV-madgraph-pythia8',
                         'DYJetsToLL_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8',
                         'DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8',
                         #'TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8',
                         #'TTToHadronic_TuneCP5_PSweights_13TeV-powheg-pythia8',
                         #'TTTo2L2Nu_TuneCP5_PSweights_13TeV-powheg-pythia8',
                         'DYJetsToLL_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8',
                         'DYJetsToLL_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8'
                        ]
                if year == '2018':
                    y = ['GluGluToHHTo2B2ZTo2L2J_node_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8',
                         'VBFHHTo2B2ZTo2L2J_CV_1_C2V_1_C3_1_dipoleRecoilOff-TuneCP5_PSweights_13TeV-madgraph-pythia8',
                         #'GluGluToHHTo2B2ZTo2L2J_node_SM_TuneCP5_PSWeights_13TeV-madgraph-pythia8',
                         #'GluGluToRadionToHHTo2B2ZTo2L2J_M-260_narrow_TuneCP5_PSWeights_13TeV-madgraph-pythia8',
                         #'GluGluToRadionToHHTo2B2ZTo2L2J_M-600_narrow_TuneCP5_PSWeights_13TeV-madgraph-pythia8',
                         #'GluGluToRadionToHHTo2B2ZTo2L2J_M-1000_narrow_TuneCP5_PSWeights_13TeV-madgraph-pythia8',
                         #'GluGluToRadionToHHTo2B2ZTo2L2J_M-3000_narrow_TuneCP5_PSWeights_13TeV-madgraph-pythia8',
                         #'DYJetsToLL_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8',
                         #'DYJetsToLL_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8',
                         #'DYJetsToLL_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8',
                         #'DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8',
                         #'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8',
                         #'TTToHadronic_TuneCP5_13TeV-powheg-pythia8',
                         #'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8'
                        ]
                for idx, y in enumerate(y):
                    print(f'Yield {y}: {(event_yields[y]*normalizations[y]).sum()}')

    logging.info('Finished getting bins and event yields.')
    for key in df_dict:
        print(key, len(df_dict[key]))
    
    return pd.DataFrame(df_dict)

def calculate_sys(histograms, normalizations, year, filter_categories=True, print_yields=False):
    with open(f'{year}_sample_reference_new.json', 'r') as infile:
        file_to_category = json.load(infile)

    categories = set(file_to_category.values())
    print("CATS", categories)
    if filter_categories:
        for category in ['Data','QCD', 'Radion', 'Graviton', 'NonRes', 'NonResVBF']:
        #for category in ['QCD', 'NonResVBF', 'Radion', 'Graviton', 'NonRes', 'NonResSM']:
            categories.remove(category)

    print("CATS AFTER", categories)
    df_dict = {}
    df_dict['sample_name'] = []
    df_dict['bins'] = []
    df_dict['var'] = []
    for category in categories:
        df_dict[category] = []
        df_dict['var_'+category] = []
    df_dict['Other'] = []
    df_dict['var_Other'] = []
    df_dict['up'] = []
    df_dict['down'] = []

    logging.info('Getting bins and event yields.')

    arb_key = next(iter(histograms))

    for idx, (name, roothist) in enumerate(tqdm(histograms[arb_key])):
        name = name.decode("utf-8")
        name = name.replace(";1", "")

        # TODO make this more robust
        if "genEventSumw" == name:
            continue

#        for idx, (name, roothist) in enumerate(tqdm(histograms[arb_key])):
#            name = name.decode("utf-8")
#            name = name.replace(";1", "")

        event_yields = {}
        event_variances = {}
        event_sys_up = {}
        event_sys_down = {}

        # TODO Only one for loop here.
        for key, value in histograms.items():
             # THIS IS WHERE YOU PRINT THINGS WHEN THINGS GO WRONG
#            print(key)
            if "Single" in key:continue
            if "Double" in key:continue
            if "EGamma" in key:continue
#            print(value)
#            print(name)
            sys_up = np.zeros(len(value[idx][1].numpy()[0]))
            sys_down = np.zeros(len(value[idx][1].numpy()[0]))
            for histogram in value:
                hist_name = histogram[0].decode("utf-8").lower()
                if "qcd_c" in hist_name: continue
                if "up" in hist_name:
#                    print("each hist", hist_name)
#                    print("up hist", (np.sum(histogram[1].numpy()[0])-np.sum(value[idx][1].numpy()[0]))/ np.sum(value[idx][1].numpy()[0])  )
#                    print("nom hist", value[idx][1].numpy()[0])
                    sys_up = sys_up + np.sqrt((histogram[1].numpy()[0] - value[idx][1].numpy()[0])**2)
#                    print("sys up", sys_up)
                    #print(histogram[1].numpy()[0]-value[idx][1].numpy()[0])
                if "down" in hist_name:
                    sys_down = sys_down + np.sqrt((value[idx][1].numpy()[0]-histogram[1].numpy()[0])**2)

            event_sys_up[key] = sys_up
            event_sys_down[key] = sys_down

            # Getting the idx from the top for loop, this loop goes over each sample, each key is the sample
            event_yields[key] = value[idx][1].numpy()[0]
            event_variances[key] = value[idx][1].variances
        output = normalize_event_yields(event_yields, normalizations, file_to_category)
        output_var = normalize_event_yields(event_variances, normalizations, file_to_category, var=True)
        output_sys_up = normalize_event_yields(event_sys_up, normalizations, file_to_category)
        output_sys_down = normalize_event_yields(event_sys_down, normalizations, file_to_category)
        #output['Other'] = output['VV'] + output['SingleTop'] + output['Wjets'] + output['ttV']
        #output_var['Other'] = output_var['VV'] + output_var['SingleTop'] + output_var['Wjets'] + output_var['ttV']

        output['Other'] = output['VV'] + output['SingleTop'] + output['ttV']

        output_var['Other'] = output_var['VV'] + output_var['SingleTop'] + output_var['ttV']
        output_sys_up['Other'] = output_sys_up['VV'] + output_sys_up['SingleTop'] + output_sys_up['ttV']
        output_sys_down['Other'] = output_sys_down['VV'] + output_sys_down['SingleTop'] + output_sys_down['ttV']

        total_var = output_var['Other'] + output_var['SMHiggs'] + output_var['DY'] + output_var['TT']
        total_sys_up = output_sys_up['Other'] + output_sys_up['SMHiggs'] + output_sys_up['DY'] + output_sys_up['TT']
        total_sys_down = output_sys_down['Other'] + output_sys_down['SMHiggs'] + output_sys_down['DY'] + output_sys_down['TT']
        if print_yields:
            if name == 'BDTscore':
                print("yooo cHHH1", output['cHHH1'].sum())

        if year == '2016':
            output['cHHH1'] = output['cHHH1']*1000 #scale 2016 to 1pb


        for category in output:
            df_dict[category].append(output[category])
            df_dict['var_'+category].append(output_var[category])
        df_dict['var'].append(total_var)
        df_dict['bins'].append(roothist.numpy()[1])
        df_dict['sample_name'].append(name)
        df_dict['up'].append(total_sys_up)
        df_dict['down'].append(total_sys_down)


    logging.info('Finished getting bins and event yields.')
#    for key in df_dict:
#        print(key, len(df_dict[key]))
    
    return pd.DataFrame(df_dict)

def btag_ratio(all_event_yields, year, filepath, overwrite):
    names_for_njets = np.array(['ngood_jets', 'ngood_jets_btagSF'])
    df_subset = (all_event_yields.set_index('sample_name')
                                 .loc[names_for_njets]
                                 .reset_index())

    njets, njets_btagSF = *[df_subset.iloc[idx] for idx in range(df_subset.shape[0])],
    sum_func = lambda x: (x['DY'] + x['TT'] + x['SMHiggs'] + x['Other'])

    logging.info('Calculating btag renormalizations by jet bin.')

    sum_njets = sum_func(njets)
    sum_njets_btagSF = sum_func(njets_btagSF)

    sum_high_njet = sum(sum_njets[10:])
    sum_high_njet_btagSF = sum(sum_njets_btagSF[10:])
    ratio_high_njet = sum_high_njet/sum_high_njet_btagSF

    btag_ratio = sum_njets/sum_njets_btagSF
    btag_ratio[10:]=[ratio_high_njet]*len(btag_ratio[10:])
    btag_ratio[np.isnan(btag_ratio)] = -999
    print(f'btag Ratio: {btag_ratio}')

    weights = {str(idx): ratio for idx, ratio in enumerate(btag_ratio)}
    btag_weights = {'year': int(year), 'weights': weights}

    if overwrite:
        mode = 'w'
    else:
        mode = 'a'
    with open(filepath, mode) as f:
        f.write(str(btag_weights).replace('\'', '\"') + '\n')

    logging.info('Saved btag renormalizations to JSON.')

def btag_ratio_plot(all_event_yields, year, outdir=''):
    names_for_njets = np.array(['ngood_jets', 'ngood_jets_btagSF', 'ngood_jets_nobtagSF'])
    df_subset = (all_event_yields.set_index('sample_name')
                                 .loc[names_for_njets]
                                 .reset_index())

    njets, njets_btagSF, njets_nobtagSF = *[df_subset.iloc[idx] for idx in range(df_subset.shape[0])],
    sum_func = lambda x: (x['DY'] + x['TT'] + x['SMHiggs'] + x['Other'])

    sum_njets = sum_func(njets)
    sum_njets_btagSF = sum_func(njets_btagSF)
    sum_njets_nobtagSF = sum_func(njets_nobtagSF)

    sum_high_njet = sum(sum_njets[10:])
    sum_high_njet_btagSF = sum(sum_njets_btagSF[10:])
    sum_high_njet_nobtagSF = sum(sum_njets_nobtagSF[10:])

    sum_njets[10:]=[sum_high_njet]*len(sum_njets[10:])
    sum_njets_btagSF[10:]=[sum_high_njet_btagSF]*len(sum_njets_btagSF[10:])
    sum_njets_nobtagSF[10:]=[sum_high_njet_nobtagSF]*len(sum_njets_nobtagSF[10:])

    logging.info('Making btag ratio plot.')

    bins = njets['bins']
    x = np.arange(len(bins)-1)

    fig, axarr = plt.subplots(2, dpi=150, figsize=(6, 5), sharex=True,
                                  gridspec_kw={'hspace': 0.08, 'height_ratios': (0.8,0.2)},
                                  constrained_layout=False)
    upper = axarr[0]
    lower = axarr[1]

    ns, bins, patches = upper.hist(x=x, bins=bins, weights=sum_njets_nobtagSF,
                          histtype='step', linestyle=('solid'), color='black',
                          linewidth=1, label=['no SF']
                          )
    ns1, bins, patches = upper.hist(x=x, bins=bins, weights=sum_njets_btagSF,
                          histtype='step', linestyle=('solid'), color='blue',
                          linewidth=1, label=['btag SF']
                          )
    ns2, bins, patches = upper.hist(x=x, bins=bins, weights=sum_njets,
                          histtype='step', linestyle=('solid'), color='crimson',
                          linewidth=1, label=['btag SF and nJet corr.']
                          )

    upper.set_xlim([3.5, 10.5])
    upper.set_ylabel("Events/bin", y=1, ha='right')
    upper.legend()

    lower.step(bins[1:], ns1 / ns,
               color='blue',linewidth=1
              )
    lower.step(bins[1:], ns2 / ns,
               color='crimson',linewidth=1
              )

    lower.set_ylim(0.5, 1.5)
    lower.set_xlabel('n Jets', x=1, ha='right')
    lower.set_ylabel('Ratio (/no SF)')

    fig.savefig(os.path.join(outdir, f'njets_btag_ratios_{year}.png'), bbox_inches='tight')

def estimate_background(all_event_yields, tol=1e-16, maxiter=100, disp=False):
    names_for_bkgd_est = np.array(['Zlep_cand_mass_DYcontrol_QCD_C', 'Zlep_cand_mass_TTcontrol_QCD_C',
                          'Zlep_cand_mass_DYcontrol', 'Zlep_cand_mass_TTcontrol',
                          'Zlep_cand_mass_QCD_B','Zlep_cand_mass_QCD_D'])
    df = all_event_yields.set_index('sample_name')
    df_subset = df.loc[names_for_bkgd_est]
    df_subset = df_subset.reset_index()

    dy_c, tt_c, dy, tt, qcd_b, qcd_d = *[df_subset.iloc[idx] for idx in range(df_subset.shape[0])],


    #residual_func = lambda x, y, z, a, b: (x['Data']
    #                                    - (x['DY'] * y + a + x['TT'] * z + b + x['SMHiggs'] + x['Other']))

    residual_func = lambda x, y, z, a, b: ((x['Data'] 
                                         - (x['DY'] * y + a + x['TT'] * z + b + x['SMHiggs'] + x['Other'])) >0)*((x['Data'] 
                                         - (x['DY'] * y + a + x['TT'] * z + b + x['SMHiggs'] + x['Other'])))

    #residual_func_all = lambda x, y, z, a, b: (x['Data'] 
    #                                    - (x['DY'] * y + a + x['TT'] * z + b + x['SMHiggs'] + x['Other']))

    def optimizer(dDY, dTT, dDY_qcd_b, dTT_qcd_b, dDY_qcd_d, dTT_qcd_d):
        counter = 0
        qcd_norm = 100
        dy_norm = 100
        tt_norm = 100
        error = 100

        dy_data = dy['Data']
        tt_data = tt['Data']

        while (error > tol) and (counter < maxiter):
            tt_c_shape = residual_func(tt_c, dy_norm, tt_norm, 0, 0)
            updated_tt_norm = np.sum((tt_data - (qcd_norm * tt_c_shape + dy_norm*tt['DY'] + tt['SMHiggs'] + tt['Other'])) / np.sum(tt['TT']))

            dy_c_shape = residual_func(dy_c, dy_norm, updated_tt_norm, 0, 0)
            updated_dy_norm = np.sum((dy_data - (qcd_norm * dy_c_shape + updated_tt_norm*dy['TT'] + dy['SMHiggs'] + dy['Other'])) / np.sum(dy['DY']))

            qcd_b_val = np.sum(residual_func(qcd_b, updated_dy_norm, updated_tt_norm, dDY_qcd_b, dTT_qcd_b))
            qcd_d_val = np.sum(residual_func(qcd_d, updated_dy_norm, updated_tt_norm, dDY_qcd_d, dTT_qcd_d))

            updated_qcd_norm = qcd_b_val / qcd_d_val
            error = np.sqrt((updated_qcd_norm - qcd_norm)**2 + np.abs(updated_dy_norm - dy_norm)**2 + np.abs(updated_tt_norm - tt_norm)**2)

            #print(f'DY norm: {dy_norm}, TT norm: {tt_norm}, fBD: {qcd_norm}, err: {current_err}')

            qcd_norm = updated_qcd_norm
            dy_norm = updated_dy_norm
            tt_norm = updated_tt_norm

            counter += 1

        if disp:
            print(f'qcd_norm: {qcd_norm:.3f} \n'
                  f'dy_norm: {dy_norm:.3f} \n'
                  f'tt_norm: {tt_norm:.3f} \n'
                  f'Error: {error} \n'
                  f'Converged: {error <= tol} \n'
                  f'Iterations: {counter}')

        return qcd_norm, dy_norm, tt_norm

    qcd_norms = []
    dy_norms = []
    tt_norms = []

    errDY = np.sqrt(dy['var_DY'])
    errTT = np.sqrt(tt['var_TT'])
    errDY_qcd_b = np.sqrt(qcd_b['var_DY'])
    errTT_qcd_b = np.sqrt(qcd_b['var_TT'])
    errDY_qcd_d = np.sqrt(qcd_d['var_DY'])
    errTT_qcd_d = np.sqrt(qcd_d['var_TT'])

    for _ in range (0, 1000):
        randDY = np.random.normal(0, 1, size=dy['Data'].shape)
        randTT = np.random.normal(0, 1, size=tt['Data'].shape)
        randDY_qcd_b = np.random.normal(0, 1, size=qcd_b['Data'].shape)
        randTT_qcd_b = np.random.normal(0, 1, size=qcd_b['Data'].shape)
        randDY_qcd_d = np.random.normal(0, 1, size=qcd_d['Data'].shape)
        randTT_qcd_d = np.random.normal(0, 1, size=qcd_d['Data'].shape)

        dDY = np.multiply(errDY,randDY)
        dTT = np.multiply(errTT,randTT)
        dDY_qcd_b = np.multiply(errDY_qcd_b,randDY_qcd_b)
        dTT_qcd_b = np.multiply(errTT_qcd_b,randTT_qcd_b)
        dDY_qcd_d = np.multiply(errDY_qcd_d,randDY_qcd_d)
        dTT_qcd_d = np.multiply(errTT_qcd_d,randTT_qcd_d)

        qcd_norm, dy_norm, tt_norm = optimizer(dDY, dTT, dDY_qcd_b, dTT_qcd_b, dDY_qcd_d, dTT_qcd_d)

        qcd_norms.append(qcd_norm)
        dy_norms.append(dy_norm)
        tt_norms.append(tt_norm)

    print(f'dy_norm: {np.mean(dy_norms):.3g} +/- {np.std(dy_norms):.1g} \n'
          f'tt_norm: {np.mean(tt_norms):.3g} +/- {np.std(tt_norms):.1g} \n'
          f'qcd_norm: {np.mean(qcd_norms):.3g} +/- {np.std(qcd_norms):.1g} \n')

    norms = (np.mean(qcd_norms), np.mean(dy_norms), np.mean(tt_norms))

    return norms

def data_mc_residual(x, norm1=1, norm2=1):
    return (x['Data'] - (x['DY']  * norm1 + x['TT'] * norm2 + x['SMHiggs'] + x['Other']))

def scale_cregions (df, qcd_norm, dy_norm, tt_norm):
    df = df.copy()
    c_samples_bool = df['sample_name'].str.contains('QCD_C')
    c_samples_to_drop = df[c_samples_bool & ~df['sample_name'].str.contains('Zlep_cand_mass_QCD_C')]['sample_name']
    c_samples_to_drop = df[c_samples_bool & ~df['sample_name'].str.contains('Zjet_cand_mass_QCD_C')]['sample_name']
    c_samples = df[c_samples_bool]['sample_name']
    df_subset = df[c_samples_bool].set_index('sample_name')
    residuals = (data_mc_residual(df_subset, dy_norm, tt_norm)) * qcd_norm
    df['QCD_estimate'] = df.apply(lambda x: np.zeros(shape=x['Data'].shape[0]), axis=1)
    df = df.set_index('sample_name')
    df.loc[residuals.index.str.replace('_QCD_C', ''), 'QCD_estimate'] = residuals.values
    df = df.drop(c_samples_to_drop).reset_index()
    return df

def new_plotting(event_yields, bkgd_norm, year, channel, outdir='', print_yields=False, datacard=False, bdtscores_to_csv=False):
    fig, axarr = plt.subplots(2, dpi=150, figsize=(6, 5), sharex=True,
                              gridspec_kw={'hspace': 0.05, 'height_ratios': (0.8,0.2)},
                              constrained_layout=False)

    upper = axarr[0]
    lower = axarr[1]

    if print_yields:
        if event_yields['sample_name']=="event_yield_A":
            print('DY yield: ', event_yields['DY'])
            print('TT yield: ', event_yields['TT'])

    # This gives the copy warning.
    event_yields['DY'] *= bkgd_norm[1]
    event_yields['TT'] *= bkgd_norm[2]

    # For comparing to data cards
    if event_yields['sample_name']=="BDTscore":
#    if event_yields['sample_name']=="Zlep_cand_mass_DYcontrol":
        print('DY yield: ', event_yields['DY'].sum())
        print('TT yield: ', event_yields['TT'].sum())
        print('SingleTop yield: ', event_yields['SingleTop'].sum())
        print('SMHiggs yield: ', event_yields['SMHiggs'].sum())
        print('WJets yield: ', event_yields['Wjets'].sum())
        print('VV yield: ', event_yields['VV'].sum())
        print('ttV yield: ', event_yields['ttV'].sum())
        print('QCD yield: ', event_yields['QCD_estimate'].sum())
        print('Other yield: ', event_yields['Other'].sum())
        print('Data yield: ', event_yields['Data'].sum())
#        print('Signal yield: ', event_yields['NonResHHH1'].sum()*0.031047*0.004)

    mc_categories = ['DY', 'TT', 'SMHiggs', 'QCD_estimate']
    MC = event_yields[mc_categories].sum()
    Data = event_yields['Data']
    Other = event_yields['Other']
    name = event_yields['sample_name']
    bins = event_yields['bins']

    #Signal = event_yields['cHHH1'] #* 0.031047 * 0.004
    Signal = event_yields['VBF1'] #* 0.031047 * 0.004

    blinding_bins = round(len(bins) * 0.8)
    print('before', len(bins))
    print('after', blinding_bins)

    if event_yields['sample_name']=="BDTscore":
        if not datacard:
            # slice depends on how many total bins in the BDT score distribution
            Data[blinding_bins:] = 0

    MC += Other

    # The first bin has a value of 0 and will give a warning.
    ratio = Data/MC
    ratio_sys_up = event_yields['up']/MC
    ratio_sys_down = event_yields['down']/MC
    ratio_sys_up[blinding_bins:]=0
    ratio_sys_down[blinding_bins:]=0
#    print("up", ratio_sys_up)
#    print("down", ratio_sys_down)
#    print("up ratio", ratio+ratio_sys_up)
#    print("down ratio", ratio-ratio_sys_down)
    binc = bins[:-1] + np.diff(bins) * 0.5
    binup = bins[1:]
    xerr = np.diff(bins)*0.5

    upper.errorbar(binc, Data, xerr = None, yerr = np.sqrt(Data), fmt = 'o',
                   zorder=10, color='black', label='Data', markersize=3)

    if event_yields['sample_name']=="BDTscore":
        if bdtscores_to_csv:
            asdf = np.column_stack((binc,Signal,MC))
            np.savetxt(f'BDTscores_{year}{channel}.csv', asdf, delimiter=',', fmt='%f')

    all_weights = np.vstack([event_yields['SMHiggs'],
                             event_yields['Other'],
                             event_yields['QCD_estimate'],
                             event_yields['DY'],
                             event_yields['TT']]).transpose()
    all_x = np.vstack([binc] * all_weights.shape[1]).transpose()

    COLORMAP = {'SMhiggs': COLORS[4],
                'Other': COLORS[3],
                'DY': COLORS[1],
                'TT': COLORS[0],
                'QCD': COLORS[2],
                'QCD_estimate':COLORS[5]}

    labels = ['SMhiggs', 'Other', 'QCD', 'DY', 'TT']
    plotting_colors = [COLORMAP[s] for s in labels]

    upper.hist(x=all_x, bins=bins, weights=all_weights,
               histtype='stepfilled', edgecolor='black', zorder=1,
               stacked=True, color=plotting_colors, label=labels)

    #sig_weight = Signal.transpose()
    #sig_x = ([binc] * sig_weight.shape[1]).transpose()
    #print("sigx", sig_x)

    #upper.hist(x=sig_x, bins=bins, weights=sig_weight,
    #           histtype='step', zorder=1)

    upper.stairs(
        edges= bins,
        values= Signal,
        # hatch="///",
        #label="cHHH1 (1pb)",
        label="VBF cv,c2v,c3=1 (1pb)",
        facecolor="orchid",
        linewidth=1,
        color="orchid",
    )

    upper.fill_between(binup, MC - np.sqrt(event_yields['var']), MC + np.sqrt(event_yields['var']), step='pre', hatch='///', alpha=0, zorder=2, label="MC Stat Err")

    upper.set_yscale("log")
    upper.set_ylim([0.01, 1000000])

    #Set parameters that will be used to make the plots prettier
    max_y = upper.get_ylim()[1]
    max_x = max(bins)
    min_x = min(bins)
    x_range = max_x - min_x
    lower_label = min_x - x_range*0.05
    upper_label = max_x - x_range*0.35


    #X and Y labels (Do not use the central matplotlob default)
    upper.set_ylabel("Events/bin", y=1, ha='right')
    upper.tick_params(axis='both', which='major', direction='in',
                      bottom=True, right=False, top=False, left=True,
                      color='black')
    upper.tick_params(axis='both', which='minor', direction='in',
                      bottom=True, right=False, top=False, left=True,
                      color='black')
    lower.set_xlabel(name, x=1, ha='right')
    lower.set_ylabel("Data/MC", fontsize = 10)
    lower.set_ylim(0.5, 1.5)
    yerr = np.sqrt(Data) / MC

    chi2 = 0
    nBins = 0
    ratio_check = np.isfinite(ratio)

    for indx, r in enumerate(ratio):
        if ratio_check[indx] and yerr[indx] > 0:
            chi2 = chi2 + ((r-1)*(r-1))/(yerr[indx]*yerr[indx])
            nBins = nBins + 1


    lower.errorbar(binc, ratio, yerr = yerr, marker = '.', color = 'black', linestyle ='none')
    #lower.fill_between(binc, 1+ratio_sys_up, 1-ratio_sys_down, step='mid', alpha=0.5, color='slategrey')
    lower.plot([min_x, max_x],[1,1],linestyle=':', color = 'black')
    lower.xaxis.set_minor_locator(AutoMinorLocator())
    lower.yaxis.set_minor_locator(AutoMinorLocator())
    lower.tick_params(axis='both', which='major', direction='in', length=4,
                      bottom=True, right=False, top=False, left=True,
                      color='black')
    lower.tick_params(axis='both', which='minor', direction='in', length=2,
                      bottom=True, right=False, top=False, left=True,
                      color='black')

    lower.grid(visible=True, which='both', axis='y', linestyle=':')
    lower.grid(visible=True, which='major', axis='x', linestyle=':')


    cms = upper.text(
        lower_label, max_y*1.08, u"CMS $\it{Preliminary}$",
        fontsize=16, fontweight='bold',
    )

    upper.text(
        upper_label, max_y*1.08, f'{LUMI[year]:.1f} fb$^{{-1}}$ (13 TeV)',
        fontsize=14,
    )

    if channel == 'electron':
        text_channel = r'ee '+u'channel'
    elif channel == 'muon':
        text_channel = r'$\mu \mu$ '+u'channel'

    upper.text(
        lower_label*0.95, max_y*0.25, text_channel,
        fontsize=14,
    )

    upper.text(
        lower_label*0.95,max_y*0.1,r'$\chi^{2}$/ndf = '+f'{chi2:.2f}/{nBins} = {chi2/nBins:.2f}',
        fontsize=8,
    )

    upper.legend(bbox_to_anchor=(1, 1), loc=1, fontsize=9, ncol=2, frameon=False)
    fig.savefig(os.path.join(outdir, f'{name}_{year}.png'), bbox_inches='tight')
    plt.close()
    #plt.show()

def plot_systematics(for_plot, year, channel, outdir=''):
    log=True

    for idx in range(1, len(for_plot), 2):
        fig, axes = plt.subplots(nrows=2, dpi=150, figsize=(5, 5), gridspec_kw={'height_ratios': [5, 1]})
        binc = (for_plot.iloc[0].bins[:-1] + for_plot.iloc[0].bins[1:]) / 2

        axes[0].hist(binc, bins=for_plot.iloc[0].bins, weights=for_plot.iloc[0].Sys, log=log, label='BDT score',
                 histtype='step', color='black')
        axes[0].hist(binc, bins=for_plot.iloc[0].bins, weights=for_plot.iloc[idx].Sys, log=log, label='Up',
                 histtype='step', linestyle='--', color='blue')
        axes[0].hist(binc, bins=for_plot.iloc[0].bins, weights=for_plot.iloc[idx + 1].Sys, log=log, label='Down',
                 histtype='step', linestyle='--', color='red')

        axes[1].step(binc, for_plot.iloc[idx].Sys/for_plot.iloc[0].Sys, color='blue')#linestyle='--', color='blue')
        axes[1].step(binc, for_plot.iloc[idx + 1].Sys/for_plot.iloc[0].Sys, color='red')#linestyle='--', color='red')
        axes[1].plot([0, 1], [1,1],linestyle=':', color='black')

        axes[0].set_ylabel("Events/bin", y=1, ha='right')
        axes[0].tick_params(axis='both', which='major', direction='in',
                          bottom=True, right=False, top=False, left=True,
                          color='black')
        axes[0].tick_params(axis='both', which='minor', direction='in',
                          bottom=True, right=False, top=False, left=True,
                          color='black')
        axes[1].set_xlabel(for_plot.iloc[idx].systematic_name, x=1, ha='right')
        axes[1].set_ylabel("Ratio", fontsize = 10)

        yacine1 = np.abs(np.max(for_plot.iloc[idx].Sys/for_plot.iloc[0].Sys) - 1)
        yacine2 = np.abs(np.min(for_plot.iloc[idx + 1].Sys/for_plot.iloc[0].Sys) - 1)
        nick = np.maximum(yacine1, yacine2)
        axes[1].set_ylim((1-nick)*0.9,(1+nick)*1.1)

    #    axes[1].xaxis.set_minor_locator(AutoMinorLocator())
    #    axes[1].yaxis.set_minor_locator(AutoMinorLocator())
        axes[1].tick_params(axis='both', which='major', direction='in', length=4,
                          bottom=True, right=False, top=False, left=True,
                          color='black')
        axes[1].tick_params(axis='both', which='minor', direction='in', length=2,
                          bottom=True, right=False, top=False, left=True,
                          color='black')

        axes[1].grid(visible=True, which='both', axis='y', linestyle=':')
        axes[1].grid(visible=True, which='major', axis='x', linestyle=':')

        max_y = axes[0].get_ylim()[1]
        max_x = max(binc)
        min_x = min(binc)
        x_range = max_x - min_x
        lower_label = min_x - x_range*0.05
        upper_label = max_x - x_range*0.35

        cms = axes[0].text(
            lower_label, max_y*1.08, u"CMS $\it{Preliminary}$",
            fontsize=16, fontweight='bold',
        )

        axes[0].text(
            upper_label, max_y*1.08, f'{LUMI[year]:.1f} fb$^{{-1}}$ (13 TeV)',
            #upper_label, max_y*1.08, '36 fb$^{{-1}}$ (13 TeV)',
            fontsize=14,
        )

        if channel == 'electron':
            text_channel = r'ee '+u'channel'
        elif channel == 'muon':
            text_channel = r'$\mu \mu$ '+u'channel'

        axes[0].text(
            0.05, max_y*0.65, text_channel,
            fontsize=10, color='dimgrey'
        )

        axes[0].legend(bbox_to_anchor=(1, 1), loc=1, fontsize=9, frameon=False)

        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        #os.mkdir(outdir)
        fig.savefig(os.path.join(outdir, f'{for_plot.iloc[idx].systematic_name}_{year}.png'), bbox_inches='tight')
        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Vivan\'s plotting tool')
    parser.add_argument('--sample_dir', type=str, required=False,
                        default='/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v1/2016/',
                        help='Directory containing sample files.')
    parser.add_argument('--hist_dir', type=str, required=False,
                        default='/eos/user/v/vinguyen/coffeafiles/2016-fixed-rename/',
                        help='Directory containing histogram files.')
    parser.add_argument('--xfile', type=str, required=False,
                        default='/afs/cern.ch/work/v/vinguyen/private/CMSSW_10_6_4/src/PhysicsTools/MonoZ/data/xsections_2016.yaml',
                        help='File containing cross sections.')
    parser.add_argument('--year', type=str, required=False,
                        default='2016',
                        help='Run year')
    parser.add_argument('--series', action='store_true', required=False,
                        help='Make plots in series.')
    parser.add_argument('--outdir', type=str, required=False, default=None,
                        help='Path to save plots.')
    parser.add_argument('--nonorm', action='store_true', required=False,
                        help='Turns off using background normalizations.')
    parser.add_argument('--filter', dest='filter_categories', action='store_true', required=False,
                        help='Filters QCD, NonResVBF, Radion, Graviton, NonRes, NonResSM categories.')
    parser.add_argument('--channel', type=str, choices=['muon', 'electron'], required=True,
                        help='Muon or Electron channel')
    parser.add_argument('--btag_filename', type=str, required=False, default='btag_weights',
                        help='File containing btag weights by jet bin')
    parser.add_argument('--btag', action='store_true', required=False,
                        help='Calculates btag weights by jet bin')
    parser.add_argument('--btag_overwrite', action='store_true', required=False,
                        help='Overwrite btag weights file.')
    parser.add_argument('--yields', action='store_true', required=False,
                        help='Print yields for select samples.')
    parser.add_argument('--finalselection', action='store_true', required=False,
                        help='Apply background norms at preselection.')
    parser.add_argument('--bdtscores', action='store_true', required=False,
                        help='Store BDT scores to csv file for binning optimization.')
    parser.add_argument('--datacard', action='store_true', required=False,
                        help='Plot full BDT score discribution, save QCD C region to root file.')
    args = parser.parse_args()


    if args.btag and not args.nonorm:
        raise ValueError('Cannot compute btag weights with normalizations.')

    if args.outdir is not None and not os.path.isdir(args.outdir):
        print(f'{args.outdir} does not exist. Making new directory.')
        os.mkdir(args.outdir)
    if args.outdir is None:
        outdir = ''
    else:
        outdir = args.outdir

    logging.info(f'Histogram directory: {args.hist_dir}')
    logging.info(f'Sample directory: {args.sample_dir}')
    logging.info(f'Cross-section file: {args.xfile}')
    logging.info(f'Year: {args.year}')

    xsections = get_xsections(args.xfile)
    sample_paths, hist_paths = cleanup(args.sample_dir, args.hist_dir, xsections)
    histograms = get_histograms(hist_paths, args.year, args.channel)
    normalizations = get_normalizations(sample_paths, xsections, list(histograms.keys()), args.year)

    df = get_bins_and_event_yields(histograms, normalizations, args.year, filter_categories=args.filter_categories, print_yields=args.yields)
#    df_sys = calculate_sys(histograms, normalizations, args.year, filter_categories=True, print_yields=args.yields)

#    df_sys['Sys'] = df_sys[['DY','TT','SMHiggs','SingleTop','ttV']].sum(axis=1)
#    df_sys.to_csv('isa.csv', index=False)
##    df_nom = pd.DataFrame({'nom': [df_sys['Sys'].iloc[0]] * len(df_sys['Sys'])})
#    df_nom = df_sys.iloc[0][['DY','TT','SMHiggs','SingleTop','ttV']].to_frame().sum(axis=1).values
#    #df_sys['err'] = df_sys['Sys'] - df_nom['nom']
#    #print("histogram names", df_sys['sample_name'])
#    #print("df sys sum", df_sys['Sys'].shape)
#    #print("df sys nom", nom)
#    #print("df sys err", df_sys['err'])
#    #testdfd = {col: [np.fromstring(df_sys[col].iloc[i][1:-1], sep='\n') for i, _ in enumerate(df_sys['Sys'])] for col in df_sys if col != 'sample_name'}
#    #test = pd.DataFrame(testdfd)
#    #test['sample_name'] = df_sys['sample_name']
#    dumb = {'dumb': [df_sys['Sys'].iloc[0]] * len(df_sys['Sys'])}
#    dfdumb = pd.DataFrame(dumb)
#    df_sys['new'] = np.abs((df_sys['Sys'] - dfdumb['dumb'])/ dfdumb['dumb'])
#    minbin = []
#    maxbin = []
#
#    for a in df_sys[['new']].values:
#        #print(a)
#        minbin.append(np.round(np.min(a[0]),3))
#        maxbin.append(np.round(np.max(a[0]),3))
#        #print(np.min(a[0]))
#        #print(np.max(a[0]))
#
#    df_sys['min'] = minbin
#    df_sys['max'] = maxbin
#
#    dropc = df_sys[df_sys["sample_name"].str.contains("QCD_C") == False]
#
#    dftable = (dropc.copy()
#              .reset_index()
#              .assign(systematic_name=lambda x: x.sample_name.str.split('_').str[2:].str.join('_')))
#
#    for_table = dftable[['systematic_name', 'min', 'max']].iloc[1::].to_latex(index=False)
#
#    with open(f'latextable{args.year}{args.channel}.txt', 'w') as f:
#        f.write(for_table)
#
#    for_plot = (dropc.copy()
#                .reset_index()
#                .assign(systematic_name=lambda x: x.sample_name.str.split('_').str[2:].str.join('_')
#                        .str.replace('Up', '').str.replace('Down', '')))
#
#    plot_systematics(for_plot, args.year, args.channel, outdir= f'systematicstest{args.channel}{args.year}')

    if args.nonorm:
        df['QCD_estimate'] = df.apply(lambda x: np.zeros(shape=x['Data'].shape[0]), axis=1)
        bkgd_norm = np.array([1.0, 1.0, 1.0])
       # btag_ratio_plot(df, args.year, outdir=outdir)
        if args.btag:
            btag_path = os.path.join(os.getcwd(), Path(args.btag_filename).stem + '.jsonl')
            btag_ratio(df, args.year, btag_path, args.btag_overwrite)
    else:
        if args.finalselection:
            #for tt bar reweight and QCD positive bins only
            if args.channel == 'muon':
                if args.year == '2016': bkgd_norm = (1.44, 1.34, 0.989)
                if args.year == '2017': bkgd_norm = (1.50, 1.88, 1.26)
                if args.year == '2018': bkgd_norm = (1.51, 1.68, 1.41)
            if args.channel == 'electron':
                if args.year == '2016': bkgd_norm = (0.997, 1.43, 1.01)
                if args.year == '2017': bkgd_norm = (1.35, 1.81, 1.17)
                if args.year == '2018': bkgd_norm = (1.10, 1.63, 1.12)
            #if args.channel == 'muon':
            #    if args.year == '2016': bkgd_norm = (1.47, 1.32, 1.00)
            #    if args.year == '2017': bkgd_norm = (1.52, 1.81, 1.21)
            #    if args.year == '2018': bkgd_norm = (1.53, 1.64, 1.38)
            #if args.channel == 'electron':
            #    if args.year == '2016': bkgd_norm = (0.996, 1.41, 1.03)
            #    if args.year == '2017': bkgd_norm = (1.29, 1.74, 1.11)
            #    if args.year == '2018': bkgd_norm = (1.08, 1.58, 1.09)
            #for tt bar reweight
            #if args.channel == 'muon':
            #    if args.year == '2016': bkgd_norm = (1.47, 1.32, 1.01)
            #    if args.year == '2017': bkgd_norm = (1.52, 1.8, 1.24)
            #    if args.year == '2018': bkgd_norm = (1.53, 1.64, 1.41)
            #if args.channel == 'electron':
            #    if args.year == '2016': bkgd_norm = (0.97, 1.41, 1.05)
            #    if args.year == '2017': bkgd_norm = (1.26, 1.74, 1.15)
            #    if args.year == '2018': bkgd_norm = (1.08, 1.58, 1.09)
            #before ttbar reweight
            #if args.channel == 'muon':
            #    if args.year == '2016': bkgd_norm = (1.46, 1.33, 0.974)
            #    if args.year == '2017': bkgd_norm = (1.52, 1.81, 1.22)
            #    if args.year == '2018': bkgd_norm = (1.53, 1.65, 1.12)
            #if args.channel == 'electron':
            #    if args.year == '2016': bkgd_norm = (1.01, 1.42, 1.01)
            #    if args.year == '2017': bkgd_norm = (1.25, 1.74, 1.14)
            #    if args.year == '2018': bkgd_norm = (1.08, 1.59, 1.04)
        else:
            bkgd_norm = estimate_background(df)

        df = scale_cregions(df, *bkgd_norm)

    logging.info('Making plots.')

    if not args.series:
        num_cpus = os.cpu_count()
        batch_size = 1 #len(all_bins) // num_cpus + 1
        (Parallel(n_jobs=num_cpus, batch_size=batch_size)
         (delayed(new_plotting)(df.iloc[rowidx], bkgd_norm, args.year, args.channel, outdir=outdir, print_yields=args.yields, datacard=args.datacard, bdtscores_to_csv=args.bdtscores)
         for rowidx in range(df.shape[0])))
    else:
        for rowidx in range(df.shape[0]):
            new_plotting(df.iloc[rowidx], bkgd_norm, args.year, args.channel, outdir=outdir, print_yields=args.yields, datacard=args.datacard, bdtscores_to_csv=args.bdtscores)

    logging.info(f'Finished making plots and saved to {outdir}.')
    if args.datacard:
        fname = f'QCD_estimate_{args.year}.root'
        f = uproot.recreate(fname, compression=uproot.ZLIB(4))
        for rowidx in range(df.shape[0]):
            qcd_estimate = df.iloc[rowidx]['QCD_estimate']
            name = df.iloc[rowidx]['sample_name']
            bins = df.iloc[rowidx]['bins']
            binc = bins[:-1] + np.diff(bins) * 0.5
            f[f'{name}'] = np.histogram(binc, bins=bins, weights=qcd_estimate)
        f.close()

        logging.info(f'QCD estimate histograms saved to root file {fname}.')

if __name__ == '__main__':
	main()
