#!/usr/bin/env/ python
#Plotter for HH variables
#%config InlineBackend.figure_format = 'retina'
import sys
print(sys.version)
print(sys.version_info)

import os
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
from scipy import interpolate
from scipy.optimize import curve_fit, fsolve
import uproot
import uproot_methods

COLORS=list(sns.color_palette('Set2').as_hex())
#plt.style.use('physics.mplstyle')


def get_histograms(hist_dir):
    histograms={}

    for filename in os.listdir(hist_dir):
        samplename = filename.split('_WS')[0]

        fn_sample = os.path.join(hist_dir, filename)
        f_sample = uproot.open(fn_sample)
        histogram_sample = f_sample.allitems( filterclass=lambda cls: issubclass(cls, uproot_methods.classes.TH1.Methods))

        histograms[samplename] = histogram_sample
    return histograms

def xs_scale(xsections, ufile, proc):
    xsec  = xsections[proc]["xsec"]
    xsec *= xsections[proc]["kr"]
    xsec *= xsections[proc]["br"]
    xsec *= 1000.0
    #print (proc, xsec)
    assert xsec > 0, "{} has a null cross section!".format(proc)
    scale = 36000/ufile["Runs"].array("genEventSumw").sum()
    return scale

def get_xsections(infile):
    with open(infile) as f:
        try:
            xsections = yaml.safe_load(f.read())
        except yaml.YAMLError as exc:
            print(exc)
    return xsections

def get_normalizations(samples_directory, xsections, histogram_names):
    norm_dict = {}

    for fn in os.listdir(samples_directory):
        if "Single" in fn: continue
        fn_sample = os.path.join(samples_directory, fn)
        _proc = os.path.basename(fn).replace(".root","")
        _file = uproot.open(fn_sample)
        if ("DoubleEG") in fn:
            _scale = 1
        elif ("DoubleMuon") in fn:
            _scale = 1
        else:
            _scale  = xs_scale(xsections, ufile=_file, proc=_proc)
        for idx, name in enumerate(histogram_names):
            if name in fn:
                norm_dict[name] = _scale
                del histogram_names[idx]
                break

    return(norm_dict)

############### THESE NUMBERS SCALE BY LUMI/GENSUMWEIGHT. THE XSEC IS ALREADY SCALED IN THE PLOT.................. FROM COFFEA OKAY

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

def normalize_event_yields(event_yields, normalizations):
# Apply genweight/lumi
    Data =  ( event_yields['DoubleMuon_Run2016B']
        +event_yields['DoubleMuon_Run2016C']
        +event_yields['DoubleMuon_Run2016D']
        +event_yields['DoubleMuon_Run2016E']
        +event_yields['DoubleMuon_Run2016F']
        +event_yields['DoubleMuon_Run2016G']
        +event_yields['DoubleMuon_Run2016H']
        )

#    DY = 0
#    TT = 0
#    SMHiggs = 0
#    for sample in event_yields:
#	    if 'DY' in sample:
#		    DY += normalizations[sample] * event_yields[sample]
#	    elif 'WToLNu' in sample:
#		    Wjets += normalizations[sample] * event_yields[sample]
#	    else:
#		    SMHiggs += ...
    DY =   ( normalizations['DYJetsToLL_M-10to50']*event_yields['DYJetsToLL_M-10to50']
            +1.26826142e-07*event_yields['DYToLL_0J']
            +1.60262147e-07*event_yields['DYToLL_1J']
            +3.6089398e-07*event_yields['DYToLL_2J']
            )

    TT = 0.0004714*event_yields['TT']

#    SMHiggs = ( 1.001376*event_yields['histogram_ggZH_HToBB_ZToLL']
#            +8.526278706823073*event_yields['histogram_GluGluHToBB']
#            #+3165.5765176496025*event_yields['histogram_GluGluHToWWTo2L2Nu']
#            +0.4364072814562913*event_yields['histogram_GluGluHToZZTo4L']
#            #+210.7766369298098*event_yields['histogram_GluGluZH_HToWW']
#            +20.879793367043888*event_yields['histogram_VBFHToBB']
#            +74.1920460729453*event_yields['histogram_VBFHToWWTo2L2Nu']
#            +0.01975231809072535*event_yields['histogram_VBF_HToZZTo4L']
#            +0.03149248440712202*event_yields['histogram_WminusH_HToZZTo4L']
#            +0.03329174080543033*event_yields['histogram_WplusH_HToZZTo4L']
#            +41.765804251004504*event_yields['histogram_ZH_HToBB_ZToLL']
#            +1.9776479867231058*event_yields['histogram_WH_HToBB_WToLNu']
#            )
#
#    SingleTop = ( 12.567585731103017*event_yields['histogram_ST_s-channel_4f']
#            +163.98185299274917*event_yields['histogram_ST_t-channel_antitop_4f']
#            +153.81183643335171*event_yields['histogram_ST_t-channel_top_4f']
#            +186.15065654670195*event_yields['histogram_ST_tW_antitop_5f']
#            +185.62225741173017*event_yields['histogram_ST_tW_top_5f']
#            )

#    VV = ( 4.079802092006373*event_yields['histogram_WWTo1L1Nu2Q']
#            +0.832523649032172*event_yields['histogram_WWTo4Q_4f']
#            +0.9135511958978306*event_yields['histogram_WZTo1L1Nu2Q']
#            +11.668208997480251*event_yields['histogram_WZTo1L3Nu']
#            +0.8630237624695615*event_yields['histogram_WZTo2L2Q']
#            +1.2560781747809395*event_yields['histogram_WZTo2Q2Nu']
#            +1.6364191282153688*event_yields['histogram_WZTo3LNu']
#            +1.4776727876795839*event_yields['histogram_ZZTo2L2Q']
#            +0.7307601577135954*event_yields['histogram_ZZTo2Q2Nu']
#            +2.130537286919005*event_yields['histogram_ZZTo4L']
#            +0.729250548985396*event_yields['histogram_ZZTo4Q']
#            +6.815110101472653*event_yields['histogram_VVTo2L2Nu']
#            )

#    ttV = ( 0.7634861609544465*event_yields['histogram_ttHJetToNonbb']
#            +1.090205294107825*event_yields['histogram_ttHJetTobb']
#            +1.4608520317358018*event_yields['histogram_ttHToNonbb']
#            +2.019714054725842*event_yields['histogram_ttHTobb']
#            +6.1098061413845555*event_yields['histogram_TTWJetsToLNu']
#            +23.455275139374876*event_yields['histogram_TTWJetsToQQ']
#            +17.438734129906862*event_yields['histogram_TTZToLLNuNu_M-10']
#            +54.49853453413941*event_yields['histogram_TTZToQQ']
#          )

#    Wjets = ( 0.6301590987380689*event_yields['histogram_WToLNu_0J']
#            +0.16810126746176954*event_yields['histogram_WToLNu_1J']
#            +0.09175339649542336*event_yields['histogram_WToLNu_2J']
#            )

    return Data, DY, TT

def get_bins_and_event_yields(histograms, normalizations):
    all_bins = []
    things_to_plot = {'DY': [], 'Data':[], 'TT': []}
    names = []
    for idx, (name, roothist) in enumerate(histograms['DoubleMuon_Run2016B']):
        if "genEventSumw" == name:
            print("bye")
            continue

        event_yields = {}
        for key, value in histograms.items():
            event_yields[key] = np.abs(value[idx][1].numpy())[0]# * normalizations[key]
        Data, DY, TT = normalize_event_yields(event_yields, normalizations)
        things_to_plot['DY'].append(DY)
        things_to_plot['Data'].append(Data)
        things_to_plot['TT'].append(TT)
        bins = roothist.numpy()[1]
        all_bins.append(roothist.numpy()[1])
        names.append(name)
    return names, all_bins, things_to_plot

def new_plotting(idx, name, bins, Data=None, DY=None, TT=None):
    fig, axarr = plt.subplots(2, dpi=150, figsize=(6, 5), sharex=True,
                              gridspec_kw={'hspace': 0.05, 'height_ratios': (0.8,0.2)},
                              constrained_layout=False)
    upper = axarr[0]
    lower = axarr[1]
    name = name.decode("utf-8")
    name = name.replace(";1", "")


    MC = DY+TT
    Data = Data
    ratio = Data/MC
    binc = np.array([ 0.5*(bins[j]+bins[j+1])for j in range(bins.shape[0]-1)])
    binc = bins[:-1] + np.diff(bins) * 0.5
    xerr = np.diff(bins)*0.5

    upper.errorbar(binc, Data, xerr = None, yerr = np.sqrt(Data), fmt = 'o',
                   zorder=10, color='black', label='Data', markersize=3)

    all_weights = np.vstack([DY, TT]).transpose()
    all_x = np.vstack([binc] * all_weights.shape[1]).transpose()
    labels = ['TT', 'DY'][::-1]

    upper.hist(x=all_x, bins=bins, weights=all_weights,
               histtype='stepfilled', edgecolor='black', zorder=1,
               stacked=True, color=COLORS[:all_weights.shape[1]], label=labels)

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
    lower.set_ylim(0, 2)
    yerr = np.sqrt(Data) / MC / 1000
    lower.errorbar(binc, ratio, yerr = yerr, marker = '.', color = 'black', linestyle ='none')
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

    lumi = upper.text(
        upper_label, max_y*1.08, r"%.1f fb$^{-1}$ (13 TeV)" % 36,
        fontsize=14,
    )

    upper.legend()
    fig.savefig("{}.png".format(re.sub('_QCD','',name)),bbox_inches='tight')
    #fig.savefig("{}.pdf".format(re.sub('_QCD','',name)))
    plt.close()
    #plt.show()

# Plotting
def plotting_func(idx, name, roothist, histograms, normalizations):
    fig, axarr = plt.subplots(2, dpi=150, figsize=(6, 5), sharex=True,
                              gridspec_kw={'hspace': 0,'height_ratios': (0.8,0.2)},
                              constrained_layout=True)
    upper = axarr[0]
    lower = axarr[1]
    name = name.decode("utf-8")
    name = name.replace(";1", "")
    print(name)

    if "genEventSumw" == name:
        print("bye")
        return

    event_yields = {}
    for key, value in histograms.items():
        event_yields[key] = np.abs(value[idx][1].numpy())[0]# * normalizations[key]
    Data, DY, TT = normalize_event_yields(event_yields, normalizations)

 #   Other = VV + SingleTop + Wjets +ttV

    MC = (DY+TT)

    ratio = Data/MC

    bins = roothist.numpy()[1]
    binc = np.array([ 0.5*(bins[j]+bins[j+1])for j in range(bins.shape[0]-1)])
    binc = bins[:-1] + np.diff(bins) * 0.5
    xerr = np.diff(bins)*0.5

    upper.errorbar(binc, Data, xerr = None, yerr = np.sqrt(Data), fmt = 'o',
                   zorder=10, color='black', label='Data', markersize=3)

    all_weights = np.vstack([DY, TT]).transpose()
    all_x = np.vstack([binc] * all_weights.shape[1]).transpose()
    labels = ['TT', 'DY'][::-1]
#     colors = [mcd.XKCD_COLORS["xkcd:cobalt"].upper(),
#               mcd.XKCD_COLORS["xkcd:green"].upper(), mcd.XKCD_COLORS["xkcd:periwinkle"].upper(),
#               mcd.XKCD_COLORS["xkcd:orange"].upper()][::-1]
    upper.hist(x=all_x, bins=bins, weights=all_weights,
               histtype='stepfilled', edgecolor='black', zorder=1,
               stacked=True, color=COLORS[:all_weights.shape[1]], label=labels)

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
    upper.set_xlabel(name, x=1, ha='right')
    upper.set_ylabel("Events/bin", y=1, ha='right')

    lower.set_ylabel("Data/MC", fontsize = 10)
    lower.set_ylim(0, 2)
    yerr = np.sqrt(Data) / MC / 1000
    lower.errorbar(binc, ratio, yerr = yerr, marker = '.', color = 'black', linestyle ='none')
    lower.plot([min_x, max_x],[1,1],linestyle=':', color = 'black')
    lower.grid(visible=True, which='major', axis='both')
    cms = upper.text(
        lower_label, max_y*1.08, u"CMS $\it{Preliminary}$",
        fontsize=16, fontweight='bold',
    )

    lumi = upper.text(
        upper_label, max_y*1.08, r"%.1f fb$^{-1}$ (13 TeV)" % 36,
        fontsize=14,
    )

    upper.legend()
    fig.savefig("{}.png".format(re.sub('_QCD','',name)),bbox_inches='tight')
    #fig.savefig("{}.pdf".format(re.sub('_QCD','',name)))
    plt.close()
    #plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Vivan\'s plotting tool')
    parser.add_argument('--sample_dir', type=str, required=False, default=None,
                        help='Directory containing sample files.')
    parser.add_argument('--hist_dir', type=str, required=False, default=None,
                        help='Directory containing histogram files.')
    parser.add_argument('--xfile', type=str, required=False, default=None,
                        help='File containing cross sections.')
    parser.add_argument('--parallel', action='store_true', required=False,
                        help='Make plots in parallel.')
    args = parser.parse_args()

    if args.hist_dir is None:
        directory = "/eos/user/v/vinguyen/coffeafiles/qcdtest_sf/"
    else:
        directory = args.hist_dir
    if args.sample_dir is None:
        samples_directory = "/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/test/qcdtest/merged/"
    else:
        samples_directory = args.sample_dir
    if args.xfile is None:
        xfile = '/afs/cern.ch/work/v/vinguyen/private/CMSSW_10_6_4/src/PhysicsTools/MonoZ/data/xsections_2016.yaml'
    else:
        xfile = args.xfile
    histograms = get_histograms(directory)
    xsections = get_xsections(xfile)
    normalizations = get_normalizations(samples_directory, xsections, list(histograms.keys()))

    #TODO Cant pickle TH1F
    if args.parallel:
        names, all_bins, things_to_plot = get_bins_and_event_yields(histograms, normalizations)

        num_cpus = os.cpu_count()
        batch_size = len(all_bins) // num_cpus + 1
        (Parallel(n_jobs=num_cpus, batch_size=batch_size)
         (delayed(new_plotting)(idx, name, bins, Data=things_to_plot['Data'][idx], DY=things_to_plot['DY'][idx], TT=things_to_plot['TT'][idx])
         for idx, (name, bins) in enumerate(zip(names, all_bins))))
        #Parallel(n_jobs=num_cpus, backend='multiprocessing', batch_size=batch_size)(delayed(plotting_func)(idx, name, roothist, histograms, normalizations)
        #         for idx, (name, roothist) in enumerate(histograms['DoubleMuon_Run2016B']))
    else:
        for idx, (name, roothist) in enumerate(histograms['DoubleMuon_Run2016B']):
            plotting_func(idx, name, roothist, histograms, normalizations)

if __name__ == '__main__':
	main()
