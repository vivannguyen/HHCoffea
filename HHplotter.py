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

COLORS=list(sns.color_palette('Set2').as_hex())
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

        if year == '2016':
            if any([s in filename for s in skip]): continue
#            if 'SingleElectron' in filename: continue
#            if 'DoubleEG' in filename: continue

        if year == '2017':
            if any([s in filename for s in skip]): continue
            if 'TTTo' in filename: continue
            #if 'SingleElectron' in filename: continue
            #if 'DoubleEG' in filename: continue

        if year == '2018':
            if any([s in filename for s in skip]): continue
            if 'ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-madgraph-pythia8' in filename: continue
            if 'TTTo' in filename: continue
            #if 'EGamma' in filename: continue

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
        if 'TTTo' in fn : continue
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
            for idx, name in enumerate(histogram_names):
                if name == Path(fn).stem:
                    norm_dict[name] = _scale
                    break
    logging.info('Finished obtaining normalizations.')
    logging.info(f'Number normalizations in memory: {len(norm_dict)}.')
# These normalizations scale by lumi/genweight. The xsec is already applied in the plots from coffea
    print('copy after deleting', len(histogram_names))
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

def normalize_event_yields(event_yields, normalizations, file_to_category):
    categorized_yields = {}

    for sample in event_yields:
        category = file_to_category[sample]
        if category not in categorized_yields:
            try:
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
            categorized_yields[category] += normalizations[sample] * event_yields[sample]

    return categorized_yields

def get_bins_and_event_yields(histograms, normalizations, year, filter_categories=False):
    with open(f'{year}_sample_reference.json', 'r') as infile:
        file_to_category = json.load(infile)

    categories = set(file_to_category.values())
    if filter_categories:
        for category in ['QCD', 'NonResVBF', 'Radion', 'Graviton', 'NonRes', 'NonResSM']:
            categories.remove(category)

    df_dict = {}
    df_dict['sample_name'] = []
    df_dict['bins'] = []
    for category in categories:
        df_dict[category] = []
    df_dict['Other'] = []

    logging.info('Getting bins and event yields.')

    arb_key = next(iter(histograms))

    for idx, (name, roothist) in enumerate(tqdm(histograms[arb_key])):
        name = name.decode("utf-8")
        name = name.replace(";1", "")

        # TODO make this more robust
        if "genEventSumw" == name:
            continue
        event_yields = {}
        # TODO Only one for loop here.
        for key, value in histograms.items():
            event_yields[key] = np.abs(value[idx][1].numpy())[0]
        output = normalize_event_yields(event_yields, normalizations, file_to_category)
        output['Other'] = output['VV']  + output['SingleTop'] + output['Wjets'] + output['ttV']

        for category in output:
            df_dict[category].append(output[category])

        df_dict['bins'].append(roothist.numpy()[1])
        df_dict['sample_name'].append(name)

    logging.info('Finished getting bins and event yields.')
    return pd.DataFrame(df_dict)

def estimate_background(all_event_yields, tol=1e-16, maxiter=50, disp=False, sigma=1):
    names_for_bkgd_est = np.array(['Zlep_cand_mass_DYcontrol_QCD_C', 'Zlep_cand_mass_TTcontrol_QCD_C',
                          'Zlep_cand_mass_DYcontrol', 'Zlep_cand_mass_TTcontrol',
                          'Zlep_cand_mass_QCD_B','Zlep_cand_mass_QCD_D'])
    df = all_event_yields.set_index('sample_name')
    df_subset = df.loc[names_for_bkgd_est]
    df_subset = df_subset.reset_index()

    dy_c, tt_c, dy, tt, qcd_b, qcd_d = *[df_subset.iloc[idx] for idx in range(df_subset.shape[0])],


    residual_func = lambda x, y, z, s: (x['Data'] + s * np.sqrt(x['Data'])
                                        - (x['DY'] * y + x['TT'] * z + x['SMHiggs'] + x['Other']))

    def optimizer(sigma):
        counter = 0
        qcd_norm = 100
        dy_norm = 100
        tt_norm = 100
        error = 100

        dy_data = dy['Data'] + sigma * np.sqrt(dy['Data'])
        tt_data = tt['Data'] + sigma * np.sqrt(tt['Data'])

        while (error > tol) and (counter < maxiter):
            dy_c_shape = residual_func(dy_c, 1, 1, sigma)
            tt_c_shape = residual_func(tt_c, 1, 1, sigma)

            updated_dy_norm = np.sum((dy_data - (qcd_norm * dy_c_shape + dy['TT'] + dy['SMHiggs'] + dy['Other'])) / np.sum(dy['DY']))
            updated_tt_norm = np.sum((tt_data - (qcd_norm * tt_c_shape + tt['DY'] + tt['SMHiggs'] + tt['Other'])) / np.sum(tt['TT']))

            qcd_b_val = np.sum(residual_func(qcd_b, dy_norm, tt_norm, sigma))
            qcd_d_val = np.sum(residual_func(qcd_d, dy_norm, tt_norm, sigma))

            updated_qcd_norm = qcd_b_val / qcd_d_val
            error = np.sqrt((updated_qcd_norm - qcd_norm)**2 + np.abs(updated_dy_norm - dy_norm)**2 + np.abs(updated_tt_norm - tt_norm)**2)

            #print(f'DY norm: {dy_norm}, TT norm: {tt_norm}, fBD: {qcd_norm}, err: {current_err}')

            qcd_norm = updated_qcd_norm
            dy_norm = updated_dy_norm
            tt_norm = updated_tt_norm

            counter += 1

        if disp and sigma == 0:
            print(f'qcd_norm: {qcd_norm} \n'
                  f'dy_norm: {dy_norm} \n'
                  f'tt_norm: {tt_norm} \n'
                  f'Error: {error} \n'
                  f'Converged: {error <= tol} \n'
                  f'Iterations: {counter}')

        return qcd_norm, dy_norm, tt_norm

    norms = optimizer(0)
    norms_upper = optimizer(sigma)
    norms_lower = optimizer(-sigma)
    return norms, norms_upper, norms_lower

def data_mc_residual(x, norm1=1, norm2=1):
    return (x['Data'] - (x['DY']  * norm1 + x['TT'] * norm2 + x['SMHiggs'] + x['Other']))

def scale_cregions (df, qcd_norm, dy_norm, tt_norm):
    df = df.copy()
    c_samples_bool = df['sample_name'].str.contains('QCD_C')
    c_samples_to_drop = df[c_samples_bool & ~df['sample_name'].str.contains('Zlep_cand_mass_QCD_C')]['sample_name']
    c_samples = df[c_samples_bool]['sample_name']
    df_subset = df[c_samples_bool].set_index('sample_name')
    residuals = (data_mc_residual(df_subset, dy_norm, tt_norm)) * qcd_norm
    df['QCD_estimate'] = df.apply(lambda x: np.zeros(shape=x['Data'].shape[0]), axis=1)
    df = df.set_index('sample_name')
    df.loc[residuals.index.str.replace('_QCD_C', ''), 'QCD_estimate'] = residuals.values
    df = df.drop(c_samples_to_drop).reset_index()
    return df

def new_plotting(event_yields, bkgd_norm, year, outdir=''):
    fig, axarr = plt.subplots(2, dpi=150, figsize=(6, 5), sharex=True,
                              gridspec_kw={'hspace': 0.05, 'height_ratios': (0.8,0.2)},
                              constrained_layout=False)
    upper = axarr[0]
    lower = axarr[1]

    # This gives the copy warning.
    event_yields['DY'] *= bkgd_norm[1]
    event_yields['TT'] *= bkgd_norm[2]

    mc_categories = ['DY', 'TT', 'SMHiggs', 'QCD_estimate']
    MC = event_yields[mc_categories].sum()
    Data = event_yields['Data']
    Other = event_yields['Other']
    name = event_yields['sample_name']
    bins = event_yields['bins']

    MC += Other

    # The first bin has a value of 0 and will give a warning.
    ratio = Data/MC
    binc = np.array([ 0.5*(bins[j]+bins[j+1])for j in range(bins.shape[0]-1)])
    binc = bins[:-1] + np.diff(bins) * 0.5
    xerr = np.diff(bins)*0.5

    upper.errorbar(binc, Data, xerr = None, yerr = np.sqrt(Data), fmt = 'o',
                   zorder=10, color='black', label='Data', markersize=3)
    all_weights = np.vstack([event_yields['SMHiggs'],
                             event_yields['Other'],
                             event_yields['QCD_estimate'],
                             event_yields['DY'],
                             event_yields['TT']]).transpose()
    all_x = np.vstack([binc] * all_weights.shape[1]).transpose()

    COLORMAP = {'SMhiggs': COLORS[0],
                'Other': COLORS[1],
                'DY': COLORS[2],
                'TT': COLORS[3],
                'QCD': COLORS[4],
                'QCD_estimate':COLORS[5]}

    labels = ['SMhiggs', 'Other', 'QCD', 'DY', 'TT']
    plotting_colors = [COLORMAP[s] for s in labels]

    upper.hist(x=all_x, bins=bins, weights=all_weights,
               histtype='stepfilled', edgecolor='black', zorder=1,
               stacked=True, color=plotting_colors, label=labels)

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

    upper.text(
        upper_label, max_y*1.08, f'{LUMI[year]:.1f} fb$^{-1}$ (13 TeV)',
        fontsize=14,
    )

    upper.legend()
    fig.savefig(os.path.join(outdir, f'{name}_{year}.png'), bbox_inches='tight')
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
    parser.add_argument('--year', type=str, required=False, default=None,
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
    args = parser.parse_args()

    if args.hist_dir is None:
        directory = "/eos/user/v/vinguyen/coffeafiles/2016-fixed-rename/"
    else:
        directory = args.hist_dir
    if args.sample_dir is None:
        samples_directory = "/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v1/2016/"
    else:
        samples_directory = args.sample_dir
    if args.xfile is None:
        xfile = '/afs/cern.ch/work/v/vinguyen/private/CMSSW_10_6_4/src/PhysicsTools/MonoZ/data/xsections_2016.yaml'
    else:
        xfile = args.xfile
    if args.year is None:
        year = '2016'
    else:
        year = args.year

    if args.outdir is not None and not os.path.isdir(args.outdir):
        print(f'{args.outdir} does not exist. Making new directory.')
        os.mkdir(args.outdir)
    if args.outdir is None:
        outdir = ''
    else:
        outdir = args.outdir

    logging.info(f'Histogram directory: {directory}')
    logging.info(f'Sample directory: {samples_directory}')
    logging.info(f'Cross-section file: {xfile}')
    logging.info(f'Year: {year}')

    xsections = get_xsections(xfile)
    sample_paths, hist_paths = cleanup(samples_directory, directory, xsections)
    histograms = get_histograms(hist_paths, year, args.channel)
    normalizations = get_normalizations(sample_paths, xsections, list(histograms.keys()), year)

    df = get_bins_and_event_yields(histograms, normalizations, year, filter_categories=args.filter_categories)

    if args.nonorm:
        df['QCD_estimate'] = df.apply(lambda x: np.zeros(shape=x['Data'].shape[0]), axis=1)
        bkgd_norm = np.array([1.0, 1.0, 1.0])
    else:
        bkgd_norm, bkgd_norm_upper, bkgd_norm_lower = estimate_background(df, disp=True, sigma=2)
        df = scale_cregions(df, *bkgd_norm)

    logging.info('Making plots.')

    if not args.series:
        num_cpus = os.cpu_count()
        batch_size = 1 #len(all_bins) // num_cpus + 1
        (Parallel(n_jobs=num_cpus, batch_size=batch_size)
         (delayed(new_plotting)(df.iloc[rowidx], bkgd_norm, year, outdir=outdir)
         for rowidx in range(df.shape[0])))
    else:
        for rowidx in range(df.shape[0]):
            new_plotting(df.iloc[rowidx], bkgd_norm, year, outdir=outdir)

    logging.info(f'Finished making plots and saved to {outdir}.')

if __name__ == '__main__':
	main()
