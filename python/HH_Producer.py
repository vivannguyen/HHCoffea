"""
HH_Producer.py
Workspace producers using coffea.
"""
import json

from coffea.hist import Hist, Bin, export1d
from coffea.processor import ProcessorABC, LazyDataFrame, dict_accumulator
from uproot3 import recreate

import awkward as ak
import numpy as np
import pandas as pd
import uproot3 as uproot
import xgboost as xgb
import yaml


class HH_NTuple(ProcessorABC):
    """
    A coffea Processor which produces a workspace.
    This applies selections and produces histograms from kinematics.
    """

    def __init__(self, isMC, era=2017, sample="DY", do_syst=False, syst_var='', weight_syst=False, haddFileName=None, flag=False, njetw=None, ttreweight=None):

        zlep_bin = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 132, 146, 164, 184, 209, 239, 275, 318, 370, 432]

        histograms = {
            'Zlep_cand_mass': {
                'target': 'Zlep_cand_mass',
                'name'  : 'Zlep_cand_mass',
                'region': ['signal'],
                'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': zlep_bin}
            },
            'Higgsbb_cand_mass': {
                'target': 'Higgsbb_cand_mass',
                'name'  : 'Higgsbb_cand_mass',
                'region': ['signal'],
                'axis': {'label': 'Higgsbb_cand_mass', 'n_or_arr': 35, 'lo': 100, 'hi': 800}
            },
            'Higgsbb_cand_pt': {
                'target': 'Higgsbb_cand_pt',
                'name'  : 'Higgsbb_cand_pt',
                'region': ['signal'],
                'axis': {'label': 'Higgsbb_cand_pt', 'n_or_arr': 20, 'lo': 0, 'hi': 500}
            },
            'HiggsZZ_cand_mass': {
                'target': 'HiggsZZ_cand_mass',
                'name'  : 'HiggsZZ_cand_mass',
                'region': ['signal'],
                'axis': {'label': 'HiggsZZ_cand_mass', 'n_or_arr': 35, 'lo': 100, 'hi': 800}
            },
            'leading_Hbb_pt': {
                'target': 'leading_Hbb_pt',
                'name'  : 'leading_Hbb_pt',
                'region': ['signal'],
                'axis': {'label': 'leading_Hbb_pt', 'n_or_arr': 20, 'lo': 30, 'hi': 430}
            },
            'leading_Hbb_btag': {
                'target': 'leading_Hbb_btag',
                'name'  : 'leading_Hbb_btag',
                'region': ['signal'],
                'axis': {'label': 'leading_Hbb_btag', 'n_or_arr': 20, 'lo': 0.3, 'hi': 1}
            },
            'trailing_Hbb_pt': {
                'target': 'trailing_Hbb_pt',
                'name'  : 'trailing_Hbb_pt',
                'region': ['signal'],
                'axis': {'label': 'trailing_Hbb_pt', 'n_or_arr': 20, 'lo': 30, 'hi': 230}
            },
            'trailing_Hbb_btag': {
                'target': 'trailing_Hbb_btag',
                'name'  : 'trailing_Hbb_btag',
                'region': ['signal'],
                'axis': {'label': 'trailing_Hbb_btag', 'n_or_arr': 20, 'lo': 0.3, 'hi': 1}
            },
            'met_pt': {
                'target': 'met_pt',
                'name'  : 'met_pt',
                'region': ['signal'],
                'axis': {'label': 'met_pt', 'n_or_arr': 15, 'lo': 0, 'hi': 75}
            },
            'dR_l1l2': {
                'target': 'dR_l1l2',
                'name'  : 'dR_l1l2',
                'region': ['signal'],
                'axis': {'label': 'dR_l1l2', 'n_or_arr': 25, 'lo': 0.5, 'hi': 5.5}
            },
            'dR_l1b1': {
                'target': 'dR_l1b1',
                'name'  : 'dR_l1b1',
                'region': ['signal'],
                'axis': {'label': 'dR_l1b1', 'n_or_arr': 25, 'lo': 0.5, 'hi': 5.5}
            },
            'dR_l1b2': {
                'target': 'dR_l1b2',
                'name'  : 'dR_l1b2',
                'region': ['signal'],
                'axis': {'label': 'dR_l1b2', 'n_or_arr': 25, 'lo': 0.5, 'hi': 5.5}
            },
            'dR_l1j1': {
                'target': 'dR_l1j1',
                'name'  : 'dR_l1j1',
                'region': ['signal'],
                'axis': {'label': 'dR_l1j1', 'n_or_arr': 25, 'lo': 0.5, 'hi': 5.5}
            },
            'dR_l1j2': {
                'target': 'dR_l1j2',
                'name'  : 'dR_l1j2',
                'region': ['signal'],
                'axis': {'label': 'dR_l1j2', 'n_or_arr': 25, 'lo': 0.5, 'hi': 5.5}
            },
## QCD C Region
            'Zlep_cand_mass_QCD_C': {
                'target': 'Zlep_cand_mass',
                'name'  : 'Zlep_cand_mass_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': zlep_bin}
            },
            'Higgsbb_cand_mass_QCD_C': {
                'target': 'Higgsbb_cand_mass',
                'name'  : 'Higgsbb_cand_mass_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'Higgsbb_cand_mass', 'n_or_arr': 35, 'lo': 100, 'hi': 800}
            },
            'Higgsbb_cand_pt_QCD_C': {
                'target': 'Higgsbb_cand_pt',
                'name'  : 'Higgsbb_cand_pt_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'Higgsbb_cand_pt', 'n_or_arr': 20, 'lo': 0, 'hi': 500}
            },
            'HiggsZZ_cand_mass_QCD_C': {
                'target': 'HiggsZZ_cand_mass',
                'name'  : 'HiggsZZ_cand_mass_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'HiggsZZ_cand_mass', 'n_or_arr': 35, 'lo': 100, 'hi': 800}
            },
            'leading_Hbb_pt_QCD_C': {
                'target': 'leading_Hbb_pt',
                'name'  : 'leading_Hbb_pt_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'leading_Hbb_pt', 'n_or_arr': 20, 'lo': 30, 'hi': 430}
            },
            'leading_Hbb_btag_QCD_C': {
                'target': 'leading_Hbb_btag',
                'name'  : 'leading_Hbb_btag_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'leading_Hbb_btag', 'n_or_arr': 20, 'lo': 0.3, 'hi': 1}
            },
            'trailing_Hbb_pt_QCD_C': {
                'target': 'trailing_Hbb_pt',
                'name'  : 'trailing_Hbb_pt_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'trailing_Hbb_pt', 'n_or_arr': 20, 'lo': 30, 'hi': 230}
            },
            'trailing_Hbb_btag_QCD_C': {
                'target': 'trailing_Hbb_btag',
                'name'  : 'trailing_Hbb_btag_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'trailing_Hbb_btag', 'n_or_arr': 20, 'lo': 0.3, 'hi': 1}
            },
            'met_pt_QCD_C': {
                'target': 'met_pt',
                'name'  : 'met_pt_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'met_pt', 'n_or_arr': 15, 'lo': 0, 'hi': 75}
            },
            'dR_l1l2_QCD_C': {
                'target': 'dR_l1l2',
                'name'  : 'dR_l1l2_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_l1l2', 'n_or_arr': 25, 'lo': 0.5, 'hi': 5.5}
            },
            'dR_l1b1_QCD_C': {
                'target': 'dR_l1b1',
                'name'  : 'dR_l1b1_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_l1b1', 'n_or_arr': 25, 'lo': 0.5, 'hi': 5.5}
            },
            'dR_l1b2_QCD_C': {
                'target': 'dR_l1b2',
                'name'  : 'dR_l1b2_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_l1b2', 'n_or_arr': 25, 'lo': 0.5, 'hi': 5.5}
            },
            'dR_l1j1_QCD_C': {
                'target': 'dR_l1j1',
                'name'  : 'dR_l1j1_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_l1j1', 'n_or_arr': 25, 'lo': 0.5, 'hi': 5.5}
            },
            'dR_l1j2_QCD_C': {
                'target': 'dR_l1j2',
                'name'  : 'dR_l1j2_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_l1j2', 'n_or_arr': 25, 'lo': 0.5, 'hi': 5.5}
            },
# BDT score
            'h_bdtscore' : {
                'target': 'BDTscore',
                'name': 'BDTscore',
                'region': ['signal'],
                #'axis': {'label': 'BDTscore', 'n_or_arr': bdt_bin}
                'axis': {'label': 'BDTscore', 'n_or_arr': 25, 'lo': 0., 'hi': 1.}
            },
            'h_bdtscore_QCD_C': {
                'target': 'BDTscore',
                'name'  : 'BDTscore_QCD_C',
                'region': ['QCD_C'],
                #'axis': {'label': 'BDTscore', 'n_or_arr': bdt_bin}
                'axis': {'label': 'BDTscore', 'n_or_arr': 25, 'lo': 0., 'hi': 1.}
            },
        }
        selection = {
                "signal" : [
                    "event.good_event{sys}     ==  1",
#                    "event.ngood_jets{sys}     >  3",
#                    "event.ngood_bjets{sys}     >  0",
                    "event.ngood_bjetsM{sys}     >  1",
                    "event.lep_category{sys}    == 1",
                    "event.event_category{sys}    == 1",
                    "event.leading_lep_pt{sys}  > 20",
                    "event.trailing_lep_pt{sys} > 10",
                    "event.Zlep_cand_mass{sys} > 15",
                    "event.Zlep_cand_pt{sys} > 60",
                    "event.leading_Hbb_pt{sys} > 20",
                    "event.trailing_Hbb_pt{sys} > 20",
                    "event.leading_jet_pt{sys} > 20",
                    "event.trailing_jet_pt{sys} > 20",
                    "event.met_pt{sys} < 75 ",
                    "event.dR_j1b1{sys} > 0",
                    "event.dR_j1b2{sys} > 0",
                    "event.dR_j2b1{sys} > 0",
                    "event.dR_j2b2{sys} > 0"
                ],
                "signal_btag" : [
                    "event.good_event{sys}     ==  1",
#                    "event.ngood_jets{sys}     >  3",
                    "event.lep_category{sys}    == 1",
                    "event.event_category{sys}    == 1",
                    "event.leading_lep_pt{sys}  > 20",
                    "event.trailing_lep_pt{sys} > 10",
                    "event.Zlep_cand_mass{sys} > 15",
                    "event.Zlep_cand_pt{sys} > 60",
                    "event.leading_Hbb_pt{sys} > 20",
                    "event.trailing_Hbb_pt{sys} > 20",
                    "event.leading_jet_pt{sys} > 20",
                    "event.trailing_jet_pt{sys} > 20",
                    "event.dR_j1b1{sys} > 0",
                    "event.dR_j1b2{sys} > 0",
                    "event.dR_j2b1{sys} > 0",
                    "event.dR_j2b2{sys} > 0"
                ],
                "QCD_B" : [
                    "event.good_event{sys}     ==  1",
#                    "event.ngood_jets{sys}     >  3",
                    "event.ngood_bjets{sys}     >  0",
                    "event.lep_category{sys}    == 1",
                    "event.event_category{sys}    == 2",
                    "event.leading_lep_pt{sys}  > 20",
                    "event.trailing_lep_pt{sys} > 10",
                    "event.Zlep_cand_mass{sys} > 15",
                    "event.Zlep_cand_pt{sys} > 60",
                    "event.leading_Hbb_pt{sys} > 20",
                    "event.trailing_Hbb_pt{sys} > 20",
                    "event.leading_jet_pt{sys} > 20",
                    "event.trailing_jet_pt{sys} > 20",
                    "event.dR_j1b1{sys} > 0",
                    "event.dR_j1b2{sys} > 0",
                    "event.dR_j2b1{sys} > 0",
                    "event.dR_j2b2{sys} > 0"
                ],
                "QCD_C" : [
                    "event.good_event{sys}     ==  1",
#                    "event.ngood_jets{sys}     >  3",
#                    "event.ngood_bjets{sys}     >  0",
                    "event.ngood_bjetsM{sys}     >  1",
                    "event.lep_category{sys}    == 1",
                    "event.event_category{sys}    == 3",
                    "event.leading_lep_pt{sys}  > 20",
                    "event.trailing_lep_pt{sys} > 10",
                    "event.Zlep_cand_mass{sys} > 15",
                    "event.Zlep_cand_pt{sys} > 60",
                    "event.leading_Hbb_pt{sys} > 20",
                    "event.trailing_Hbb_pt{sys} > 20",
                    "event.leading_jet_pt{sys} > 20",
                    "event.trailing_jet_pt{sys} > 20",
                    "event.met_pt{sys} < 75 ",
                    "event.dR_j1b1{sys} > 0",
                    "event.dR_j1b2{sys} > 0",
                    "event.dR_j2b1{sys} > 0",
                    "event.dR_j2b2{sys} > 0"
                ],
                "QCD_D" : [
                    "event.good_event{sys}     ==  1",
#                    "event.ngood_jets{sys}     >  3",
                    "event.ngood_bjets{sys}     >  0",
                    "event.lep_category{sys}    == 1",
                    "event.event_category{sys}    == 4",
                    "event.leading_lep_pt{sys}  > 20",
                    "event.trailing_lep_pt{sys} > 10",
                    "event.Zlep_cand_mass{sys} > 15",
                    "event.Zlep_cand_pt{sys} > 60",
                    "event.leading_Hbb_pt{sys} > 20",
                    "event.trailing_Hbb_pt{sys} > 20",
                    "event.leading_jet_pt{sys} > 20",
                    "event.trailing_jet_pt{sys} > 20",
                    "event.dR_j1b1{sys} > 0",
                    "event.dR_j1b2{sys} > 0",
                    "event.dR_j2b1{sys} > 0",
                    "event.dR_j2b2{sys} > 0"
                ],
                "DYcontrol" : [
                    "event.good_event{sys}     ==  1",
#                    "event.ngood_jets{sys}     >  3",
                    "event.ngood_bjets{sys}     >  0",
                    "event.lep_category{sys}    == 1",
                    "event.event_category{sys}    == 1",
                    "event.leading_lep_pt{sys}  > 20",
                    "event.trailing_lep_pt{sys} > 10",
                    "event.Zlep_cand_pt{sys} > 60",
                    "event.leading_Hbb_pt{sys} > 20",
                    "event.trailing_Hbb_pt{sys} > 20",
                    "event.leading_jet_pt{sys} > 20",
                    "event.trailing_jet_pt{sys} > 20",
                    "event.Zlep_cand_mass{sys} > 80",
                    "event.Zlep_cand_mass{sys} < 100",
                    "event.dR_j1b1{sys} > 0",
                    "event.dR_j1b2{sys} > 0",
                    "event.dR_j2b1{sys} > 0",
                    "event.dR_j2b2{sys} > 0"
                ],
                "DYcontrol_QCD_C" : [
                    "event.good_event{sys}     ==  1",
#                    "event.ngood_jets{sys}     >  3",
                    "event.ngood_bjets{sys}     >  0",
                    "event.lep_category{sys}    == 1",
                    "event.event_category{sys}    == 3",
                    "event.leading_lep_pt{sys}  > 20",
                    "event.trailing_lep_pt{sys} > 10",
                    "event.Zlep_cand_pt{sys} > 60",
                    "event.leading_Hbb_pt{sys} > 20",
                    "event.trailing_Hbb_pt{sys} > 20",
                    "event.leading_jet_pt{sys} > 20",
                    "event.trailing_jet_pt{sys} > 20",
                    "event.Zlep_cand_mass{sys} > 80",
                    "event.Zlep_cand_mass{sys} < 100",
                    "event.dR_j1b1{sys} > 0",
                    "event.dR_j1b2{sys} > 0",
                    "event.dR_j2b1{sys} > 0",
                    "event.dR_j2b2{sys} > 0"
                ],
                "TTcontrol" : [
                    "event.good_event{sys}     ==  1",
#                    "event.ngood_jets{sys}     >  3",
                    "event.ngood_bjets{sys}     >  0",
                    "event.lep_category{sys}    == 1",
                    "event.event_category{sys}    == 1",
                    "event.leading_lep_pt{sys}  > 20",
                    "event.trailing_lep_pt{sys} > 10",
                    "event.leading_Hbb_pt{sys} > 20",
                    "event.trailing_Hbb_pt{sys} > 20",
                    "event.leading_jet_pt{sys} > 20",
                    "event.trailing_jet_pt{sys} > 20",
                    "event.Zlep_cand_mass{sys} > 100",
                    "event.met_pt{sys} > 100",
                    "event.dR_j1b1{sys} > 0",
                    "event.dR_j1b2{sys} > 0",
                    "event.dR_j2b1{sys} > 0",
                    "event.dR_j2b2{sys} > 0"
                ],
                "TTcontrol_QCD_C" : [
                    "event.good_event{sys}     ==  1",
#                    "event.ngood_jets{sys}     >  3",
                    "event.ngood_bjets{sys}     >  0",
                    "event.lep_category{sys}    == 1",
                    "event.event_category{sys}    == 3",
                    "event.leading_lep_pt{sys}  > 20",
                    "event.trailing_lep_pt{sys} > 10",
                    "event.leading_Hbb_pt{sys} > 20",
                    "event.trailing_Hbb_pt{sys} > 20",
                    "event.leading_jet_pt{sys} > 20",
                    "event.trailing_jet_pt{sys} > 20",
                    "event.Zlep_cand_mass{sys} > 100",
                    "event.met_pt{sys} > 100",
                    "event.dR_j1b1{sys} > 0",
                    "event.dR_j1b2{sys} > 0",
                    "event.dR_j2b1{sys} > 0",
                    "event.dR_j2b2{sys} > 0"
                ],
            }
        self._flag = flag
        self.do_syst = do_syst
        self.era = era
        self.histograms = histograms
        self.selection = selection
        self.isMC = isMC
        self.sample = sample
        self.syst_var, self.syst_suffix = (syst_var, f'_sys_{syst_var}') if do_syst and syst_var else ('', '')
        self.weight_syst = weight_syst
        self._accumulator = dict_accumulator({
            name: Hist('Events', Bin(name=name, **axis))
            for name, axis in ((self.naming_schema(hist['name'], region), hist['axis'])
                               for _, hist in list(histograms.items())
                               for region in hist['region'])
        })
        self.outfile = haddFileName

        self.ttreweight = ttreweight
        self.njet_weights = None
        if njetw is not None:
            for line in open(njetw, 'r'):
                json_read = json.loads(line)
                if json_read['year'] == era:
                    self.njet_weights = np.fromiter(json_read['weights'].values(), dtype=np.float64)
                    self.njet_weights[self.njet_weights == -999] = 1
                    self.njet_weights = np.concatenate([self.njet_weights, np.tile(self.njet_weights[-1], 100)])
                    break
        #print('NJET WEIGHTS', self.njet_weights)

    def __repr__(self):
        return f'{self.__class__.__name__}(era: {self.era}, isMC: {self.isMC}, sample: {self.sample}, do_syst: {self.do_syst}, syst_var: {self.syst_var}, weight_syst: {self.weight_syst}, output: {self.outfile})'

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df, *args):
        features = ['met_pt','Higgsbb_cand_pt','Higgsbb_cand_mass','HiggsZZ_cand_mass','Zlep_cand_mass',
                    'leading_Hbb_pt','leading_Hbb_btag','trailing_Hbb_pt','trailing_Hbb_btag',
                    'dR_l1l2','dR_l1j1','dR_l1j2','dR_l1b1','dR_l1b2']

        year_str = str(self.era)

        X = df[features]
        X = ak.to_numpy(X).tolist()
        #load BDT model
        model = xgb.XGBClassifier()
        #model.load_model(f'models/{year_str}-uu-2btag-met75')
        model.load_model(f'{year_str}newfinalselecttest')
        bdtscore=model.predict_proba(X)[:,1]
        df['BDTscore']=bdtscore

        output = self.accumulator.identity()

        weight = self.weighting(df)
        nobtag_weight = weight
        btag_weight = self.btag_weighting(df, weight)

        if self.era == 2016: ttbar_ratio = 31268737/30821062
        if self.era == 2017: ttbar_ratio = 120435564254/118580050004
        if self.era == 2018: ttbar_ratio = 253757886068/249652709023

        if self.njet_weights is not None:
            weight = self.my_btag_weighting(df, btag_weight, self.njet_weights)
        if self.ttreweight:
            weight = self.tt_weighting(df,weight)
            weight = self.my_tt_weighting(df, weight, ttbar_ratio)

        nott_weight = weight #tt weight happens after btag weight applied
        tt_weight = self.tt_weighting(df,weight)


        for h, hist in list(self.histograms.items()):
            for region in hist['region']:
                name = self.naming_schema(hist['name'], region)
                selec = self.passbut(df, hist['target'], region)
                if name == 'ngood_jets_btagSF':
                    output[name].fill(**{
                        'weight':btag_weight[selec],
                        name: df[hist['target']][selec]#.flatten()
                    })
                elif name == 'ngood_jets_nobtagSF':
                    output[name].fill(**{
                        'weight':nobtag_weight[selec],
                        name: df[hist['target']][selec]#.flatten()
                    })
                elif name == 'Zlep_cand_mass_tt_weight':
                    output[name].fill(**{
                        'weight':tt_weight[selec],
                        name: df[hist['target']][selec]#.flatten()
                    })
                elif name == 'Zlep_cand_mass_nott_weight':
                    output[name].fill(**{
                        'weight':nott_weight[selec],
                        name: df[hist['target']][selec]#.flatten()
                    })
                else:
                    output[name].fill(**{
                        'weight':weight[selec],
                        name: df[hist['target']][selec]#.flatten()
                    })

        return output#, bdtscore

    def postprocess(self, accumulator):
        return accumulator

    def passbut(self, event: LazyDataFrame, excut: str, cat: str):
        """Backwards-compatible passbut."""
        return eval('&'.join('(' + cut.format(sys=('' if self.weight_syst else self.syst_suffix)) + ')' for cut in self.selection[cat] ))#if excut not in cut))

    def weighting(self, event: LazyDataFrame):
        weight = 1.0
        try:
            weight = event.xsecscale
        except:
            return "ERROR: weight branch doesn't exist"

        if self.isMC:
            if "puWeight" in self.syst_suffix:
                if "Up" in self.syst_suffix:
                    weight = weight * event.puWeightUp
                else:
                    weight = weight * event.puWeightDown
            else:
                weight = weight * event.puWeight

            # PDF uncertainty
            if "PDF" in self.syst_suffix:
                try:
                    if "Up" in self.syst_suffix:
                        weight = weight * event.pdfw_Up
                    else:
                        weight = weight * event.pdfw_Down
                except:
                    pass

            # QCD Scale weights
            if "QCDScale0" in self.syst_suffix:
                try:
                    if "Up" in self.syst_suffix:
                        weight = weight * event.QCDScale0wUp
                    else:
                        weight = weight * event.QCDScale0wDown
                except:
                    pass
            if "QCDScale1" in self.syst_suffix:
                try:
                    if "Up" in self.syst_suffix:
                        weight = weight * event.QCDScale1wUp
                    else:
                        weight = weight * event.QCDScale1wDown
                except:
                    pass
            if "QCDScale2" in self.syst_suffix:
                try:
                    if "Up" in self.syst_suffix:
                        weight = weight * event.QCDScale2wUp
                    else:
                        weight = weight * event.QCDScale2wDown
                except:
                    pass

            #Muon SF
            if "MuonSF" in self.syst_suffix:
                if "Up" in self.syst_suffix:
                    weight = weight * event.w_muon_SFUp
                else:
                    weight = weight * event.w_muon_SFDown
            else:
                weight = weight * event.w_muon_SF

            # Electron SF
            if "ElectronSF" in self.syst_suffix:
                if "Up" in self.syst_suffix:
                    weight = weight * event.w_electron_SFUp
                else:
                    weight = weight * event.w_electron_SFDown
            else:
                weight = weight * event.w_electron_SF

            #Prefire Weight
            try:
                if "PrefireWeight" in self.syst_suffix:
                    if "Up" in self.syst_suffix:
                        weight = weight * event.PrefireWeight_Up
                    else:
                        weight = weight * event.PrefireWeight_Down
                else:
                    weight = weight * event.PrefireWeight
            except:
                pass

            #TriggerSFWeight
            if "TriggerSFWeight" in self.syst_suffix:
                if "Up" in self.syst_suffix:
                    weight = weight * event.TriggerSFWeight_Up
                else:
                    weight = weight * event.TriggerSFWeight_Down
            else:
                weight = weight * event.TriggerSFWeight

            # Signal Sample reweighting
            try:
                weight = weight * event.weight_cHHH1
            except:
                pass

        return weight

    def btag_weighting(self, event: LazyDataFrame, weight):
        if self.isMC:
            
            btag_mapping = {"_sys_hfUp": "_sys_up_hf", "_sys_hfDown": "_sys_down_hf",
            "_sys_lfUp": "_sys_up_lf", "_sys_lfDown": "_sys_down_lf",
            "_sys_cferr1Up": "_sys_up_cferr1", "_sys_cferr1Down": "_sys_down_cferr1",
            "_sys_cferr2Up": "_sys_up_cferr2", "_sys_cferr2Down": "_sys_down_cferr2",
            "_sys_hfstats1_2016Up": "_sys_up_hfstats1", "_sys_hfstats1_2016Down": "_sys_down_hfstats1",
            "_sys_hfstats2_2016Up": "_sys_up_hfstats2", "_sys_hfstats2_2016Down": "_sys_down_hfstats2",
            "_sys_lfstats1_2016Up": "_sys_up_lfstats1", "_sys_lfstats1_2016Down": "_sys_down_lfstats1",
            "_sys_lfstats2_2016Up": "_sys_up_lfstats2", "_sys_lfstats2_2016Down": "_sys_down_lfstats2",
            "_sys_hfstats1_2017Up": "_sys_up_hfstats1", "_sys_hfstats1_2017Down": "_sys_down_hfstats1",
            "_sys_hfstats2_2017Up": "_sys_up_hfstats2", "_sys_hfstats2_2017Down": "_sys_down_hfstats2",
            "_sys_lfstats1_2017Up": "_sys_up_lfstats1", "_sys_lfstats1_2017Down": "_sys_down_lfstats1",
            "_sys_lfstats2_2017Up": "_sys_up_lfstats2", "_sys_lfstats2_2017Down": "_sys_down_lfstats2",
            "_sys_hfstats1_2018Up": "_sys_up_hfstats1", "_sys_hfstats1_2018Down": "_sys_down_hfstats1",
            "_sys_hfstats2_2018Up": "_sys_up_hfstats2", "_sys_hfstats2_2018Down": "_sys_down_hfstats2",
            "_sys_lfstats1_2018Up": "_sys_up_lfstats1", "_sys_lfstats1_2018Down": "_sys_down_lfstats1",
            "_sys_lfstats2_2018Up": "_sys_up_lfstats2", "_sys_lfstats2_2018Down": "_sys_down_lfstats2"}

            btag_name = "w_btag_SF" + btag_mapping.get(self.syst_suffix, self.syst_suffix)
            if hasattr(event,btag_name):
                weight = weight * getattr(event, btag_name)
            else:
                weight = weight * event.w_btag_SF

        return weight

    def tt_weighting(self, event: LazyDataFrame, weight):
        if self.isMC:
            try:
                weight = weight * event.ttbarweight_nominal
            except:
                pass

        return weight

    def my_btag_weighting(self, event: LazyDataFrame, weight, njet_weights):
        if self.isMC:
            weight = weight * njet_weights[event.ngood_jets]

        return weight

    def my_tt_weighting(self, event: LazyDataFrame, weight, ttbar_ratio):
        if self.isMC:
            try:
                weight = weight * event.ttbarweight_nominal * ttbar_ratio
            except:
                pass

        return weight

    def naming_schema(self, name, region):
     return f'{name}{self.syst_suffix}'
