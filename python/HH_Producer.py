"""
HH_Producer.py
Workspace producers using coffea.
"""
import json

from coffea.hist import Hist, Bin, export1d
from coffea.processor import ProcessorABC, LazyDataFrame, dict_accumulator
from uproot3 import recreate
import numpy as np

class WSProducer(ProcessorABC):
    """
    A coffea Processor which produces a workspace.
    This applies selections and produces histograms from kinematics.
    """

    histograms = NotImplemented
    selection = NotImplemented

    def __init__(self, isMC, era=2017, sample="DY", do_syst=False, syst_var='', weight_syst=False, haddFileName=None, flag=False, njetw=None):
        self._flag = flag
        self.do_syst = do_syst
        self.era = era
        self.isMC = isMC
        self.sample = sample
        self.syst_var, self.syst_suffix = (syst_var, f'_sys_{syst_var}') if do_syst and syst_var else ('', '')
        self.weight_syst = weight_syst
        self._accumulator = dict_accumulator({
            name: Hist('Events', Bin(name=name, **axis))
            for name, axis in ((self.naming_schema(hist['name'], region), hist['axis'])
                               for _, hist in list(self.histograms.items())
                               for region in hist['region'])
        })
        self.outfile = haddFileName

        self.njet_weights = None
        if njetw is not None:
            for line in open(njetw, 'r'):
                json_read = json.loads(line)
                if json_read['year'] == era:
                    self.njet_weights = np.fromiter(json_read['weights'].values(), dtype=np.float64)
                    self.njet_weights[self.njet_weights == -999] = 1
                    self.njet_weights = np.concatenate([self.njet_weights, np.tile(self.njet_weights[-1], 20)])
                    break
        #print('NJET WEIGHTS', self.njet_weights)

    def __repr__(self):
        return f'{self.__class__.__name__}(era: {self.era}, isMC: {self.isMC}, sample: {self.sample}, do_syst: {self.do_syst}, syst_var: {self.syst_var}, weight_syst: {self.weight_syst}, output: {self.outfile})'

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df, *args):
        output = self.accumulator.identity()

        weight = self.weighting(df)
        nobtag_weight = weight
        btag_weight = self.btag_weighting(df, weight)
        if self.njet_weights is not None:
            weight = self.my_btag_weighting(df, btag_weight, self.njet_weights)
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
                else:
                    output[name].fill(**{
                        'weight':weight[selec],
                        name: df[hist['target']][selec]#.flatten()
                    })

        return output

    def postprocess(self, accumulator):
        return accumulator

    def passbut(self, event: LazyDataFrame, excut: str, cat: str):
        """Backwards-compatible passbut."""
        return eval('&'.join('(' + cut.format(sys=('' if self.weight_syst else self.syst_suffix)) + ')' for cut in self.selection[cat] ))#if excut not in cut))

class HH_NTuple(WSProducer):

    zlep_bin = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 132, 146, 164, 184, 209, 239, 275, 318, 370, 432]

    histograms = {
        'Zlep_cand_mass_QCD_B': {
            'target': 'Zlep_cand_mass',
            'name'  : 'Zlep_cand_mass_QCD_B',  # name to write to histogram
            'region': ['QCD_B'],
            'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': zlep_bin}
        },
        'Zlep_cand_mass_QCD_C': {
            'target': 'Zlep_cand_mass',
            'name'  : 'Zlep_cand_mass_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': zlep_bin}
        },
        'Zlep_cand_mass_QCD_D': {
            'target': 'Zlep_cand_mass',
            'name'  : 'Zlep_cand_mass_QCD_D',  # name to write to histogram
            'region': ['QCD_D'],
            'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': zlep_bin}
        },
        'Zlep_cand_mass_DYcontrol': {
            'target': 'Zlep_cand_mass',
            'name'  : 'Zlep_cand_mass_DYcontrol',  # name to write to histogram
            'region': ['DYcontrol'],
            'axis': {'label': 'Zlep_cand_mass', 'n_or_arr':  40, 'lo': 80, 'hi': 100}
        },
        'Zlep_cand_mass_DYcontrol_QCD_C': {
            'target': 'Zlep_cand_mass',
            'name'  : 'Zlep_cand_mass_DYcontrol_QCD_C',  # name to write to histogram
            'region': ['DYcontrol_QCD_C'],
            'axis': {'label': 'Zlep_cand_mass', 'n_or_arr':  40, 'lo': 80, 'hi': 100}
        },
        'Zlep_cand_mass_TTcontrol': {
            'target': 'Zlep_cand_mass',
            'name'  : 'Zlep_cand_mass_TTcontrol',  # name to write to histogram
            'region': ['TTcontrol'],
            'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': 70, 'lo': 0, 'hi': 700}
        },
        'Zlep_cand_mass_TTcontrol_QCD_C': {
            'target': 'Zlep_cand_mass',
            'name'  : 'Zlep_cand_mass_TTcontrol_QCD_C',  # name to write to histogram
            'region': ['TTcontrol_QCD_C'],
            'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': 70, 'lo': 0, 'hi': 700}
        },
        'Zlep_cand_mass': {
            'target': 'Zlep_cand_mass',
            'name'  : 'Zlep_cand_mass',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': zlep_bin}
        },
        'Zlep_cand_pt': {
            'target': 'Zlep_cand_pt',
            'name'  : 'Zlep_cand_pt',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'Zlep_cand_pt', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
        },
        'Zlep_cand_eta': {
            'target': 'Zlep_cand_eta',
            'name'  : 'Zlep_cand_eta',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'Zlep_cand_eta', 'n_or_arr': 190, 'lo': -9, 'hi': 9}
        },
        'leading_lep_pt': {
            'target': 'leading_lep_pt',
            'name'  : 'leading_lep_pt',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'leading_lep_pt', 'n_or_arr': 50, 'lo': 0, 'hi': 500}
        },
        'leading_lep_eta': {
            'target': 'leading_lep_eta',
            'name'  : 'leading_lep_eta',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'leading_lep_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
        },
        'trailing_lep_pt': {
            'target': 'trailing_lep_pt',
            'name'  : 'trailing_lep_pt',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'trailing_lep_pt', 'n_or_arr': 50, 'lo': 0, 'hi': 500}
        },
        'trailing_lep_eta': {
            'target': 'trailing_lep_eta',
            'name'  : 'trailing_lep_eta',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'trailing_lep_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
        },
        'Zjet_cand_mass': {
            'target': 'Zjet_cand_mass',
            'name'  : 'Zjet_cand_mass',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'Zjet_cand_mass', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
        },
        'Higgsbb_cand_mass': {
            'target': 'Higgsbb_cand_mass',
            'name'  : 'Higgsbb_cand_mass',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'Higgsbb_cand_mass', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
        },
        'HiggsZZ_cand_mass': {
            'target': 'HiggsZZ_cand_mass',
            'name'  : 'HiggsZZ_cand_mass',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'HiggsZZ_cand_mass', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
        },
        'met_pt': {
            'target': 'met_pt',
            'name'  : 'met_pt',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'met_pt', 'n_or_arr': 60, 'lo': 0, 'hi': 600}
        },
        'dR_l1l2': {
            'target': 'dR_l1l2',
            'name'  : 'dR_l1l2',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l1l2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_j1j2': {
            'target': 'dR_j1j2',
            'name'  : 'dR_j1j2',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_j1j2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_b1b2': {
            'target': 'dR_b1b2',
            'name'  : 'dR_b1b2',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_b1b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l1b1': {
            'target': 'dR_l1b1',
            'name'  : 'dR_l1b1',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l1b1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l1b2': {
            'target': 'dR_l1b2',
            'name'  : 'dR_l1b2',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l1b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l2b1': {
            'target': 'dR_l2b1',
            'name'  : 'dR_l2b1',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l2b1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l2b2': {
            'target': 'dR_l2b2',
            'name'  : 'dR_l2b2',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l2b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l1j1': {
            'target': 'dR_l1j1',
            'name'  : 'dR_l1j1',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l1j1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l1j2': {
            'target': 'dR_l1j2',
            'name'  : 'dR_l1j2',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l1j2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l2j1': {
            'target': 'dR_l2j1',
            'name'  : 'dR_l2j1',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l2j1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l2j2': {
            'target': 'dR_l2j2',
            'name'  : 'dR_l2j2',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'dR_l2j2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'cosThetaCS': {
            'target': 'cosThetaCS',
            'name'  : 'cosThetaCS',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'cosThetaCS', 'n_or_arr': 20, 'lo': -1, 'hi': 1}
        },
        'cosThetabHbb': {
            'target': 'cosThetabHbb',
            'name'  : 'cosThetabHbb',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'cosThetabHbb', 'n_or_arr': 20, 'lo': -1, 'hi': 1}
        },
        'cosThetaZjjHzz': {
            'target': 'cosThetaZjjHzz',
            'name'  : 'cosThetaZjjHzz',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'cosThetaZjjHzz', 'n_or_arr': 20, 'lo': -1, 'hi': 1}
        },
        'cosThetaZllHzz': {
            'target': 'cosThetaZllHzz',
            'name'  : 'cosThetaZllHzz',  # name to write to histogram
            'region': ['signal'],
            'axis': {'label': 'cosThetaZllHzz', 'n_or_arr': 20, 'lo': -1, 'hi': 1}
        },
# QCD C Region
        'Zlep_cand_pt_QCD_C': {
            'target': 'Zlep_cand_pt',
            'name'  : 'Zlep_cand_pt_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'Zlep_cand_pt', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
        },
        'Zlep_cand_eta_QCD_C': {
            'target': 'Zlep_cand_eta',
            'name'  : 'Zlep_cand_eta_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'Zlep_cand_eta', 'n_or_arr': 190, 'lo': -9, 'hi': 9}
        },
        'leading_lep_pt_QCD_C': {
            'target': 'leading_lep_pt',
            'name'  : 'leading_lep_pt_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'leading_lep_pt', 'n_or_arr': 50, 'lo': 0, 'hi': 500}
        },
        'leading_lep_eta_QCD_C': {
            'target': 'leading_lep_eta',
            'name'  : 'leading_lep_eta_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'leading_lep_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
        },
        'trailing_lep_pt_QCD_C': {
            'target': 'trailing_lep_pt',
            'name'  : 'trailing_lep_pt_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'trailing_lep_pt', 'n_or_arr': 50, 'lo': 0, 'hi': 500}
        },
        'trailing_lep_eta_QCD_C': {
            'target': 'trailing_lep_eta',
            'name'  : 'trailing_lep_eta_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'trailing_lep_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
        },
        'Zjet_cand_mass_QCD_C': {
            'target': 'Zjet_cand_mass',
            'name'  : 'Zjet_cand_mass_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'Zjet_cand_mass', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
        },
        'Higgsbb_cand_mass_QCD_C': {
            'target': 'Higgsbb_cand_mass',
            'name'  : 'Higgsbb_cand_mass_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'Higgsbb_cand_mass', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
        },
        'HiggsZZ_cand_mass_QCD_C': {
            'target': 'HiggsZZ_cand_mass',
            'name'  : 'HiggsZZ_cand_mass_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'HiggsZZ_cand_mass', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
        },
        'met_pt_QCD_C': {
            'target': 'met_pt',
            'name'  : 'met_pt_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'met_pt', 'n_or_arr': 60, 'lo': 0, 'hi': 600}
        },
        'dR_l1l2_QCD_C': {
            'target': 'dR_l1l2',
            'name'  : 'dR_l1l2_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'dR_l1l2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_j1j2_QCD_C': {
            'target': 'dR_j1j2',
            'name'  : 'dR_j1j2_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'dR_j1j2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_b1b2_QCD_C': {
            'target': 'dR_b1b2',
            'name'  : 'dR_b1b2_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'dR_b1b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l1b1_QCD_C': {
            'target': 'dR_l1b1',
            'name'  : 'dR_l1b1_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'dR_l1b1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l1b2_QCD_C': {
            'target': 'dR_l1b2',
            'name'  : 'dR_l1b2_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'dR_l1b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l2b1_QCD_C': {
            'target': 'dR_l2b1',
            'name'  : 'dR_l2b1_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'dR_l2b1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l2b2_QCD_C': {
            'target': 'dR_l2b2',
            'name'  : 'dR_l2b2_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'dR_l2b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l1j1_QCD_C': {
            'target': 'dR_l1j1',
            'name'  : 'dR_l1j1_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'dR_l1j1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l1j2_QCD_C': {
            'target': 'dR_l1j2',
            'name'  : 'dR_l1j2_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'dR_l1j2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l2j1_QCD_C': {
            'target': 'dR_l2j1',
            'name'  : 'dR_l2j1_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'dR_l2j1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'dR_l2j2_QCD_C': {
            'target': 'dR_l2j2',
            'name'  : 'dR_l2j2_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'dR_l2j2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
        },
        'cosThetaCS_QCD_C': {
            'target': 'cosThetaCS',
            'name'  : 'cosThetaCS_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'cosThetaCS', 'n_or_arr': 20, 'lo': -1, 'hi': 1}
        },
        'cosThetabHbb_QCD_C': {
            'target': 'cosThetabHbb',
            'name'  : 'cosThetabHbb_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'cosThetabHbb', 'n_or_arr': 20, 'lo': -1, 'hi': 1}
        },
        'cosThetaZjjHzz_QCD_C': {
            'target': 'cosThetaZjjHzz',
            'name'  : 'cosThetaZjjHzz_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'cosThetaZjjHzz', 'n_or_arr': 20, 'lo': -1, 'hi': 1}
        },
        'cosThetaZllHzz_QCD_C': {
            'target': 'cosThetaZllHzz',
            'name'  : 'cosThetaZllHzz_QCD_C',  # name to write to histogram
            'region': ['QCD_C'],
            'axis': {'label': 'cosThetaZllHzz', 'n_or_arr': 20, 'lo': -1, 'hi': 1}
        },
        'ngood_jets': {
            'target': 'ngood_jets',
            'name'  : 'ngood_jets',  # name to write to histogram
            'region': ['signal_btag'],
            'axis': {'label': 'ngood_jets', 'n_or_arr': 21, 'lo': -0.5, 'hi': 20.5}
        },
        'ngood_jets_btagSF': {
            'target': 'ngood_jets',
            'name'  : 'ngood_jets_btagSF',  # name to write to histogram
            'region': ['signal_btag'],
            'axis': {'label': 'ngood_jets', 'n_or_arr': 21, 'lo': -0.5, 'hi': 20.5}
        },
        'ngood_jets_btagSF_nobtagSF': {
            'target': 'ngood_jets',
            'name'  : 'ngood_jets_nobtagSF',  # name to write to histogram
            'region': ['signal_btag'],
            'axis': {'label': 'ngood_jets', 'n_or_arr': 21, 'lo': -0.5, 'hi': 20.5}
        },
    }
    selection = {
            "signal" : [
                "event.ngood_bjets     >  0",
                "event.lep_category    == 1",
                "event.event_category    == 1",
                "event.leading_lep_pt  > 20",
                "event.trailing_lep_pt > 10",
                "event.Zlep_cand_mass > 15",
                "event.leading_Hbb_pt > 20",
                "event.trailing_Hbb_pt > 20",
                "event.leading_jet_pt > 20",
                "event.trailing_jet_pt > 20"
            ],
            "signal_btag" : [
                "event.lep_category    == 1",
                "event.event_category    == 1",
                "event.leading_lep_pt  > 20",
                "event.trailing_lep_pt > 10",
                "event.Zlep_cand_mass > 15",
                "event.leading_Hbb_pt > 20",
                "event.trailing_Hbb_pt > 20",
                "event.leading_jet_pt > 20",
                "event.trailing_jet_pt > 20"
            ],
            "QCD_B" : [
                "event.ngood_bjets     >  0",
                "event.lep_category    == 1",
                "event.event_category    == 2",
                "event.leading_lep_pt  > 20",
                "event.trailing_lep_pt > 10",
                "event.Zlep_cand_mass > 15",
                "event.leading_Hbb_pt > 20",
                "event.trailing_Hbb_pt > 20",
                "event.leading_jet_pt > 20",
                "event.trailing_jet_pt > 20"
            ],
            "QCD_C" : [
                "event.ngood_bjets     >  0",
                "event.lep_category    == 1",
                "event.event_category    == 3",
                "event.leading_lep_pt  > 20",
                "event.trailing_lep_pt > 10",
                "event.Zlep_cand_mass > 15",
                "event.leading_Hbb_pt > 20",
                "event.trailing_Hbb_pt > 20",
                "event.leading_jet_pt > 20",
                "event.trailing_jet_pt > 20"
            ],
            "QCD_D" : [
                "event.ngood_bjets     >  0",
                "event.lep_category    == 1",
                "event.event_category    == 4",
                "event.leading_lep_pt  > 20",
                "event.trailing_lep_pt > 10",
                "event.Zlep_cand_mass > 15",
                "event.leading_Hbb_pt > 20",
                "event.trailing_Hbb_pt > 20",
                "event.leading_jet_pt > 20",
                "event.trailing_jet_pt > 20"
            ],
            "DYcontrol" : [
                "event.ngood_bjets     >  0",
                "event.lep_category    == 1",
                "event.event_category    == 1",
                "event.leading_lep_pt  > 20",
                "event.trailing_lep_pt > 10",
                "event.leading_Hbb_pt > 20",
                "event.trailing_Hbb_pt > 20",
                "event.leading_jet_pt > 20",
                "event.trailing_jet_pt > 20",
                "event.Zlep_cand_mass > 80",
                "event.Zlep_cand_mass < 100",
            ],
            "DYcontrol_QCD_C" : [
                "event.ngood_bjets     >  0",
                "event.lep_category    == 1",
                "event.event_category    == 3",
                "event.leading_lep_pt  > 20",
                "event.trailing_lep_pt > 10",
                "event.leading_Hbb_pt > 20",
                "event.trailing_Hbb_pt > 20",
                "event.leading_jet_pt > 20",
                "event.trailing_jet_pt > 20",
                "event.Zlep_cand_mass > 80",
                "event.Zlep_cand_mass < 100",
            ],
            "TTcontrol" : [
                "event.ngood_bjets     >  0",
                "event.lep_category    == 1",
                "event.event_category    == 1",
                "event.leading_lep_pt  > 20",
                "event.trailing_lep_pt > 10",
                "event.leading_Hbb_pt > 20",
                "event.trailing_Hbb_pt > 20",
                "event.leading_jet_pt > 20",
                "event.trailing_jet_pt > 20",
                "event.Zlep_cand_mass > 100",
                "event.met_pt > 100"
            ],
            "TTcontrol_QCD_C" : [
                "event.ngood_bjets     >  0",
                "event.lep_category    == 1",
                "event.event_category    == 3",
                "event.leading_lep_pt  > 20",
                "event.trailing_lep_pt > 10",
                "event.leading_Hbb_pt > 20",
                "event.trailing_Hbb_pt > 20",
                "event.leading_jet_pt > 20",
                "event.trailing_jet_pt > 20",
                "event.Zlep_cand_mass > 100",
                "event.met_pt > 100"
            ],
        }


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

            #Muon SF
            if "MuonSF" in self.syst_suffix:
                if "Up" in self.syst_suffix:
                    weight = weight * event.w_muon_SFUp
                else:
                    weight = weight * event.w_muon_SFDown
            else:
                weight = weight * event.w_muon_SF

            # Electron SF
            if "ElecronSF" in self.syst_suffix:
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
                    weight = weight * event.TriggerSFWeightUp
                else:
                    weight = weight * event.TriggerSFWeightDown
            else:
                weight = weight * event.TriggerSFWeight

        return weight

    def btag_weighting(self, event: LazyDataFrame, weight):
        if self.isMC:
            weight = weight * event.w_btag_SF

        return weight

    def my_btag_weighting(self, event: LazyDataFrame, weight, njet_weights):
        if self.isMC:
            weight = weight * njet_weights[event.ngood_jets]
        return weight

    def naming_schema(self, name, region):
     return f'{name}{self.syst_suffix}'
