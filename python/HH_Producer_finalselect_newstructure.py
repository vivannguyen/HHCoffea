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
        #bdt_bin = [0, 0.0531, 0.2825, 0.5391, 0.5808, 0.6296, 0.6745, 0.6827, 0.6948,0.7358, 0.7952, 1.0]

        if era == 2016:
            bdt_bin = [0.0, 0.04002083, 0.08004167, 0.1200625 , 0.16008333, 0.20010417,
                   0.240125  , 0.28014583, 0.32016667, 0.3601875 , 0.40020833,
                          0.44022917, 0.48025   , 0.52027083, 0.56029167, 0.6003125 ,
                                 0.64033333, 0.68035417, 0.720375  , 0.76514583, 0.80041667,
                                        0.8404375 , 0.88045833, 0.93772917, 0.97, 1.0     ]
        if era == 2017:
            bdt_bin = [0.0, 0.04003125, 0.0800625 , 0.12009375, 0.160125  , 0.20015625,
                   0.2401875 , 0.28021875, 0.32025   , 0.36028125, 0.4003125 ,
                          0.44034375, 0.480375  , 0.52040625, 0.5604375 , 0.60796875,
                                 0.6405    , 0.68053125, 0.7205625 , 0.76059375, 0.800625  ,
                                        0.84065625, 0.8806875 , 0.92071875, 0.975, 1.0     ]
        if era == 2018:
            bdt_bin = [0.0, 0.04014878, 0.08029756, 0.1202515 , 0.16059511, 0.19998878,
                   0.24089267, 0.27978058, 0.32119023, 0.36133901, 0.40039801,
                          0.43938897, 0.48178534, 0.52518412, 0.55975514, 0.59993947,
                                 0.64238046, 0.67845264, 0.72033426, 0.76282679, 0.79952829,
                                        0.84312435, 0.88327313, 0.9047344 , 0.97125334, 1.0]

        histograms = {
            'Zlep_cand_mass_QCD_B': {
                'target': 'Zlep_cand_mass',
                'name'  : 'Zlep_cand_mass_QCD_B',
                'region': ['QCD_B'],
                'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': zlep_bin}
            },
            'Zlep_cand_mass_QCD_C': {
                'target': 'Zlep_cand_mass',
                'name'  : 'Zlep_cand_mass_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': zlep_bin}
            },
            'Zlep_cand_mass_QCD_D': {
                'target': 'Zlep_cand_mass',
                'name'  : 'Zlep_cand_mass_QCD_D',
                'region': ['QCD_D'],
                'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': zlep_bin}
            },
            'Zlep_cand_mass_DYcontrol': {
                'target': 'Zlep_cand_mass',
                'name'  : 'Zlep_cand_mass_DYcontrol',
                'region': ['DYcontrol'],
                'axis': {'label': 'Zlep_cand_mass', 'n_or_arr':  40, 'lo': 80, 'hi': 100}
            },
            'Zlep_cand_mass_DYcontrol_QCD_C': {
                'target': 'Zlep_cand_mass',
                'name'  : 'Zlep_cand_mass_DYcontrol_QCD_C',
                'region': ['DYcontrol_QCD_C'],
                'axis': {'label': 'Zlep_cand_mass', 'n_or_arr':  40, 'lo': 80, 'hi': 100}
            },
            'Zlep_cand_mass_TTcontrol': {
                'target': 'Zlep_cand_mass',
                'name'  : 'Zlep_cand_mass_TTcontrol',
                'region': ['TTcontrol'],
                'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': 70, 'lo': 0, 'hi': 700}
            },
            'Zlep_cand_mass_TTcontrol_QCD_C': {
                'target': 'Zlep_cand_mass',
                'name'  : 'Zlep_cand_mass_TTcontrol_QCD_C',
                'region': ['TTcontrol_QCD_C'],
                'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': 70, 'lo': 0, 'hi': 700}
            },
            'Zlep_cand_mass': {
                'target': 'Zlep_cand_mass',
                'name'  : 'Zlep_cand_mass',
                'region': ['signal'],
                'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': zlep_bin}
            },
            'Zlep_cand_pt': {
                'target': 'Zlep_cand_pt',
                'name'  : 'Zlep_cand_pt',
                'region': ['signal'],
                'axis': {'label': 'Zlep_cand_pt', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
            },
            'Zlep_cand_eta': {
                'target': 'Zlep_cand_eta',
                'name'  : 'Zlep_cand_eta',
                'region': ['signal'],
                'axis': {'label': 'Zlep_cand_eta', 'n_or_arr': 190, 'lo': -9, 'hi': 9}
            },
            'Zlep_cand_phi': {
                'target': 'Zlep_cand_phi',
                'name'  : 'Zlep_cand_phi',
                'region': ['signal'],
                'axis': {'label': 'Zlep_cand_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'leading_lep_pt': {
                'target': 'leading_lep_pt',
                'name'  : 'leading_lep_pt',
                'region': ['signal'],
                'axis': {'label': 'leading_lep_pt', 'n_or_arr': 50, 'lo': 0, 'hi': 500}
            },
            'leading_lep_eta': {
                'target': 'leading_lep_eta',
                'name'  : 'leading_lep_eta',
                'region': ['signal'],
                'axis': {'label': 'leading_lep_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
            },
            'leading_lep_phi': {
                'target': 'leading_lep_phi',
                'name'  : 'leading_lep_phi',
                'region': ['signal'],
                'axis': {'label': 'leading_lep_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'trailing_lep_pt': {
                'target': 'trailing_lep_pt',
                'name'  : 'trailing_lep_pt',
                'region': ['signal'],
                'axis': {'label': 'trailing_lep_pt', 'n_or_arr': 50, 'lo': 0, 'hi': 500}
            },
            'trailing_lep_eta': {
                'target': 'trailing_lep_eta',
                'name'  : 'trailing_lep_eta',
                'region': ['signal'],
                'axis': {'label': 'trailing_lep_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
            },
            'trailing_lep_phi': {
                'target': 'trailing_lep_phi',
                'name'  : 'trailing_lep_phi',
                'region': ['signal'],
                'axis': {'label': 'trailing_lep_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'Zjet_cand_mass': {
                'target': 'Zjet_cand_mass',
                'name'  : 'Zjet_cand_mass',
                'region': ['signal'],
                'axis': {'label': 'Zjet_cand_mass', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
            },
            'Zjet_cand_pt': {
                'target': 'Zjet_cand_pt',
                'name'  : 'Zjet_cand_pt',
                'region': ['signal'],
                'axis': {'label': 'Zjet_cand_pt', 'n_or_arr': 100, 'lo': 0, 'hi': 1000}
            },
            'Zjet_cand_eta': {
                'target': 'Zjet_cand_eta',
                'name'  : 'Zjet_cand_eta',
                'region': ['signal'],
                'axis': {'label': 'Zjet_cand_eta', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'Zjet_cand_phi': {
                'target': 'Zjet_cand_phi',
                'name'  : 'Zjet_cand_phi',
                'region': ['signal'],
                'axis': {'label': 'Zjet_cand_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'HH_cand_pt': {
                'target': 'HH_cand_pt',
                'name'  : 'HH_cand_pt',
                'region': ['signal'],
                'axis': {'label': 'HH_cand_pt', 'n_or_arr': 150, 'lo': 0, 'hi': 1500}
            },
            'HH_cand_eta': {
                'target': 'HH_cand_eta',
                'name'  : 'HH_cand_eta',
                'region': ['signal'],
                'axis': {'label': 'HH_cand_eta', 'n_or_arr': 130, 'lo': -6, 'hi': 6}
            },
            'HH_cand_phi': {
                'target': 'HH_cand_phi',
                'name'  : 'HH_cand_phi',
                'region': ['signal'],
                'axis': {'label': 'HH_cand_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'HH_cand_mass': {
                'target': 'HH_cand_mass',
                'name'  : 'HH_cand_mass',
                'region': ['signal'],
                'axis': {'label': 'HH_cand_mass', 'n_or_arr': 100, 'lo': 0, 'hi': 1000}
            },
            'Higgsbb_cand_mass': {
                'target': 'Higgsbb_cand_mass',
                'name'  : 'Higgsbb_cand_mass',
                'region': ['signal'],
                'axis': {'label': 'Higgsbb_cand_mass', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
            },
            'Higgsbb_cand_pt': {
                'target': 'Higgsbb_cand_pt',
                'name'  : 'Higgsbb_cand_pt',
                'region': ['signal'],
                'axis': {'label': 'Higgsbb_cand_pt', 'n_or_arr': 100, 'lo': 0, 'hi': 1000}
            },
            'Higgsbb_cand_eta': {
                'target': 'Higgsbb_cand_eta',
                'name'  : 'Higgsbb_cand_eta',
                'region': ['signal'],
                'axis': {'label': 'Higgsbb_cand_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
            },
            'Higgsbb_cand_phi': {
                'target': 'Higgsbb_cand_phi',
                'name'  : 'Higgsbb_cand_phi',
                'region': ['signal'],
                'axis': {'label': 'Higgsbb_cand_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'HiggsZZ_cand_mass': {
                'target': 'HiggsZZ_cand_mass',
                'name'  : 'HiggsZZ_cand_mass',
                'region': ['signal'],
                'axis': {'label': 'HiggsZZ_cand_mass', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
            },
            'HiggsZZ_cand_pt': {
                'target': 'HiggsZZ_cand_pt',
                'name'  : 'HiggsZZ_cand_pt',
                'region': ['signal'],
                'axis': {'label': 'HiggsZZ_cand_pt', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
            },
            'HiggsZZ_cand_eta': {
                'target': 'HiggsZZ_cand_eta',
                'name'  : 'HiggsZZ_cand_eta',
                'region': ['signal'],
                'axis': {'label': 'HiggsZZ_cand_eta', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'HiggsZZ_cand_phi': {
                'target': 'HiggsZZ_cand_phi',
                'name'  : 'HiggsZZ_cand_phi',
                'region': ['signal'],
                'axis': {'label': 'HiggsZZ_cand_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'leading_Hbb_pt': {
                'target': 'leading_Hbb_pt',
                'name'  : 'leading_Hbb_pt',
                'region': ['signal'],
                'axis': {'label': 'leading_Hbb_pt', 'n_or_arr': 100, 'lo': 0, 'hi': 1000}
            },
            'leading_Hbb_eta': {
                'target': 'leading_Hbb_eta',
                'name'  : 'leading_Hbb_eta',
                'region': ['signal'],
                'axis': {'label': 'leading_Hbb_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
            },
            'leading_Hbb_phi': {
                'target': 'leading_Hbb_phi',
                'name'  : 'leading_Hbb_phi',
                'region': ['signal'],
                'axis': {'label': 'leading_Hbb_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'leading_Hbb_btag': {
                'target': 'leading_Hbb_btag',
                'name'  : 'leading_Hbb_btag',
                'region': ['signal'],
                'axis': {'label': 'leading_Hbb_btag', 'n_or_arr': 20, 'lo': 0, 'hi': 1}
            },
            'trailing_Hbb_pt': {
                'target': 'trailing_Hbb_pt',
                'name'  : 'trailing_Hbb_pt',
                'region': ['signal'],
                'axis': {'label': 'trailing_Hbb_pt', 'n_or_arr': 100, 'lo': 0, 'hi': 1000}
            },
            'trailing_Hbb_eta': {
                'target': 'trailing_Hbb_eta',
                'name'  : 'trailing_Hbb_eta',
                'region': ['signal'],
                'axis': {'label': 'trailing_Hbb_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
            },
            'trailing_Hbb_phi': {
                'target': 'trailing_Hbb_phi',
                'name'  : 'trailing_Hbb_phi',
                'region': ['signal'],
                'axis': {'label': 'trailing_Hbb_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'trailing_Hbb_btag': {
                'target': 'trailing_Hbb_btag',
                'name'  : 'trailing_Hbb_btag',
                'region': ['signal'],
                'axis': {'label': 'trailing_Hbb_btag', 'n_or_arr': 20, 'lo': 0, 'hi': 1}
            },
            'leading_jet_pt': {
                'target': 'leading_jet_pt',
                'name'  : 'leading_jet_pt',
                'region': ['signal'],
                'axis': {'label': 'leading_jet_pt', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
            },
            'leading_jet_eta': {
                'target': 'leading_jet_eta',
                'name'  : 'leading_jet_eta',
                'region': ['signal'],
                'axis': {'label': 'leading_jet_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
            },
            'leading_jet_phi': {
                'target': 'leading_jet_phi',
                'name'  : 'leading_jet_phi',
                'region': ['signal'],
                'axis': {'label': 'leading_jet_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'leading_jet_qgl': {
                'target': 'leading_jet_qgl',
                'name'  : 'leading_jet_qgl',
                'region': ['signal'],
                'axis': {'label': 'leading_jet_qgl', 'n_or_arr': 20, 'lo': 0, 'hi': 1}
            },
            'trailing_jet_pt': {
                'target': 'trailing_jet_pt',
                'name'  : 'trailing_jet_pt',
                'region': ['signal'],
                'axis': {'label': 'trailing_jet_pt', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
            },
            'trailing_jet_eta': {
                'target': 'trailing_jet_eta',
                'name'  : 'trailing_jet_eta',
                'region': ['signal'],
                'axis': {'label': 'trailing_jet_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
            },
            'trailing_jet_phi': {
                'target': 'trailing_jet_phi',
                'name'  : 'trailing_jet_phi',
                'region': ['signal'],
                'axis': {'label': 'trailing_jet_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'trailing_jet_qgl': {
                'target': 'trailing_jet_qgl',
                'name'  : 'trailing_jet_qgl',
                'region': ['signal'],
                'axis': {'label': 'trailing_jet_qgl', 'n_or_arr': 20, 'lo': 0, 'hi': 1}
            },
            'met_pt': {
                'target': 'met_pt',
                'name'  : 'met_pt',
                'region': ['signal'],
                'axis': {'label': 'met_pt', 'n_or_arr': 60, 'lo': 0, 'hi': 600}
            },
            'met_phi': {
                'target': 'met_phi',
                'name'  : 'met_phi',
                'region': ['signal'],
                'axis': {'label': 'met_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'dR_l1l2': {
                'target': 'dR_l1l2',
                'name'  : 'dR_l1l2',
                'region': ['signal'],
                'axis': {'label': 'dR_l1l2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_j1j2': {
                'target': 'dR_j1j2',
                'name'  : 'dR_j1j2',
                'region': ['signal'],
                'axis': {'label': 'dR_j1j2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_b1b2': {
                'target': 'dR_b1b2',
                'name'  : 'dR_b1b2',
                'region': ['signal'],
                'axis': {'label': 'dR_b1b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_l1b1': {
                'target': 'dR_l1b1',
                'name'  : 'dR_l1b1',
                'region': ['signal'],
                'axis': {'label': 'dR_l1b1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_l1b2': {
                'target': 'dR_l1b2',
                'name'  : 'dR_l1b2',
                'region': ['signal'],
                'axis': {'label': 'dR_l1b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_l2b1': {
                'target': 'dR_l2b1',
                'name'  : 'dR_l2b1',
                'region': ['signal'],
                'axis': {'label': 'dR_l2b1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_l2b2': {
                'target': 'dR_l2b2',
                'name'  : 'dR_l2b2',
                'region': ['signal'],
                'axis': {'label': 'dR_l2b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_l1j1': {
                'target': 'dR_l1j1',
                'name'  : 'dR_l1j1',
                'region': ['signal'],
                'axis': {'label': 'dR_l1j1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_l1j2': {
                'target': 'dR_l1j2',
                'name'  : 'dR_l1j2',
                'region': ['signal'],
                'axis': {'label': 'dR_l1j2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_l2j1': {
                'target': 'dR_l2j1',
                'name'  : 'dR_l2j1',
                'region': ['signal'],
                'axis': {'label': 'dR_l2j1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_l2j2': {
                'target': 'dR_l2j2',
                'name'  : 'dR_l2j2',
                'region': ['signal'],
                'axis': {'label': 'dR_l2j2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_j1b1': {
                'target': 'dR_j1b1',
                'name'  : 'dR_j1b1',
                'region': ['signal'],
                'axis': {'label': 'dR_j1b1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_j1b2': {
                'target': 'dR_j1b2',
                'name'  : 'dR_j1b2',
                'region': ['signal'],
                'axis': {'label': 'dR_j1b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_j2b1': {
                'target': 'dR_j2b1',
                'name'  : 'dR_j2b1',
                'region': ['signal'],
                'axis': {'label': 'dR_j2b1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_j2b2': {
                'target': 'dR_j2b2',
                'name'  : 'dR_j2b2',
                'region': ['signal'],
                'axis': {'label': 'dR_j2b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'cosThetaCS': {
                'target': 'cosThetaCS',
                'name'  : 'cosThetaCS',
                'region': ['signal'],
                'axis': {'label': 'cosThetaCS', 'n_or_arr': 20, 'lo': -1, 'hi': 1}
            },
            'cosThetabHbb': {
                'target': 'cosThetabHbb',
                'name'  : 'cosThetabHbb',
                'region': ['signal'],
                'axis': {'label': 'cosThetabHbb', 'n_or_arr': 20, 'lo': -1, 'hi': 1}
            },
            'cosThetaZjjHzz': {
                'target': 'cosThetaZjjHzz',
                'name'  : 'cosThetaZjjHzz',
                'region': ['signal'],
                'axis': {'label': 'cosThetaZjjHzz', 'n_or_arr': 20, 'lo': -1, 'hi': 1}
            },
            'cosThetaZllHzz': {
                'target': 'cosThetaZllHzz',
                'name'  : 'cosThetaZllHzz',
                'region': ['signal'],
                'axis': {'label': 'cosThetaZllHzz', 'n_or_arr': 20, 'lo': -1, 'hi': 1}
            },
            'phi1': {
                'target': 'phi1',
                'name'  : 'phi1',
                'region': ['signal'],
                'axis': {'label': 'phi1', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'phi1_Zjj': {
                'target': 'phi1_Zjj',
                'name'  : 'phi1_Zjj',
                'region': ['signal'],
                'axis': {'label': 'phi1_Zjj', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
# QCD C Region
            'Zlep_cand_pt_QCD_C': {
                'target': 'Zlep_cand_pt',
                'name'  : 'Zlep_cand_pt_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'Zlep_cand_pt', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
            },
            'Zlep_cand_eta_QCD_C': {
                'target': 'Zlep_cand_eta',
                'name'  : 'Zlep_cand_eta_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'Zlep_cand_eta', 'n_or_arr': 190, 'lo': -9, 'hi': 9}
            },
            'Zlep_cand_phi_QCD_C': {
                'target': 'Zlep_cand_phi',
                'name'  : 'Zlep_cand_phi_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'Zlep_cand_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'leading_lep_pt_QCD_C': {
                'target': 'leading_lep_pt',
                'name'  : 'leading_lep_pt_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'leading_lep_pt', 'n_or_arr': 50, 'lo': 0, 'hi': 500}
            },
            'leading_lep_eta_QCD_C': {
                'target': 'leading_lep_eta',
                'name'  : 'leading_lep_eta_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'leading_lep_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
            },
            'leading_lep_phi_QCD_C': {
                'target': 'leading_lep_phi',
                'name'  : 'leading_lep_phi_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'leading_lep_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'trailing_lep_pt_QCD_C': {
                'target': 'trailing_lep_pt',
                'name'  : 'trailing_lep_pt_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'trailing_lep_pt', 'n_or_arr': 50, 'lo': 0, 'hi': 500}
            },
            'trailing_lep_eta_QCD_C': {
                'target': 'trailing_lep_eta',
                'name'  : 'trailing_lep_eta_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'trailing_lep_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
            },
            'trailing_lep_phi_QCD_C': {
                'target': 'trailing_lep_phi',
                'name'  : 'trailing_lep_phi_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'trailing_lep_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'Zjet_cand_mass_QCD_C': {
                'target': 'Zjet_cand_mass',
                'name'  : 'Zjet_cand_mass_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'Zjet_cand_mass', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
            },
            'Zjet_cand_pt_QCD_C': {
                'target': 'Zjet_cand_pt',
                'name'  : 'Zjet_cand_pt_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'Zjet_cand_pt', 'n_or_arr': 100, 'lo': 0, 'hi': 1000}
            },
            'Zjet_cand_eta_QCD_C': {
                'target': 'Zjet_cand_eta',
                'name'  : 'Zjet_cand_eta_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'Zjet_cand_eta', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'Zjet_cand_phi_QCD_C': {
                'target': 'Zjet_cand_phi',
                'name'  : 'Zjet_cand_phi_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'Zjet_cand_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'HH_cand_pt_QCD_C': {
                'target': 'HH_cand_pt',
                'name'  : 'HH_cand_pt_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'HH_cand_pt', 'n_or_arr': 150, 'lo': 0, 'hi': 1500}
            },
            'HH_cand_eta_QCD_C': {
                'target': 'HH_cand_eta',
                'name'  : 'HH_cand_eta_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'HH_cand_eta', 'n_or_arr': 130, 'lo': -6, 'hi': 6}
            },
            'HH_cand_phi_QCD_C': {
                'target': 'HH_cand_phi',
                'name'  : 'HH_cand_phi_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'HH_cand_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'HH_cand_mass_QCD_C': {
                'target': 'HH_cand_mass',
                'name'  : 'HH_cand_mass_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'HH_cand_mass', 'n_or_arr': 100, 'lo': 0, 'hi': 1000}
            },
            'Higgsbb_cand_mass_QCD_C': {
                'target': 'Higgsbb_cand_mass',
                'name'  : 'Higgsbb_cand_mass_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'Higgsbb_cand_mass', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
            },
            'Higgsbb_cand_pt_QCD_C': {
                'target': 'Higgsbb_cand_pt',
                'name'  : 'Higgsbb_cand_pt_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'Higgsbb_cand_pt', 'n_or_arr': 100, 'lo': 0, 'hi': 1000}
            },
            'Higgsbb_cand_eta_QCD_C': {
                'target': 'Higgsbb_cand_eta',
                'name'  : 'Higgsbb_cand_eta_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'Higgsbb_cand_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
            },
            'Higgsbb_cand_phi_QCD_C': {
                'target': 'Higgsbb_cand_phi',
                'name'  : 'Higgsbb_cand_phi_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'Higgsbb_cand_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'HiggsZZ_cand_mass_QCD_C': {
                'target': 'HiggsZZ_cand_mass',
                'name'  : 'HiggsZZ_cand_mass_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'HiggsZZ_cand_mass', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
            },
            'HiggsZZ_cand_pt_QCD_C': {
                'target': 'HiggsZZ_cand_pt',
                'name'  : 'HiggsZZ_cand_pt_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'HiggsZZ_cand_pt', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
            },
            'HiggsZZ_cand_eta_QCD_C': {
                'target': 'HiggsZZ_cand_eta',
                'name'  : 'HiggsZZ_cand_eta_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'HiggsZZ_cand_eta', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'HiggsZZ_cand_phi_QCD_C': {
                'target': 'HiggsZZ_cand_phi',
                'name'  : 'HiggsZZ_cand_phi_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'HiggsZZ_cand_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'leading_Hbb_pt_QCD_C': {
                'target': 'leading_Hbb_pt',
                'name'  : 'leading_Hbb_pt_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'leading_Hbb_pt', 'n_or_arr': 100, 'lo': 0, 'hi': 1000}
            },
            'leading_Hbb_eta_QCD_C': {
                'target': 'leading_Hbb_eta',
                'name'  : 'leading_Hbb_eta_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'leading_Hbb_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
            },
            'leading_Hbb_phi_QCD_C': {
                'target': 'leading_Hbb_phi',
                'name'  : 'leading_Hbb_phi_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'leading_Hbb_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'leading_Hbb_btag_QCD_C': {
                'target': 'leading_Hbb_btag',
                'name'  : 'leading_Hbb_btag_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'leading_Hbb_btag', 'n_or_arr': 20, 'lo': 0, 'hi': 1}
            },
            'trailing_Hbb_pt_QCD_C': {
                'target': 'trailing_Hbb_pt',
                'name'  : 'trailing_Hbb_pt_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'trailing_Hbb_pt', 'n_or_arr': 100, 'lo': 0, 'hi': 1000}
            },
            'trailing_Hbb_eta_QCD_C': {
                'target': 'trailing_Hbb_eta',
                'name'  : 'trailing_Hbb_eta_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'trailing_Hbb_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
            },
            'trailing_Hbb_phi_QCD_C': {
                'target': 'trailing_Hbb_phi',
                'name'  : 'trailing_Hbb_phi_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'trailing_Hbb_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'trailing_Hbb_btag_QCD_C': {
                'target': 'trailing_Hbb_btag',
                'name'  : 'trailing_Hbb_btag_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'trailing_Hbb_btag', 'n_or_arr': 20, 'lo': 0, 'hi': 1}
            },
            'leading_jet_pt_QCD_C': {
                'target': 'leading_jet_pt',
                'name'  : 'leading_jet_pt_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'leading_jet_pt', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
            },
            'leading_jet_eta_QCD_C': {
                'target': 'leading_jet_eta',
                'name'  : 'leading_jet_eta_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'leading_jet_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
            },
            'leading_jet_phi_QCD_C': {
                'target': 'leading_jet_phi',
                'name'  : 'leading_jet_phi_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'leading_jet_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'leading_jet_qgl_QCD_C': {
                'target': 'leading_jet_qgl',
                'name'  : 'leading_jet_qgl_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'leading_jet_qgl', 'n_or_arr': 20, 'lo': 0, 'hi': 1}
            },
            'trailing_jet_pt_QCD_C': {
                'target': 'trailing_jet_pt',
                'name'  : 'trailing_jet_pt_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'trailing_jet_pt', 'n_or_arr': 80, 'lo': 0, 'hi': 800}
            },
            'trailing_jet_eta_QCD_C': {
                'target': 'trailing_jet_eta',
                'name'  : 'trailing_jet_eta_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'trailing_jet_eta', 'n_or_arr': 70, 'lo': -3, 'hi': 3}
            },
            'trailing_jet_phi_QCD_C': {
                'target': 'trailing_jet_phi',
                'name'  : 'trailing_jet_phi_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'trailing_jet_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'trailing_jet_qgl_QCD_C': {
                'target': 'trailing_jet_qgl',
                'name'  : 'trailing_jet_qgl_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'trailing_jet_qgl', 'n_or_arr': 20, 'lo': 0, 'hi': 1}
            },
            'met_pt_QCD_C': {
                'target': 'met_pt',
                'name'  : 'met_pt_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'met_pt', 'n_or_arr': 60, 'lo': 0, 'hi': 600}
            },
            'met_phi_QCD_C': {
                'target': 'met_phi',
                'name'  : 'met_phi_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'met_phi', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },

            'dR_l1l2_QCD_C': {
                'target': 'dR_l1l2',
                'name'  : 'dR_l1l2_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_l1l2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_j1j2_QCD_C': {
                'target': 'dR_j1j2',
                'name'  : 'dR_j1j2_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_j1j2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_b1b2_QCD_C': {
                'target': 'dR_b1b2',
                'name'  : 'dR_b1b2_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_b1b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_l1b1_QCD_C': {
                'target': 'dR_l1b1',
                'name'  : 'dR_l1b1_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_l1b1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_l1b2_QCD_C': {
                'target': 'dR_l1b2',
                'name'  : 'dR_l1b2_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_l1b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_l2b1_QCD_C': {
                'target': 'dR_l2b1',
                'name'  : 'dR_l2b1_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_l2b1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_l2b2_QCD_C': {
                'target': 'dR_l2b2',
                'name'  : 'dR_l2b2_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_l2b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_l1j1_QCD_C': {
                'target': 'dR_l1j1',
                'name'  : 'dR_l1j1_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_l1j1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_l1j2_QCD_C': {
                'target': 'dR_l1j2',
                'name'  : 'dR_l1j2_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_l1j2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_l2j1_QCD_C': {
                'target': 'dR_l2j1',
                'name'  : 'dR_l2j1_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_l2j1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_l2j2_QCD_C': {
                'target': 'dR_l2j2',
                'name'  : 'dR_l2j2_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_l2j2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_j1b1_QCD_C': {
                'target': 'dR_j1b1',
                'name'  : 'dR_j1b1_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_j1b1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_j1b2_QCD_C': {
                'target': 'dR_j1b2',
                'name'  : 'dR_j1b2_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_j1b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_j2b1_QCD_C': {
                'target': 'dR_j2b1',
                'name'  : 'dR_j2b1_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_j2b1', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'dR_j2b2_QCD_C': {
                'target': 'dR_j2b2',
                'name'  : 'dR_j2b2_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'dR_j2b2', 'n_or_arr': 70, 'lo': 0, 'hi': 7}
            },
            'cosThetaCS_QCD_C': {
                'target': 'cosThetaCS',
                'name'  : 'cosThetaCS_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'cosThetaCS', 'n_or_arr': 20, 'lo': -1, 'hi': 1}
            },
            'cosThetabHbb_QCD_C': {
                'target': 'cosThetabHbb',
                'name'  : 'cosThetabHbb_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'cosThetabHbb', 'n_or_arr': 20, 'lo': -1, 'hi': 1}
            },
            'cosThetaZjjHzz_QCD_C': {
                'target': 'cosThetaZjjHzz',
                'name'  : 'cosThetaZjjHzz_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'cosThetaZjjHzz', 'n_or_arr': 20, 'lo': -1, 'hi': 1}
            },
            'cosThetaZllHzz_QCD_C': {
                'target': 'cosThetaZllHzz',
                'name'  : 'cosThetaZllHzz_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'cosThetaZllHzz', 'n_or_arr': 20, 'lo': -1, 'hi': 1}
            },
            'phi1_QCD_C': {
                'target': 'phi1',
                'name'  : 'phi1_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'phi1', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
            'phi1_Zjj_QCD_C': {
                'target': 'phi1_Zjj',
                'name'  : 'phi1_Zjj_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'phi1_Zjj', 'n_or_arr': 90, 'lo': -4, 'hi': 4}
            },
# For btag Event Weight
            'ngood_jets': {
                'target': 'ngood_jets',
                'name'  : 'ngood_jets',
                'region': ['signal_btag'],
                'axis': {'label': 'ngood_jets', 'n_or_arr': 21, 'lo': -0.5, 'hi': 20.5}
            },
            'ngood_jets_btagSF': {
                'target': 'ngood_jets',
                'name'  : 'ngood_jets_btagSF',
                'region': ['signal_btag'],
                'axis': {'label': 'ngood_jets', 'n_or_arr': 21, 'lo': -0.5, 'hi': 20.5}
            },
            'ngood_jets_btagSF_nobtagSF': {
                'target': 'ngood_jets',
                'name'  : 'ngood_jets_nobtagSF',
                'region': ['signal_btag'],
                'axis': {'label': 'ngood_jets', 'n_or_arr': 21, 'lo': -0.5, 'hi': 20.5}
            },
# For tt Event Weight
            'Zlep_cand_mass_tt_weight': {
                'target': 'Zlep_cand_mass',
                'name'  : 'Zlep_cand_mass_tt_weight',  # name to write to histogram
                'region': ['signal'],
                'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': zlep_bin}
            },
            'Zlep_cand_mass_nott_weight': {
                'target': 'Zlep_cand_mass',
                'name'  : 'Zlep_cand_mass_nott_weight',  # name to write to histogram
                'region': ['signal'],
                'axis': {'label': 'Zlep_cand_mass', 'n_or_arr': zlep_bin}
            },
# BDT score
            'h_bdtscore' : {
                'target': 'BDTscore',
                'name': 'BDTscore',
                'region': ['signal'],
                'axis': {'label': 'BDTscore', 'n_or_arr': bdt_bin}
                #'axis': {'label': 'BDTscore', 'n_or_arr': 100, 'lo': 0., 'hi': 1.}
            },
            'h_bdtscore_QCD_C': {
                'target': 'BDTscore',
                'name'  : 'BDTscore_QCD_C',
                'region': ['QCD_C'],
                'axis': {'label': 'BDTscore', 'n_or_arr': bdt_bin}
                #'axis': {'label': 'BDTscore', 'n_or_arr': 100, 'lo': 0., 'hi': 1.}
            },
        }
        selection = {
                "signal" : [
                    "event.ngood_bjetsT     >  0",
                    "event.lep_category    == 1",
                    "event.event_category    == 1",
                    "event.leading_lep_pt  > 20",
                    "event.trailing_lep_pt > 10",
                    "event.Zlep_cand_mass > 15",
                    "event.leading_Hbb_pt > 20",
                    "event.trailing_Hbb_pt > 20",
                    "event.leading_jet_pt > 20",
                    "event.trailing_jet_pt > 20",
                    "event.met_pt < 75 ",
                    "event.dR_j1b1 > 0",
                    "event.dR_j1b2 > 0",
                    "event.dR_j2b1 > 0",
                    "event.dR_j2b2 > 0"
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
                    "event.trailing_jet_pt > 20",
                    "event.dR_j1b1 > 0",
                    "event.dR_j1b2 > 0",
                    "event.dR_j2b1 > 0",
                    "event.dR_j2b2 > 0"
                ],
                "QCD_B" : [
                    "event.ngood_bjetsT     >  0",
                    "event.lep_category    == 1",
                    "event.event_category    == 2",
                    "event.leading_lep_pt  > 20",
                    "event.trailing_lep_pt > 10",
                    "event.Zlep_cand_mass > 15",
                    "event.leading_Hbb_pt > 20",
                    "event.trailing_Hbb_pt > 20",
                    "event.leading_jet_pt > 20",
                    "event.trailing_jet_pt > 20",
                    "event.dR_j1b1 > 0",
                    "event.dR_j1b2 > 0",
                    "event.dR_j2b1 > 0",
                    "event.dR_j2b2 > 0"
                ],
                "QCD_C" : [
                    "event.ngood_bjetsT     >  0",
                    "event.lep_category    == 1",
                    "event.event_category    == 3",
                    "event.leading_lep_pt  > 20",
                    "event.trailing_lep_pt > 10",
                    "event.Zlep_cand_mass > 15",
                    "event.leading_Hbb_pt > 20",
                    "event.trailing_Hbb_pt > 20",
                    "event.leading_jet_pt > 20",
                    "event.trailing_jet_pt > 20",
                    "event.met_pt < 75 ",
                    "event.dR_j1b1 > 0",
                    "event.dR_j1b2 > 0",
                    "event.dR_j2b1 > 0",
                    "event.dR_j2b2 > 0"
                ],
                "QCD_D" : [
                    "event.ngood_bjetsT     >  0",
                    "event.lep_category    == 1",
                    "event.event_category    == 4",
                    "event.leading_lep_pt  > 20",
                    "event.trailing_lep_pt > 10",
                    "event.Zlep_cand_mass > 15",
                    "event.leading_Hbb_pt > 20",
                    "event.trailing_Hbb_pt > 20",
                    "event.leading_jet_pt > 20",
                    "event.trailing_jet_pt > 20",
                    "event.dR_j1b1 > 0",
                    "event.dR_j1b2 > 0",
                    "event.dR_j2b1 > 0",
                    "event.dR_j2b2 > 0"
                ],
                "DYcontrol" : [
                    "event.ngood_bjetsT     >  0",
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
                    "event.dR_j1b1 > 0",
                    "event.dR_j1b2 > 0",
                    "event.dR_j2b1 > 0",
                    "event.dR_j2b2 > 0"
                ],
                "DYcontrol_QCD_C" : [
                    "event.ngood_bjetsT     >  0",
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
                    "event.dR_j1b1 > 0",
                    "event.dR_j1b2 > 0",
                    "event.dR_j2b1 > 0",
                    "event.dR_j2b2 > 0"
                ],
                "TTcontrol" : [
                    "event.ngood_bjetsT     >  0",
                    "event.lep_category    == 1",
                    "event.event_category    == 1",
                    "event.leading_lep_pt  > 20",
                    "event.trailing_lep_pt > 10",
                    "event.leading_Hbb_pt > 20",
                    "event.trailing_Hbb_pt > 20",
                    "event.leading_jet_pt > 20",
                    "event.trailing_jet_pt > 20",
                    "event.Zlep_cand_mass > 100",
                    "event.met_pt > 100",
                    "event.dR_j1b1 > 0",
                    "event.dR_j1b2 > 0",
                    "event.dR_j2b1 > 0",
                    "event.dR_j2b2 > 0"
                ],
                "TTcontrol_QCD_C" : [
                    "event.ngood_bjetsT     >  0",
                    "event.lep_category    == 1",
                    "event.event_category    == 3",
                    "event.leading_lep_pt  > 20",
                    "event.trailing_lep_pt > 10",
                    "event.leading_Hbb_pt > 20",
                    "event.trailing_Hbb_pt > 20",
                    "event.leading_jet_pt > 20",
                    "event.trailing_jet_pt > 20",
                    "event.Zlep_cand_mass > 100",
                    "event.met_pt > 100",
                    "event.dR_j1b1 > 0",
                    "event.dR_j1b2 > 0",
                    "event.dR_j2b1 > 0",
                    "event.dR_j2b2 > 0"
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
                    self.njet_weights = np.concatenate([self.njet_weights, np.tile(self.njet_weights[-1], 20)])
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
        model.load_model(f'models/{year_str}-uu-dR')
        bdtscore=model.predict_proba(X)[:,1]
        df['BDTscore']=bdtscore

        output = self.accumulator.identity()

        weight = self.weighting(df)
        nobtag_weight = weight
        btag_weight = self.btag_weighting(df, weight)

        if self.era == 2016: ttbar_ratio = 1.0096804
        if self.era == 2017: ttbar_ratio = 1.0095114
        if self.era == 2018: ttbar_ratio = 1.0108063

        #if self.era == 2016: ttbar_ratio = 1.0089811
        #if self.era == 2017: ttbar_ratio = 1.0097912
        #if self.era == 2018: ttbar_ratio = 1.0101884

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
                    weight = weight * event.TriggerSFWeightUp
                else:
                    weight = weight * event.TriggerSFWeightDown
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
