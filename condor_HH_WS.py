import os
import re

from coffea.processor import run_uproot_job, futures_executor

from python.HH_Producer import *
from python.SumWeights import *

import uproot3 as uproot
import argparse

import time

parser = argparse.ArgumentParser("")
parser.add_argument('--isMC', type=int, default=1, help="")
parser.add_argument('--jobNum', type=int, default=1, help="")
parser.add_argument('--era', type=str, default="2018", help="")
parser.add_argument('--doSyst', type=int, default=1, help="")
parser.add_argument('--infile', type=str, default=None, help="")
parser.add_argument('--dataset', type=str, default="X", help="")
parser.add_argument('--nevt', type=str, default=-1, help="")
parser.add_argument('--njetw', type=str, default=None, help="")

options = parser.parse_args()


def inputfile(nanofile):
    tested = False
    forceaaa = False
    pfn = os.popen("edmFileUtil -d %s" % (nanofile)).read()
    pfn = re.sub("\n", "", pfn)
    print((nanofile, " -> ", pfn))
    if (os.getenv("GLIDECLIENT_Group", "") != "overflow" and
            os.getenv("GLIDECLIENT_Group", "") != "overflow_conservative" and not
            forceaaa):
        if not tested:
            print("Testing file open")
            testfile = uproot.open(pfn)
            if testfile:
                print("Test OK")
                nanofile = pfn
            else:
                if "root://cms-xrd-global.cern.ch/" not in nanofile:
                    nanofile = "root://cms-xrd-global.cern.ch/" + nanofile
                forceaaa = True
        else:
            nanofile = pfn
    else:
        if "root://cms-xrd-global.cern.ch/" not in nanofile:
            nanofile = "root://cms-xrd-global.cern.ch/" + nanofile
    return nanofile


#options.dataset='QCD'

pre_selection = ""

if float(options.nevt) > 0:
    print((" passing this cut and : ", options.nevt))
    pre_selection += ' && (Entry$ < {})'.format(options.nevt)

pro_syst = ["ElectronEn", "MuonEn", "jesAbsolute", "jesBBEC1", "jesEC2","jesFlavorQCD","jesHF","jesRelativeBal"]
if options.era == '2016': pro_syst.extend(["jesAbsolute_2016","jesBBEC1_2016","jesEC2_2016","jesHF_2016","jesRelativeSample_2016"])
if options.era == '2017': pro_syst.extend(["jesAbsolute_2017","jesBBEC1_2017","jesEC2_2017","jesHF_2017","jesRelativeSample_2017"])
if options.era == '2018': pro_syst.extend(["jesAbsolute_2018","jesBBEC1_2018","jesEC2_2018","jesHF_2018","jesRelativeSample_2018"])

ext_syst = ["puWeight", "PDF", "MuonSF", "ElectronSF", "TriggerSFWeight", "QCDScale0w", "QCDScale1w", "QCDScale2w"]

# additional ext_syst with different up and down naming convention
btag_syst = ["w_btag_SF_sys_up_hf", "w_btag_SF_sys_up_lf", "w_btag_SF_sys_up_cferr1", "w_btag_SF_sys_up_cferr2",
             "w_btag_SF_sys_down_hf", "w_btag_SF_sys_down_lf", "w_btag_SF_sys_down_cferr1", "w_btag_SF_sys_down_cferr2",
             "w_btag_SF_sys_up_hfstats1", "w_btag_SF_sys_up_hfstats2", "w_btag_SF_sys_up_lfstats1", "w_btag_SF_sys_up_lfstats2",
             "w_btag_SF_sys_down_hfstats1", "w_btag_SF_sys_down_hfstats2", "w_btag_SF_sys_down_lfstats1", "w_btag_SF_sys_down_lfstats2"
             ]

tt_syst = ["ttbarweight_up", "ttbarweight_down"]

ext_syst_more = btag_syst + tt_syst

modules_era = []

modules_era.append(HH_NTuple(isMC=options.isMC, era=int(options.era), do_syst=1, syst_var='', sample=options.dataset, njetw=options.njetw))#,
#                         haddFileName="tree_%s.root" % str(options.jobNum)))
if options.isMC and options.doSyst==1:
   for sys in pro_syst:
       for var in ["Up", "Down"]:
           modules_era.append(HH_NTuple(options.isMC, str(options.era), do_syst=1,
                                    syst_var=sys + var, sample=options.dataset, njetw=options.njetw))#,
                                    #haddFileName=f"tree_{options.jobNum}_{sys}{var}.root"))

   for sys in ext_syst:
       for var in ["Up", "Down"]:
           modules_era.append(
               HH_NTuple(
                   options.isMC, str(options.era),
                   do_syst=1, syst_var=sys + var,
                   weight_syst=True,
                   sample=options.dataset,
                   njetw=options.njetw#,
#                   haddFileName=f"tree_{options.jobNum}_{sys}{var}.root",
               )
           )

   for sys in ext_syst_more:
       modules_era.append(
           HH_NTuple(
               options.isMC, str(options.era),
               do_syst=1, syst_var=sys,
               weight_syst=True,
               sample=options.dataset,
               njetw=options.njetw#,
           )
       )

for i in modules_era:
    print("modules : ", i)

print("Selection : ", pre_selection)
tstart = time.time()
f = uproot.recreate("tree_%s_WS.root" % str(options.jobNum))
for instance in modules_era:
    output = run_uproot_job(
        {instance.sample: [options.infile]},
        treename='Events',
        processor_instance=instance,
        executor=futures_executor,
        executor_args={'workers': 10},
        chunksize=500000
    )
    for h, hist in output.items():
        f[h] = export1d(hist)
        #print(f'wrote {h} to tree_{options.jobNum}_WS.root')

modules_gensum = []

if options.isMC:
    modules_gensum.append(GenSumWeight(isMC=options.isMC, era=int(options.era), do_syst=1, syst_var='', sample=options.dataset))#,
    #                         haddFileName="tree_%s.root" % str(options.jobNum)))
    
    for instance in modules_gensum:
        output = run_uproot_job(
            {instance.sample: [options.infile]},
            treename='Runs',
            processor_instance=instance,
            executor=futures_executor,
            executor_args={'workers': 10},
            chunksize=500000000
        )
        for h, hist in output.items():
            f[h] = export1d(hist)
            #print(f'wrote {h} to tree_{options.jobNum}_WS.root')

