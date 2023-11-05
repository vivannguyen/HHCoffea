import os
import re

from coffea.processor import run_uproot_job, futures_executor

# for local
from python.HH_Producer import *
from python.SumWeights import *

# for condor
#from HH_Producer import *
#from SumWeights import *

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
parser.add_argument('--ttreweight', type=int, default=None, help="")
parser.add_argument('--EFTbenchmark', type=str, default=None, help="")

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

#f_scores = uproot.recreate('BDTscores.root', compression=uproot.ZLIB(4))
#f_scores["Events"] = uproot.newtree({"BDTscores": "float"})

#options.dataset='QCD'

pre_selection = ""

if float(options.nevt) > 0:
    print((" passing this cut and : ", options.nevt))
    pre_selection += ' && (Entry$ < {})'.format(options.nevt)

#pro_syst = ["ElectronEn", "MuonEn"] #, "", "jer"]
pro_syst = ["ElectronEn", "MuonEn", "jer","jesAbsolute", "jesBBEC1", "jesEC2","jesFlavorQCD","jesHF","jesRelativeBal"]#, "met_ptUnclustEn", "met_phiUnclustEn"]
#pro_syst = []
if options.era == '2016':
    pro_syst.extend(["jesAbsolute_2016","jesBBEC1_2016","jesEC2_2016","jesHF_2016","jesRelativeSample_2016"])
if options.era == '2017':
    pro_syst.extend(["jesAbsolute_2017","jesBBEC1_2017","jesEC2_2017","jesHF_2017","jesRelativeSample_2017"])
if options.era == '2018':
    pro_syst.extend(["jesAbsolute_2018","jesBBEC1_2018","jesEC2_2018","jesHF_2018","jesRelativeSample_2018"])

#ext_syst = ["QCDScale0w", "QCDScale1w", "QCDScale2w"]
ext_syst = ["puWeight", "PrefireWeight", "PDF", "MuonSF", "ElectronSF", "TriggerSFWeight", "QCDScale0w", "QCDScale1w", "QCDScale2w",
            "hf", "lf", "cferr1", "cferr2"]

if options.era == '2016':
    ext_syst.extend(["hfstats1_2016","hfstats2_2016","lfstats1_2016","lfstats2_2016"])
if options.era == '2017':
    ext_syst.extend(["hfstats1_2017","hfstats2_2017","lfstats1_2017","lfstats2_2017"])
if options.era == '2018':
    ext_syst.extend(["hfstats1_2018","hfstats2_2018","lfstats1_2018","lfstats2_2018"])

# extension to ext_syst

#btag_syst_up = ["w_btag_SF_sys_up_hf", "w_btag_SF_sys_up_lf", "w_btag_SF_sys_up_cferr1", "w_btag_SF_sys_up_cferr2"]
#btag_syst_down = ["w_btag_SF_sys_down_hf", "w_btag_SF_sys_down_lf", "w_btag_SF_sys_down_cferr1", "w_btag_SF_sys_down_cferr2"]

#btag_uncorr_syst_up = ["w_btag_SF_sys_up_hfstats1", "w_btag_SF_sys_up_hfstats2", "w_btag_SF_sys_up_lfstats1", "w_btag_SF_sys_up_lfstats2"]
#btag_uncorr_syst_down = ["w_btag_SF_sys_down_hfstats1", "w_btag_SF_sys_down_hfstats2", "w_btag_SF_sys_down_lfstats1", "w_btag_SF_sys_down_lfstats2"]

#btag_syst = btag_syst_up + btag_syst_down + btag_uncorr_syst_up + btag_uncorr_syst_down
#btag_syst = btag_syst_up + btag_syst_down

modules_era = []

modules_era.append(HH_NTuple(isMC=options.isMC, era=int(options.era), do_syst=1, syst_var='', sample=options.dataset, njetw=options.njetw, ttreweight=options.ttreweight))#, eftbenchmark=options.EFTbenchmark))#,
#                         haddFileName="tree_%s.root" % str(options.jobNum)))
if options.isMC and options.doSyst==1:
   for sys in pro_syst:
       for var in ["Up", "Down"]:
           modules_era.append(HH_NTuple(options.isMC, int(options.era), do_syst=1,
                                    syst_var=sys + var, sample=options.dataset, njetw=options.njetw, ttreweight=options.ttreweight))#, eftbenchmark=options.EFTbenchmark))#,
#                                    haddFileName=f"tree_{options.jobNum}_{sys}{var}.root"))

   for sys in ext_syst:
       for var in ["Up", "Down"]:
           modules_era.append(
               HH_NTuple(
                   options.isMC, int(options.era),
                   do_syst=1, syst_var=sys + var,
                   weight_syst=True,
                   sample=options.dataset,
                   njetw=options.njetw,
                   ttreweight=options.ttreweight
                   #eftbenchmark=options.EFTbenchmark
#                   haddFileName=f"tree_{options.jobNum}_{sys}{var}.root",
               )
           )

#   for sys in btag_syst:
#       modules_era.append(
#           HH_NTuple(
#               options.isMC, int(options.era),
#               do_syst=1, syst_var=sys,
#               weight_syst=True,
#               sample=options.dataset,
#               njetw=options.njetw#,
#           )
#       )

for i in modules_era:
    print("modules : ", i)

print("Selection : ", pre_selection)
tstart = time.time()
f = uproot.recreate("tree_%s_WS.root" % str(options.era))

for instance in modules_era:
#    output, bdtoutput = run_uproot_job(
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

#    f_scores["Events"].extend({"BDTscores": bdtoutput})

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
