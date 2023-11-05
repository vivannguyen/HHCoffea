import os, sys
import argparse
import logging
import pwd
import subprocess
import shutil
import time
import glob

logging.basicConfig(level=logging.DEBUG)

script_TEMPLATE = """#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc6_amd64_gcc630

cd {cmssw_base}/src/
#eval `scramv1 runtime -sh`
echo
echo $_CONDOR_SCRATCH_DIR
cd   $_CONDOR_SCRATCH_DIR
echo
echo "... start job at" `date "+%Y-%m-%d %H:%M:%S"`
echo "----- directory before running:"
ls -lR .
echo "----- CMSSW BASE, python path, pwd:"
echo "+ CMSSW_BASE  = $CMSSW_BASE"
echo "+ PYTHON_PATH = $PYTHON_PATH"
echo "+ PWD         = $PWD"
cp $2 temp_$1.root

INFILE=$2
echo "INFILE $INFILE"
python condor_HH_WS_forcondor.py --jobNum=$1 --isMC={ismc} --era={era} --infile=$INFILE --njetw={njetw} --ttreweight={ttreweight}

echo "----- transfer output to eos :"
xrdcp -s -f tree_{era}_WS.root {eosoutdir}/{sample}_WS_selections.root

echo "----- directory after running :"

ls -lR .

echo " ------ THE END (everyone dies !) ----- "
"""


condor_TEMPLATE = """
executable            = {jobdir}/script.sh
arguments             = $(ProcId) $(jobid)
transfer_input_files  = {transfer_file}
output                = $(ClusterId).$(ProcId).out
error                 = $(ClusterId).$(ProcId).err
log                   = $(ClusterId).$(ProcId).log
initialdir            = {jobdir}
transfer_output_files = ""
Requirements = HasSingularity
+JobFlavour           = "{queue}"
+SingularityImage = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest"
queue jobid from {jobdir}/inputfiles.dat
"""

#python condor_HH_WS_forcondor.py --jobNum=$1 --isMC={ismc} --era={era} --infile=$INFILE --njetw={njetw} --ttreweight={ttreweight}
#NUM_EXT="${{INFILE##*_}}"
#echo $NUM_EXT
#NUM="${{NUM_EXT%.root}}"
#echo "LOOK AT ME $NUM"
#xrdcp -s -f tree_{era}_WS.root {eosoutdir}/{sample_dir}_tree_${{NUM}}_WS.root

#xrdcp -s -f tree_{era}_WS.root {eosoutdir}/{sample_dir}_WS.root
#xrdcp -s -f {sample_dir}_tree_${{NUM}}_WS.root {eosoutdir}
def main():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-t"   , "--tag"   , type=str, default="Exorcism"  , help="production tag", required=True)
    parser.add_argument("-isMC", "--isMC"  , type=int, default=1          , help="")
    parser.add_argument("-q"   , "--queue" , type=str, default="espresso", help="")
    parser.add_argument("-e"   , "--era"   , type=str, default="2017"     , help="")
    parser.add_argument("-njetw", "--njetw"  , type=str, default=None          , help="")
    parser.add_argument("-tt", "--ttreweight"  , type=int, default=0          , help="")
    parser.add_argument("-f"   , "--force" , action="store_true"          , help="recreate files and jobs")
    parser.add_argument("-s"   , "--submit", action="store_true"          , help="submit only")
    parser.add_argument("-dry" , "--dryrun", action="store_true"          , help="running without submission")

    options = parser.parse_args()

    cmssw_base = os.environ['CMSSW_BASE']
    indir = "/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/test/{year}v11_merge/".format(year=options.era)
    #indir = "/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/test/{year}DYv8_sys_merge/".format(year=options.era)
    #indir = "/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v6/{year}/".format(year=options.era)
#    if options.isMC==0:
#        indir = "/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v6/{year}/".format(year=options.era)
#    else:
#        indir = "/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/test/{year}v10_merge/".format(year=options.era)
        #indir = "/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v8/{year}/".format(year=options.era)
    ######indir = "/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v8/{year}/".format(year=options.era)
    #indir = "/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/test/zac3/"
    eosbase = "/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/vivan/{tag}/"
    #eosbase = "/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/vivan/{tag}/{year}/{sample}/"
    #eosbase2 = "/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/vivan/{tag}/{year}/"

    def do_the_thing(sample, sample_dir=''):
        sample_joined = os.path.join(sample_dir, sample.replace('.root', ''))
        if "merged" in sample:
            return
        if "WS" in sample:
            return
        if options.isMC==0:
            if "Single" not in sample and "Double" not in sample and "EGamma" not in sample:
                return
        #if "Single" in sample or "Double" in sample or "EGamma" in sample or "02Apr2020_ver1-v1" in sample:
         #   return
#        if "Single" not in sample and "Double" not in sample and "EGamma" not in sample and "02Apr2020_ver1-v1" in sample:
#            return
        else:
            if "DY2JetsToLL_M-50_LHEZpT_150-250" not in sample:
                return
            if "Single" in sample or "Double" in sample or "EGamma" in sample:
                return
            if "VBFHHTo2B2ZTo2L2J" in sample:
                return
            if "node_1" in sample:
                return
            if "node_2" in sample:
                return
            if "node_3" in sample:
                return
            if "node_4" in sample:
                return
            if "node_5" in sample:
                return
            if "node_6" in sample:
                return
            if "node_7" in sample:
                return
            if "node_8" in sample:
                return
            if "node_9" in sample:
                return
            if "node_SM" in sample:
                return
            if "ttHTobb_M125" in sample:
                return
            if "ttHToNonbb_M125" in sample:
                return
#        if "VBF_HToZZTo4L" not in sample:
#            return
#        if "SingleElectron_Run2016B-02Apr2020_ver1-v1" in sample:
#            return
#        if "SingleMuon_Run2016B-02Apr2020_ver1-v1" in sample:
#            return
#        if "DY2JetsToLL_M-50_LHEZpT_150-250" not in sample:
#            return
#        if "Single" not in sample and "Double" not in sample and "EGamma" not in sample:
#            return

        sample = sample.replace('.root', '')
        jobs_dir = '_'.join(['jobs', options.tag,'_'.join([sample_dir,sample])])
        logging.info("-- sample_name : " + sample_joined)

        if os.path.isdir(jobs_dir):
            if not options.force:
                logging.error(" " + jobs_dir + " already exist !")
                return
            else:
                logging.warning(" " + jobs_dir + " already exists, forcing its deletion!")
                shutil.rmtree(jobs_dir)
                #os.system("rm -rf {}".format(jobs_dir))
                os.mkdir(jobs_dir)
        else:
            os.mkdir(jobs_dir)

        with open(os.path.join(jobs_dir, 'inputfiles.dat'), 'w') as infiles:
            infiles.write(os.path.join(indir, "{sample}.root".format(sample=sample_joined)))
            infiles.close()

        time.sleep(1)
        if sample_dir == '':
            #eosoutdir = eosbase2.format(tag=options.tag, year=options.era)
            eosoutdir = eosbase.format(tag=options.tag, year=options.era, sample=sample)
        else:
            eosoutdir = eosbase.format(tag=options.tag, year=options.era, sample=sample_dir)

        if '/eos/cms' in eosoutdir:
            eosoutdir = eosoutdir.replace('/eos/cms', 'root://eoscms.cern.ch/')
            os.system("eos mkdir -p {}".format(eosoutdir.replace('root://eoscms.cern.ch/','')))
        else:
            raise NameError(eosoutdir)

        with open(os.path.join(jobs_dir, "script.sh"), "w") as scriptfile:
            script = script_TEMPLATE.format(
                cmssw_base=cmssw_base,
                ismc=options.isMC,
                era=options.era,
                eosoutdir=eosoutdir,
                sample_dir=sample_dir,
                sample=sample,
                njetw=options.njetw,
                ttreweight=options.ttreweight,
            )
            scriptfile.write(script)
            scriptfile.close()

        with open(os.path.join(jobs_dir, "condor.sub"), "w") as condorfile:
            condor = condor_TEMPLATE.format(
                transfer_file= ",".join([
                    "../condor_HH_WS_forcondor.py",
                    "../python/SumWeights.py",
                    "../python/HH_Producer.py",
                    "../btag_weights.jsonl",
                    "../2016newfinalselecttest",
                    "../2017newfinalselecttest",
                    "../2018newfinalselecttest",
                    "../2016newfinalselecttest-ee",
                    "../2017newfinalselecttest-ee",
                    "../2018newfinalselecttest-ee"
                ]),
                jobdir=jobs_dir,
                queue=options.queue
            )
            condorfile.write(condor)
            condorfile.close()

        if options.dryrun:
            return
        
        htc = subprocess.Popen(
            "condor_submit " + os.path.join(jobs_dir, "condor.sub"),
            shell  = True,
            stdin  = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            close_fds=True
        )
        out, err = htc.communicate()
        exit_status = htc.returncode
        logging.info("condor submission status : {}".format(exit_status))


    for sample in os.listdir(indir):
        if os.path.isfile(os.path.join(indir, sample)):
            do_the_thing(sample)
        elif os.path.isdir(os.path.join(indir, sample)):
            for fi in os.listdir(os.path.join(indir, sample)):
                do_the_thing(fi, sample)

if __name__ == "__main__":
    main()
