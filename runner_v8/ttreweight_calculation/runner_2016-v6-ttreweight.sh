python3 condor_HH_WS.py --isMC=1 --era=2016 --doSyst=0 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7-top-reweighting/2016/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8.root --njetw btag_weights.jsonl
mv tree_2016_WS.root output2016/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8_WS_selections.root
echo "finished TT_TuneCUETP8M2T4_13TeV-powheg-pythia8.root"

#python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v6/2016/DYToLL_1J_13TeV-amcatnloFXFX-pythia8.root --njetw btag_weights.jsonl
#mv tree_2016_WS.root output2016/DYToLL_1J_13TeV-amcatnloFXFX-pythia8_WS_selections.root
#echo "finished DYToLL_1J_13TeV-amcatnloFXFX-pythia8.root"
