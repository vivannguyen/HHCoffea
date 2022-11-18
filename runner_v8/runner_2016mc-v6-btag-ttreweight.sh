python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8_WS_selections.root
echo "finished DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/DYToLL_0J_13TeV-amcatnloFXFX-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/DYToLL_0J_13TeV-amcatnloFXFX-pythia8_WS_selections.root
echo "finished DYToLL_0J_13TeV-amcatnloFXFX-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/DYToLL_1J_13TeV-amcatnloFXFX-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/DYToLL_1J_13TeV-amcatnloFXFX-pythia8_WS_selections.root
echo "finished DYToLL_1J_13TeV-amcatnloFXFX-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/DYToLL_2J_13TeV-amcatnloFXFX-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/DYToLL_2J_13TeV-amcatnloFXFX-pythia8_WS_selections.root
echo "finished DYToLL_2J_13TeV-amcatnloFXFX-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/GluGluHToBB_M125_13TeV_amcatnloFXFX_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/GluGluHToBB_M125_13TeV_amcatnloFXFX_pythia8_WS_selections.root
echo "finished GluGluHToBB_M125_13TeV_amcatnloFXFX_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/GluGluHToWWTo2L2Nu_M125_13TeV_powheg_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/GluGluHToWWTo2L2Nu_M125_13TeV_powheg_pythia8_WS_selections.root
echo "finished GluGluHToWWTo2L2Nu_M125_13TeV_powheg_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8_WS_selections.root
echo "finished GluGluHToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/GluGluToHHTo2B2ZTo2L2J_node_cHHH0_TuneCUETP8M1_PSWeights_13TeV-powheg-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/GluGluToHHTo2B2ZTo2L2J_node_cHHH0_TuneCUETP8M1_PSWeights_13TeV-powheg-pythia8_WS_selections.root
echo "finished GluGluToHHTo2B2ZTo2L2J_node_cHHH0_TuneCUETP8M1_PSWeights_13TeV-powheg-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/GluGluToHHTo2B2ZTo2L2J_node_cHHH1_TuneCUETP8M1_PSWeights_13TeV-powheg-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/GluGluToHHTo2B2ZTo2L2J_node_cHHH1_TuneCUETP8M1_PSWeights_13TeV-powheg-pythia8_WS_selections.root
echo "finished GluGluToHHTo2B2ZTo2L2J_node_cHHH1_TuneCUETP8M1_PSWeights_13TeV-powheg-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/GluGluToHHTo2B2ZTo2L2J_node_cHHH2p45_TuneCUETP8M1_PSWeights_13TeV-powheg-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/GluGluToHHTo2B2ZTo2L2J_node_cHHH2p45_TuneCUETP8M1_PSWeights_13TeV-powheg-pythia8_WS_selections.root
echo "finished GluGluToHHTo2B2ZTo2L2J_node_cHHH2p45_TuneCUETP8M1_PSWeights_13TeV-powheg-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/GluGluToHHTo2B2ZTo2L2J_node_cHHH5_TuneCUETP8M1_PSWeights_13TeV-powheg-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/GluGluToHHTo2B2ZTo2L2J_node_cHHH5_TuneCUETP8M1_PSWeights_13TeV-powheg-pythia8_WS_selections.root
echo "finished GluGluToHHTo2B2ZTo2L2J_node_cHHH5_TuneCUETP8M1_PSWeights_13TeV-powheg-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/GluGluZH_HToWW_M125_13TeV_powheg_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/GluGluZH_HToWW_M125_13TeV_powheg_pythia8_WS_selections.root
echo "finished GluGluZH_HToWW_M125_13TeV_powheg_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/ST_s-channel_4f_InclusiveDecays_13TeV-amcatnlo-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/ST_s-channel_4f_InclusiveDecays_13TeV-amcatnlo-pythia8_WS_selections.root
echo "finished ST_s-channel_4f_InclusiveDecays_13TeV-amcatnlo-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8_WS_selections.root
echo "finished ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8_WS_selections.root
echo "finished ST_t-channel_top_4f_InclusiveDecays_TuneCP5_PSweights_13TeV-powheg-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/ST_tW_antitop_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/ST_tW_antitop_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1_WS_selections.root
echo "finished ST_tW_antitop_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/ST_tW_top_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/ST_tW_top_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1_WS_selections.root
echo "finished ST_tW_top_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/TTWJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/TTWJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8_WS_selections.root
echo "finished TTWJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/TTWJetsToQQ_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/TTWJetsToQQ_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8_WS_selections.root
echo "finished TTWJetsToQQ_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/TTZToLLNuNu_M-10_TuneCUETP8M1_13TeV-amcatnlo-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/TTZToLLNuNu_M-10_TuneCUETP8M1_13TeV-amcatnlo-pythia8_WS_selections.root
echo "finished TTZToLLNuNu_M-10_TuneCUETP8M1_13TeV-amcatnlo-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/TTZToQQ_TuneCUETP8M1_13TeV-amcatnlo-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/TTZToQQ_TuneCUETP8M1_13TeV-amcatnlo-pythia8_WS_selections.root
echo "finished TTZToQQ_TuneCUETP8M1_13TeV-amcatnlo-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7-top-reweighting/2016/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8_WS_selections.root
echo "finished TT_TuneCUETP8M2T4_13TeV-powheg-pythia8.root"

#python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/VBFHHTo2B2ZTo2L2J_CV_1_5_C2V_1_C3_1_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
#mv tree_2016_WS.root output2016/VBFHHTo2B2ZTo2L2J_CV_1_5_C2V_1_C3_1_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8_WS_selections.root
#echo "finished VBFHHTo2B2ZTo2L2J_CV_1_5_C2V_1_C3_1_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8.root"
#
#python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/VBFHHTo2B2ZTo2L2J_CV_1_C2V_0_C3_1_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
#mv tree_2016_WS.root output2016/VBFHHTo2B2ZTo2L2J_CV_1_C2V_0_C3_1_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8_WS_selections.root
#echo "finished VBFHHTo2B2ZTo2L2J_CV_1_C2V_0_C3_1_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8.root"
#
#python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/VBFHHTo2B2ZTo2L2J_CV_1_C2V_1_C3_0_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
#mv tree_2016_WS.root output2016/VBFHHTo2B2ZTo2L2J_CV_1_C2V_1_C3_0_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8_WS_selections.root
#echo "finished VBFHHTo2B2ZTo2L2J_CV_1_C2V_1_C3_0_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8.root"
#
#python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/VBFHHTo2B2ZTo2L2J_CV_1_C2V_1_C3_1_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
#mv tree_2016_WS.root output2016/VBFHHTo2B2ZTo2L2J_CV_1_C2V_1_C3_1_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8_WS_selections.root
#echo "finished VBFHHTo2B2ZTo2L2J_CV_1_C2V_1_C3_1_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8.root"
#
#python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/VBFHHTo2B2ZTo2L2J_CV_1_C2V_1_C3_2_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
#mv tree_2016_WS.root output2016/VBFHHTo2B2ZTo2L2J_CV_1_C2V_1_C3_2_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8_WS_selections.root
#echo "finished VBFHHTo2B2ZTo2L2J_CV_1_C2V_1_C3_2_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8.root"
#
#python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/VBFHHTo2B2ZTo2L2J_CV_1_C2V_2_C3_1_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
#mv tree_2016_WS.root output2016/VBFHHTo2B2ZTo2L2J_CV_1_C2V_2_C3_1_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8_WS_selections.root
#echo "finished VBFHHTo2B2ZTo2L2J_CV_1_C2V_2_C3_1_dipoleRecoilOff-TuneCUETP8M1_PSweights_13TeV-madgraph-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/VBFHToBB_M-125_13TeV_powheg_pythia8_weightfix.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/VBFHToBB_M-125_13TeV_powheg_pythia8_weightfix_WS_selections.root
echo "finished VBFHToBB_M-125_13TeV_powheg_pythia8_weightfix.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/VBFHToWWTo2L2Nu_M125_13TeV_powheg_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/VBFHToWWTo2L2Nu_M125_13TeV_powheg_pythia8_WS_selections.root
echo "finished VBFHToWWTo2L2Nu_M125_13TeV_powheg_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/VBF_HToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/VBF_HToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8_WS_selections.root
echo "finished VBF_HToZZTo4L_M125_13TeV_powheg2_JHUgenV6_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/VVTo2L2Nu_13TeV_amcatnloFXFX_madspin_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/VVTo2L2Nu_13TeV_amcatnloFXFX_madspin_pythia8_WS_selections.root
echo "finished VVTo2L2Nu_13TeV_amcatnloFXFX_madspin_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/WH_HToBB_WToLNu_M125_13TeV_amcatnloFXFX_madspin_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/WH_HToBB_WToLNu_M125_13TeV_amcatnloFXFX_madspin_pythia8_WS_selections.root
echo "finished WH_HToBB_WToLNu_M125_13TeV_amcatnloFXFX_madspin_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/WToLNu_0J_13TeV-amcatnloFXFX-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/WToLNu_0J_13TeV-amcatnloFXFX-pythia8_WS_selections.root
echo "finished WToLNu_0J_13TeV-amcatnloFXFX-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/WToLNu_1J_13TeV-amcatnloFXFX-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/WToLNu_1J_13TeV-amcatnloFXFX-pythia8_WS_selections.root
echo "finished WToLNu_1J_13TeV-amcatnloFXFX-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/WToLNu_2J_13TeV-amcatnloFXFX-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/WToLNu_2J_13TeV-amcatnloFXFX-pythia8_WS_selections.root
echo "finished WToLNu_2J_13TeV-amcatnloFXFX-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/WWTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/WWTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8_WS_selections.root
echo "finished WWTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/WWTo4Q_4f_13TeV_amcatnloFXFX_madspin_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/WWTo4Q_4f_13TeV_amcatnloFXFX_madspin_pythia8_WS_selections.root
echo "finished WWTo4Q_4f_13TeV_amcatnloFXFX_madspin_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/WZTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/WZTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8_WS_selections.root
echo "finished WZTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/WZTo1L3Nu_13TeV_amcatnloFXFX_madspin_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/WZTo1L3Nu_13TeV_amcatnloFXFX_madspin_pythia8_WS_selections.root
echo "finished WZTo1L3Nu_13TeV_amcatnloFXFX_madspin_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8_WS_selections.root
echo "finished WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/WZTo2Q2Nu_13TeV_amcatnloFXFX_madspin_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/WZTo2Q2Nu_13TeV_amcatnloFXFX_madspin_pythia8_WS_selections.root
echo "finished WZTo2Q2Nu_13TeV_amcatnloFXFX_madspin_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/WZTo3LNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/WZTo3LNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8_WS_selections.root
echo "finished WZTo3LNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/WminusH_HToZZTo4L_M125_13TeV_powheg2-minlo-HWJ_JHUgenV6_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/WminusH_HToZZTo4L_M125_13TeV_powheg2-minlo-HWJ_JHUgenV6_pythia8_WS_selections.root
echo "finished WminusH_HToZZTo4L_M125_13TeV_powheg2-minlo-HWJ_JHUgenV6_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/WplusH_HToZZTo4L_M125_13TeV_powheg2-minlo-HWJ_JHUgenV6_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/WplusH_HToZZTo4L_M125_13TeV_powheg2-minlo-HWJ_JHUgenV6_pythia8_WS_selections.root
echo "finished WplusH_HToZZTo4L_M125_13TeV_powheg2-minlo-HWJ_JHUgenV6_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/ZH_HToBB_ZToLL_M125_13TeV_powheg_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/ZH_HToBB_ZToLL_M125_13TeV_powheg_pythia8_WS_selections.root
echo "finished ZH_HToBB_ZToLL_M125_13TeV_powheg_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8_WS_selections.root
echo "finished ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/ZZTo2Q2Nu_13TeV_amcatnloFXFX_madspin_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/ZZTo2Q2Nu_13TeV_amcatnloFXFX_madspin_pythia8_WS_selections.root
echo "finished ZZTo2Q2Nu_13TeV_amcatnloFXFX_madspin_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/ZZTo4L_13TeV-amcatnloFXFX-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/ZZTo4L_13TeV-amcatnloFXFX-pythia8_WS_selections.root
echo "finished ZZTo4L_13TeV-amcatnloFXFX-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/ZZTo4Q_13TeV_amcatnloFXFX_madspin_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/ZZTo4Q_13TeV_amcatnloFXFX_madspin_pythia8_WS_selections.root
echo "finished ZZTo4Q_13TeV_amcatnloFXFX_madspin_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/ggZH_HToBB_ZToLL_M125_13TeV_powheg_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/ggZH_HToBB_ZToLL_M125_13TeV_powheg_pythia8_WS_selections.root
echo "finished ggZH_HToBB_ZToLL_M125_13TeV_powheg_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/ttHJetToNonbb_M125_13TeV_amcatnloFXFX_madspin_pythia8_mWCutfix.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/ttHJetToNonbb_M125_13TeV_amcatnloFXFX_madspin_pythia8_mWCutfix_WS_selections.root
echo "finished ttHJetToNonbb_M125_13TeV_amcatnloFXFX_madspin_pythia8_mWCutfix.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/ttHJetTobb_M125_13TeV_amcatnloFXFX_madspin_pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/ttHJetTobb_M125_13TeV_amcatnloFXFX_madspin_pythia8_WS_selections.root
echo "finished ttHJetTobb_M125_13TeV_amcatnloFXFX_madspin_pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8_WS_selections.root
echo "finished ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8.root"

python3 condor_HH_WS.py --isMC=1 --era=2016 --infile=/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/jlidrych/v7/2016/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8.root --njetw btag_weights.jsonl --ttreweight=1 --doSyst=1
mv tree_2016_WS.root output2016/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8_WS_selections.root
echo "finished ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8.root"
