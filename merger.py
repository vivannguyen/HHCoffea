import os, glob
from array import array
import sys
from termcolor import colored




indir = "/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/test/qcdtestv2/"
outdir = "/eos/cms/store/group/phys_higgs/HiggsExo/HH_bbZZ_bbllqq/test/qcdtestv2/merged"

def main():
    pattern = "WZ"
    for sample in os.listdir(indir):
        print sample
        print colored(" -- " + sample, "blue")
        if "merged" in sample:
            continue
        #if pattern not in sample:
        #    continue
        in_files = glob.glob("{indir}{sample}/*.root".format(sample=sample, indir=indir))
        str1 = " "
        in_files = str1.join(in_files)
        out_file = "{outdir}/{sample}.root".format(sample=sample, outdir=outdir)
        if len(in_files) == 0:
            print colored(" -- [warning] empty directory for " + sample, "red")
            continue
        #print("python haddnano.py {} {}".format(out_file,in_files))
        #os.system("python haddnano.py {} {}".format(out_file,in_files))
        os.system("python haddnano.py {} {}".format("out.root",in_files))
        os.system("mv out.root {}".format(out_file))

if __name__ == "__main__":
   main()
