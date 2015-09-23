import os 
import sys
import shutil

if __name__ == '__main__':
    path = 'eos/cms/store/group/phys_susy/razor/run2/RazorNtupleV1.5/PHYS14_25ns/v7/sixie/'
  #  path = 'eos/cms/store/group/phys_susy/razor/run2/Run2RazorNtupleV1.14/MC/RunIISpring15DR74_50ns/v3/sixie/'
    indir = ['QCD_Pt-170to300_Tune4C_13TeV_pythia8/razorNtuplerV1p5_PHYS14_25ns_v7_v1/150213_165922/0000/', 'SMS-T1bbbb_2J_mGl-1500_mLSP-100_Tune4C_13TeV-madgraph-tauola/razorNtuplerV1p5_PHYS14_25ns_v7_v1/150212_173952/0000/', 'ZJetsToNuNu_HT-200to400_Tune4C_13TeV-madgraph-tauola/razorNtuplerV1p5_PHYS14_25ns_v7_v1/150212_174727/0000/','TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola/razorNtuplerV1p5_PHYS14_25ns_v7_v1/150212_174432/0000/', 'WJetsToLNu_13TeV-madgraph-pythia8-tauola/razorNtuplerV1p5_PHYS14_25ns_Phys14DR-PU4bx50_PHYS14_25_V1-v1_v7_v2/150603_201201/0000/'] 
    
  #  indir = ['', '', 'QCD_Pt_170to300_TuneCUETP8M1_13TeV_pythia8/Run2RazorNtuplerV1p14_ToCERN_MC_50ns_RunIISpring15DR74-Asympt50ns_MCRUN2_74_V9A-v2_v3_v1/150724_042550/0000/']
  #  outpath = 'new/'        
    outpath = 'old/'
  #  outdir = ['TTJets_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/', 'WJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/', 'QCD_Pt_170to300_TuneCUETP8M1_13TeV_pythia8/', 'DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/']
    outdir = ['QCD_Pt-170to300_Tune4C_13TeV_pythia8', 'SMS-T1bbbb_2J_mGl-1500_mLSP-100_Tune4C_13TeV-madgraph-tauola', 'ZJetsToNuNu_HT-200to400_Tune4C_13TeV-madgraph-tauola', 'TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola', 'WJetsToLNu_13TeV-madgraph-pythia8-tauola']

    d_in = int(sys.argv[1])
    i = int(sys.argv[2])
    f = int(sys.argv[3])
    din = path + indir[d_in]
    dout = outpath + outdir[d_in]
    print dout
    for j in range(i, f):
        filename = '/razorNtuple_' + str(j) + '.root'
        print filename
        if os.path.isfile(din + filename):
            shutil.copy(din + filename, dout + filename)

