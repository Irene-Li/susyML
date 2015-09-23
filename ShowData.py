import h5py 
import os
from matplotlib import pyplot as plt
import numpy as np

# A method that plots histograms of all the features 
def show_data(feature_list, yscale, *arg):
    n_args = len(arg)
    print "comparing", n_args, "datasets" 
    print feature_list
    for (i, feat) in enumerate(feature_list):
        print feat
        plt.yscale(yscale)
        for data in arg:
            plt.hist(data[:, i], label = feat,  bins = 100, alpha = 0.4, normed = True, histtype = 'stepfilled')
        plt.show()

if __name__ == '__main__':
	feature_list = ['jetPt:0', 'jetPt:1', 'jetEta:0', 'jetEta:1', 'jetMass:0', 'jetMass:1', 'sumMET', 'metPt', 'R2', 'MR']
	data_route = 'train/data'
	label = 'NADEdataset'

	cuts = [(0, 0, 0, 0)]

	for cut in cuts:

		sumcut = cut[0] 
		ptcut = cut[1] 
		mrcut = cut[2]
		r2cut = cut[3] 

		dataset = '%s_sumMET%d_metPt%d_MR%d_Rsq%d.hdf5'%(label, sumcut, ptcut, mrcut, r2cut * 100)
		print dataset
		datafile = os.path.join('../Data', dataset)
		f = h5py.File(datafile, 'r')
		data = f[data_route][()]
		r2 = lambda x: np.log(np.exp(x) + (r2cut - 0.002) - (r2cut - 0.0001))
		data[:, -2] = r2(data[:, -2])
		show_data(feature_list, 'linear', data)



