import os
import numpy as np
import h5py
from DatasetProcessor import DatasetProcessor
from FileProcessor import FileProcessor 
from matplotlib import pyplot as plt 

def view_data(filename, data_route, bins = 100):
	f = h5py.File(filename, 'r')
	data = f[data_route][()]
	features = data.shape[1] 

	for i in range(6, features):
		print 'feature:', i
		plt.hist(data[:, i], bins = 100, color = 'blue', alpha = 0.5)
		plt.show()

if __name__ == '__main__':
	path = '../../Data/'

	# Test the file processor 
	dirs = ['old/ZJetsToNuNu_HT-200to400_Tune4C_13TeV-madgraph-tauola']
	dirs = [path + d for d in dirs]
	print dirs 

	fprocessor = FileProcessor(dirs)
	leaves = [3, 2, 2]

	fprocessor.process(leaves, overwrite = False, parallel = False)

	# Initialise the processor 
	path = '../../Data/'
	files = ['old/ZJetsToNuNu_HT-200to400_Tune4C_13TeV-madgraph-tauola.hdf5']

	files = [path + f for f in files] 
	print files

	dprocessor = DatasetProcessor(files)

	# Perform cuts in the dataset
	sumcut = 0 
	ptcut = 0 
	mrcut = 0 
	r2cut = 0.09

	dprocessor.hadronicCuts(sumcut = sumcut, ptcut = ptcut, mrcut = mrcut, r2cut = r2cut)

	# Make into SOM datasets 
	filename = '%sSOMDataset_sumMET%d_metPt%d_MR%d_Rsq%d.hdf5'%(path, sumcut, ptcut, mrcut, r2cut * 100)
	trn_samples = [1000]
	cv_samples = [10]
	tst_samples = [10000]
	xSections = [100]

	dprocessor.SOMdatasets(trn_samples, cv_samples, tst_samples, xSections, filename)

	# Make datasets for NADE
	ID = lambda x:x #the idensity function 
	eta = lambda x:np.arctanh(x/2.4 * 0.99) 
	jetpt = lambda x:np.log(x - 39.5)
	metpt = lambda x:np.log(x - (ptcut - 0.05))
	summet = lambda x:np.log(x - (sumcut - 1))
	mr = lambda x: np.log(x) 
	r2 = lambda x: np.log(x - (r2cut - 0.002)) 
	funcs = [jetpt, jetpt, eta, eta, ID, ID, summet, metpt, r2, mr]

	filename = '%sNADEdataset_sumMET%d_metPt%d_MR%d_Rsq%d.hdf5'%(path, sumcut, ptcut, mrcut, r2cut * 100)
	samples = np.array([10000])
	trn_ratio = 0.1

	dprocessor.NADEdatasets(samples, trn_ratio, funcs, filename)

	view_data(filename, 'test/data')


