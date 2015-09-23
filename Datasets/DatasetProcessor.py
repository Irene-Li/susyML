import os
import numpy as np
import h5py


class DatasetProcessor(object):
	'''
	A processor that processes a list of files and select 
	'''
	def __init__(self, files):
		'''
		Initialise with a list of files to be processed
		'''
		for f in files:
			if not os.path.isfile(f): 
				print f, 'does not exist'
		self.files = files

	# ------------------------------------------------------
	# Method for training datasets preparation
	# ------------------------------------------------------


	@staticmethod
	def transform(data, funcs):
		assert data.shape[1] == len(funcs)
		for i in range(data.shape[1]):
			data[:, i] = funcs[i](data[:, i])
		return data 

	# Replace all the zeros, could be useful if NADE dataset ever needs to be made without cuts
	@staticmethod
	def replace_zeros(dataset, mus = None, sigmas = None): 
	    samples = dataset.shape[0] 
	    features = dataset.shape[1] 
	    maxs = dataset.max(0)
	    mmax = maxs.max()
	    dataset[dataset == 0] = mmax
	    mins = dataset.min(0)
	    if mus!= None and sigmas!= None:
	        means = mus
	        stds = sigmas
	    else:
	        stds = (maxs - mins)/6
	        means = 2 * mins - maxs
	    for i in range(features): 
	        for j in range(samples):
	            if dataset[j, i] == mmax:
	                dataset[j, i] = np.random.normal(means[i], stds[i])
	    return (mins, means, stds, dataset)


	@staticmethod
	def make_samples(datasets, trn_n, tst_n, cv_n, features, weights = None):
		'''
		datasets: a set with integers as keys 
		trn_n, tst_n, cv_n: number of samples selected from each type
		weights: weight added to each type
		features: number of features
		'''

		assert len(trn_n) == len(tst_n)
		assert len(trn_n) == len(cv_n)

		dim = features # dimension of the dataset
		if isinstance(weights, np.ndarray) or isinstance(weights, list):
			dim += 1

		types = len(trn_n)
		trndata = np.zeros((np.sum(trn_n), dim), dtype = np.float32)
		tstdata = np.zeros((np.sum(tst_n), dim), dtype = np.float32)
		cvdata = np.zeros((np.sum(cv_n), dim), dtype = np.float32)

		trn_index = 0 
		tst_index = 0 
		cv_index = 0
		for i in range(types):
			np.random.shuffle(datasets[i])
			trndata[trn_index: trn_index + trn_n[i], 0:features] = datasets[i][0:trn_n[i]]
			tstdata[tst_index: tst_index + tst_n[i], 0:features] = datasets[i][trn_n[i]: trn_n[i] + tst_n[i]] 
			cvdata[cv_index: cv_index + cv_n[i], 0:features] = datasets[i][trn_n[i] + tst_n[i]: trn_n[i] + tst_n[i] + cv_n[i]]

			if isinstance(weights, np.ndarray) or isinstance(weights, list):
				trndata[trn_index: trn_index + trn_n[i], features] = weights[i] 
				tstdata[tst_index: tst_index + tst_n[i], features] = i 
				cvdata[cv_index:cv_index + cv_n[i], features] = i

			trn_index += trn_n[i]  
			tst_index += tst_n[i] 
			cv_index += cv_n[i]

		return trndata, tstdata, cvdata

	# --------------------------------------------------------
	# Methods for performing cuts in a dataset
	# --------------------------------------------------------

	@staticmethod
	def cut(data, veto, req, rem, sumcut = 0, ptcut = 0, mrcut = 0, r2cut = 0):
		'''
		a method that performs cuts in a dataset 
		veto: reject event if feature is nonzero 
		req: reject event if feature is zero 
		red: redundant information
		ptcut: cut in Rsq
		'''
		samples = data.shape[0] #number of datapoints 
		features = data.shape[1] #number of features 
		cuts = np.array([sumcut, ptcut, r2cut, mrcut])

		cut_data = np.zeros((samples, len(rem)))
		count = 0 
		for i in range(samples):
			sample = data[i]
			if np.sum(sample[veto] != 0) == 0 and np.sum(sample[req] == 0) == 0: 
				if np.sum(sample[-4:] < cuts) == 0: # all the cuts are satisfied
					cut_data[count] = sample[rem]
					count += 1

		return cut_data, count

	@staticmethod
	def select(datafile, label, veto, req, rem, sumcut = 0, ptcut = 0, mrcut = 0, r2cut = 0, overwrite = False):
		'''
		a method that performs cut in a dataset, writes to a separate file and returns the filename
		datafile: the name of the hdf5 file where data is stored under the dataset named 'data' 
		'''

		result = '%s_%s_sumMET%d_metPt%d_MR%d_Rsq%d.hdf5'%(datafile[0:-5], label, sumcut, ptcut, mrcut, r2cut * 100) 

		if (overwrite or not os.path.isfile(result)):
			f = h5py.File(datafile, 'r')
			dset = f['data']
			samples = dset.attrs['samples']
			eff1 = dset.attrs['eff']
			n_features =  len(rem)

			dataset = np.zeros((samples, n_features), dtype = 'float32')
			batchsize = 4000

			count = 0 
			index = 0

			while index < samples:
				bs = min(batchsize, samples - index)
				dataset[count:count+bs], selected = DatasetProcessor.cut(dset[index:index+bs], veto, req, rem, sumcut = sumcut, ptcut = ptcut, mrcut = mrcut, r2cut = r2cut)
				count += selected
				index += bs
			dataset = dataset[0:count]
			f.close()

			eff2 = count/float(index)

			f = h5py.File(result, 'a')
			if "data" in f.keys():
				del f['data'] 
			dset = f.create_dataset('data', data = dataset, dtype = 'float32')
			dset.attrs['eff1'] = eff1
			dset.attrs['eff2'] = eff2
			dset.attrs['sumcut'] = sumcut 
			dset.attrs['ptcut'] = ptcut
			dset.attrs['mrcut'] = mrcut
			dset.attrs['r2cut'] = r2cut
			dset.attrs['samples'] = dataset.shape[0]
			dset.attrs['nfeatures'] = n_features
			f.close() 

			print 'saved to file:', result

		return result

	def hadronicCuts(self, sumcut = 0, ptcut = 0, mrcut = 0, r2cut = 0, overwrite = False):
		'''
		A method that select the hadronic events with specified cuts and save them to files, the file lists are also updated
		overwrite: a flag that indicates whether existing files need to be overwritten
		'''

		# Select diject events with no muons and no electrons 
		n_features = 17
		veto_indices = [9, 10, 11, 12]
		req_indices = []
		red_indices = [2, 5, 8]
		rem_indices = [i for i in np.arange(n_features) if not i in veto_indices and not i in red_indices]
		n_features = len(rem_indices)

		cut_files = [] 

		for f in self.files:
			terminal_file = DatasetProcessor.select(f, 'hadronic', veto_indices, req_indices, rem_indices, sumcut = sumcut, ptcut = ptcut, mrcut = mrcut, r2cut = r2cut, overwrite = overwrite)
			cut_files.append(terminal_file)

		# Replace the class argument with 
		self.files = cut_files 

	def unboxedCuts(self, sumcut = 0, ptcut = 0, mrcut = 0, r2cut = 0, overwrite = False):
		# Get all the events 
		n_features = 17 
		veto_indices = [] 
		req_indices = [] 
		rem_indices = np.arange(n_features)
		cut_files = [] 

		for f in self.files:
			terminal_file = DatasetProcessor.select(f, 'unboxed', veto_indices, req_indices, rem_indices, sumcut = sumcut, ptcut = ptcut, mrcut = mrcut, r2cut = r2cut, overwrite = overwrite)
			cut_files.append(terminal_file)

		# replace the class argument with the new files 
		self.files = cut_files


	def SOMdatasets(self, xSections, luminosity, max_n, filename):
		'''
		A method that
			- selects a certain number of samples from each file
			- for training data, the last column is the weight of each data point 
			- for validation and test data, the last column indicates the type 
			- packs into file with name given by filename
		'''
		datasets = {} 
		n_types = len(xSections)
		samples = np.zeros(n_types)
		eff1 = np.zeros(n_types)
		eff2 = np.zeros(n_types)

		for f_i, f in enumerate(self.files):
			if os.path.isfile(f):
				h5file = h5py.File(f, 'r')
				datasets[f_i] = h5file['data'][()]
				samples[f_i] = h5file['data'].attrs['samples']
				eff1[f_i] = h5file['data'].attrs['eff1']
				eff2[f_i] = h5file['data'].attrs['eff2']
				n_features = h5file['data'].attrs['nfeatures']
				h5file.close()
			else:
				print f, 'is not a file'

		# Found out which type of data is the limiting factor
		n = xSections * eff1 * eff2 * 1000 * luminosity
		lim = np.argmin(samples/n) # the limiting factor
		trn_samples = np.minimum(samples/2.0, n).astype(int)
		trn_samples = np.minimum(trn_samples, trn_samples[lim])
		trn_samples = np.minimum(trn_samples, max_n)
		cv_samples = np.ones(4)
		tst_samples = samples - trn_samples - cv_samples
		weights = (n/trn_samples).astype(int)
		weights_tst = n/tst_samples
		weights_cv = n/cv_samples
		print 'SOM --------------------------------------'
		print 'training samples:', trn_samples
		print 'cross validation samples:', cv_samples
		print 'test samples:', tst_samples
		print 'cross sections:', xSections
		print 'eff1:', eff1
		print 'eff2:', eff2
		print 'weights:', weights
		print 'data file:', filename

		# Split into training datasets and testing datasets
		trndata, tstdata, cvdata = self.make_samples(datasets, trn_samples, tst_samples, cv_samples, n_features, weights)

		# Save data to file 
		f = h5py.File(filename, 'a')
		if 'data' in f.keys():
			print 'data exists'
			del f['data']

		grp = f.create_group('data')
		trn = grp.create_dataset('training', data = trndata, dtype = 'float32')
		trn.attrs['samples'] = trn_samples
		trn.attrs['weights'] = weights
		tst = grp.create_dataset('testing', data = tstdata, dtype = 'float32')
		tst.attrs['samples'] = tst_samples
		tst.attrs['weights'] = weights_tst
		cv = grp.create_dataset('validation', data = cvdata, dtype = 'float32')
		cv.attrs['samples'] = cv_samples
		cv.attrs['weights'] = weights_cv

		grp.attrs['eff1'] = eff1
		grp.attrs['eff2'] = eff2 
		grp.attrs['nfeatures'] = n_features
		grp.attrs['types'] = n_types
		f.close()

		print 'SOM --------------------------------------'

	def NADEdatasets(self, xSections, trn_total, funcs, filename, sig_index):
		'''
		samples: xSections of the datasets
		trn_total: total number of samples for training
		funcs: transforming functions 
		filename: terminal file 

		This method: 
			- selects a certain number of samples from each file, according to trn_ratio
			- cv_ratio = trn_ratio, test data is whatever remains
			- transforms all the variables according to funcs 
			- save to file with name given by filename
		'''
		datasets = {} 
		n_types = len(xSections)

		samples = np.zeros(n_types)
		eff1 = np.zeros(n_types)
		eff2 = np.zeros(n_types)

		print 'NADE -------------------------------------'

		for f_i, f in enumerate(self.files):
			if os.path.isfile(f):
				h5file = h5py.File(f, 'r')
				datasets[f_i] = h5file['data'][()]
				samples[f_i] = h5file['data'].attrs['samples']
				eff1[f_i] = h5file['data'].attrs['eff1']
				eff2[f_i] = h5file['data'].attrs['eff2']
				n_features = h5file['data'].attrs['nfeatures']
				h5file.close()

				# transform the datasets 
				datasets[f_i] = self.transform(datasets[f_i], funcs)
			else: 
				print f, 'is not a file'
		print 'samples:', samples

		# Decide how many samples to get from each dataset
		ratios = xSections * eff1 * eff2 
		trn_samples = (ratios * trn_total/np.sum(ratios[0:-1])).astype(int) 
		cv_samples = (trn_samples * 0.5).astype(int)
		tst_samples = ((samples[0] - trn_samples[0] - cv_samples[0])/ratios[0] * ratios).astype(int)

		print 'training samples:', trn_samples 
		print 'validation samples:', cv_samples 
		print 'test samples:', tst_samples

		trndata, tstdata, cvdata = self.make_samples(datasets, trn_samples, tst_samples, cv_samples, n_features)

		f = h5py.File(filename, 'a')

		print 'existing keys:', f.keys()
		if "train" in f.keys():
			del f['train'] 
			del f['validation'] 
			del f['test'] 
			del f['sig']
		grp = f.create_group('train')
		grp.create_dataset('data', data = trndata)
		grp.attrs['samples'] = trn_samples
		f.create_group('validation').create_dataset('data', data = cvdata)
		f['validation'].attrs['samples'] = cv_samples
		f.create_group('test').create_dataset('data', data = tstdata)
		f['validation'].attrs['samples'] = tst_samples
		f.create_group('sig').create_dataset('data', data = datasets[sig_index])
		f['sig'].attrs['samples'] = datasets[sig_index].shape[0]
		f.attrs['types'] = n_types 
		f.attrs['eff1'] = eff1
		f.attrs['eff2'] = eff2
		f.attrs['nfeatures'] = n_features
		f.close() 

		print 'NADE -------------------------------------'


if __name__ == '__main__':

	# Initialise the processor 
	path = '/afs/cern.ch/work/y/yuting/public/razor/'

	files = ['new/QCD_Pt_170to300_TuneCUETP8M1_13TeV_pythia8.hdf5',
	 'old/ZJetsToNuNu_HT-200to400_Tune4C_13TeV-madgraph-tauola.hdf5',
	 'old/TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola.hdf5',
	'old/SMS-T1bbbb_2J_mGl-1500_mLSP-100_Tune4C_13TeV-madgraph-tauola.hdf5'] 

	files = [path + f for f in files] 

	# Perform cuts in the dataset
	cuts = [(0, 0, 0, 0.1), (0, 0, 0, 0.05), (0, 0, 0, 0), (0, 65, 0, 0)]
	trns = [20000, 30000, 40000, 30000]

	for (i, cut) in enumerate(cuts):

		processor = DatasetProcessor(files)

		sumcut = cut[0] 
		ptcut = cut[1] 
		mrcut = cut[2]
		r2cut = cut[3] 

		print 'sumcut:', sumcut, 'ptcut:', ptcut, 'mrcut:', mrcut, 'r2cut:', r2cut

		processor.hadronicCuts(sumcut = sumcut, ptcut = ptcut, mrcut = mrcut, r2cut = r2cut, overwrite = False)

		# Make datasets for SOM 
		filename = '%sSOMDataset_sumMET%d_metPt%d_MR%d_Rsq%d.hdf5'%(os.path.join(path, 'Data/'), sumcut, ptcut, mrcut, r2cut * 100)
		xSections = [117267, 100, 424, 0.014] # in pb
		luminosity = 10 # in fb^-1
		max_trn = 50000 #maximum number of samples from one type of data

		processor.SOMdatasets(xSections, luminosity, max_trn, filename)

		# Make datasets for NADE
		ID = lambda x:x #the idensity function 
		eta = lambda x:np.arctanh(x/2.4 * 0.99) 
		jetpt = lambda x:np.log(x - 39.5)
		metpt = lambda x:np.log(x - (ptcut - 0.05))
		summet = lambda x:np.log(x - (sumcut - 1))
		mr = lambda x: np.log(x) 
		r2 = lambda x: np.log(x - (r2cut - 0.0001)) 
		funcs = [jetpt, jetpt, eta, eta, ID, ID, summet, metpt, r2, mr]

		filename = '%sNADEdataset_sumMET%d_metPt%d_MR%d_Rsq%d.hdf5'%(os.path.join(path, 'Data/'), sumcut, ptcut, mrcut, r2cut * 100)
		trn_total = trns[i]
		sig_index = 3

		processor.NADEdatasets(xSections, trn_total, funcs, filename, sig_index)

	for (i, cut) in enumerate(cuts):

		processor = DatasetProcessor(files)

		sumcut = cut[0] 
		ptcut = cut[1] 
		mrcut = cut[2]
		r2cut = cut[3] 

		print 'sumcut:', sumcut, 'ptcut:', ptcut, 'mrcut:', mrcut, 'r2cut:', r2cut

		processor.unboxedCuts(sumcut = sumcut, ptcut = ptcut, mrcut = mrcut, r2cut = r2cut, overwrite = False)

		# Make datasets for SOM 
		filename = '%sSOMall_sumMET%d_metPt%d_MR%d_Rsq%d.hdf5'%(os.path.join(path, 'Data/'), sumcut, ptcut, mrcut, r2cut * 100)
		xSections = [117267, 100, 424, 0.014] # in pb
		luminosity = 10 # in fb^-1
		max_trn = 50000 #maximum number of samples from one type of data

		processor.SOMdatasets(xSections, luminosity, max_trn, filename)
















































