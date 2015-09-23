import numpy as np 
import theano.tensor as T 
from theano import function, shared 
import time
import h5py
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, NoNorm
from mpl_toolkits.mplot3d import Axes3D
import os

class SOM(object):

	'''
	Properties of the map: 
		- Variables
			- features 
			- dim: dimensionality of the map 
			- units: number of units in the map 
		- theano symbolic shared variables
			- shape: shape of the map
			- grid: coordinates of all the cells in the map
			- codebook: weights of all the cells

	Training parameters: 
		- Variables
			- epochs
			- lrate_i, lrate_f
			- sigma_i, sigma_f 
			- threshold
		- theano symbolic shared variables
			- sigma: contains the current sigma 
			- lrate: contains the current lrate 
			- sigma_factor: the factor by which sigma gets updated at each epoch
			- lrate_factor: the factor by which lrate gets updates at each epoch

	Properties of the datasets:
		- Variables
			- means: means of the datasets
			- stds: standard deviation of the datasets
			- fires: index of the cell that fires (produced in training)
			- test: hits in each cell by different types of data (produced in testing)

	Functions: 
		- __init__(shape, features, filename = None)
			- Initialise the map 
		- set_params(epochs = 5, sigma = (6, 0.001), lrate = (0.2, 0.001), threshold = 4)
			- Set training parameters 
		- train_theano(data): 
			- Rescale the data and train the map
		- test_theano(data, types):
			- Test with data
		- save_map(filename)
			- Save the entire content of the map
		- save_results(filename) 
			- Save the results from testing
	'''
	def __init__(self, datafile, razor = True, trn_route = 'data/training', tst_route = 'data/testing', shape = (10, 10, 10), filename = None):
		self.datafile = datafile 
		self.trn_route = trn_route 
		self.tst_route = tst_route 

		if isinstance(filename, str):
			f = h5py.File(filename, 'r')
			grp = f['map'] 
			self.features = grp.attrs['features']
			shape = grp.attrs['shape']
			self.units = np.prod(shape)
			self.means = grp.attrs['means'][()]
			self.stds = grp.attrs['stds'][()]

			# Initialise the shared variables
			self.dim = shared(grp.attrs['dim'][()], name = 'dim')
			self.shape = shared(shape, name = 'shape')
			self.grid = shared(grp['grid'][()], name = 'grid')
			self.codebook = shared(grp['codebook'][()], name = 'grid')

			# Set the parameters 
			self.epochs = grp.attrs['epochs']
			self.threshold = grp.attrs['threshold']
			self.lrate_i, self.lrate_f = grp.attrs['lrate']
			self.sigma_i, self.sigma_f = grp.attrs['sigma']

			f.close()

		else:
			self.units = np.prod(shape)
			f = h5py.File(datafile, 'r')
			assert f[trn_route].shape[1] == f[tst_route].shape[1]
			self.features = f[trn_route].shape[1] - 1 # minus 1 because the last column is weight
			if not razor:
				self.features -= 2 #minus two for the razor variables

			# Initialise the shared variables 
			self.dim = shared(len(shape), name = 'dim')
			self.shape = shared(np.array(shape), name = 'shape')
			self.grid = shared(np.vstack(map(np.ravel, np.indices(shape))).T, name = 'grid')
			self.codebook = shared(np.random.random((self.units, self.features)), name = 'codebook')


	def set_params(self, epochs = 5, sigma = (6, 0.001), lrate = (0.2, 0.001), threshold = 4):
		'''
		epochs: the number of times all the samples get passed
		sigma: neighbourhood 
		lrate: learning rate 
		threshold: stopping condition
		'''
		self.epochs = epochs
		self.threshold = threshold

		self.lrate_i, self.lrate_f = lrate 
		self.sigma_i, self.sigma_f = sigma
		lrate_factor = (self.lrate_f/self.lrate_i) ** (1/float(self.epochs))
		sigma_factor = (self.sigma_f/self.sigma_i) ** (1/float(self.epochs))

		self.sigma = shared(self.sigma_i, name = 'sigma')
		self.lrate = shared(self.lrate_i, name = 'lrate')
		self.sigma_factor = shared(sigma_factor, name = 'sigma_factor')
		self.lrate_factor = shared(lrate_factor, name = 'lrate_factor')

	def _match(self, sample):
		diff = (T.sqr(self.codebook)).sum(axis = 1, keepdims = True) + (T.sqr(sample)).sum(axis = 1, keepdims = True) - 2 * T.dot(self.codebook, sample.T)
		bmu = T.argmin(diff)
		err = T.min(diff)
		return err, bmu

	def _update_map(self, sample, weight, winner):
		scale = T.maximum(self.shape -1, 1)
		dist = T.sqrt((T.sqr((self.grid - winner)/(scale)).sum(axis = 1, keepdims = True)/self.dim))
		gaussian = T.exp(- T.sqr(dist/self.sigma))
		return [[self.codebook, 
						sample + (self.codebook - sample) * (1 - gaussian * self.lrate) ** weight]]

	def _update_params(self):
		return [[self.lrate, self.lrate * self.lrate_factor], 
				[self.sigma, self.sigma * self.sigma_factor]]

	def train_theano(self, quiet = False, razor = True): 

		''' 
		A method that takes an np.array, scales it to have zero mean and unit standard deviation, and train the map on the data
		''' 
		# -----
		# Define symbolic variables and compile the functions
		# -----

		broadscalar = T.TensorType('float32', (True, True))
		s = T.frow('s')
		w = broadscalar('w')
		win = T.frow('win')

		match = function(
			inputs = [s], 
			outputs = self._match(s), # return err, bmu
			allow_input_downcast = True
			)

		update_map = function(
			inputs = [s, w, win], 
			outputs = [], 
			updates = self._update_map(s, w, win),
			allow_input_downcast = True
			)

		update_params = function(
			inputs = [],
			outputs = [],
			updates = self._update_params(),
			allow_input_downcast = True
			)

		# ----
		# Load training data
		# ----
		
		f = h5py.File(self.datafile, 'r')
		trndata = f[self.trn_route][()]
		weights = f[self.trn_route].attrs['weights']
		f.close()

		trndata = np.concatenate((trndata[:, 0:self.features], trndata[:, -1:]), axis = 1)

		self.means = np.mean(trndata[:, 0:self.features], axis = 0)
		self.stds = np.std(trndata[:, 0:self.features], axis = 0)
		trndata[:, 0:self.features] = (trndata[:, 0:self.features] - self.means)/self.stds + 0.5


		# ----- 
		# Training starts here 
		# ----- 

		samples = trndata.shape[0]
		self.fires = np.zeros((samples, 2)) # One for index of the neuron that fired and 

		print '-------------------------------------------------------'
		print 'Training starts....'
		print 'Number of samples:', samples, 'with weights:', weights
		print 'Epochs: {0:2d}, Sigma: ({1:4f}, {2:4f}), Lrate: ({3:4f}, {4:4f}), Threshold: {5:3d}'.format(self.epochs, self.sigma_i, self.sigma_f, self.lrate_i, self.lrate_f, self.threshold)

		total_time = 0

		for e in range(self.epochs):

			if not quiet:
				print '[ epoch: {0:5d}]'.format(e)
				print '		sigma:{0:5f}	lrate:{1:5f}'.format(self.sigma.get_value(), self.lrate.get_value())
			start = time.mktime(time.localtime())
			ordering = np.random.permutation(samples)
		
			for i in ordering:

				sample = trndata[i, 0:self.features][None, :]
				weight = np.array([[trndata[i, -1]]])
			
				error, unit = match(sample)
				
				if self.fires[i, 0] == unit:
					self.fires[i, 1] += 1
			
				else:
					self.fires[i, 0] = unit
					self.fires[i, 1] = 0
			
				if self.fires[i, 1] < self.threshold: 
					winner = self.grid.get_value()[unit][None, :]
					update_map(sample, weight, winner)
			
			update_params() 
			end = time.mktime(time.localtime())
			total_time += (end - start)
			if not quiet:
				print '		stable samples:{0:5d}'.format(np.sum(self.fires[:, 1] >= self.threshold))
				print '		time stamp:{0:5f}'.format(end - start)

		print 'average time per epoch: {0:5f}'.format(total_time/float(self.epochs))
		print 'average time per sample: {0:5f}'.format(total_time/(self.epochs * samples))
		print '-------------------------------------------------------'

	
	def test_theano(self, data = None, weights = None, quiet = False):
		'''
		data: np.array with the last column as what type of data it is 
		types: the number of types
		'''
		s = T.frow('s')
		match = function(
			inputs = [s], 
			outputs = self._match(s), # return err, bmu
			allow_input_downcast = True
			)

		# Rescale the test data the same way training data gets rescaled
		if isinstance(data, np.ndarray) and isinstance(weights, np.ndarray):
			tstdata = data 
			self.types = len(weights)
		else: 
			f = h5py.File(self.datafile, 'r')
			tstdata = f[self.tst_route][()]
			tstdata = np.concatenate((tstdata[:, 0:self.features], tstdata[:, -1:]), axis = 1)
			weights = f[self.tst_route].attrs['weights']		
			tstdata[:, 0:self.features] = (tstdata[:, 0:self.features] - self.means)/self.stds + 0.5
			self.types = len(weights)
			if not quiet:
				print 'testing on:', f[self.tst_route].attrs['samples'], 'with weights', weights
			f.close() 

		self.test = np.zeros((self.units, self.types))
		self.err = 0

		samples = tstdata.shape[0] 

		for i in range(samples):
			sample = tstdata[i, 0:self.features][None, :]
			index = tstdata[i, -1]

			err, unit = match(sample)
			self.test[unit, index] += weights[index]
			self.err += err

		self.err = self.err/float(self.units)
		if not quiet:
			print 'error:', self.err 

		return np.sum(self.test, axis = -1) # return the total number of hits in each cell

	def show_map(self):
		[x, y, z] = self.grid.get_value().T
		for t in range(self.types):
			fig = plt.figure()
			ax = fig.add_subplot(111, projection = '3d')
			ax.scatter(x, y, z, c = self.test[..., t], norm = LogNorm(), lw = 0)
			plt.show() 

	def save_map(self, resultspath): 
		'''
		A method that saves the map into hdf5 format 
		top group: 'map' 
		datasets: 'codebook' and 'grid' 
		attributes: information on the map and training params 
		'''
		mapfile = "map.hdf5"
		if not os.path.exists(resultspath):
			os.mkdir(resultspath)

		i = 1
		while os.path.exists(os.path.join(resultspath, mapfile)):
		    i += 1
		    mapfile = "%s_%d.hdf5" % ('map', i)
		print 'saving the map to', os.path.join(resultspath, mapfile)

		f = h5py.File(os.path.join(resultspath, mapfile))
		if 'map' in f.keys():
			del f['map']

		grp = f.create_group('map')	
		grp.create_dataset('codebook', data = self.codebook.get_value())
		grp.create_dataset('grid', data = self.grid.get_value())

		# Information on the map
		grp.attrs['shape'] = self.shape.get_value()
		grp.attrs['dim'] = self.dim.get_value()
		grp.attrs['features'] = self.features
		grp.attrs['means'] = self.means
		grp.attrs['stds'] = self.stds

		# Information on training
		grp.attrs['epochs'] = self.epochs
		grp.attrs['sigma'] = (self.sigma_i, self.sigma_f)
		grp.attrs['lrate'] = (self.lrate_i, self.lrate_f)
		grp.attrs['threshold'] = self.threshold
		f.close()

	def negloglikelihood(self, sig_cols):
		bkg_cols = np.array([(i not in sig_cols) for i in range(self.types)], dtype = bool)
		sig_cols = np.array([(i in sig_cols) for i in range(self.types)], dtype = bool)
		bkg = np.sum(self.test[:, bkg_cols], axis = -1)
		sig = np.sum(self.test[:, sig_cols], axis = -1)

		# Combine all the zero bins in bkg
		zeros = np.sum(bkg == 0)
		sig_temp = np.sum(sig[bkg == 0])
		sig = sig[bkg != 0]
		bkg = bkg[bkg != 0]
		sig[np.argmin(bkg)] += sig_temp

		self.llfunc = (lambda m, m0: 2 * np.sum((bkg + m0 * sig) - (bkg + m * sig) * np.log(bkg + m0 * sig)))
		return self.llfunc, zeros

	def exclusion(self, mu_range, n = 100, show = True):
		''' 
		a function that takes the outcome of the likelihood function and plot the likelihood within a certain range of mu
		mu_range: a tuple
		fun: a function
		''' 
		mu_i, mu_f = mu_range
		mu = np.linspace(mu_i, mu_f, n)
		zero = np.zeros(n)
		y = np.array(map(self.llfunc, zero, mu)) - self.llfunc(0, 0)
		ref = np.full(y.shape, 2 * 1.92)

		if show:
			plt.plot(mu, y)
			plt.plot(mu, ref)
			plt.title('exclusion plot')
			plt.xlabel('mu')
			plt.ylabel('- 2 * log(L(b | b + mu * s)/ L(b | b))')
			plt.show() 

		return mu[np.argmin(np.abs(y - ref))]

	def discovery(self, mu_range, n = 100, show = True):
		mu_i, mu_f = mu_range
		mu = np.linspace(mu_i, mu_f, n)
		zero = np.zeros(n)
		y = np.array(map(self.llfunc, mu, zero)) - np.array(map(self.llfunc, mu, mu))
		ref = np.full(y.shape, 25)

		if show:
			plt.plot(mu, y)
			plt.plot(mu, ref)
			plt.title('discovery plot')
			plt.xlabel('mu')
			plt.ylabel('- 2 * log(L(b + mu * s | b)/ L(b + mu * s | b + mu * s))')
			plt.show() 

		return mu[np.argmin(np.abs(y - ref))]







                 









	




