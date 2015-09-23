import numpy as np 
from SOM_theano import SOM 
import h5py
import os

if __name__ == '__main__':

	# Specify the training parameters
	epochs = [30]
	sigma = [(3.0, 0.0005)]
	lrate = [(0.2, 0.0001)]
	threshold = 5
	trials = 5

	datapath = '../Data'
	dataset = 'SOMDataset_sumMET0_metPt65_MR0_Rsq0.hdf5'

	datafile = os.path.join(datapath, dataset)
	resultspath = os.path.join(datapath, 'SOM', dataset[0:-5])
	if not os.path.exists(resultspath):
		os.mkdir(resultspath)

	logfile = os.path.join(resultspath, 'training.log')
	log = open(logfile, 'a')
	log.write('---- Training with no razor variables ---- \n')

	for e in epochs:
		for s in sigma:
			for l in lrate:
				for i in range(trials):
					log.write('Epochs: {0:2d}, Sigma: ({1:4f}, {2:4f}), Lrate: ({3:4f}, {4:4f}) \n'.format(e, s[0], s[1], l[0], l[1]))
					Map = SOM(datafile, razor = False, shape = (10, 10, 10))
					Map.set_params(epochs = e, lrate = l, sigma = s, threshold = threshold)
					Map.train_theano(quiet = True)
					Map.test_theano(quiet = True)
					# Map.show_map()
					_, zeros = Map.negloglikelihood([3]) 
					exc = Map.exclusion((0, 5), n = 1000, show = False)
					dis = Map.discovery((0, 5), n = 1000, show = False)
					log.write('contains {0:2d} zeros \n'.format(zeros))						
					log.write('exclusion: {0:5f} \n'.format(exc))
					log.write('discovery: {0:5f} \n \n'.format(dis))
					print 'contains {0:2d} zeros'.format(zeros)
					print 'exclusion: {0:5f} '.format(exc)
					print 'discovery: {0:5f} '.format(dis)
					# Map.save_map(resultspath)
	log.close()
	

