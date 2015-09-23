import numpy as np 
from SOM_theano import SOM 
import h5py
import os

if __name__ == '__main__':

	datapath = '../Data'

	cuts = [(0, 0, 0, 0.1)]
	labels = ['Dataset', 'all']

	for cut in cuts:
		for label in labels:

			sumcut = cut[0] 
			ptcut = cut[1] 
			mrcut = cut[2]
			r2cut = cut[3] 

			print 'sumcut:', sumcut, 'ptcut:', ptcut, 'mrcut:', mrcut, 'r2cut:', r2cut

			dataset = 'SOM%s_sumMET%d_metPt%d_MR%d_Rsq%d.hdf5'%(label, sumcut, ptcut, mrcut, r2cut * 100)
			print dataset
			mappath = os.path.join(datapath, 'SOM', dataset[0:-5])
			datafile = os.path.join(datapath, dataset)

			logfile = os.path.join(mappath, 'results.log')
			log = open(logfile, 'w')

			exclusion = [] 
			discovery = []

			for mapfile in [os.path.join(mappath, f) for f in os.listdir(mappath) if f.endswith('hdf5')]:
				print mapfile
				Map = SOM(datafile, shape = (10, 10, 10), filename = mapfile)
				log.write('Epochs: {0:2d}, Sigma: ({1:4f}, {2:4f}), Lrate: ({3:4f}, {4:4f}) \n'.format(Map.epochs, Map.sigma_i, Map.sigma_f, Map.lrate_i, Map.lrate_f))
				Map.test_theano(quiet = True)
				Map.show_map()
				_, zeros = Map.negloglikelihood([3])
				log.write('contains {0:2d} zeros \n'.format(zeros))
				exc = Map.exclusion((0, 2), n = 1000, show = True)
				dis = Map.discovery((0, 3), n = 1000, show = True)
				log.write('exclusion: {0:5f} \n'.format(exc))
				log.write('discovery: {0:5f} \n'.format(dis))
				print 'contains {0:2d} zeros'.format(zeros)
				print exc, dis
				exclusion.append(exc)
				discovery.append(dis)

			mean_exc = np.mean(exclusion)
			std_exc = np.std(exclusion)
			mean_dis = np.mean(discovery)
			std_dis = np.std(discovery)

			log.write('exclusion mean: {0:6f} std: {0:4f} \n'.format(mean_exc, std_exc))
			log.write('discovery mean: {0:6f} std: {0:4f} \n'.format(mean_dis, std_dis))

			log.close()

