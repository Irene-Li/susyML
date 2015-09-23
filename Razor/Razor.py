import os
import numpy as np
import h5py
import time
import sys

def exclusion(llfunc, mu_range, n = 100, show = True):
	''' 
	a function that takes the outcome of the likelihood function and plot the likelihood within a certain range of mu
	mu_range: a tuple
	fun: a function
	''' 
	mu_i, mu_f = mu_range
	mu = np.linspace(mu_i, mu_f, n)
	zero = np.zeros(n)
	y = np.array(map(llfunc, zero, mu)) -llfunc(0, 0)
	ref = np.zeros(y.shape, dtype = float)
	ref.fill(2 * 1.92)

	if show:
		from matplotlib import pyplot as plt
		plt.plot(mu, y)
		plt.plot(mu, ref)
		plt.title('exclusion plot')
		plt.xlabel('mu')
		plt.ylabel('- 2 * log(L(b | b + mu * s)/ L(b | b))')
		plt.show() 

	return mu[np.argmin(np.abs(y - ref))]

def discovery(llfunc, mu_range, n = 100, show = True):
	mu_i, mu_f = mu_range
	mu = np.linspace(mu_i, mu_f, n)
	zero = np.zeros(n)
	y = np.array(map(llfunc, mu, zero)) - np.array(map(llfunc, mu, mu))
	ref = np.zeros(y.shape, dtype = float)
	ref.fill(25)

	if show:
		from matplotlib import pyplot as plt
		plt.plot(mu, y)
		plt.plot(mu, ref)
		plt.title('discovery plot')
		plt.xlabel('mu')
		plt.ylabel('- 2 * log(L(b + mu * s | b)/ L(b + mu * s | b + mu * s))')
		plt.show() 

	return mu[np.argmin(np.abs(y - ref))]

def combineBins(hist, edges):
	index = hist.shape[0] - 1
	indices = np.invert(np.zeros(hist.shape, dtype = bool))
	while index >= 0:
	    if hist[index] == 0: 
	        hist[index-1] += hist[index] 
	        indices[index] = 0
	    index -= 1

	hist = hist[indices]  
	edges = edges[np.concatenate((indices, [True]))]

	return hist, edges

def between(a, b, array):
	temp1 = (array >= a)
	temp2 = (array < b)
	return temp1 & temp2


def negloglikelihood(samples, types, weights, sig_index = 3, mr_col = -2, r2_col = -3, mr_bins = 10, r2_bins = 10):
	'''
	A method that bins data in the MR-R2 plan, making sure there's no bin in which background is zero
	'''
	# Load the samples
	bkg = samples[samples[:, -1] != sig_index]
	sig = samples[samples[:, -1] == sig_index]

	# Binning in MR, making sure there are no empty bins
	l = np.min(bkg[:, mr_col])
	r = np.max(bkg[:, mr_col])
	mredges = np.arange(l, r, (r-l)/float(mr_bins))
	mredges = np.concatenate((mredges[0:-1], [np.max(samples[:, mr_col])]))

	mrhist, mredges = np.histogram(bkg[mr_col], bins = mredges)
	mrhist, mredges = combineBins(mrhist, mredges)

	# Binning in R2 on top of M2 binning, making sure there are no empty bins
	l = np.min(bkg[:, r2_col])
	r = np.max(bkg[:, r2_col])
	edges = np.arange(l, r, (r-l)/float(r2_bins))
	edges = np.concatenate((edges[0:-1], [np.max(samples[:, r2_col])]))

	r2edges = []
	bkg_hist = [] # a flatterned array of the counts in each bin due to background
	sig_hist = [] # a flatterned array of the counts in each bin due to signal

	for i in range(mredges.shape[0] - 1):
		ledge = mredges[i]
		redge = mredges[i + 1]

		# Get R2 and weights within the range of MR
		bkg_r2 = bkg[:, r2_col][between(ledge, redge, bkg[:, mr_col])]
		bkg_weights = [weights[x] for x in bkg[:, -1][between(ledge, redge, bkg[:, mr_col])]]
		sig_r2 = sig[:, r2_col][between(ledge, redge, sig[:, mr_col])]
		sig_weight = weights[sig_index]

		# Binning
		bhist, temp_edges = np.histogram(bkg_r2, bins = edges, weights = bkg_weights)
		bhist, temp_edges = combineBins(bhist, temp_edges)
		shist, temp_edges = np.histogram(sig_r2, bins = temp_edges)
		shist *= sig_weight

		# Append edges and hists to list
		r2edges.append(temp_edges)
		for bh in bhist:
			bkg_hist.append(bh)
		for sh in shist:
			sig_hist.append(sh)

	bkg_hist = np.array(bkg_hist)
	sig_hist = np.array(sig_hist)

	print 'total number of bins:', len(bkg_hist)
		
	return (lambda m, m0: 2 * np.sum((bkg_hist + m0 * sig_hist) - (bkg_hist + m * sig_hist) * np.log(bkg_hist + m0 * sig_hist)))

if __name__ == '__main__':
	mr_bins = int(sys.argv[1]) 
	r2_bins = int(sys.argv[2])


	cuts = [(0, 0, 0, 0.05), (0, 0, 0, 0.1), (0, 65, 0, 0), (0, 0, 0, 0)]
	labels = ['all', 'Dataset']
	path = '/afs/cern.ch/work/y/yuting/public/razor/'
	logfile = os.path.join(path, 'Razor/result.log')
	log = open(logfile, 'a')

	for cut in cuts:
		for label in labels:

			sumcut = cut[0] 
			ptcut = cut[1] 
			mrcut = cut[2]
			r2cut = cut[3] 
			print 'sumcut:', sumcut, 'ptcut:', ptcut, 'mrcut:', mrcut, 'r2cut:', r2cut
			log.write('sumcut: {0:3d}	ptcut: {1:3d}	mrcut: {2:3d}	r2cut: {3:4f} \n'.format(sumcut, ptcut, mrcut, r2cut))
			log.write('label: {0} \n'.format(label))
			log.write('mr_bins: {0:3d}	r2_bins: {0:3d} \n'.format(mr_bins, r2_bins))

			datafile = '/afs/cern.ch/work/y/yuting/public/razor/Data/SOM%s_sumMET%d_metPt%d_MR%d_Rsq%d.hdf5'%(label, sumcut, ptcut, mrcut, r2cut * 100)
			print datafile

			f = h5py.File(datafile, 'r')
			dataset = f['data/testing'][()]
			weights = f['data/testing'].attrs['weights']
			samples = f['data/testing'].attrs['samples']
			f.close()

			print 'samples:', samples 
			print 'weights:', weights
			types = len(samples)

			llfunc = negloglikelihood(dataset, types, weights, sig_index = 3, mr_col = -2, r2_col = -3, mr_bins = mr_bins, r2_bins = r2_bins)
			exc = exclusion(llfunc, (0, 5), n = 5000, show = False)
			dis = discovery(llfunc, (0, 5), n = 5000, show = False)
			print 'exclusion:', exc
			print 'discovery:', dis
			log.write('exclusion: {0:5f} \n'.format(exc))
			log.write('discovery: {0:5f} \n \n'.format(dis))

	log.close()


	'''
	sumcut = 0 
	ptcut = 0 
	mrcut = 0 
	r2cut = 0.10	
	50, 50 
	exclusion: 0.555111022204
	discovery: 1.39927985597

	sumcut = 0 
	ptcut = 65
	mrcut = 0 
	r2cut = 0
	50, 50
	exclusion: 0.714142828566
	discovery: 1.84636927385






	




	'''




















