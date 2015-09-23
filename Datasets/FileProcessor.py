import ROOT 
import numpy as np
import h5py 
import os
import time
from multiprocessing import Pool

# --------------------------------------------
# Functions to convert root files to hdf5 
# --------------------------------------------
 
def getMuons(tree, index, n, output):
	'''
	muonIsLoose == True
	muonPt < 2000 for event selection  
	tree: the root tree 
	index: the index in the event array where the muon variables go 
	n: number of muons 
	output: the event array
	'''
	Pt = tree.muonPt
	IsLoose = tree.muonIsLoose
	vector_pos = 0
	output_pos = index
	while vector_pos < len(Pt) and output_pos < index + n:
		if Pt[vector_pos] > 2000:
			return None, 0
		if IsLoose[vector_pos] == True:
			output[output_pos] = Pt[vector_pos]
			vector_pos += 1
		output_pos += 1
	return output, index + n

def getElectrons(tree, index, n, output):
	'''
	Get everything
	'''
	Pt = tree.elePt
	vector_pos = 0
	while vector_pos < len(Pt) and vector_pos < n:
	    output[index + vector_pos] = Pt[vector_pos]
	    vector_pos += 1
	return output, index + n

def makeVector(pt, eta, phi, e):
    # return a vector containing momention in xyz directions, energy and jet Pt
    pt = np.abs(pt)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return [px, py, pz, e]

def computeMsqr(vector):
    # Compute mass squared 
    e = vector[-1] 
    p = vector[0:-1]
    msqr = e ** 2 - np.sum(p ** 2)
    return msqr 

def getHemispheres(pt, eta, phi, e):

    # Make vectors
    vectorJets = np.array(map(makeVector, pt, eta, phi, e))
    nJets = len(pt)

    # Iterate through all the combinations and store the ones with the smallest combined mass 
    nComb = 2 ** nJets 
    megaJet1 = np.zeros(4)
    megaJet2 = np.zeros(4) 
    msqr= -1 

    for i in range(1, nComb - 1): # to get rid of 00000 and 11111

        comb = np.array([int(s) for s in np.binary_repr(i, width = nJets)], dtype = np.bool) # convert from int to a boolean array
        tempJet1 = np.sum(vectorJets[comb], axis = 0) # get the four vector of one mega jet
        tempJet2 = np.sum(vectorJets[np.invert(comb)], axis = 0) # get the four vector of the other mega jet
        tempMsqr= computeMsqr(tempJet1) + computeMsqr(tempJet2)

        if tempMsqr < msqr or msqr < 0:
            msqr = tempMsqr
            megaJet1 = tempJet1 
            megaJet2 = tempJet2

    return megaJet1, megaJet2 

def computeRazors(jetpt, jeteta, jetphi, jete, metpt, metphi):
    '''
    A method that computes the razor variables given the relevant event statistics
    '''
    Hem1, Hem2 = getHemispheres(jetpt, jeteta, jetphi, jete)

    # Compute MR
    Hem1mag = np.sqrt(np.sum(Hem1[0:3] ** 2))
    Hem2mag = np.sqrt(np.sum(Hem2[0:3] ** 2))
    mR = np.sqrt((Hem1mag + Hem2mag) ** 2 - (Hem1[2] + Hem2[2]) ** 2)
    if mR == 0:
        print 'mR = 0 error'
        print 'megajet1:', Hem1 
        print 'megajet2:', Hem2

    # Compute Rsq 
    metpx = metpt * np.cos(metphi)
    metpy = metpt * np.sin(metphi)
    Hem1pt = np.sqrt(np.sum(Hem1[0:2] ** 2))
    Hem2pt = np.sqrt(np.sum(Hem2[0:2] ** 2))
    term1 = metpt/ 2 * (Hem1pt + Hem2pt)
    term2 = metpx/ 2 * (Hem1[0] + Hem2[0]) + metpy/2 * (Hem1[1] + Hem2[1])
    mTR = np.sqrt(term1 - term2)
    Rsq = (mTR / mR) ** 2

    return mR, Rsq

def getJets(tree, index, n, output):
    '''
    2 Jets that pass the cut
    Pt > 40 
    abs(Eta) < 2.4 
    Compute Rsq and MR for selected events
    '''

    # Convert all the array variables to numpy arrays
    jetPt = np.array(tree.jetPt)
    jetEta = np.array(tree.jetEta)
    jetPhi = np.array(tree.jetPhi)
    jetE = np.array(tree.jetE)
    jetMass = np.array(tree.jetMass)
    metPt = tree.metPt
    metPhi = tree.metPhi
    
    # Get the boolean array of all the events that satify the cut
    cut = (np.abs(jetEta) < 2.4) & (jetPt > 40)
    jetNum = min(np.sum(cut), n)

    # Discard if there are fewer than two jets that pass the cut
    if jetNum < 2: 
        return None, 0

    # Compute the razor variables and write into the event array
    else:
        mR, Rsq = computeRazors(jetPt[cut], jetEta[cut], jetPhi[cut], jetE[cut], metPt, metPhi)
        output[index:index + jetNum] = jetPt[cut][0:jetNum]
        output[index + n: index + n + jetNum] = jetEta[cut][0:jetNum]
        output[index + 2 * n: index + 2 * n + jetNum] = jetMass[cut][0:jetNum]
        output[-1] = mR
        output[-2] = Rsq
        return output, index + 3 * n

def getMET(tree, index, output):
    output[index] = tree.sumMET
    output[index + 1] = tree.metPt
    return output, index + 2

def tohdf5((filename, leaves)):
	'''
	Input arguments: 
		- filename: the file to be procssed 
		- leaves: number of jets, electrons and muons

	Require:
		- 2 Jets that pass the cut (Pt > 40, abs(Eta) < 2.4) for event selection
			- muonIsLoose == True for muon selection
		- muonPt < 2000 for event selection  
	'''

	# Get the root file
	print filename
	f = ROOT.TFile.Open(filename)
	tree = f.Get('ntuples/RazorEvents')
	samples = tree.GetEntries()
	print 'entries:',  samples 

	# Load the tree into numpy array
	length = leaves[0]*3 + leaves[1] + leaves[2] + 4 # length of an event, 2 for MET, 2 for razor variables
	data = np.zeros((samples, length))
	count = 0
	for i in range(samples):

		# Get the node and initialise the event array
		tree.GetEntry(i)
		event = np.zeros(length)
		index = 0

		# Get the jets
		nJets = leaves[0]
		event, index = getJets(tree, index, nJets, event)
		if event == None: 
		    continue 

		# Get the muons
		nMuons = leaves[1]
		event, index = getMuons(tree, index, nMuons, event)
		if event == None:
		    continue 

		# Get the leaves 
		nEles = leaves[2]
		event, index = getElectrons(tree, index, nEles, event)

		event, index = getMET(tree, index, event)

		data[count] = event
		count += 1

	print 'count:', count 
	eff = count/float(samples)
	print 'eff:', eff
	data = data[0:count]

	# Pack into hdf5 
	f = h5py.File(filename[0:-4] + 'hdf5', 'a')
	if 'ntuples' in f.keys():
	    del f['ntuples']
	dset = f.create_dataset('ntuples', data = data)
	dset.attrs['eff'] = eff
	f.close() 


class FileProcessor(object):
	''' 
	A processor that processes all the files in a directory and convert them to a single hdf5 file

	Class Variables: 
		- dirs: a list of directories to be processed
	'''
	def __init__(self, dirs): 
		'''
		Initialise with the target directories
		''' 
		self.dirs = dirs


	def processDir(self, indir, leaves, filename = False, overwrite = False, parallel = False): 
		'''
		a method that processes a directory or a specific file in the directory according to the defined tohdf5 function
		leaves: the number of jets, muons and electrons
		'''

		# A function that returns True if a file has been processed
		def converted(f): 
			return os.path.isfile(f[0:-4] + 'hdf5')

		# A function that returns a list of root files to be converted
		def convert(f, overwrite):
			return f.endswith('.root') and (overwrite or not converted(f))

		# Make a list of all the inputs 
		if filename:
			if convert(filename, overwrite):
			    inputs = [(filename, leaves)] 
		else:
			inputs = [(os.path.join(indir, f), leaves) for f in os.listdir(indir) if convert(os.path.join(indir, f), overwrite)]

		# Process the files
		start = time.mktime(time.localtime())
		if parallel: 
		    # Batch process the files
		    batchsize = 8 # Magical number that's apparently good on lxplus
		    pool = Pool(batchsize)
		    pool.map(tohdf5, inputs) 
		else: 
		    for t in inputs:
				tohdf5(t)   

		end = time.mktime(time.localtime())
		print 'processing time:', end - start

   	# ------------------------------------------------------------------
   	# Functions that combine the hdf5 files in a directory into one file
   	# ------------------------------------------------------------------


	def combinehdf5(self, path, directory):
		'''
		Combine all the hdf5 files in the directory into one file
		'''
		files = [os.path.join(path, directory, f) for f in os.listdir(os.path.join(path, directory)) if f.endswith(".hdf5")]

		if len(files) == 0:
		    print 'No hdf5 files in directory', directory
	   	else: 
			# count the total number of samples and calculate average efficiency
			samples = 0
			total = 0
			features = -1

			for filename in files: 
				f = h5py.File(filename, 'r')
				assert (features == -1 or features == f['ntuples'].shape[1])
				inc = f['ntuples'].shape[0] 
				eff = f['ntuples'].attrs['eff']
				features = f['ntuples'].shape[1]
				samples += inc # add the number of samples to the count of samples
				total += inc/eff # add the number of total to the count of total
				f.close()

			eff_total = samples/total
			print 'number of samples:', samples 
			print 'number of total samples:', total 
			print 'efficiency:', eff_total

			# load the samples into one hdf5 file 
			data = np.zeros((samples, features))
			first = 0
			for filename in files: 
				f = h5py.File(filename, 'r')
				count = f["ntuples"].shape[0] 
				data[first:first + count, :] = f["ntuples"][()]
				first += count
				f.close() 

			# Pack data into a master hdf5 file 
			f = h5py.File(os.path.join(path, directory) + '.hdf5', 'w')
			dset = f.create_dataset('data', data = data)
			dset.attrs['samples'] = samples 
			dset.attrs['nfeatures'] = features
			dset.attrs['eff'] = eff_total
			f.close() 

    # -----------------------------------------------------------------
    # A functin that processes all the files in the list of directories
    # -----------------------------------------------------------------

	def process(self, leaves, overwrite = False, parallel = False):

		for indir in self.dirs:
			assert not indir.endswith('/')
			(path, tail) = os.path.split(indir)

			# convert all root files to hdf5 
			print 'start processing all the files in', indir
			self.processDir(indir, leaves, overwrite = overwrite, parallel = parallel)
			print 'The directory has been processed'

			print 'Start combining the hdf5 files'
			self.combinehdf5(path, tail)
			print 'The files have been combined'

if __name__ == '__main__':
	path = '/afs/cern.ch/work/y/yuting/public/razor/'

	dirs = ['old/TTJets_MSDecaysCKM_central_Tune4C_13TeV-madgraph-tauola', 'old/SMS-T1bbbb_2J_mGl-1500_mLSP-100_Tune4C_13TeV-madgraph-tauola']
	dirs = [path + d for d in dirs] 

	leaves = [3, 2, 2]

	processor = FileProcessor(dirs)
	processor.process(leaves, overwrite = True, parallel = True)





























