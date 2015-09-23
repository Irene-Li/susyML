import os
import Instrumentation
import Backends
import Optimization
import TrainingController
import numpy as np
import Utils
import Data.utils
import scipy.stats
import gc
from Utils.DropoutMask import create_dropout_masks
from Utils.theano_helpers import floatX
import pprint
import NADE
import h5py
from matplotlib import pyplot as plt

def log_message(backends, message):
    for b in backends:
        b.write([], "", message)

class OrderlessNADE(object):
	'''
	A class that packs all the information about training 

	Class Variables:
		- params: a python dictionary with information on the model, training and dataset
			- parameters of the model: 
				- form: MoG, Bernoulli or QR 
				- n_quantiles
				- n_components
				- hlayers
				- units
				- nonlinearlity 
			- paramters on training: 
				- layerwise 
				- training_ll_stop
				- lr: learning rate 
				- decrease_constant
				- wd
				- momentum
				- epochs
				- pretraining_epochs
				- epoch_size
				- batch_size
				- show_training_stop
				- report_mixtures
				- summary_orderings
			- parameters on dataset: 
				- datasetspath: the path that contains the dataset file 
				- resultspath: the path where results are stored 
				- dataset: the hdf5 file containing the dataset 
				- training_route, validation_route, test_route: group names in the dataset hdf5 file that contains the corresponding dataset
				- samples_name: the dataset name in the group that contains the samples 
				- normalize: whether the dataset is to be normalized before training 
				- validation_loops: number of loops for validation 
				- no validation: whether validation happens
				- train_mean, train_std: mean and std of the training dataset
		- results: a python dictionairy with result files: 
			- training_log: training statistics 
			- result_file: the file containing the final model

	Class methods
		- __init__ (self, **kwargs): sets the parameters 
		- run(): run training and testing
	''' 
	def __init__(self, **kwargs):
		self.params = {
			# model paramters 
			'form': 'MoG',
			'n_quantiles': 20, 
			'n_components': 10, 
			'hlayers': 1, 
			'units': 100, 
			'nonlinearity':'RLU',

			# Training paramters 
			'layerwise': True,  
			'training_ll_stop': np.inf,  
			'lr': 0.02, 
			'decrease_constant': 0.1, 
			'wd': 0.02, 
			'momentum': 0.9, 
			'epochs': 20, 
			'pretraining_epochs': 5, 
			'epoch_size': 100, 
			'batch_size': 100,
			'show_training_stop': True, 
			'report_mixtures': True,
			'summary_orderings': 5,

			# Dataset parameters 
			'datasetspath': 'Data/NADE/Train/',
			'resultspath': 'Data/NADE/Result/',
			'dataset': '', 
			'training_route': 'train', 
			'validation_route': 'validation', 
			'test_route': 'test', 
			'samples_name': 'data', 
			'normalize': True, 
			'validation_loops': 20, 
			'no_validation': False, 
			'mean': 0, 
			'std': 0
		}

		for key in kwargs:
			if key in self.params:
				self.params[key] = kwargs[key]
			else: 
				print key, 'is not a valid option'
				print 'The valid options are:', self.params.keys()
				break

	def run(self, quiet = False, testing = True):
		# Set garbage collection threshold 
		gc.set_threshold(gc.get_threshold()[0]/5)

		# Create result directory if it doesnt already exist
		results_route = self.params['resultspath']
		if not os.path.isdir(results_route):
			os.makedirs(results_route)

		# Write all the parameters to file
		console = Backends.Console()
		textfile_log = Backends.TextFile(os.path.join(results_route, "NADE_training.log"))
		hdf5_backend = Backends.HDF5(results_route, "NADE")
		hdf5_backend.write([], "params", self.params)
		hdf5_backend.write([], "svn_revision", Utils.svn.svnversion())
		hdf5_backend.write([], "svn_status", Utils.svn.svnstatus())
		hdf5_backend.write([], "svn_diff", Utils.svn.svndiff())

		if quiet: 
			report = [textfile_log, hdf5_backend]
		else:
			report = [console, textfile_log, hdf5_backend]


		# Read datasets
		dataset_file = os.path.join(self.params['datasetspath'], self.params['dataset'])
		training_dataset = Data.BigDataset(dataset_file, self.params['training_route'], self.params['samples_name'])
		if not self.params['no_validation']:
		    validation_dataset = Data.BigDataset(dataset_file, self.params['validation_route'], self.params['samples_name'])
		test_dataset = Data.BigDataset(dataset_file, self.params['test_route'], self.params['samples_name'])
		n_visible = training_dataset.get_dimensionality(0)

		# Calculate normalsation constants
		mean, std = Data.utils.get_dataset_statistics(training_dataset)
		self.params['mean'] = mean.reshape(len(mean), 1)
		self.params['std'] = std.reshape(len(std), 1)

		if self.params['normalize']:
		    # Normalise all datasets
		    training_dataset = Data.utils.normalise_dataset(training_dataset, mean, std)
		    if not self.params['no_validation']:
		        validation_dataset = Data.utils.normalise_dataset(validation_dataset, mean, std)
		    test_dataset = Data.utils.normalise_dataset(test_dataset, mean, std)
		    hdf5_backend.write([], "normalisation/mean", mean)
		    hdf5_backend.write([], "normalisation/std", std)

		# Dataset of masks
		try:
		    masks_filename = self.params['dataset'] + "." + floatX + ".masks"
		    masks_route = os.path.join(self.params['datasetspath'], masks_filename)
		    masks_dataset = Data.BigDataset(masks_route + ".hdf5", "masks/.*", "masks")
		except:
		    create_dropout_masks(self.params['datasetspath'], masks_filename, n_visible, ks=1000)
		    masks_dataset = Data.BigDataset(masks_route + ".hdf5", "masks/.*", "masks")

		l = 1 if self.params['layerwise'] else self.params['hlayers']

		if self.params['form'] == "MoG":
		    nade_class = NADE.OrderlessMoGNADE
		    self.nade = nade_class(n_visible, self.params['units'], l, self.params['n_components'], nonlinearity= self.params['nonlinearity'])
		    loss_function = "sym_masked_neg_loglikelihood_gradient"
		    validation_loss_measurement = Instrumentation.Function("validation_loss", lambda ins:-ins.model.estimate_average_loglikelihood_for_dataset_using_masks(validation_dataset, masks_dataset, loops=self.params['validation_loops']))
		elif self.params['form'] == "Bernoulli":
		    nade_class = NADE.OrderlessBernoulliNADE
		    self.nade = nade_class(n_visible, self.params['units'], l, nonlinearity=self.params['nonlinearity'])
		    loss_function = "sym_masked_neg_loglikelihood_gradient"
		    validation_loss_measurement = Instrumentation.Function("validation_loss", lambda ins:-ins.model.estimate_average_loglikelihood_for_dataset_using_masks(validation_dataset, masks_dataset, loops = self.params['validation_loops']))
		elif self.params['form'] == "QR":
		    nade_class = NADE.OrderlessQRNADE
		    self.nade = nade_class(n_visible, self.params['units'], l, self.params['n_quantiles'], nonlinearity= self.params['nonlinearity'])
		    loss_function = "sym_masked_pinball_loss_gradient"
		    validation_loss_measurement = Instrumentation.Function("validation_loss", lambda ins: ins.model.estimate_average_pinball_loss_for_dataset(validation_dataset, masks_dataset, loops=self.params['validation_loops']))
		else:
		    raise Exception("Unknown form")

		if self.params['layerwise']:
		    # Pretrain layerwise
		    for l in xrange(1, self.params['hlayers'] + 1):
		        if l == 1:
		            self.nade.initialize_parameters_from_dataset(training_dataset)
		        else:
		            self.nade = nade_class.create_from_smaller_NADE(self.nade, add_n_hiddens=1)

		        # Configure training 
		        trainer = Optimization.MomentumSGD(self.nade, self.nade.__getattribute__(loss_function))
		        trainer.set_datasets([training_dataset, masks_dataset])
		        trainer.set_learning_rate(self.params['lr'])
		        trainer.set_datapoints_as_columns(True)
		        trainer.add_controller(TrainingController.AdaptiveLearningRate(lr, 0, epochs=self.params['pretraining_epochs']))
		        trainer.add_controller(TrainingController.MaxIterations(self.params['pretraining_epochs']))
		        trainer.add_controller(TrainingController.ConfigurationSchedule("momentum", [(2, 0), (float('inf'), self.params['momentum'])]))
		        trainer.set_updates_per_epoch(self.params['epoch_size'])
		        trainer.set_minibatch_size(self.params['batch_size'])
		        # trainer.set_weight_decay_rate(options.wd)
		        trainer.add_controller(TrainingController.NaNBreaker())
		        # Instrument the training
		        trainer.add_instrumentation(Instrumentation.Instrumentation(report,
		                                                                    Instrumentation.Function("training_loss", lambda ins: ins.get_training_loss())))
		        trainer.add_instrumentation(Instrumentation.Instrumentation(report, Instrumentation.Configuration()))
		        trainer.add_instrumentation(Instrumentation.Instrumentation(report, Instrumentation.Timestamp()))
		        # Train
		        trainer.set_context("pretraining_%d" % l)
		        trainer.train()

		else:  # No pretraining
		    self.nade.initialize_parameters_from_dataset(training_dataset)
		# Configure training
		ordering = range(n_visible)
		np.random.shuffle(ordering)
		trainer = Optimization.MomentumSGD(self.nade, self.nade.__getattribute__(loss_function))
		trainer.set_datasets([training_dataset, masks_dataset])
		trainer.set_learning_rate(self.params['lr'])
		trainer.set_datapoints_as_columns(True)
		trainer.add_controller(TrainingController.AdaptiveLearningRate(self.params['lr'], 0, epochs=self.params['epochs']))
		trainer.add_controller(TrainingController.MaxIterations(self.params['epochs']))
		if self.params['training_ll_stop'] < np.inf:
		    trainer.add_controller(TrainingController.TrainingErrorStop(-self.params['training_ll_stop']))  # Assumes that we're doing minimization so negative ll
		trainer.add_controller(TrainingController.ConfigurationSchedule("momentum", [(2, 0), (float('inf'), self.params['momentum'])]))
		trainer.set_updates_per_epoch(self.params['epoch_size'])
		trainer.set_minibatch_size(self.params['batch_size'])

		# trainer.set_weight_decay_rate(options.wd)
		trainer.add_controller(TrainingController.NaNBreaker())
		# Instrument the training
		trainer.add_instrumentation(Instrumentation.Instrumentation(report,
		                                                            Instrumentation.Function("training_loss", lambda ins: ins.get_training_loss())))
		if not self.params['no_validation']:
			temp = [textfile_log] 
			if not quiet:
				temp.append(console)
			trainer.add_instrumentation(Instrumentation.Instrumentation(temp,
			                                                            validation_loss_measurement))
			trainer.add_instrumentation(Instrumentation.Instrumentation([hdf5_backend], 
			                                                            validation_loss_measurement, at_lowest=[Instrumentation.Parameters()]))
		trainer.add_instrumentation(Instrumentation.Instrumentation(report, Instrumentation.Configuration()))
		# trainer.add_instrumentation(Instrumentation.Instrumentation([hdf5_backend], Instrumentation.Parameters(), every = 10))
		trainer.add_instrumentation(Instrumentation.Instrumentation(report, Instrumentation.Timestamp()))
		# Train
		trainer.train()

		#------------------------------------------------------------------------------
		# Report some final performance measurements
		flag = False # a flag that indicates whether training is successful
		if trainer.was_successful():
			flag = True
			np.random.seed(8341)
			hdf5_backend.write(["final_model"], "parameters", self.nade.get_parameters())
			if not self.params['no_validation']:
			    self.nade.set_parameters(hdf5_backend.read("/lowest_validation_loss/parameters"))
			config = {"wd": self.params['wd'], "units": self.params['units'], "lr": self.params['lr'], "n_quantiles": self.params['n_quantiles']}
			log_message([console, textfile_log], "Config %s" % str(config))
			if self.params['show_training_stop']:
			    training_likelihood = self.nade.estimate_loglikelihood_for_dataset(training_dataset)
			    log_message([console, textfile_log], "Training average loss\t%f" % training_likelihood)
			    hdf5_backend.write([], "training_loss", training_likelihood)

			if testing:
				val_ests = []
				test_ests = []
				for i in xrange(self.params['summary_orderings']):
				    self.nade.setup_n_orderings(n=1)
				    if not self.params['no_validation']:
				        val_ests.append(self.nade.estimate_loglikelihood_for_dataset(validation_dataset))
				    test_ests.append(self.nade.estimate_loglikelihood_for_dataset(test_dataset))
				if not self.params['no_validation']:
				    val_avgs = map(lambda x: x.estimation, val_ests)
				    val_mean, val_se = (np.mean(val_avgs), scipy.stats.sem(val_avgs))
				    log_message([console, textfile_log], "*Validation mean\t%f \t(se: %f)" % (val_mean, val_se))
				    hdf5_backend.write([], "validation_likelihood", val_mean)
				    hdf5_backend.write([], "validation_likelihood_se", val_se)
				    for i, est in enumerate(val_ests):
				        log_message([console, textfile_log], "Validation detail #%d mean\t%f \t(se: %f)" % (i + 1, est.estimation, est.se))
				        hdf5_backend.write(["results", "orderings", str(i + 1)], "validation_likelihood", est.estimation)
				        hdf5_backend.write(["results", "orderings", str(i + 1)], "validation_likelihood_se", est.se)
				test_avgs = map(lambda x: x.estimation, test_ests)
				test_mean, test_se = (np.mean(test_avgs), scipy.stats.sem(test_avgs))
				log_message([console, textfile_log], "*Test mean\t%f \t(se: %f)" % (test_mean, test_se))
				hdf5_backend.write([], "test_likelihood", test_mean)
				hdf5_backend.write([], "test_likelihood_se", test_se)
				for i, est in enumerate(test_ests):
				    log_message([console, textfile_log], "Test detail #%d mean\t%f \t(se: %f)" % (i + 1, est.estimation, est.se))
				    hdf5_backend.write(["results", "orderings", str(i + 1)], "test_likelihood", est.estimation)
				    hdf5_backend.write(["results", "orderings", str(i + 1)], "test_likelihood_se", est.se)
				hdf5_backend.write([], "final_score", test_mean)

				# Report results on ensembles of NADES
				if self.params['report_mixtures']:
				    # #
				    for components in [2, 4, 8, 16, 32]:
				        self.nade.setup_n_orderings(n=components)
				        est = self.nade.estimate_loglikelihood_for_dataset(test_dataset)
				        log_message([console, textfile_log], "Test ll mixture of nades %d components: mean\t%f \t(se: %f)" % (components, est.estimation, est.se))
				        hdf5_backend.write(["results", "mixtures", str(components)], "test_likelihood", est.estimation)
				        hdf5_backend.write(["results", "mixtures", str(components)], "test_likelihood_se", est.se)

		# Set the training log file and the hdf5 file 
		self.results = {
			'training_log': textfile_log.filename, 
			'result_file': os.path.join(hdf5_backend.path, hdf5_backend.filename),
			'successful': flag 
		}

		print self.results

	def test(self, mixture = 1):
		dataset_file = os.path.join(self.params['datasetspath'], self.params['dataset'])
		test_dataset = Data.BigDataset(dataset_file, self.params['test_route'], self.params['samples_name'])
		test_dataset = Data.utils.normalise_dataset(test_dataset, self.params['mean'].T, self.params['std'].T)
		self.nade.setup_n_orderings(n=mixture)
		est = self.nade.estimate_loglikelihood_for_dataset(test_dataset)
		return est.estimation, est.se

	def create(self, filename):
		'''
		create a nade from file for the purpose of testing, assign values to self.nade, self.mean and self.rmss
		'''
		if not os.path.isfile(filename):
		    print 'not a file'
		else: 
		    f=h5py.File(filename, 'r')

		    if f.get('lowest_validation_loss/parameters'):
		    	p5 = f['lowest_validation_loss/parameters']
		    else:
		    	p5 = f["final_model/parameters"]

		    if f.get('normalisation/mean'):
		        mean = f['normalisation/mean'][()]
		        self.params['mean'] = mean.reshape(len(mean), 1)
		    if f.get('normalisation/std'):
		        std = f['normalisation/std'][()]
		        self.params['std'] = std.reshape(len(std), 1)

		    p={}
		    for k in p5.keys():
		        p[k] = p5.get(k).value

		    clsn = p.get("__class__")
		    cls = getattr(NADE,clsn)
		    self.nade = cls.create_from_params(p)

	def test_logdensity(self, tstdata = None, bins = 10, show = False, mixture = 1):
		if not isinstance(tstdata, np.ndarray):
			dataset_file = os.path.join(self.params['datasetspath'], self.params['dataset'])
			f = h5py.File(dataset_file, 'r')
			tstdata = f[self.params['test_route']][self.params['samples_name']][()]
			f.close()
		tstdata = (tstdata.T - self.params['mean'])/self.params['std']
		self.nade.setup_n_orderings(n = mixture)
		logdensity = self.nade.logdensity(tstdata)

		print np.sum(np.isnan(logdensity))

		if show: 
			plt.hist(logdensity, bins = bins, alpha = 0.5)	
			plt.show() 

		return logdensity

	def likelihood_test(self, xSections, luminosity, mu_range, bins = 10, mixture = 1, sig_route = 'sig', show = False): # require xSections and luminosity to be in the same unit
		dataset_file = os.path.join(self.params['datasetspath'], self.params['dataset'])
		f = h5py.File(dataset_file, 'r')
		bkg_tst = f[self.params['test_route']][self.params['samples_name']][()]
		bkg_cv = f[self.params['validation_route']][self.params['samples_name']][()]
		bkg = np.append(bkg_tst, bkg_cv, axis = 0)
		sig = f[sig_route][self.params['samples_name']][()]
		eff1 = f.attrs['eff1']
		eff2 = f.attrs['eff2']
		f.close()

		n = luminosity * xSections * eff1 * eff2
		bkg_weight = np.sum(n[0:-1])/bkg.shape[0]
		sig_weight = n[-1]/sig.shape[0]

		self.negloglikelihood(bkg, sig, bins = bins, mixture = mixture, bkg_weight = bkg_weight, sig_weight = sig_weight, show = False)
		return self.exclusion(mu_range, n = 1000, show = show), self.discovery(mu_range, n = 1000, show = show)


	def negloglikelihood(self, bkg, sig, bins = 10, mixture = 1, bkg_weight = 1, sig_weight = 1, show = False):
		'''
		This returns - 2 * log[ L(b + m2 * s | b + m1 * s) ]
		'''
	
		mean = self.params['mean']
		std = self.params['std']

		# Rescale data
		bkg = ((bkg.T - mean)/std)
		sig = ((sig.T - mean)/std)


		# Estimate log density
		self.nade.setup_n_orderings(n = mixture)
		bkg_logdensity = self.nade.logdensity(bkg)
		sig_logdensity = self.nade.logdensity(sig)

		sig_n0 = len(sig_logdensity)
		bkg_n0 = len(bkg_logdensity)
		sig_logdensity = sig_logdensity[np.invert(np.isnan(sig_logdensity))]
		bkg_logdensity = bkg_logdensity[np.invert(np.isnan(bkg_logdensity))]
		sig_n1 = len(sig_logdensity)
		bkg_n2 = len(bkg_logdensity)
		sig_ratio = sig_n0/float(sig_n1)
		bkg_ratio = bkg_n0/float(bkg_n2)

		sig_weight *= sig_ratio
		bkg_weight *= bkg_ratio


		l = np.min(bkg_logdensity)
		r = np.max(bkg_logdensity)
		edges = np.arange(l, r, (r-l)/float(bins))
		edges = np.concatenate(([min(np.min(sig_logdensity), np.min(bkg_logdensity))], edges[1:]))

		b, edges = np.histogram(bkg_logdensity, bins = edges)
		index = 0
		indices = np.full(b.shape, 1, dtype = bool)
		while index < len(b):
		    if b[index] == 0: 
		        b[index+1] += b[index] 
		        indices[index] = 0
		    index += 1
		    
		b = b[indices]  
		edges = edges[np.concatenate(([True], indices))]
		    
		s, edges = np.histogram(sig_logdensity, bins = edges)
		b = b * bkg_weight
		s = s * sig_weight

		if show:
		    lefts = edges[0:-1] 
		    rights = edges[1:]
		    widths = rights - lefts 
		    
		    plt.bar(lefts, b/widths, widths, alpha = 0.4, color = 'blue', label = 'background', edgecolor = 'none', linewidth = 0)
		    plt.bar(lefts, s/widths, widths, alpha = 0.4, color = 'red', label = 'signal', edgecolor = 'none', linewidth = 0)
		    plt.legend()
		    #plt.yscale('log')
		    plt.title('Log Density Histogram')
		    plt.xlabel('log density')
		    plt.ylabel('number of data points/width')
		    plt.show() 

		self.llfunc = (lambda m, m0: 2 * np.sum((b + m0 * s) - (b + m * s) * np.log(b + m0 * s)))
		return self.llfunc, np.sum(bkg == 0)

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

# A method that does the transform
def transform(data, funcs):
		assert data.shape[1] == len(funcs)
		for i in range(data.shape[1]):
			data[:, i] = funcs[i](data[:, i])
		return data 


if __name__ == '__main__':

	xSections = np.array([117267, 100, 500, 0.014])
	Lum = 10 * 1000
	trials = 100

	cuts = [(0, 0, 0, 0.05)]

	lrates = [0.1]
	epochs = [100]
	hlayers = [4, 6]
	n_components = [10]


	for cut in cuts:

		sumcut = cut[0] 
		ptcut = cut[1] 
		mrcut = cut[2]
		r2cut = cut[3] 

		print 'sumcut:', sumcut, 'ptcut:', ptcut, 'mrcut:', mrcut, 'r2cut:', r2cut

		dataset = 'NADEdataset_sumMET%d_metPt%d_MR%d_Rsq%d.hdf5'%(sumcut, ptcut, mrcut, r2cut * 100)
		print dataset
		datasetspath = '/afs/cern.ch/work/y/yuting/public/razor/Data'
		resultspath = os.path.join('/afs/cern.ch/work/y/yuting/public/razor/Data/NADE', dataset[0:-5])

		trained = 0

		count = 0
		for lr in lrates:
			for ep in epochs:
				for hl in hlayers:
					for cp in n_components:
						if count < trained:
							count += 1
							continue
						log = open(os.path.join(resultspath, 'likelihood.log'), 'a')
						log.write('lrate: %f epochs: %d hlayers: %d n_components: %d \n'%(lr, ep, hl, cp))
						print 'lrate:', lr, 'epochs:', ep, 'hlayers:', hl, 'n_components:', cp
						success = False
						while not success:
							myNADE = OrderlessNADE(lr = lr, epochs = ep, hlayers = hl, n_components = cp, datasetspath = datasetspath, resultspath = resultspath, dataset = dataset)
							myNADE.run(testing = False, quiet = True)
							if myNADE.results['successful']:
								success = True
								log.write('result_file:%s\n'%(myNADE.results['result_file']))
								log.write('result_log:%s\n'%(myNADE.results['training_log']))
								mean, se = myNADE.test(mixture = 8)
								log.write('test_likelihood: {0:8f}             se: {1:3f} \n'.format(mean, se))
								min_exc = 10
								min_dis = 10
								for i in range(trials):
									try:
										exc, dis= myNADE.likelihood_test(xSections, Lum, (0, 5), bins = 25, mixture = 1, sig_route = 'sig')
									except ZeroDivisionError:
										log.write('Log Density of Signal are all nan \n')
									else:
										if exc < min_exc and isinstance(exc, float):
											min_exc = exc
											min_dis = dis
											log.write('trial No. %d: %f, %f \n'%(i, exc, dis))
								log.write('minimum exclusion: %f \n'%(min_exc))
								log.write('minimum discovery: %f \n \n'%(min_dis))
								print min_exc, min_dis
						log.close()


	'''
	# Calculate weights
	xSections = np.array([117267, 100, 500, 0.014])
	n = 10 * 1000 * xSections * eff1 * eff2
	print 'number expected:', n
	bkg_weight = np.sum(n[0:-1])/bkg_n 
	sig_weight = n[-1]/sig_n
	print 'bkg_weight:', bkg_weight 
	print 'sig_weight:', sig_weight 
	print 'bkg_n:', bkg_n 
	print 'sig_n:', sig_n
	
	# Create NADE and do the loglikelihood test
	filename = '../Data/NADE/NADEdataset_sumMET%d_metPt%d_MR%d_Rsq%d/NADE_3.hdf5'%(sumcut, ptcut, mrcut, r2cut * 100)
	myNADE = OrderlessNADE() 
	myNADE.create(filename)
	# myNADE.test_logdensity(sig, show = True, mixture = 1)
	threshold = 1.8

	for i in range(1000):
		myNADE.negloglikelihood(bkg, sig, bins = 25, mixture = 1, bkg_weight = bkg_weight, sig_weight = sig_weight, show = False)
		temp = myNADE.negloglikelihood_plot((0.5, 4), n = 1000, show = False) 
		if temp < threshold:
			print temp, myNADE.nade.orderings
	'''

'''
sumcut = 0 
ptcut = 0 
mrcut = 0 
r2cut = 0.10

NADE_4.hdf5

/NADE_5.hdf5


#---------------------------------------------

sumcut = 0 
ptcut = 0 
mrcut = 0 
r2cut = 0.05

for NADE_2.hdf5

#---------------------------------------------
sumcut = 0 
ptcut = 0 
mrcut = 0 
r2cut = 0

for NADE.hdf5

#---------------------------------------------
sumcut = 0 
ptcut = 65 
mrcut = 0 
r2cut = 0

for NADE.hdf5

'''




















			


