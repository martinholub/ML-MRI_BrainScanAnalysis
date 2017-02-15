# -*- coding: utf-8 -*-
import logging

def initializeLogging():
	logging.shutdown()	# This is to clear possible previous mess
	# We log to file and to script
	logging.basicConfig(format='%(asctime)s:%(name)s:%(levelname)s:	 %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG, handlers=[logging.FileHandler("ML_SVC_multijob.log"), logging.StreamHandler()])
	# Here we start logging, we use the same file as before
	logging.info('--------Starting a fresh run-----------')

initializeLogging()

logging.info('Importing modules')
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import splitext
import time

import nibabel as nib
from nilearn.image import crop_img
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MultiLabelBinarizer
from sklearn.utils import shuffle
# from seaborn import kdeplot
# from nilearn.masking import compute_epi_mask
# from nilearn.regions import RegionExtractor
# from skimage.morphology import opening, closing, dilation

from sklearn.model_selection import KFold, ParameterGrid, GridSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import hamming_loss
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

logging.info('Modules imported')

def defineVariables():
	TRAIN_DIR = './dataFull~/set_train/'
	TARGET_DIR = './dataFull~/targets.csv'
	TEST_DIR = './dataFull~/set_test/'

	## These are tied together
	MODE = 1

	ZOOM = 'N' #Set to 'N' otherwise
	binsList = [5,20,35,50,65,80] #You must load new data (ie MODE=0) when you want different bins!
	# binsList = [5,20]
	
	n_splits = 3 # Same for splits
	rnd_state = np.random.randint(1, 256**2)

	numRuns = 10  # For statistics
	TIME = time.localtime(time.time())

	# Mode 0 ~ Load everything from scratch, expensive
	# Mode 1 ~ Load full saved train and test arrays, cheap when zoomed, otherwise expensive
		
	return TRAIN_DIR, TARGET_DIR, TEST_DIR, MODE, ZOOM, binsList, n_splits, rnd_state, numRuns, TIME 

TRAIN_DIR, TARGET_DIR, TEST_DIR, MODE, ZOOM, binsList, n_splits, rnd_state, numRuns, TIME = defineVariables()

def Debug():
	raise SystemExit('Terminated for purpose of debugging')
	return
	
def filename_timestamp(name):
	return name + "_%02d%02d%02d%02d%02d" % (TIME.tm_mon, TIME.tm_mday, TIME.tm_hour, TIME.tm_min, TIME.tm_sec)

def crop(img, ZOOM = 'N'):
	img = crop_img(img)	 # crop it
	img = img.get_data()[:, :, :, 0] # make it array	
	# This is only to make the code exectuable on local machine.
	if ZOOM == 'Y':
		img = ndimage.zoom(img, 0.25, order=1, mode='reflect', prefilter=False)
		logging.info('Downsampling ON')
	else:
		pass
	return img

def cube_split(img, n_splits):
	# We want to create n_splits^n_splits cubes that are all the same size.
	# Since every dimension may not be divisible by n_splits, we want to ceil
	# the values, so that only the last cube is smaller.
	d1, d2, d3 = img.shape
	s1 = np.ceil(d1 / n_splits)
	s2 = np.ceil(d2 / n_splits)
	s3 = np.ceil(d3 / n_splits)
	cubes = [[[img[s1 * i: min(s1 * (i + 1), d1), s2 * j: min(s2 * (j + 1), d2), s3 * k: min(s3 * (k + 1), d3)]
			   for k in range(n_splits)] for j in range(n_splits)] for i in range(n_splits)]
	print(np.asarray(cubes).shape)
	return np.asarray(cubes)

def slice_plot(img, index=""):
	def show_slices(slices):
		fig, axes = plt.subplots(1, len(slices))
		for i, slice in enumerate(slices):
			axes[i].imshow(slice.T, cmap="gray", origin="lower")
			
	d1, d2, d3 = img.shape
	slice_0 = img[d1 / 2, :, :]
	slice_1 = img[:, d2 / 2, :]
	slice_2 = img[:, :, d3 / 2]

	# show_slices([slice_0, slice_1, slice_2])
	# plt.suptitle("center slices for EPI image")
	# plt.savefig(filename_timestamp("slice{}".format(index)))
	return
	
def getMeCubeHists(cubes, bins = 45, hrange = (2,255)):
	# Replacement for concatenate_cube_hists function. This one
	# is designed so that we can more easily control number of bins.
	# It also allows for scaling to 'conventional' pixel values
	
	cubes_flat = cubes.reshape(-1, ) # aka flatten
	cubes_flat = [x.reshape(-1,) for x in cubes_flat] #aka flatten lists in list
	cubes_flat = np.asarray(cubes_flat)
	cols = cubes_flat[0].shape[0]
	rows = cubes_flat.shape[0]
	
	#Add zeros to make it reshapable. Not a big deal because we will # not count them in our histograms (thus previous crop is 
	# probably redundant)
	cubes_flat = [np.pad(x, [0,cols - len(x)], mode='constant') for x in cubes_flat]
	cubes_flat = np.asarray(cubes_flat)
	
		
	# get it into (n_samples, n_featrues) shape
	cubes_flat = np.reshape(cubes_flat,(rows, cols))
	cubes_flat = MinMaxScaler(feature_range=hrange, copy=False).fit_transform(cubes_flat)  # scale to traditional values
	# It is worth trying it without scaling
	cubes_flat = np.int16(cubes_flat)  # convert to displayable values
	
	cube_hists = []
	for it, cube in enumerate(cubes_flat):
		hist, bin_edges = np.histogram(cube, bins= bins, range= hrange)
		cube_hists.append(hist)
		
	# output as vector. The code should be probably written more
	# elegantly because later I had some problems manipulating my lists
	# and arrays. I just don't seem to make it work in a better way, though
	return np.asarray(cube_hists).reshape(-1,)
	
def preprocessImg(img):

	img = crop(img, ZOOM)
	img = cube_split(img, n_splits=3)
	cube_hists = []
	for bin in binsList:
		cube_hists.append(getMeCubeHists(img, bins = bin))
	return np.asarray(cube_hists)
	
def load_data(dir):
	filelist = [x for x in listdir(dir) if splitext(x)[1] == '.nii']
	data = [None] * len(filelist)
	for f in filelist:
		print("Loading", f)
		index = int(f.replace("train_", "", 1).replace("test_", "", 1).replace(".nii", "", 1)) - 1
		img = nib.load(os.path.join(dir, f))
		cube_hists = preprocessImg(img)
		print(cube_hists.shape)
		data[index] = cube_hists

	logging.info('Data loaded')
	# If one checks data values (even before zoom) he can see that the image cointains intensity values higher then 255. This is puzzling and I suggest to scale every image between 0 and 255.
	return data
	
def load_target(dir):
	# male(0)/female(1) | young(1)/old(0) | sick(0)/healthy(0) 
	target = np.loadtxt(dir, dtype = 'uint16', delimiter = ',');
	logging.info('Target loaded')
	return target
	
def saveresult(result, name):
	filename = filename_timestamp("result") + '_' + name
	lb = ['gender', 'age','health']
	i = 0
	id = [0,1,2]
	
	with open("./" + filename + ".csv", "w") as f:
		f.write("ID,Sample,Label,Predicted\n")
		for line in result:
			idx = [int(3*i),int(3*i+1),int(3*i+2)]
			for j in id:
				boolVal = 'True' if line[j] == 1 else 'False'
				f.write("{},{},{},{}\n".format(idx[j], i, lb[j], boolVal))
			i += 1
			
	f.close()
	logging.info("Result saved as {}.csv".format(filename))	

def crossValidation(X, y, p_grid, estimator, n_splits=10, numRuns=10, rnd_state = rnd_state):
	## X should be scaled and dim reduced, see under for rest
	X, y = shuffle(X, y, random_state = rnd_state)  # Randomly permute your data, control random behaviour by fixing random state
	kf = KFold(n_splits=n_splits)  # Generate splitting strategy

	validCost = np.zeros(numRuns)
	optParam = []; ests = []
	error_score = 1	 # value to assing where scoring fails
	n_jobs = 1	# You can increase this if you have a good machone
	
	HL = 0
	
	estimator = OneVsRestClassifier(estimator, n_jobs = n_jobs)
	#y = MultiLabelBinarizer().fit_transform(y)

	for j in range(numRuns):
		# Overwrite random state to have different shuffles
		rnd_state_split = np.random.randint(1, 256**2)
		
		# This should work just fine, one could also use ShuffleSplit and initialize numRun times in a loop
		X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=rnd_state_split)

		# sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25,random_state= None)

		clf = GridSearchCV(estimator=estimator, param_grid=p_grid, cv=kf, scoring=None, n_jobs=n_jobs, error_score=error_score) # This does training and parameter optimization
		
		clf.fit(X_train, y_train)

		est = clf.best_estimator_
		y_pred = est.predict(X_valid)
		
		#print(log_loss(y_valid, y_pred))
		HL = hamming_loss(y_valid, y_pred)
		validCost[j] = HL
		optParam.append(clf.best_params_.values())

		logging.info('Run #{} finished'.format(j))
	optParam = optParam[np.argmin(np.abs(validCost))]

	return validCost, optParam, clf
	
def plotSetup():

	# Just utility function to make consistent plots
	
	plt.close('all')
	plt.rcParams['font.size'] = 10
	plt.rcParams['figure.dpi'] = 200
	plt.rcParams['savefig.dpi'] = plt.rcParams['figure.dpi']
	plt.rcParams['text.usetex'] = False
	return

def plotPerformance(name, validCosts, numsBins, num):
	# Plot Performance for multiple values of one preprocess parameter
	# Used with one estimator
	
	plotSetup()
	
	plt.figure(num= num+1)
	x = numsBins
	y = np.mean(validCosts, axis=0)
	error = np.std(validCosts, axis=0)
	filename = filename_timestamp("CVFig") + '_' + name[num]
	
	plt.ylabel("Hamming Loss on Valdiation Set")
	plt.xlabel('Number of bins')
	plt.plot(x, y, "g^")
	plt.xlim(numsBins[0]-5, numsBins[-1]+5)
	plt.title(name[num])
	plt.errorbar(x, y, yerr=error, fmt="none", ecolor='r')
	plt.savefig(filename)
	logging.info('Figure saved to file')
	plt.close('all')
	
def ScaleMultiple(X, dataTest):
	# Standard Scaling for multidimensional arrays and list combinations
	# Probably not really elegant but working. May want to adjust output 
	# of getMeCubeHists to make manipulation here easier
	
	Xs = []; dataTests = []; scalers = [];
	numVersions = X.shape[1]
	numFeatures = [len(x) for x in X[0]]
	numSamples = X.shape[0]
	numTestSamples = dataTest.shape[0]
	
	for it in range (0,numVersions):
		x_temp = X[:,it];
		x_temp = [[x] for x in x_temp]
		x_temp = np.reshape(x_temp,(numSamples, numFeatures[it]))
		scaler = StandardScaler().fit(x_temp)
		scalers.append(scaler)
		x_temp = scaler.transform(x_temp)
		
		d_temp = dataTest[:,it];
		d_temp = [[d] for d in d_temp]
		d_temp = np.reshape(d_temp,(numTestSamples, numFeatures[it]))
		d_temp = scaler.transform(d_temp)
		
		Xs.append(x_temp)
		dataTests.append(d_temp)
	
	# a = np.asarray(dataTests)
	# b = np.asarray(Xs)
	return Xs, dataTests, scalers
	
############################################

## Load train, test and target data
y = load_target(TARGET_DIR)

if MODE == 0:
	# Mode 0 ~ Load everything from scratch, expensive

	X = np.asarray(load_data(TRAIN_DIR))
	# X = np.float64(X)
	np.save('./archive~/dataTrain.npy', X)
	logging.info('dataTrain saved')

	dataTest = np.asarray(load_data(TEST_DIR))
	# dataTest = np.float64(dataTest)
	np.save('./archive~/dataTest.npy', dataTest)
	logging.info('dataTest saved')
elif MODE == 1:
	# Mode 1 ~ Load full saved train and test arrays, cheap when zoomed, otherwise expensive

	X = np.load('./archive~/dataTrain.npy')
	dataTest = np.load('./archive~/dataTest.npy')
	# dataTest = np.float64(dataTest)
	# X = np.float64(X)
	logging.info('dataTrain and dataTest Loaded')
else:
	pass

numVersions = X.shape[1]
numsBins = [len(x)/(n_splits**3) for x in X[0]]
[X, dataTest, scalers] = ScaleMultiple(X,dataTest)

## OPTIMIZE YOUR CLASSIFIER
## Define classifier
clf1 = SVC(probability=True, class_weight=None, decision_function_shape='ovr', random_state = rnd_state) 
# fix random state of classifier for all computations you do with it, improve interpretability?

clf2 = QDA()

clf3 = RandomForestClassifier(min_samples_split = 1, n_jobs = -1)

clf4 = MLPClassifier(max_iter = 400, tol = 0.0001, verbose = False, epsilon = 1e-8)

# Define parameters for search
# Use 'estimator__parameter' to access nested parameter in multiclass classifier
p_SVC = [{'estimator__kernel': ['rbf'], 'estimator__C': np.logspace(0, 2, 3), 'estimator__gamma': np.logspace(-3, 2, 6)}]
# p_SVC = [
	# {'estimator__kernel': ['rbf'], 'estimator__C': np.logspace(0, 2, 3), 'estimator__gamma': np.logspace(-3, 2, 6)},
	# {'estimator__kernel': ['poly'], 'estimator__C': np.logspace(0, 2, 3), 'estimator__gamma': np.logspace(-3, 2, 6), 'estimator__degree': np.arange(1, 6, 1)}
#]

p_QDA = [
	{'estimator__priors': [None], 'estimator__reg_param': [0]}
]

p_RFC = [
	{'estimator__class_weight': ['balanced'], 'estimator__n_estimators': np.arange(30,90,20), 'estimator__max_features' : ['sqrt', None], 'estimator__criterion' : ['gini']}
]

p_MLPC = [
	{'estimator__solver': ['adam'], 'estimator__alpha': np.logspace(-5,-5,1), 'estimator__hidden_layer_sizes': [(100,),(50,2)], 'estimator__activation': ['relu'], 'estimator__learning_rate': ['invscaling'], 'estimator__learning_rate_init': np.logspace(-2,-2,1), 'estimator__batch_size': [200], 'estimator__early_stopping': [False], 'estimator__beta_1': np.arange(0.9,1,0.1), 'estimator__beta_2': np.linspace(0.999, 0.999, 1)}
]

## put them into lists so that you can loop over them easily
estimators = [clf1]
p_grid = [p_SVC]
estimatorsStr = ['SVC']

validCosts = np.zeros((numRuns, len(estimators), numVersions))
optParamsj = []; optParams = []; clfsj = []; clfs = [];
estsTrained = []; y_preds = []

def predictOneEstimator(num = 0, validCosts = validCosts, clfs = clfs, X = X, y = y, dataTest = dataTest):

	[no_runs, no_ests, no_vers] = validCosts.shape
	costs = validCosts[:,num,:];
	costs = np.reshape(costs, (no_runs, no_vers))
	# compute stds for no_runs and make decisinon as argmin of costsMean * costsStd, to better capture the objective of low cost and low variance
	costsMean = np.mean(costs, axis=0)
	costsStd = np.std(costs, axis =0)
	costsDecide = costsMean * costsStd
	

	clf_best = clfs[num][np.argmin(np.abs(costsDecide))]
	estTrained = clf_best
	estTrained = estTrained.fit(X[np.argmin(np.abs(costsDecide))], y) 

	dataTest_optVer = dataTest[np.argmin(np.abs(costsDecide))]
	noBins = numsBins[np.argmin(np.abs(costsDecide))]

	## predict
	y_pred = estTrained.predict(dataTest_optVer)
	saveresult(y_pred, estimatorsStr[num])
	logging.info('{}'.format(estTrained))
	logging.info('No. bins = {}'.format(noBins))
	plotPerformance(estimatorsStr, costs, numsBins, num = num)
	
	return y_pred, estTrained

for i, estimator in enumerate(estimators):
	logging.info('Searching parameter grid for estimator {}'.format(estimatorsStr[i]))
	
	for j in range(0, len(binsList)):
		logging.info('Number of bins {}'.format(binsList[j]))
		validCost, optParam, clf = crossValidation(np.asarray(X[j]), y, p_grid[i], estimator, numRuns = numRuns, rnd_state= rnd_state)
		validCosts[:, i, j] = validCost
		clfsj.append(clf.best_estimator_)
		optParamsj.append(optParam)
	
	optParams.append(optParamsj)
	clfs.append(clfsj)
	logging.info('Estimator #{} finished'.format(i))
	
	y_pred, estTrained = predictOneEstimator(num = i, validCosts = validCosts, clfs = clfs, X = X, y = y, dataTest = dataTest)
	
	y_preds.append(y_pred)
	estsTrained.append(estTrained)
	
	time.sleep(2.0)
	
# y_pred, estTrained = predictOneEstimator(num = 0, validCosts, clfs, X, y, dataTest)
# saveresult(y_pred)
# logging.info('{}'.format(estTrained))


## Flush and terminate
logging.info('Script finished')
logging.shutdown()	# This should be at the very end