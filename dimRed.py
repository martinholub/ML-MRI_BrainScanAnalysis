import logging

def initializeLogging():
	logging.shutdown()	# This is to clear possible previous mess
	# We log to file and to script
	logging.basicConfig(format='%(asctime)s:%(name)s:%(levelname)s:	 %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG, handlers=[logging.FileHandler("ML_dimred.log"), logging.StreamHandler()])
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
from sklearn.decomposition import KernelPCA

logging.info('Modules imported')

def defineVariables():
	TRAIN_DIR = './dataFull~/set_train/'
	TARGET_DIR = './dataFull~/targets.csv'
	TEST_DIR = './dataFull~/set_test/'

	binsList = [5,20,35,50,65,80] #You must load new data (ie MODE=0) when you want different bins!
	# binsList = [5,20]
	
	n_splits = 3 # Same for splits
	rnd_state = np.random.randint(1, 256**2)

	TIME = time.localtime(time.time())

	# Mode 0 ~ Load everything from scratch, expensive
	# Mode 1 ~ Load full saved train and test arrays, cheap when zoomed, otherwise expensive
		
	return TRAIN_DIR, TARGET_DIR, TEST_DIR, binsList, n_splits, rnd_state,TIME 

TRAIN_DIR, TARGET_DIR, TEST_DIR, binsList, n_splits, rnd_state, TIME = defineVariables()

def plotSetup():
	# Just utility function to make consistent plots

	plt.close('all')
	plt.rcParams['font.size'] = 10
	plt.rcParams['figure.dpi'] = 100
	plt.rcParams['figure.figsize'] = [8,6]
	plt.rcParams['savefig.dpi'] = plt.rcParams['figure.dpi']
	plt.rcParams['text.usetex'] = False
	plt.rcParams['font.monospace'] = 'monospace'
	plt.rcParams['legend.fancybox'] = True
	plt.rcParams['legend.shadow'] = True
	return

def Debug():
	raise SystemExit('Terminated for purpose of debugging')
	return
	
def filename_timestamp(name):
	return name + "_%02d%02d%02d%02d%02d" % (TIME.tm_mon, TIME.tm_mday, TIME.tm_hour, TIME.tm_min, TIME.tm_sec)

def load_target(dir):
	# male(0)/female(1) | young(1)/old(0) | sick(0)/healthy(0) 
	target = np.loadtxt(dir, dtype = 'uint16', delimiter = ',');
	logging.info('Target loaded')
	return target
	
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

y = load_target(TARGET_DIR)
X = np.load('./archive~/dataTrain.npy')
dataTest = np.load('./archive~/dataTest.npy')
# dataTest = np.float64(dataTest)
# X = np.float64(X)
logging.info('dataTrain and dataTest Loaded')

numVersions = X.shape[1]
numsBins = [len(x)/(n_splits**3) for x in X[0]]
[X, dataTest, scalers] = ScaleMultiple(X,dataTest)

kpca = KernelPCA(kernel = 'rbf', gamma = 1e-4, random_state = rnd_state)

# kpca = KernelPCA(kernel = 'poly', degree = 3, gamma = 1e-3, random_state = rnd_state)

X20 = X[3] # use just 35 bin version
X20_tr = kpca.fit_transform(X20);

#plotSetup()
x_axis = np.arange(1,X20.shape[1]+1,1)
x_axis_tr = np.arange(1,X20_tr.shape[1]+1,1)

plt.figure(1)
# 0 ~ male (0) / female (1), 
# 1 ~ young (1) / old (0) 
# 2 ~ sick (0) / healthy (1). 
label1 = y[:,2] ==1;
label2 = y[:,2] ==0;
plt.plot(X20_tr[label1,1], X20_tr[label1,2], 'ro', label = 'Healthy')
plt.plot(X20_tr[label2,1], X20_tr[label2,2], 'go', label = 'Sick')
plt.xlabel("1st transformed component")
plt.ylabel("2nd transformed component")
plt.legend()
plt.title('More Realistic Classification Task')
plt.show()


