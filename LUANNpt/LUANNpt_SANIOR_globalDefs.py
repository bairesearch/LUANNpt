"""LUANNpt_SANIOR_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
LUANNpt SANIOR globalDefs

"""

LUANNvectorised = True	#execute column operations in parallel	#optional #True: requires high GPU RAM, False: requires high GPU time
useCNNlayers = True	#mandatory

debugPrintActivationOutput = True

linkFilters = False	#TODO

trainLastLayerOnly = True	#True: default LUANN, False: standard backprop comparison for debug

batchSize = 1	#save GPU RAM
numberOfLayers = 4	#number of columns in each permutation
trainNumberOfEpochs = 10
		
# Define network hyperparameters
numberOfClasses = 10  # Number of CIFAR-10 classes
imageSizeChannels = 3
imageSizeWidth = 32
imageSizeHeight = 32
imageSize = imageSizeWidth*imageSizeHeight
numberOfFeatures = imageSize*imageSizeChannels	# Number of CIFAR-10 features

if(useCNNlayers):
	useCNNlayersConverge = True	#mandatory
	if(useCNNlayersConverge):
		CNNkernelSize = 3
		CNNstride = 2	#decresse pixel width  by 2x every layer
		CNNpadding = 0	#CHECKTHIS
	else:
		CNNkernelSize = 5
		CNNstride = 1	#maintain pixel width per layer
		CNNpadding = 'same'
	hiddenLayerSize = 1	#number of Conv2D channels channels per layer (column)
	inputLayerSize = imageSizeChannels

SMANNuseSoftmax = True	#experimental (select activation function that will gradually reduce probability of activation of higher level columns in each permutation)
usePositiveWeights = True	#required
if(usePositiveWeights):
	usePositiveWeightsClampModel = False	#mandatory False as hidden layer weights untrained but all initialised as positive

if(usePositiveWeights):
	thresholdActivations = True	#make activations go to zero at later layers in columns
	if(thresholdActivations):
		import math
		def calculateDefaultLinearWeightsStdv(in_features):
			if(SMANNuseSoftmax):
				stdv = 0.003
			else:
				stdv = 10.0	#0.2
			return stdv
		def calculateDefaultConv2DweightsStdv(in_features):
			if(SMANNuseSoftmax):
				stdv = 0.003
			else:
				stdv = 10.0	#0.2
			return stdv
		#min activation for RelU:
		if(useCNNlayers):
			thresholdActivationsMin = calculateDefaultConv2DweightsStdv(hiddenLayerSize)
		else:
			thresholdActivationsMin = calculateDefaultLinearWeightsStdv(hiddenLayerSize)
		#need to assume input layer activations are normalised, elses some inputs will be entirely suppressed while others will be accepted

inputLayerInList = True
outputLayerInList = False


if(LUANNvectorised):
	useLinearSublayers = True	#recommended	#pass input through multiple independent column permutations
	linearSublayersNumber = 10	#number of unique columns in each layer	#requires high GPU RAM
else:
	useLinearSublayers = False
	linearSublayersNumber = 10	#100	#number of unique columns in each layer	#requires high GPU time
		


workingDrive = '/large/source/ANNpython/LUANNpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelSANIOR'
