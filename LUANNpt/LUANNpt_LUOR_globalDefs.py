"""LUANNpt_LUOR_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
LUANNpt LUOR globalDefs

"""

LUANNvectorised = False	#execute column operations in parallel	#optional
useCNNlayers = False	#optional	#else Linear layers

trainLastLayerOnly = True	#True: default LUANN, False: standard backprop comparison for debug

batchSize = 32
numberOfLayersMax = 10	#Maximum number of columns in each permutation
numberOfLayersMin = 3	#Minimum number of columns in each permutation
trainNumberOfEpochs = 10
		
# Define network hyperparameters
numberOfUniqueColumns = 100  # Number of unique columns (layers) to select permutations from
numberOfClasses = 10  # Number of CIFAR-10 classes
imageSizeChannels = 3
imageSizeWidth = 32
imageSizeHeight = 32
imageSize = imageSizeWidth*imageSizeHeight
numberOfFeatures = imageSize*imageSizeChannels	# Number of CIFAR-10 features

if(useCNNlayers):
	CNNkernelSize = 5
	hiddenLayerSize = 16	#number of Conv2D channels channels per layer (column)
	inputLayerSize = imageSizeChannels
else:
	hiddenLayerSize = 256
	inputLayerSize = numberOfFeatures

SMANNuseSoftmax = False	#experimental (select activation function that will gradually reduce probability of activation of higher level columns in each permutation)
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
else:
	useLinearSublayers = False
linearSublayersNumber = 100
		


workingDrive = '/large/source/ANNpython/LUANNpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelLUOR'
