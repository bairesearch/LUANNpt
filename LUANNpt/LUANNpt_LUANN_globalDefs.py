"""LUANNpt_LUANN_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
LUANNpt LUANN globalDefs

"""

debugSmallNetwork = False

useLUANNonly = True
if(useLUANNonly):
	trainLastLayerOnly = True	#True: default LUANN, False: standard backprop comparison for debug

	inputLayerInList = True
	outputLayerInList = True

	useLinearSublayers = True	#recommended	#pass input through multiple independent column permutations
	linearSublayersNumber = 1000
		
	SMANNuseSoftmax = False
	if(SMANNuseSoftmax):
		usePositiveWeights = False	#required
		if(usePositiveWeights):
			usePositiveWeightsClampModel = False	#mandatory False as hidden layer weights untrained but all initialised as positive
	useInbuiltCrossEntropyLossFunction = True	#optional	#for LUANNpt_LUANNmodel only
	
	if(trainLastLayerOnly):
		batchSize = 64
		numberOfLayers = 4
		if(SMANNuseSoftmax):
			hiddenLayerSize = 1000
			normaliseActivationSparsity = True	#increases performance
		else:
			hiddenLayerSize = 100
			normaliseActivationSparsity = False
	
workingDrive = '/large/source/ANNpython/LUANNpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelLUANN'

