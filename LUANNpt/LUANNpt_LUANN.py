"""LUANNpt_LUANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
LUANNpt large untrained artificial neural network

"""

from ANNpt_globalDefs import *
import LUANNpt_LUANNmodel
import ANNpt_data

def createModel(dataset):
	datasetSize = ANNpt_data.getDatasetSize(dataset, printSize=True)
	numberOfFeatures = ANNpt_data.countNumberFeatures(dataset)
	numberOfClasses, numberOfClassSamples = ANNpt_data.countNumberClasses(dataset)
	
	print("creating new model")
	config = LUANNpt_LUANNmodel.SMANNconfig(
		batchSize = batchSize,
		numberOfLayers = numberOfLayers,
		hiddenLayerSize = hiddenLayerSize,
		inputLayerSize = numberOfFeatures,
		outputLayerSize = numberOfClasses,
		linearSublayersNumber = linearSublayersNumber,
		numberOfFeatures = numberOfFeatures,
		numberOfClasses = numberOfClasses,
		datasetSize = datasetSize,
		numberOfClassSamples = numberOfClassSamples
	)
	model = LUANNpt_LUANNmodel.SMANNmodel(config)
	return model
	
