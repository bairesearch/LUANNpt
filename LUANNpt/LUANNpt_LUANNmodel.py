"""LUANNpt_LUANNmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
LUANNpt large untrained artificial neural network model

"""

import torch as pt
from torch import nn
from ANNpt_globalDefs import *
from torchmetrics.classification import Accuracy
import ANNpt_linearSublayers


class SMANNconfig():
	def __init__(self, batchSize, numberOfLayers, hiddenLayerSize, inputLayerSize, outputLayerSize, linearSublayersNumber, numberOfFeatures, numberOfClasses, datasetSize, numberOfClassSamples):
		self.batchSize = batchSize
		self.numberOfLayers = numberOfLayers
		self.hiddenLayerSize = hiddenLayerSize
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.linearSublayersNumber = linearSublayersNumber
		self.numberOfFeatures = numberOfFeatures
		self.numberOfClasses = numberOfClasses
		self.datasetSize = datasetSize		
		self.numberOfClassSamples = numberOfClassSamples
		
class SMANNmodel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		layersLinearList = []
		layersActivationList = []
		for layerIndex in range(config.numberOfLayers):
			linear = ANNpt_linearSublayers.generateLinearLayer(self, layerIndex, config, parallelStreams=True)
			layersLinearList.append(linear)
		for layerIndex in range(config.numberOfLayers):
			activation = ANNpt_linearSublayers.generateActivationLayer(self, layerIndex, config)
			layersActivationList.append(activation)
		self.layersLinear = nn.ModuleList(layersLinearList)
		self.layersActivation = nn.ModuleList(layersActivationList)
	
		if(useInbuiltCrossEntropyLossFunction):
			self.lossFunction = nn.CrossEntropyLoss()
		else:
			self.lossFunction = nn.NLLLoss()	#nn.CrossEntropyLoss == NLLLoss(log(softmax(x)))
		self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.config.outputLayerSize, top_k=1)
		
		ANNpt_linearSublayers.weightsSetPositiveModel(self)
				
	def forward(self, trainOrTest, x, y, optim=None, l=None):
		if(useLUANNonly):
			x = x.unsqueeze(dim=1).repeat(1, self.config.linearSublayersNumber, 1)
		for layerIndex in range(self.config.numberOfLayers):
			if(trainLastLayerOnly):
				x = x.detach()
			x = ANNpt_linearSublayers.executeLinearLayer(self, layerIndex, x, self.layersLinear[layerIndex], parallelStreams=useLUANNonly)
			if(debugSmallNetwork):
				print("layerIndex = ", layerIndex)
				print("x after linear = ", x)
			if(layerIndex == self.config.numberOfLayers-1):
				if(not useInbuiltCrossEntropyLossFunction):
					x = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, x, self.layersActivation[layerIndex], parallelStreams=useLUANNonly)	#CHECKTHIS
					x = pt.log(x)
			else:
				x = ANNpt_linearSublayers.executeActivationLayer(self, layerIndex, x, self.layersActivation[layerIndex], parallelStreams=useLUANNonly)
			if(debugSmallNetwork):
				print("x after activation = ", x)
		loss = self.lossFunction(x, y)
		accuracy = self.accuracyFunction(x, y)
		accuracy = accuracy.detach().cpu().numpy()
		
		return loss, accuracy


