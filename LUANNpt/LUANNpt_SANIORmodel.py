"""LUANNpt_SANIORmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
LUANNpt SANI object recognition model (excitatory)

"""

from ANNpt_globalDefs import *

import torch as pt
from torch import nn
from ANNpt_globalDefs import *
from torchmetrics.classification import Accuracy
import ANNpt_linearSublayers

import random

class SANIORconfig():
	def __init__(self, batchSize, numberOfLayers, hiddenLayerSize, inputLayerSize, outputLayerSize, linearSublayersNumber, numberOfFeatures, numberOfClasses, imageSize):
		self.batchSize = batchSize
		self.numberOfLayers = numberOfLayers
		self.hiddenLayerSize = hiddenLayerSize
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.linearSublayersNumber = linearSublayersNumber
		self.numberOfFeatures = numberOfFeatures
		self.numberOfClasses = numberOfClasses
		self.imageSize = imageSize

# Define the network architecture
if(LUANNvectorised):
	class SANIORmodel(nn.Module):
		def __init__(self, config,):
			super(SANIORmodel, self).__init__()
			self.config = config
			self.columnSize = config.hiddenLayerSize	#Number of neurons per column
			self.numberOfColumnPermutations = config.linearSublayersNumber	#Number of random permutations of columns
			self.numberOfLayers = config.numberOfLayers	#number of columns in each permutation
			self.numberOfClasses = config.numberOfClasses	#Number of CIFAR-10 classes
			if(useCNNlayers):
				self.imageSize = config.imageSize
				self.numberOfChannelsHidden = config.hiddenLayerSize

			#declare columns
			self.columnsLinear = nn.ModuleList()
			for layerIndex in range(self.numberOfLayers):
				linear = ANNpt_linearSublayers.generateLinearLayer(self, layerIndex, config, parallelStreams=True)
				self.columnsLinear.append(linear)

			self.activationFunction = ANNpt_linearSublayers.generateActivationFunction()
			
			if(useCNNlayers):
				outputInFeatures = pow(self.numberOfColumnPermutations, self.numberOfLayers)
				self.output = nn.Linear(outputInFeatures, self.numberOfClasses)
			#else:
			#	self.output = nn.Linear(self.numberOfColumnPermutations*self.columnSize, self.numberOfClasses)
				
			self.lossFunction = nn.CrossEntropyLoss()
			self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.numberOfClasses, top_k=1)	#CHECKTHIS

		def forward(self, trainOrTest, x, y, optim=None, l=None):
			#if(not useCNNlayers):
			#	x = pt.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
			h = x.unsqueeze(dim=1)
			
			# Propagate input through the network columns
			for l in range(self.numberOfLayers):
				hLast = h
				if(useCNNlayers):
					h = h.repeat(1, self.numberOfColumnPermutations, 1, 1, 1)
				#else:
				#	h = h.repeat(1, self.numberOfColumnPermutations, 1)
				
				linear = self.columnsLinear[l]
				h = ANNpt_linearSublayers.executeLinearLayer(self, l, h, linear, parallelStreams=True)
				h = ANNpt_linearSublayers.executeActivationLayer(self, l, h, self.activationFunction, parallelStreams=True)
			
				'''
				if(linkFilters):
					for every pixel:
						for every sample in hLast:
							calculate ideal h to hLast mapping
				'''
				
				if(useCNNlayers):
					h = pt.reshape(h, (h.shape[0]*h.shape[1], 1, h.shape[2], h.shape[3], h.shape[4]))
				#else:
				#	h = pt.reshape(h, (h.shape[0]*h.shape[1], 1, h.shape[2]))

			if(useCNNlayers):
				h = pt.reshape(h, (h.shape[1], h.shape[0], h.shape[2], h.shape[3], h.shape[4]))
			#else:
			#	h = pt.reshape(h, (h.shape[1], h.shape[0], h.shape[2]))
				
			loss, accuracy = calculateOutput(self, h, y)
			
			return loss, accuracy
else:
	class SANIORmodel(nn.Module):
		def __init__(self, config,):
			super(SANIORmodel, self).__init__()
			self.config = config
			self.columnSize = config.hiddenLayerSize	#Number of neurons per column
			self.numberOfColumnPermutations = config.linearSublayersNumber	#Number of random permutations of columns
			self.numberOfLayers = config.numberOfLayers	#number of columns in each permutation
			self.numberOfClasses = config.numberOfClasses	#Number of CIFAR-10 classes
			self.numberOfFeatures = config.numberOfFeatures
			if(useCNNlayers):
				self.imageSize = config.imageSize
				self.numberOfChannelsHidden = config.hiddenLayerSize
				
			#declare columns
			self.columns = nn.ModuleList()
			for l in range(self.numberOfLayers):
				columnsLayer = nn.ModuleList()
				for c in range(self.numberOfColumnPermutations):
					column = ANNpt_linearSublayers.generateLinearLayer(self, l, config, parallelStreams=False)
					columnsLayer.append(column)
					if(usePositiveWeights):
						column.weight.data = pt.abs(column.weight.data)
				self.columns.append(columnsLayer)
				
			if(useCNNlayers):
				outputInFeatures = pow(self.numberOfColumnPermutations, self.numberOfLayers)
				self.output = nn.Linear(outputInFeatures, self.numberOfClasses)
			#else:
			#	self.output = nn.Linear(self.numberOfColumnPermutations*self.columnSize, self.numberOfClasses)

			self.activationFunction = ANNpt_linearSublayers.generateActivationFunction()
			
			self.lossFunction = nn.CrossEntropyLoss()
			self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.numberOfClasses, top_k=1)	#CHECKTHIS


		def forward(self, trainOrTest, x, y, optim=None, l=None):
			#if(not useCNNlayers):
			#	x = pt.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
			
			outputList = []
			self.forwardLayer(0, x, y, outputList)
			
			h = pt.stack(outputList, dim=1)
			loss, accuracy = calculateOutput(self, h, y)
						
			return loss, accuracy

		def forwardLayer(self, layerIndex, x, y, outputList):
			columnsLayer = self.columns[layerIndex]
			for c in range(self.numberOfColumnPermutations):
				# Propagate input through the network column
				col = columnsLayer[c]
				h = x
				h = col(h)
				h = self.activationFunction(h)
						
				if(layerIndex == self.numberOfLayers-1):
					outputList.append(h)
				else:
					self.forwardLayer(layerIndex+1, h, y, outputList)
			
			'''
			if(linkFilters):
				for every pixel:
					calculate ideal h to hLast (x) mapping
			'''
			
def calculateOutput(self, h, y):
	if(trainLastLayerOnly):
		h = h.detach()
	
	# Flatten columns' output and connect to output layer
	if(useCNNlayers):
		h = pt.reshape(h, (h.shape[0], h.shape[1]*h.shape[2]*h.shape[3]*h.shape[4]))
	else:
		h = pt.reshape(h, (h.shape[0], h.shape[1]*h.shape[2]))
	
	out = self.output(h)

	loss = self.lossFunction(out, y)
	accuracy = self.accuracyFunction(out, y)
	accuracy = accuracy.detach().cpu().numpy()

	return loss, accuracy
		
