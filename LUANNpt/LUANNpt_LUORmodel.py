"""LUANNpt_LUORmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
LUANNpt large untrained object recognition model (excitatory)

Summary (not LUANNvectorised):

Create a PyTorch neural network, trained with the huggingface CIFAR-10 dataset. The network has only excitatory weights, and only uses the softmax activation function. Declare a large number of independent layers ("columns"), and they are stored in a python list. Each column is of size H (i.e. has H neurons). Declare P random permutations of columns (sets of layers), where each permutation contains L columns.

Perform preprocessing of the network: The entire CIFAR-10 training set is sent through each permutation (p) of columns. If zero activation is detected at a particular column in a permutation (over all training example), then the permutation is severed (culled or shortened) at this layer. 

Perform training of the network: The last (unsevered) layer of each permutation is then connected to the output layer of the network (containing class targets) using a large linear layer (input size = number of permutations P x layer size H, output size = number of class targets C).

"""

from ANNpt_globalDefs import *

import torch as pt
from torch import nn
from ANNpt_globalDefs import *
from torchmetrics.classification import Accuracy
import ANNpt_linearSublayers

import random

class LUORconfig():
	def __init__(self, batchSize, numberOfLayersMax, numberOfLayersMin, hiddenLayerSize, inputLayerSize, outputLayerSize, linearSublayersNumber, numberOfFeatures, numberOfClasses, numberOfUniqueColumns, imageSize):
		self.batchSize = batchSize
		self.numberOfLayersMax = numberOfLayersMax
		self.numberOfLayersMin = numberOfLayersMin
		self.hiddenLayerSize = hiddenLayerSize
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.linearSublayersNumber = linearSublayersNumber
		self.numberOfFeatures = numberOfFeatures
		self.numberOfClasses = numberOfClasses
		self.numberOfUniqueColumns = numberOfUniqueColumns
		self.imageSize = imageSize

# Define the network architecture
if(LUANNvectorised):
	class LUORmodel(nn.Module):	#LargeUntrainedExcitatoryNet
		def __init__(self, config,):
			super(LUORmodel, self).__init__()
			self.config = config
			self.columnSize = config.hiddenLayerSize	#Number of neurons per column
			self.numberOfColumnPermutations = config.linearSublayersNumber	#Number of random permutations of columns
			self.numberOfLayersMax = config.numberOfLayersMax	#Maximum number of columns in each permutation
			self.numberOfLayersMin = config.numberOfLayersMin	#Minimum number of columns in each permutation
			self.numberOfClasses = config.numberOfClasses	#Number of CIFAR-10 classes
			if(useCNNlayers):
				self.imageSize = config.imageSize
				self.numberOfChannelsHidden = config.hiddenLayerSize

			#declare columns
			self.columnsLinear = nn.ModuleList()
			for layerIndex in range(self.numberOfLayersMax):
				linear = ANNpt_linearSublayers.generateLinearLayer(self, layerIndex, config, parallelStreams=True)
				self.columnsLinear.append(linear)
				
			self.activationFunction = ANNpt_linearSublayers.generateActivationFunction()
			
			if(useCNNlayers):
				if(useCNNlayersConverge):
					self.output = nn.Linear(self.numberOfColumnPermutations*self.numberOfChannelsHidden, self.numberOfClasses)
				else:
					self.output = nn.Linear(self.numberOfColumnPermutations*self.numberOfChannelsHidden*self.imageSize, self.numberOfClasses)
			else:
				self.output = nn.Linear(self.numberOfColumnPermutations*self.columnSize, self.numberOfClasses)
				
			self.lossFunction = nn.CrossEntropyLoss()
			self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.numberOfClasses, top_k=1)	#CHECKTHIS

		def forward(self, trainOrTest, x, y, optim=None, l=None):
			if(not useCNNlayers):
				x = pt.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
				
			# Propagate input through the network columns
			if(useCNNlayers):
				h = x.unsqueeze(dim=1).repeat(1, self.numberOfColumnPermutations, 1, 1, 1)
			else:
				h = x.unsqueeze(dim=1).repeat(1, self.numberOfColumnPermutations, 1)
			for l in range(self.numberOfLayersMax):
				linear = self.columnsLinear[l]
				h = ANNpt_linearSublayers.executeLinearLayer(self, l, h, linear, parallelStreams=True)
				h = ANNpt_linearSublayers.executeActivationLayer(self, l, h, self.activationFunction, parallelStreams=True)
			if(trainLastLayerOnly):
				h = h.detach()

			loss, accuracy = calculateOutput(self, h, y)
			
			return loss, accuracy
			
else:
	class LUORmodel(nn.Module):	#LargeUntrainedExcitatoryNet
		def __init__(self, config,):
			super(LUORmodel, self).__init__()
			self.config = config
			self.numberOfUniqueColumns = config.numberOfUniqueColumns	 #Number of unique columns (layers) to select permutations from
			self.columnSize = config.hiddenLayerSize	#Number of neurons per column
			self.numberOfColumnPermutations = config.linearSublayersNumber	#Number of random permutations of columns
			self.numberOfLayersMax = config.numberOfLayersMax	#Maximum number of columns in each permutation
			self.numberOfLayersMin = config.numberOfLayersMin	#Minimum number of columns in each permutation
			self.numberOfClasses = config.numberOfClasses	#Number of CIFAR-10 classes
			self.numberOfFeatures = config.numberOfFeatures
			if(useCNNlayers):
				self.imageSize = config.imageSize
				self.numberOfChannelsHidden = config.hiddenLayerSize
				
			#declare columns
			self.columns = nn.ModuleList()
			for l in range(self.numberOfLayersMax):
				columnsLayer = nn.ModuleList()
				for c in range(self.numberOfUniqueColumns):
					column = ANNpt_linearSublayers.generateLinearLayer(self, l, config, parallelStreams=False)
					print("column = ", column)
					columnsLayer.append(column)
					if(usePositiveWeights):
						column.weight.data = pt.abs(column.weight.data)
				self.columns.append(columnsLayer)
			if(useCNNlayers):
				if(useCNNlayersConverge):
					self.output = nn.Linear(self.numberOfColumnPermutations*self.numberOfChannelsHidden, self.numberOfClasses)
				else:
					self.output = nn.Linear(self.numberOfColumnPermutations*self.numberOfChannelsHidden*self.imageSize, self.numberOfClasses)
			else:
				self.output = nn.Linear(self.numberOfColumnPermutations*self.columnSize, self.numberOfClasses)

			#declare permutations
			self.permuations = []
			self.permuationsActive = []
			for p in range(self.numberOfColumnPermutations):
				# create random column permutations (can sample same column more than once; recurrent connections)
				permutation = []
				for l in range(self.numberOfLayersMax):
					columnsLayer = self.columns[l]
					permLayer = random.choices(range(self.numberOfUniqueColumns), k=1)
					permutationLayer = columnsLayer[permLayer[0]]
					#print("permutationLayer = ", permutationLayer)
					permutation.append(permutationLayer)
				self.permuations.append(permutation)
				
				permutationActive = [False]*self.numberOfUniqueColumns
				self.permuationsActive.append(permutationActive)

			self.activationFunction = ANNpt_linearSublayers.generateActivationFunction()
			
			self.lossFunction = nn.CrossEntropyLoss()
			self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.numberOfClasses, top_k=1)	#CHECKTHIS

		def preprocessMarkActivePermutations(self, x):
			if(not useCNNlayers):
				x = pt.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
			for p in range(self.numberOfColumnPermutations):
				permutation = self.permuations[p]
				permuationActive = self.permuationsActive[p]
				h = x
				for l in range(self.numberOfLayersMax):
					#print("l = ", l, ", h = ", h)
					col = permutation[l]
					colActive = permuationActive[l]
					h = col(h)
					#print("l = ", l, ", h = ", h)
					h = self.activationFunction(h)
					if(pt.sum(h) == 0):
						break
					else:
						permuationActive[l] = True
				if(thresholdActivations):
					if(debugPrintActivationOutput):
						print("max l = ", l)
						
		def preprocessSeverInactivePermutations(self):
			permutationsCulled = []
			for p in range(self.numberOfColumnPermutations):
				permutationCulled = []
				permutation = self.permuations[p]
				permuationActive = self.permuationsActive[p]
				for l in range(self.numberOfLayersMax):
					col = permutation[l]
					colActive = permuationActive[l]
					if(colActive):
						permutationCulled.append(col)
						maxActiveColumn = l
					else:
						break
				if(len(permutationCulled) >= self.numberOfLayersMin):
					permutationsCulled.append(permutationCulled)
			self.permutations = permutationsCulled

		def forward(self, trainOrTest, x, y, optim=None, l=None):
			if(not useCNNlayers):
				x = pt.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
				
			hAllP = []
			for p in range(len(self.permuations)):
				permutation = self.permuations[p]
				# Propagate input through the network columns
				h = x
				for l in range(len(permutation)):
					col = permutation[l]
					h = col(h)
					h = self.activationFunction(h)
					if(pt.sum(h) == 0):
					    break	#fast exit to save processing time
				if(trainLastLayerOnly):
					h = h.detach()
				hAllP.append(h)

			hAllP = pt.stack(hAllP, dim=1)
			loss, accuracy = calculateOutput(self, hAllP, y)
			
			return loss, accuracy
			
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
		
