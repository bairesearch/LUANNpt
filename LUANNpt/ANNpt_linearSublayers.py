"""ANNpt_linearSublayers.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ANNpt linear sublayers

"""

import torch as pt
from torch import nn
from ANNpt_globalDefs import *

class LinearSegregated(nn.Module):
	def __init__(self, in_features, out_features, number_sublayers):
		super().__init__()
		if(useCNNlayers):
			self.segregatedLinear = nn.Conv2d(in_channels=in_features*number_sublayers, out_channels=out_features*number_sublayers, kernel_size=CNNkernelSize, padding='same', groups=number_sublayers)
		else:	
			self.segregatedLinear = nn.Conv1d(in_channels=in_features*number_sublayers, out_channels=out_features*number_sublayers, kernel_size=1, groups=number_sublayers)
		self.number_sublayers = number_sublayers
		
	def forward(self, x):
		#x.shape = batch_size, number_sublayers, in_features
		if(useCNNlayers):
			x = x.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])
		else:
			x = x.view(x.shape[0], x.shape[1]*x.shape[2], 1)
		x = self.segregatedLinear(x)
		if(useCNNlayers):
			x = x.view(x.shape[0], self.number_sublayers, x.shape[1]//self.number_sublayers, x.shape[2], x.shape[3])
		else:
			x = x.view(x.shape[0], self.number_sublayers, x.shape[1]//self.number_sublayers)
		#x.shape = batch_size, number_sublayers, out_features
		return x

def generateLinearLayer(self, layerIndex, config, parallelStreams=False):
	if(inputLayerInList and layerIndex == 0):
		in_features = config.inputLayerSize
	else:
		in_features = config.hiddenLayerSize
	if(outputLayerInList and layerIndex == config.numberOfLayers-1):
		out_features = config.outputLayerSize
	else:
		out_features = config.hiddenLayerSize

	if(getUseLinearSublayers(self, layerIndex)):
		linear = LinearSegregated(in_features=in_features, out_features=out_features, number_sublayers=config.linearSublayersNumber)
	else:
		if(useCNNlayers):
			linear = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=CNNkernelSize, padding='same')
		else:
			if(parallelStreams):
				in_features = config.hiddenLayerSize*config.linearSublayersNumber
			linear = nn.Linear(in_features=in_features, out_features=out_features)

	weightsSetLayer(self, layerIndex, linear)

	return linear

def generateActivationFunction():
	if(SMANNuseSoftmax):
		if(thresholdActivations):
			activation = OffsetSoftmax(thresholdActivationsMin)
		else:
			activation = nn.Softmax(dim=1)
	else:
		if(thresholdActivations):
			activation = OffsetReLU(thresholdActivationsMin)
		else:
			activation = nn.ReLU()
	return activation

def generateActivationLayer(self, layerIndex, config):
	return generateActivationFunction()

def executeLinearLayer(self, layerIndex, x, linear, parallelStreams=False):
	weightsFixLayer(self, layerIndex, linear)	#otherwise need to constrain backprop weight update function to never set weights below 0
	if(getUseLinearSublayers(self, layerIndex)):
		#perform computation for each sublayer independently
		if(not parallelStreams):
			x = x.unsqueeze(dim=1).repeat(1, self.config.linearSublayersNumber, 1)
		x = linear(x)
	else:
		if(parallelStreams):
			x = pt.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
			#print("x.shape = ", x.shape)
		x = linear(x)
	return x

def executeActivationLayer(self, layerIndex, x, activationFunction, parallelStreams=False):
	if(getUseLinearSublayers(self, layerIndex)):
		if(SMANNuseSoftmax):
			#TODO: find an optimised parallel implementation of executeActivationLayer:getUseLinearSublayers:SMANNuseSoftmax:activationFunction
			xSublayerList = []
			for sublayerIndex in range(self.config.linearSublayersNumber):
				xSublayer = x[:, sublayerIndex]
				xSublayer = activationFunction(xSublayer)
				xSublayerList.append(xSublayer)
			if(parallelStreams):
				x = pt.stack(xSublayerList, dim=1)
			else:
				x = pt.concat(xSublayerList, dim=1)
		else:
			x = activationFunction(x)
			if(not parallelStreams):
				if(useCNNlayers):
					x = pt.reshape(x, (x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4]))
				else:
					x = pt.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
	else:
		x = activationFunction(x)
	return x

def getUseLinearSublayers(self, layerIndex):
	result = False
	if(useLinearSublayers):
		if(outputLayerInList):
			if(layerIndex != self.config.numberOfLayers-1):	#final layer does not useLinearSublayers
				result = True
		else:
			result = True
	return result

def weightsSetLayer(self, layerIndex, linear):
	weightsSetPositiveLayer(self, layerIndex, linear)
	if(useCustomWeightInitialisation):
		if(getUseLinearSublayers(self, layerIndex)):
			nn.init.normal_(linear.segregatedLinear.weight, mean=Wmean, std=WstdDev)
		else:
			nn.init.normal_(linear.weight, mean=Wmean, std=WstdDev)
	if(useCustomBiasInitialisation):
		if(getUseLinearSublayers(self, layerIndex)):
			nn.init.constant_(linear.segregatedLinear.bias, 0)
		else:
			nn.init.constant_(linear.bias, 0)

def weightsFixLayer(self, layerIndex, linear):
	if(not trainLastLayerOnly):
		weightsSetPositiveLayer(self, layerIndex, linear)
			
def weightsSetPositiveLayer(self, layerIndex, linear):
	if(usePositiveWeights):
		if(not usePositiveWeightsClampModel):
			if(getUseLinearSublayers(self, layerIndex)):
				weights = linear.segregatedLinear.weight #only positive weights allowed
				weights = pt.abs(weights)
				linear.segregatedLinear.weight = pt.nn.Parameter(weights)
			else:
				weights = linear.weight #only positive weights allowed
				weights = pt.abs(weights)
				linear.weight = pt.nn.Parameter(weights)
		
def weightsSetPositiveModel(self):
	if(usePositiveWeights):
		if(usePositiveWeightsClampModel):
			for p in self.parameters():
				p.data.clamp_(0)

class OffsetReLU(nn.Module):
	def __init__(self, offset):
		super(OffsetReLU, self).__init__()
		self.offset = offset

	def forward(self, x):
		print("OffsetReLU: x = ", x)
		#print("self.offset = ", self.offset)
		x = pt.max(pt.zeros_like(x), x - self.offset)
		return x

class OffsetSoftmax(nn.Module):
	def __init__(self, offset):
		super(OffsetSoftmax, self).__init__()
		self.offset = offset
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		print("OffsetSoftmax: x = ", x)
		#print("self.offset = ", self.offset)
		x = self.softmax(x)
		print("OffsetSoftmax: x after softmax = ", x)
		x = pt.max(pt.zeros_like(x), x - self.offset)
		return x




