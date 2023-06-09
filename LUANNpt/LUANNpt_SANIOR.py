"""LUANNpt_SANIOR.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
LUANNpt SANI object recognition (excitatory)

"""

from ANNpt_globalDefs import *
import LUANNpt_SANIORmodel
import ANNpt_data

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random

from tqdm.auto import tqdm


def createModel(dataset):
	print("creating new model")
	config = LUANNpt_SANIORmodel.SANIORconfig(
		batchSize = batchSize,
		numberOfLayers = numberOfLayers,
		hiddenLayerSize = hiddenLayerSize,
		inputLayerSize = inputLayerSize,
		outputLayerSize = numberOfClasses,
		linearSublayersNumber = linearSublayersNumber,
		numberOfFeatures = numberOfFeatures,
		numberOfClasses = numberOfClasses,
		imageSize = imageSize,
	)
	model = LUANNpt_SANIORmodel.SANIORmodel(config)
	return model
	
# Preprocess the network
def preprocessLUANNpermutations(dataset, model):
	if(not LUANNvectorised):
		print("preprocessLUANNpermutations")
		loader = ANNpt_data.createDataLoaderImage(dataset)	#required to reset dataloader and still support tqdm modification
		loop = tqdm(loader, leave=True)
		for batchIndex, batch in enumerate(loop):
			x, y = batch
			x = x.to(device)
			model.preprocessMarkActivePermutations(x)
		model.preprocessSeverInactivePermutations()
		
			
