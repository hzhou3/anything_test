import torch
from torch import nn
import numpy as np



def cosineSimilarity(gt, pred):

	'''
	calculate cosine similarity of two tensors
	'''
	cos = nn.CosineSimilarity(dim=1, eps=1e-6)
	return cos(gt, pred)



def mse(gt, pred):
	mseloss = nn.MSELoss()

	return mseloss(gt, pred)



if __name__ == '__main__':

	batch_size = 4

	# same tensor
	gt = np.array([1, 2, 3, 4, 5.0]) 
	pred = np.array([1, 2, 3, 4, 5.0])

	gt = torch.from_numpy(np.tile(gt, (batch_size, 1)))
	pred = torch.from_numpy(np.tile(pred, (batch_size, 1)))

	print("gt.shape == pred.shape: {}".format(gt.shape == pred.shape))
	
	sim = cosineSimilarity(gt, pred)
	m = mse(gt, pred)
	print(sim, m)

	# different tensor by scaling down by 5
	gt = np.array([1, 2, 3, 4, 5.0]) 
	pred = np.array([1, 2, 3, 4, 5.0]) / 5.0

	gt = torch.from_numpy(np.tile(gt, (batch_size, 1)))
	pred = torch.from_numpy(np.tile(pred, (batch_size, 1)))

	print("gt.shape == pred.shape: {}".format(gt.shape == pred.shape))
	
	sim = cosineSimilarity(gt, pred)
	m = mse(gt, pred)
	print(sim, m)