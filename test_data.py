import numpy as np
import torch
# data = np.load('data.npz')
# print(type(data), data)

# print(type(data['params']), data['params'].shape, data['params'])
# print(type(data['edges']), data['edges'].shape, data['edges'])



data3 = np.load('data_3.npz')
p = data3['points']
edges = data3['edges']
edge2 = p[:,1:,:] - p[:,:-1,:]
edges = torch.from_numpy(edges)
edge2 = torch.from_numpy(edge2)
torch.allclose(edge2,edges)
