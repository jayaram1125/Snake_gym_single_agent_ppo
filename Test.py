import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gym
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.vector.async_vector_env import AsyncVectorEnv
import random

import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim


from torch.nn.parallel import DistributedDataParallel as DDP

'''
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()



def MLP(sizes,weights_std,output_layer_weights_std,activation=nn.Tanh,output_activation = nn.Identity):
	nnlayers =[]
	for j in range(len(sizes)-1):
		act = activation if j<len(sizes)-2 else output_activation 
		nnlayers += [nn.Linear(sizes[j],sizes[j+1]),act()]
	model =  nn.Sequential(*nnlayers)

	def init_weights_and_biases(m):
		if type(m) == nn.Linear:
			print("within init_weights_and_biases")
			std = weights_std
			if(sizes[-1]== m.out_features): # If the layer is output linear layer
				std = output_layer_weights_std			
			nn.init.orthogonal_(m.weight,std)
			nn.init.constant_(m.bias.data,0)
	model.apply(init_weights_and_biases)
	return model

class MLPActorCrtic(nn.Module):
	def __init__ (self,obs_dim,act_dim,hidden_sizes,activation = nn.Tanh):
		super().__init__()
		self.pi_logits_net = MLP([obs_dim]+hidden_sizes+[act_dim],np.sqrt(2),output_layer_weights_std = 0.01,activation=nn.Tanh)
		self.v_logits_net =  MLP([obs_dim]+hidden_sizes+[1],np.sqrt(2),output_layer_weights_std = 1,activation=nn.Tanh)

	def step(self,obs,a=None,grad_condition=False):
		with torch.set_grad_enabled(grad_condition):
			pi_logits = self.pi_logits_net(obs)
			pi = Categorical(logits = pi_logits)
			if a == None:
				a = pi.sample()
				logp_a = pi.log_prob(a)
			else:
				logp_a = pi.log_prob(a)	
			v_logits = self.v_logits_net(obs)
			v = torch.squeeze(v_logits,-1)
		return a,v,logp_a,pi.entropy()


if rank == 0:
	ac = MLPActorCrtic(401,4,[256,256],activation = nn.Tanh)    
	comm.send(ac, dest=1, tag=11)
elif rank == 1:
	ac = comm.recv(source=0, tag=11)			
	print("rank=%d"%rank)
	print(ac)'''

"""run.py:"""
#!/usr/bin/env python

#B = torch.tensor([1, 2, 3, 4])
#print(B.shape)

'''

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '27035'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)




def main():
	world_size = 2
	mp.spawn(method,args=(world_size,),nprocs=world_size,join=True)'''


class ToyModel(nn.Module):
	def __init__(self):
		super(ToyModel, self).__init__()
		self.net1 = nn.Linear(10, 10)
		self.relu = nn.ReLU()
		self.net2 = nn.Linear(10, 5)

	def forward(self, x):
		return self.net2(self.relu(self.net1(x)))

class A:
	def __init__(self):
		self.a = 1
		self.b = 2

	def func1(self):
		print(self.a)

	def func2(self):
		print(self.b)

	def train(self):
		
		print('X')

		for i in range(0,2): 
			if __name__ == '__main__':
				size = 2
				processes = []

				for rank in range(size):
					p = mp.Process(target= init_process, args=(rank,size,compute))
					p.start()
					processes.append(p)

				for p in processes:
					p.join()

def init_process(rank, size, fn, backend = "gloo"):
	""" Initialize the distributed environment. """
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '29503'
	dist.init_process_group(backend, rank=rank, world_size=size)
	fn(rank,size)

def cleanup():
	print("in cleanup")
	dist.destroy_process_group()


def compute(rank,size):
	print("Inside compute called from",rank)
	model = ToyModel()
	ddp_model = DDP(model)

	loss_fn = nn.MSELoss()
	optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

	optimizer.zero_grad()
	outputs = ddp_model(torch.randn(20, 10))
	labels = torch.randn(20, 5)
	loss_fn(outputs, labels).backward()
	optimizer.step()

	#print(ob.a)
	#print(ob.b)

	cleanup()


'''
def method(rank, size):
	setup(rank, size)
	print("run called from rank =%d"%rank)
	model = ToyModel()
	ddp_model = DDP(model)

	loss_fn = nn.MSELoss()
	optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

	optimizer.zero_grad()
	outputs = ddp_model(torch.randn(20, 10))
	labels = torch.randn(20, 5)
	loss_fn(outputs, labels).backward()
	optimizer.step()

	print(ob.a)
	print(ob.b)
	cleanup()'''


ob = A()
ob.train()

