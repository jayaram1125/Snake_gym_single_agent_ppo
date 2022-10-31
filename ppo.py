import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gym
import Snake
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.vector.sync_vector_env import SyncVectorEnv
import random

import os

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP



from gym.envs.registration import register
register(
	id='Snake-v0',
	entry_point='Snake:SnakeGameEnv'
)


def layer_init(m,std=np.sqrt(2)):
	#print("within init_weights_and_biases")
	nn.init.orthogonal_(m.weight,std)
	nn.init.constant_(m.bias.data,0)
	return m

class MLPActorCrtic(nn.Module):
	def __init__ (self,act_dim):
		super().__init__()	
		self.network = nn.Sequential(
			layer_init(nn.Conv2d(3, 32, 8, stride=4)),
			nn.ReLU(),
			layer_init(nn.Conv2d(32, 64, 4, stride=2)),
			nn.ReLU(),
			layer_init(nn.Conv2d(64, 64, 3, stride=1)),
			nn.ReLU(),
			nn.Flatten(),
			layer_init(nn.Linear(46*46*64, 512)),
			nn.ReLU(),
		)
		self.policy = layer_init(nn.Linear(512, 4), std=0.01)
		self.value =  layer_init(nn.Linear(512, 1), std=1)


	def step(self,obs,a=None,grad_condition=False):
		with torch.set_grad_enabled(grad_condition):
			pi_logits = self.policy(self.network(obs))
			pi = Categorical(logits = pi_logits)
			if a == None:
				a = pi.sample()
				logp_a = pi.log_prob(a)
			else:
				logp_a = pi.log_prob(a)
			v_logits = self.value(self.network(obs))
			v = torch.squeeze(v_logits,-1)
		return a,v,logp_a,pi.entropy()



class PPO: 
	def __init__(self):
		self.num_envs  = 4
		self.num_updates = 500
		self.num_timesteps = 128
		self.gamma = 0.99
		self.lamda = 0.95
		self.mini_batch_size = 4
		self.learning_rate = 2.5e-4
		self.clip_coef = 0.2
		self.entropy_coef=0.01
		self.value_coef= 0.5
		self.max_grad_norm =0.5
		self.update = 0
		self.epochs = 4



	def capped_cubic_video_schedule(self,episode_id: int) -> bool:
		if self.update  < self.num_updates-20:
			if episode_id < 1000:
				return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
			else:
				return episode_id % 1000 == 0	
		else:
			return episode_id %1 == 0


	def make_env(self,env_index):
		def func():
			snake_game_env = gym.make('Snake-v0')
			snake_game_env = gym.wrappers.RecordEpisodeStatistics(snake_game_env)
			if env_index == 0:
				snake_game_env = gym.wrappers.RecordVideo(snake_game_env, "video",episode_trigger = self.capped_cubic_video_schedule)
			return snake_game_env
		return func


	def calculate_gae(self,last_values,last_dones):
		next_nonterminal = None
		last_gae = 0 

		last_index = self.num_envs-1

		for step in reversed(range(self.batch_size)):
			if step >= self.batch_size-self.num_envs and step <= self.batch_size-1: 
				next_nonterminal = 1.0-last_dones[last_index]
				next_values = last_values[last_index]
				last_index = last_index-1
			else:
				next_nonterminal = 1.0-self.batch_dones[step+1]
				next_values = 1.0-self.batch_values[step+1]
	
			delta = self.batch_rewards[step]+self.gamma*next_nonterminal*next_values-self.batch_values[step] 			
			self.batch_advantages[step] = last_gae = delta +self.gamma*next_nonterminal*self.lamda*last_gae
			
		self.batch_returns = self.batch_advantages+self.batch_values 	



	def step(self):
		#print("Step function enter:")	
		list_actions = []
		list_values = []
		list_logprobs_ac = []
		list_entropies_agent = []

		list_obs = []
		list_rewards = []
		list_dones =[]


		last_values = None
		last_dones = None

		for i in range(0,self.num_timesteps):

			list_obs.append(torch.from_numpy(self.next_obs).type(torch.float32).to(self.device))
			list_dones.append(torch.from_numpy(self.next_dones).type(torch.float32).to(self.device))

			#print("----------------------------------TIMESTEP NO:%d---------------------------------------"%i)
			#print(self.next_obs.shape)

			actions,values,logprobs_ac,entropies_agent = self.actor_critic.step(
				torch.as_tensor(self.next_obs.reshape(self.next_obs.shape[0],self.next_obs.shape[3],self.next_obs.shape[1],self.next_obs.shape[2]),dtype = torch.float32).to(self.device))
			
			self.next_obs,rewards,self.next_dones,infos = self.snake_game_envs.step(actions.cpu().numpy())
			
			#print(infos)

			if infos[0] and infos[0]["episode"]:
				r = infos[0]["episode"]
				self.trainf.write("\n"+str(r))


			list_actions.append(actions)
			list_values.append(values)
			
			list_logprobs_ac.append(logprobs_ac)
			list_entropies_agent.append(entropies_agent)
			list_rewards.append(torch.from_numpy(rewards).type(torch.float32).to(self.device))

		_,next_values,_,_ = self.actor_critic.step(torch.as_tensor(
			self.next_obs.reshape(self.next_obs.shape[0],self.next_obs.shape[3],self.next_obs.shape[1],self.next_obs.shape[2]),dtype = torch.float32).to(self.device))

		

		self.batch_size	= self.num_timesteps*self.num_envs
		self.data_store_dict["batch_size"] = self.batch_size


		self.batch_actions = torch.Tensor(self.batch_size).to(self.device)
		torch.cat(list_actions, out= self.batch_actions)
		self.data_store_dict["batch_actions"] = self.batch_actions


		self.batch_values = torch.Tensor(self.batch_size).to(self.device)
		torch.cat(list_values, out=self.batch_values)
		self.data_store_dict["batch_values"] = self.batch_values


		self.batch_logprobs_ac = torch.Tensor(self.batch_size).to(self.device)
		torch.cat(list_logprobs_ac, out=self.batch_logprobs_ac)
		self.data_store_dict["batch_logprobs_ac"] = self.batch_logprobs_ac


		self.batch_entropies_agent = torch.Tensor(self.batch_size).to(self.device)
		torch.cat(list_entropies_agent, out=self.batch_entropies_agent)
		self.data_store_dict["batch_entropies_agent"] = self.batch_entropies_agent


		self.batch_obs = torch.Tensor(self.batch_size,400,400,3).to(self.device)
		torch.cat(list_obs, out = self.batch_obs)
		self.data_store_dict["batch_obs"]= self.batch_obs



		self.batch_rewards = torch.Tensor(self.batch_size).to(self.device)
		#print("Device id for batch rewards tensor",self.batch_rewards.get_device())
		torch.cat(list_rewards, out = self.batch_rewards).to(self.device)
		self.data_store_dict["batch_rewards"] = self.batch_rewards


		self.batch_dones = torch.Tensor(self.batch_size).to(self.device)
		torch.cat(list_dones, out = self.batch_dones).to(self.device)
		self.data_store_dict["batch_dones"] = self.batch_dones

	
		self.batch_advantages = torch.zeros_like(self.batch_values).to(self.device)

		self.batch_returns = torch.zeros_like(self.batch_values).to(self.device)

		self.calculate_gae(next_values,self.next_dones)	

		self.data_store_dict["batch_advantages"]=self.batch_advantages
		self.data_store_dict["batch_returns"] = self.batch_returns


		#print("Step function exit:")


	def train(self):
		seed = 0
		torch.manual_seed(seed)
		np.random.seed(seed)
	

		#creating multiple aysnc enviroments that run in parallel
		env_fns = [self.make_env(i) for i in range(self.num_envs)]
		self.snake_game_envs = SyncVectorEnv(env_fns)
		assert self.snake_game_envs.num_envs ==  self.num_envs ,"Number of envs are not created as expected!"	


		print("observation_space.shape=")
		print(self.snake_game_envs.single_observation_space.shape)
		
		print("number_of_actions=")
		print(self.snake_game_envs.action_space.nvec[0])


		

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		

		self.actor_critic = MLPActorCrtic(self.snake_game_envs.action_space.nvec[0]).to(self.device)

		self.actor_critic.share_memory()

		self.data_store_dict = {}


		self.next_obs = self.snake_game_envs.reset()

		self.next_dones = np.zeros(self.num_envs)

		self.data_store_dict["epochs"] = self.epochs

		self.data_store_dict["mini_batch_size"] = self.mini_batch_size

		self.data_store_dict["max_grad_norm"] = self.max_grad_norm

		self.data_store_dict["entropy_coef"] = self.entropy_coef

		self.data_store_dict["value_coef"] = self.value_coef

		self.data_store_dict["clip_coef"] = self.clip_coef

		self.trainf = open('TrainLog.txt','a')

		for update in range(1,self.num_updates+1):

			self.update = update  #Used in video recording schedule
			
			print("***********************Update num**********************:%d"%update)

			multiple = 1.0-(update-1.0)/self.num_updates
			self.lr_current = multiple*self.learning_rate
			self.data_store_dict["lr_current"] = self.lr_current
			
			self.step() #step the environment and actor critic to get one batch of data

			self.batch_indices = [i for i in range(0,self.batch_size)]
			
			random.shuffle(self.batch_indices)

			self.data_store_dict["batch_indices"] =self.batch_indices


			torch.set_num_threads(1)

			if __name__ == '__main__':
				size = 1
				run_method(compute_gradients_and_optimize,size,self.actor_critic,self.data_store_dict)
			
		ppo_obj.trainf.close()



def setup(rank,world_size):
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '29530'
	dist.init_process_group("gloo", rank=rank, world_size=world_size)

def run_method(fn,world_size,actor_critic,data_store_dict):
	mp.spawn(fn,args=(world_size,actor_critic,data_store_dict),nprocs = world_size,join = True)


def compute_gradients_and_optimize(rank,world_size,actor_critic,data_store_dict):
	#print("enter compute_gradients_and_optimize with rank:",rank)
	setup(rank, world_size)

	actor_critic = actor_critic.to(rank)

	ddp_model = DDP(actor_critic,device_ids= [rank])
	#ddp_model = DDP(actor_critic)


	optimizer = Adam(ddp_model.parameters(), lr= data_store_dict["lr_current"])
	
	sub_batch_size =  data_store_dict["batch_size"] //world_size
	sub_batch_train_start_index = rank*sub_batch_size 
	sub_batch_train_stop_index = sub_batch_train_start_index+sub_batch_size

	#print("before for")

	epochs = data_store_dict["epochs"]
	mini_batch_size = data_store_dict["mini_batch_size"]

	for epoch in range(epochs):
		#print("enter for")

		i = sub_batch_train_start_index 

		#print("rank:",rank)
		#print("sub_batch_train_start_index=")
		#print(sub_batch_train_start_index)

		while (i < sub_batch_train_stop_index):
			#print("enter while")

			start = i
			end = i+ mini_batch_size
			
			#print("rank:",rank)
			#print("start=")
			#print(start)


			#print("rank:",rank)
			#print("end =")
			#print(end)


			slice = data_store_dict["batch_indices"][start:end]

			#print("slice:")
			#print(slice)

			mini_batch_obs = data_store_dict["batch_obs"][slice].to(rank)
		

			mini_batch_actions = data_store_dict["batch_actions"][slice].to(rank)

			mini_batch_logp_a = data_store_dict["batch_logprobs_ac"][slice].to(rank)

			mini_batch_returns = data_store_dict["batch_returns"][slice].to(rank)

			mini_batch_values = data_store_dict["batch_values"][slice].to(rank)
				
			mb_obs_size = list(mini_batch_obs.size())	

			_,new_v,new_logp_a,entropy = actor_critic.step(
				mini_batch_obs.reshape(mb_obs_size[0],mb_obs_size[3],mb_obs_size[1],mb_obs_size[2]).to(rank),mini_batch_actions,grad_condition=True)


			mini_batch_advantages = data_store_dict["batch_advantages"][slice].to(rank)



			mini_batch_advantages_mean = mini_batch_advantages.mean()
			mini_batch_advantages_std = mini_batch_advantages.std()
			mini_batch_advantages = (mini_batch_advantages - mini_batch_advantages_mean)/(mini_batch_advantages_std + 1e-8)

			

			logratio = new_logp_a-mini_batch_logp_a

			ratio = logratio.exp()
			
			ploss1 = -mini_batch_advantages*ratio
			ploss2 = -mini_batch_advantages* torch.clamp(ratio, 1 - data_store_dict["clip_coef"], 1 + data_store_dict["clip_coef"]) 
			ploss = torch.max(ploss1,ploss2).mean()

			vloss1 = (new_v-mini_batch_returns)**2
			vloss2 = (torch.clamp(new_v-mini_batch_values,-data_store_dict["clip_coef"],data_store_dict["clip_coef"])-mini_batch_returns)**2
			vloss = 0.5*torch.max(vloss1,vloss2).mean()

			entropy_loss = entropy.mean()


			loss = ploss - data_store_dict["entropy_coef"]*entropy_loss + data_store_dict["value_coef"]*vloss  

			#print("rank:",rank)
			#print("loss =")
			#print(loss)


			optimizer.zero_grad()
			loss.backward()


			nn.utils.clip_grad_norm_(ddp_model.parameters(), data_store_dict["max_grad_norm"])
			optimizer.step()


			i = i+mini_batch_size
	
		#print("exit compute_gradients_and_optimize with rank:",rank)	


if __name__ == '__main__':
	ppo_obj = PPO() 	
	ppo_obj.train()




