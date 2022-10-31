import gym 
from gym import logger, spaces
import numpy as np
import Box2D
from enum import IntEnum
import math
import random

from getkey import getkey,keys

import pygame

from typing import Optional

from Box2D import b2Vec2
    
from Box2D.b2 import(
	fixtureDef,
	polygonShape)


BLACK = (0,0,0) #background 

DARK_GREEN = (0,100,0)#Snake head color
GREEN = (0, 255, 0) #Snake body color

RED = (255, 0, 0) #Fruit color
BROWN = (156,102,31) #Grid color



COLOR_CODE_BLACK = 0
COLOR_CODE_BROWN = 1
COLOR_CODE_RED = 2
COLOR_CODE_GREEN = 3
COLOR_CODE_DARK_GREEN = 4



DEG2RAD = math.pi/180.0
RAD2DEG = 180.0/math.pi



class SnakeDirection(IntEnum):
	LEFT =0,
	RIGHT =1,
	UP = 2,
	DOWN = 3



class SnakeGameEnv(gym.Env):	
	"""
	**********Action Space:***************
	The action is a `ndarray` with shape `(1,)` which can take values `{0,1,2,3}` indicating the direction
	of the head of the Snake to be steered.
	 Num    Action                                      Value
	  0      Steer Snake head to the left direction       0
	  1      Steer Snake head to the right                1  
	  2      Steer Snake head to the Upward direction     2 
	  3      Steer Snake head to the Downward direction   3		


	**********Observation Space:**********
	RGB array of shape 400x400x3 having pixel intensity values scaled by 255.0

	  
	  Note: 
	  1.Snake's body length can range from 0 to 383 units.
	  2.The object values are normalized to have zero mean , unit variance  and be in the interval [-10,10]
	    before passing into the neural network


     ******Rewards:**********************
	  +1 reward if snake eats fruit and grows its body unit 
	  
	  if dcurrent is not = -1
	  (-dcurrent)/d0 reward if snake is alive and away from fruit
	  	dcurrent = current distance between snake head and fruit
		d0 = distance between snake head and fruit at the beginning of the episode or after the fruit is moved to a new location
	  
	  -5 if  dcurrent is -1 or dprev-dcurrent =0  (snake blocked it self ,it could unblock or might not)


      -10 reward if snake dies 


	 ******Episode Termination:**********
     The episode terminates  if:
	    1) Termination:The snake head crosses the maze (unsuccessful end of the game)
	    2) Termination:The snake head crosses any one of the body units (unsuccessful end of the game)
	    3) Termination: Snake fills the entire play area with its head and body and no place to position the fruit. It touches its body (successful end of the game)	
	"""

	

	def __init__(self,render_mode: Optional[str] = "rgb_array"):

		self.metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"]}
		
		#print("render_mode=",render_mode)
		assert render_mode is None or render_mode in self.metadata["render_modes"]

		self.render_mode = render_mode


		self.displayX = 400.0
		self.displayY = 400.0

		self.window =None
		if self.render_mode == "human":
			import pygame

			pygame.init()
			pygame.display.init()
			self.window = pygame.display.set_mode((self.displayX, self.displayY))
			self.clock = pygame.time.Clock()


		self.body_width = 40.0
		self.scale = 10.0
		self.world = Box2D.b2World(gravity=(0,0))


		self.is_game_over = False
		self.fruit_eaten = False
		self.screen = None
		self.action_space = spaces.Discrete(4)


		self.total_rows = int(self.displayX/self.body_width)
		self.total_cols = int(self.displayY/self.body_width)


		low = np.full(((400,400,3)),0)
		high = np.full(((400,400,3)),255)


		self.observation_space = spaces.Box(low, high)   

		self.create_maze()
		
		self.fruit = None
		self.snakeobj =None

		self.d0 = 0  # distance from head to fruit at the start of the episode or after the fruit is moved to a new location
		self.dcurrent = 0
		self.dprev =0



	def create_maze(self):
		vertices = [b2Vec2(0.0,0.0), b2Vec2(self.displayX/self.scale,0.0), b2Vec2(self.displayX/self.scale,self.displayY/self.scale),b2Vec2(0.0,self.displayY/self.scale),
				b2Vec2(0.0,0.0), b2Vec2(self.body_width/self.scale,self.body_width/self.scale),b2Vec2(self.displayX/self.scale-self.body_width/self.scale,self.body_width/self.scale),b2Vec2(self.displayX/self.scale-self.body_width/self.scale,self.displayY/self.scale-self.body_width/self.scale),b2Vec2(self.body_width/self.scale,self.displayY/self.scale-self.body_width/self.scale),b2Vec2(self.body_width/self.scale,self.body_width/self.scale)]
		chain = Box2D.b2ChainShape(vertices_chain=vertices)

		
		#displayX == displayY assumed in the design .All statements below work only if this assumption is considered		
		self.maze_collision_bound_1 = self.body_width/self.scale/2
		self.maze_collision_bound_2 = self.displayX/self.scale-self.body_width/self.scale/2
		self.maze = self.world.CreateStaticBody(fixtures=fixtureDef(shape=chain))

		#Maze indices in the observations 2D array
		self.maze_indices = []

		start = 0
		end = int(self.displayX/self.body_width)
		
		#top side of maze
		for j in range(start,end):
			self.maze_indices.append([start,j])


		#left side of the maze
		begin = start+1
		for i in range(begin, end):
			self.maze_indices.append([i,start])


		#bottom side of maze	
		for j in range(begin, end):
			self.maze_indices.append([end-1,j])


		#right side of the maze
		for i in range(begin,end-1):
			self.maze_indices.append([i,end-1])

		
		self.play_area_set = set()

		play_area_bound_1 =  int(self.body_width/self.scale + 2)
		play_area_bound_2 =  int(self.displayX/self.scale-self.body_width/self.scale -2)   
		self.cols  = play_area_bound_2//4

		self.area_pos_dict = {}

		for i in range(play_area_bound_1,play_area_bound_2+4,4):
			for j in range(play_area_bound_1,play_area_bound_2+4,4):
				#print("(%d,%d)"%(i,j))
				#i*self.cols+j helps to create unique key in the play area set
				self.play_area_set.add(i//4+(j//4)*self.cols)
				self.area_pos_dict[i//4+(j//4)*self.cols] =(i,j)

				

	class Snake:
		def __init__(self,env_ref,list_occupied_area_sets):
			#print("enter snake __init__")
			self.env_ref = env_ref
			sampled_pos = self.env_ref.sample_position_from_play_area(list_occupied_area_sets)
			sampled_angle = self.env_ref.sample_angle()
			self.head = self.env_ref.world.CreateStaticBody(position = b2Vec2(sampled_pos[0],sampled_pos[1]),angle = sampled_angle[0] ,fixtures=fixtureDef(shape=polygonShape(box=(2,2)))) 
						
			self.body = []
			self.head_pos_set = set()
			self.head_pos_set.add(self.head.position[0]//4+(self.head.position[1]//4)*self.env_ref.cols)
			self.body_pos_set = set()



		def move_snake(self,next_direction):

			prev_head_position =  self.head.position.copy()
			prev_head_angle = self.head.angle	

				
			if next_direction == SnakeDirection.UP and round(self.head.angle*RAD2DEG) != 270: 
				self.head.position[1] = self.head.position[1]-self.env_ref.body_width/self.env_ref.scale
				self.head.angle = math.pi/2

			elif next_direction == SnakeDirection.DOWN and round(self.head.angle*RAD2DEG) != 90: 
				self.head.position[1] = self.head.position[1]+self.env_ref.body_width/self.env_ref.scale
				self.head.angle = 3*math.pi/2
					
			elif next_direction == SnakeDirection.RIGHT and round(self.head.angle*RAD2DEG) != 180: 				 
				self.head.position[0] = self.head.position[0]+self.env_ref.body_width/self.env_ref.scale
				self.head.angle = 0
			
			elif next_direction == SnakeDirection.LEFT and round(self.head.angle*RAD2DEG) != 0:
				self.head.position[0] = self.head.position[0]-self.env_ref.body_width/self.env_ref.scale
				self.head.angle = math.pi

	
			self.head_pos_set.clear()
			self.head_pos_set.add(self.head.position[0]//4+(self.head.position[1]//4)*self.env_ref.cols)


		
			#Update the body positions only if the head is moved .Snake cannot move in opposite direction	
			if(prev_head_position != self.head.position and len(self.body)>0):		 	
				self.body_pos_set.clear() 
				for i in range(len(self.body)- 1, 0, -1):
					self.body[i].position = self.body[i-1].position
					self.body[i].angle = self.body[i-1].angle
					self.body_pos_set.add(self.body[i].position[0]//4+(self.body[i].position[1]//4)*self.env_ref.cols)


				self.body[0].position = prev_head_position
				self.body[0].angle =  prev_head_angle
				self.body_pos_set.add(self.body[0].position[0]//4+(self.body[0].position[1]//4)*self.env_ref.cols)
	

	 
		def destroy_snake(self):
			if(self.head != None):
				self.env_ref.world.DestroyBody(self.head)
				self.head = None
				self.head_pos_set.clear()
			for i in range(0,len(self.body)):
				#print("enter destroy body:")
				self.env_ref.world.DestroyBody(self.body[i])
			self.body.clear()
			self.body_pos_set.clear()
			#print("Snake Destroyed:")	
	

		def increase_snake_length(self):

			new_body_unit_position_x = 0
			new_body_unit_position_y = 0

			last_unit_angle = round(self.body[-1].angle*RAD2DEG) if len(self.body)>0 else round(self.head.angle*RAD2DEG) 
			last_unit_position_x = self.body[-1].position[0] if len(self.body)>0 else  self.head.position[0]
			last_unit_position_y = self.body[-1].position[1] if len(self.body)>0 else  self.head.position[1]

			distance_delta = self.env_ref.body_width/self.env_ref.scale

			if(last_unit_angle == 0):
				new_body_unit_position_x = last_unit_position_x-distance_delta
				new_body_unit_position_y = last_unit_position_y

			elif(last_unit_angle == 90):
				new_body_unit_position_x = last_unit_position_x
				new_body_unit_position_y = last_unit_position_y+distance_delta

			elif(last_unit_angle == 180):
				new_body_unit_position_x = last_unit_position_x+distance_delta
				new_body_unit_position_y = last_unit_position_y

			elif(last_unit_angle == 270):	
				new_body_unit_position_x = last_unit_position_x
				new_body_unit_position_y = last_unit_position_y-distance_delta

			self.body.append(self.env_ref.world.CreateStaticBody(position = b2Vec2(new_body_unit_position_x,new_body_unit_position_y),angle= last_unit_angle,fixtures=fixtureDef(shape=polygonShape(box=(2,2)))))
			self.body_pos_set.add(new_body_unit_position_x//4+(new_body_unit_position_y//4)*self.env_ref.cols)



	def sample_position_from_play_area(self,list_occupied_area_sets):
		remaining_area_set = self.play_area_set -list_occupied_area_sets[0]
		for i in range(1,len(list_occupied_area_sets)):
			remaining_area_set = remaining_area_set-list_occupied_area_sets[i]

		#print("remaining_area_set=")
		#print(remaining_area_set)
		
		sampled_pos = None
		if(len(remaining_area_set)==0):
			self.is_game_over = True
		else: 
			output = random.sample(list(remaining_area_set), 1)
			sampled_pos = self.area_pos_dict[output[0]]

		return sampled_pos	 

	def sample_angle(self):
		angles = [0.0,math.pi/2,3*math.pi/2,math.pi]
		sampled_angle = random.sample(angles,1)
		return sampled_angle	  		


	def create_fruit(self):
		sampled_pos = self.sample_position_from_play_area([set()])
		self.fruit = self.world.CreateStaticBody(position=b2Vec2(sampled_pos[0],sampled_pos[1]),angle= 0,fixtures=fixtureDef(shape=polygonShape(box=(2,2))))
	
	def destroy_fruit(self):
		self.world.DestroyBody(self.fruit)


	def move_fruit_to_another_location(self):
		sampled_pos = self.sample_position_from_play_area([self.snakeobj.head_pos_set,self.snakeobj.body_pos_set])   
		if(sampled_pos):	
			self.fruit.position[0] = sampled_pos[0]
			self.fruit.position[1] = sampled_pos[1]		
			#print('new fruit position= (%d,%d)'%(self.fruit.position[0],self.fruit.position[1]))
		#destroy fruit object when play area is filled with snake
		else:
			self.destroy_fruit()


	def create_snake(self):
		fruit_area_set = set()
		fruit_area_set.add(self.fruit.position[0]//4 + (self.fruit.position[1]//4)*self.cols)
		self.snakeobj = self.Snake(self,[fruit_area_set])
	

	def check_contact(self,snakeobj):
		snake_collided_with_maze = (snakeobj.head.position[0]== self.maze_collision_bound_1 or snakeobj.head.position[1]==self.maze_collision_bound_1 or
			snakeobj.head.position[0]== self.maze_collision_bound_2 or snakeobj.head.position[1]== self.maze_collision_bound_2)

		#Checking contact with Maze or snake itself
		if(snake_collided_with_maze or (snakeobj.head.position[0]//4+(snakeobj.head.position[1]//4)*self.cols) in snakeobj.body_pos_set):
			snakeobj.destroy_snake()
			self.is_game_over = True
		#Checking contact with fruit	
		elif(self.fruit.position[0] == snakeobj.head.position[0] and self.fruit.position[1] == snakeobj.head.position[1]):
			snakeobj.increase_snake_length()
			self.fruit_eaten = True
			self.dcurrent = 0
			self.move_fruit_to_another_location()




	def create_observations(self):

		'''
			Row index and col index of the matrix are adjusted so as matrix represents the 
			image frame.  
			Graphics coordinate system of Box2d and Pygame  is not same as matrix indexing

		'''
	
		obs = np.full((self.total_rows,self.total_cols), COLOR_CODE_BLACK)
		#print(obs.shape)

		#print(self.maze_indices)

		for i in range(0,len(self.maze_indices)):
			obs[self.maze_indices[i][0]][self.maze_indices[i][1]] = COLOR_CODE_BROWN

		if(self.fruit != None):	
			obs[int(self.fruit.position[1]//4)][int(self.fruit.position[0]//4)] =  COLOR_CODE_RED

		for i in range(0,len(self.snakeobj.body)):
			if(self.snakeobj.body[i] != None):
				obs[int(self.snakeobj.body[i].position[1]//4)][int(self.snakeobj.body[i].position[0]//4)] = COLOR_CODE_GREEN

		if(self.snakeobj.head != None):
			obs[int(self.snakeobj.head.position[1]//4)][int(self.snakeobj.head.position[0]//4)] = COLOR_CODE_DARK_GREEN


		return obs


	def calculate_distance_head_and_fruit(self,obs):
		

		headX = int(self.snakeobj.head.position[1]//4)
		headY = int(self.snakeobj.head.position[0]//4)


		source = [headX, headY,0]    

		visited = []

		for i in range(len(obs)):
			visited.append([])
			for j in range(len(obs[0])): 
				visited[i].append(False)

		delta =[[-1,0],[1,0],[0,-1],[0,1]]
               
		queue = []
		queue.append(source)
		visited[source[0]][source[1]] = True


		while len(queue) != 0:
			source = queue.pop(0)
 
        	#Fruit found,return its distance from snake head
			#print(obs[source[0]][source[1]])
			if (obs[source[0]][source[1]] == COLOR_CODE_RED):
				return source[2]

			for i in range(0,len(delta)):
				x = source[0]+delta[i][0]
				y = source[1]+delta[i][1]

				if (x>=0 and y>=0 and x< len(obs) and y < len(obs[0]) and (obs[x][y] == COLOR_CODE_BLACK or obs[x][y] == COLOR_CODE_RED) and visited[x][y] == False):
					queue.append([x,y,source[2] + 1])
					visited[x][y] = True

		return -1




	def reset(self, seed=None, return_info=False, options=None):
		super().reset(seed=seed)
		#print("--------------------------RESET CALLED -----------------")
		if(self.fruit):
			self.destroy_fruit()
			self.fruit = None

		if(self.snakeobj and self.snakeobj.head):	
			self.snakeobj.destroy_snake()

	
		self.is_game_over = False

		self.create_fruit()

		self.create_snake()


		self.fruit_eaten = False


		obs = self.create_observations()

		#print(obs)
		#print(self.d0)

		self.dcurrent = self.calculate_distance_head_and_fruit(obs)
		self.d0 = self.dcurrent
		self.dprev  = 0

		return_obs =  self.render()

		return_obs = return_obs/255.0
		#print(return_obs.shape)

		return return_obs


	def step(self,action):

		assert self.action_space.contains(action),"action = %d is invalid" %action	
			

		self.snakeobj.move_snake(action)
		
		self.check_contact(self.snakeobj)

		reward = 0.0

		if(self.is_game_over):
			reward  = -1
		elif(self.fruit_eaten):
			reward  = 1
			self.fruit_eaten = False 
			obs = self.create_observations()
			self.dcurrent =  self.calculate_distance_head_and_fruit(obs)
			self.d0 = self.dcurrent
			self.dprev = 0 
		else:
			self.dprev = self.dcurrent
			obs = self.create_observations()
			self.dcurrent =  self.calculate_distance_head_and_fruit(obs)
			if(self.dcurrent == -1 or (self.dprev-self.dcurrent ==0)):
				reward  = -0.5   #snake is at same place or blocked its path to fruit, -ve reward for wrong move and not moving in zig zag manner when it is long
			elif(self.dcurrent<self.dprev):
				reward  = 0.2
			else:
				reward  =-0.2

		#print("BEfore")
		#print("AFter")
		
		#print("Within step func, obs=")
		#print(obs)

		#print("dprev=",self.dprev)
		#print("dcurrent=",self.dcurrent)

		info = {}

		return_obs = self.render()

		return_obs = return_obs/255.0

		return return_obs,reward,self.is_game_over,info
	


	def render(self, mode: str="rgb_array"):
		# create the display surface object
		# of specific dimension..e(X,Y).

		assert mode is not None
		#print("Enter check 1:")


		surf = pygame.Surface((self.displayX, self.displayY))
		surf.fill(BLACK)
		pygame.transform.scale(surf, (self.scale, self.scale))
		

		pygame.draw.rect(surf,BROWN,pygame.Rect(0,0,self.displayX,self.displayY),int(self.body_width))

		if self.fruit:
			pygame.draw.rect(surf,RED,pygame.Rect(self.fruit.position[0]*self.scale-self.body_width/2,self.fruit.position[1]*self.scale-self.body_width/2,self.body_width,self.body_width))
		
		if self.snakeobj: 
			if self.snakeobj.head :			
				pygame.draw.rect(surf,DARK_GREEN,pygame.Rect(self.snakeobj.head.position[0]*self.scale-self.body_width/2,self.snakeobj.head.position[1]*self.scale-self.body_width/2,self.body_width,self.body_width))  
		
			for i in range(0,len(self.snakeobj.body)):
				if self.snakeobj.body[i]:
					pygame.draw.rect(surf,GREEN,pygame.Rect(self.snakeobj.body[i].position[0]*self.scale-self.body_width/2,self.snakeobj.body[i].position[1]*self.scale-self.body_width/2,self.body_width,self.body_width))  
		
		if mode == "human":
			assert self.window is not None
			self.window.blit(surf, surf.get_rect())
			pygame.event.pump()
			pygame.display.update()

			self.clock.tick(self.metadata["render_fps"])
		else:
			return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))

	def close(self):
		if self.window is not None:
			import pygame
			pygame.display.quit()
			pygame.quit()

#Test code
if __name__ == "__main__":
	'''obj = SnakeGameEnv()
	obj.reset()
	
	#print(reward)
	#print(is_game_over)
	#print(info)

	def render(mode: str="rgb_array"):
		# create the display surface object
		# of specific dimension..e(X,Y).

		assert mode is not None
		#print("Enter check 1:")


		running = True
		
		pygame.init()
		pygame.display.init()
		window = pygame.display.set_mode((400,400))

		while running:
			surf = pygame.Surface((400,400))
			surf.fill(BLACK)
			pygame.transform.scale(surf,(10,10))


			pygame.draw.rect(surf,BROWN,pygame.Rect(0,0,obj.displayX,obj.displayY),int(obj.body_width))
			

			if obj.fruit:
				pygame.draw.rect(surf,RED,pygame.Rect(obj.fruit.position[0]*obj.scale-obj.body_width/2,obj.fruit.position[1]*obj.scale-obj.body_width/2,obj.body_width,obj.body_width))
			
			if obj.snakeobj: 
				if obj.snakeobj.head :			
					pygame.draw.rect(surf,DARK_GREEN,pygame.Rect(obj.snakeobj.head.position[0]*obj.scale-obj.body_width/2,obj.snakeobj.head.position[1]*obj.scale-obj.body_width/2,obj.body_width,obj.body_width))  
			
				for i in range(0,len(obj.snakeobj.body)):
					if obj.snakeobj.body[i]:
						pygame.draw.rect(surf,GREEN,pygame.Rect(obj.snakeobj.body[i].position[0]*obj.scale-obj.body_width/2,obj.snakeobj.body[i].position[1]*obj.scale-obj.body_width/2,obj.body_width,obj.body_width))  
			

			window.blit(surf, surf.get_rect())
			pygame.event.pump()
			pygame.display.update()

				
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					running = False

	action =0 
			
	while(True):
		render()

		key = getkey()
		if key == keys.LEFT:
			print('LEFT')
			action = 0 
		elif key == keys.RIGHT:
			print('RIGHT')
			action = 1
		elif key == keys.UP:
			print('UP')
			action = 2
		elif key == keys.DOWN:
			print('DOWN')
			action = 3

		_,reward,is_game_over,info =obj.step(action)
		print("reward=",reward)	
		print("is_game_over=",is_game_over)
		print("info=",info)'''
