import random
import numpy as np

EXPLORATION_RATE = 1
EXPLORATION_RATE_DECAY = 0.9999999999
EXPLORATION_RATE_MIN = 0.1
LEARNING_RATE = 0.1

class Bird():
	def __init__(self,x_dim, y_dim, v_dim, action_dim):
		self.exploration_rate = EXPLORATION_RATE
		self.exploration_rate_decay = EXPLORATION_RATE_DECAY
		self.exploration_rate_min = EXPLORATION_RATE_MIN
		self.curr_step = 0
		self.action_dim = action_dim
		self.q_table= np.zeros((x_dim,y_dim,v_dim)+(action_dim,))
		self.learning_rate = LEARNING_RATE
		#self.reward_function = 
		self.dicount_factor = 0.99
		self.initial_state = '0_0_0_0'


	def choose_action(self, state):

		# EXPLORATION
		if (np.random.rand() < self.exploration_rate):
			action_idx = np.random.randint(self.action_dim)
		
		# EXPLOITATION
		else: 
			action_idx = np.argmax(self.q_table[state])
		self.exploration_rate = max(self.exploration_rate * self.exploration_rate_decay, self.exploration_rate_min)
		return action_idx

	def updateQtable(self, current_state, next_state, action, reward):
		self.q_table[current_state][action]+= self.learning_rate * (reward + self.dicount_factor * np.max(self.q_table[next_state] - self.q_table[current_state][action]))


	def get_state(self, x, y, v, pipe):
		
		pass
	
	def training(self, episodes):
		totalsteps = 0
		for e in(1, episodes+1):
			done = False
			episode_steps = 0
			#env.reset()
			state = self.initial_state
			while (not done) and (episode_steps < 2000):
				#env.render()
				action = self.choose_action(self, state)
				next_state, reward, done =  self.act(self, state, action)
				self.updateQtable(self, state, next_state, action, reward)
				state = next_state
				episode_steps +=1
			
			#print('Episode number: {}'.format(episode_no))
            #print('Time step: {}'.format(time_step))
            #print('Selected Action: {}'.format(action))
           	#print('Current State: {}'.format(str(state_value)))
            #print('Reward Obtained: {}'.format(reward_gain))
            #print('Best Q Value: {}'.format(best_q_value))
            #print('Learning rate: {}'.format(learning_rate))
            #print('Explore rate: {}'.format(explore_rate))






	







		