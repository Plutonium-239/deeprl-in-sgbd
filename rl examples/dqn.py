import torch
import torch.nn as nn
import numpy as np
from collections import deque

gamma = 0.99


class DQN(nn.module):
	"""docstring for DQN nn.module"""
	def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
		super(DQN,nn.module).__init__()
		self.input_dims = input_dims
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.n_actions = n_actions
		# Defining the NN Architecture (Currently just linear, can include CNNs)
		self.layers = nn.Sequential(
			nn.Linear(*self.input_dims, self.fc1_dims),
			nn.ReLU(),
			nn.Linear(self.fc1_dims, self.fc2_dims),
			nn.ReLU(),
			nn.Linear(self.fc2_dims, self.n_actions)
			)
		self.optimzer = torch.optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()
		self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
		self.to(self.device)

	def forward(self, state):
		return self.layers(state)

class Agent(object):
	"""docstring for Agent"""
	def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, 
		max_mem_size = 100_000, epsilon_final = 0.01, epsilon_decay = 5e-4):
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_final = epsilon_final
		self.epsilon_decay = epsilon_decay
		self.lr = lr
		self.action_space = list(range(n_actions))
		self.mem_size = max_mem_size
		self.batch_size = batch_size
		self.mem_counter = 0

		self.q_eval = DQN(lr=lr, input_dims=input_dims, fc1_dims=256, fc2_dims=256, n_actions=n_actions)

		self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
		self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
		self.action_memory = np.zeros((self.mem_size, *input_dims), dtype=np.int32)
		self.reward_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
		self.terminal_memory = np.zeros((self.mem_size, *input_dims), dtype=np.bool)

	def store_transition(self, state, action, reward, state_, done):
		index = self.mem_counter%self.mem_size
		self.state_memory[index] = state
		self.new_state_memory[index] = state_
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.terminal_memory[index] = done

		self.mem_counter += 1

	def choose_action(self, observation):
		if np.random.random() > self.epsilon:
			state = torch.tensor([observation]).to(self.q_eval.device)
			actions = self.q_eval.forward(state)
			action = torch.argmax(actions).item()
		else:
			action = np.random.choice(self.action_space)
		return action

	def learn(self):
		if self.mem_counter < self.batch_size:
			return

		self.q_eval.optimzer.zero_grad()
		max_mem = min(self.mem_counter, self.mem_size)
		batch = np.random.choice(max_mem, self.batch_size, replace=False)
		
		batch_index = np.arange(self.batch_size, dtype=np.int32)

		state_batch = torch.tensor(self.state_memory).to(self.q_eval.device)
		new_state_batch = torch.tensor(self.new_state_memory).to(self.q_eval.device)
