from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import deque
import gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

device = torch.device('cuda')

class Policy(nn.Module):
	def __init__(self, state_size = 4, hidden_size = 16, action_size = 2):
		super(Policy, self).__init__()
		self.fc1 = nn.Linear(state_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, action_size)
		self.optimizer = torch.optim.Adam(self.parameters(), lr= 1e-2)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.softmax(x, dim=1)

	def act(self, state):
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		probs = self.forward(state).cpu()
		m = Categorical(probs)
		action = m.sample()
		return action.item(), m.log_prob(action)


policy = Policy().to(device)

env = gym.make('CartPole-v0')

def reinforce(n_episodes = 10000, max_t = 1000, gamma = 1.0, print_every = 100):
	scores_deque = deque(maxlen = 100)
	scores = []
	num_steps = []
	avg_numsteps = []

	for episode in range(1, n_episodes+1):
		saved_log_probs = []
		rewards = []
		state = env.reset()
		for t in range(max_t):
			action, log_prob = policy.act(state)
			saved_log_probs.append(log_prob)
			state, reward, done, _ = env.step(action)
			rewards.append(reward)
			if done:
				num_steps.append(t)
				avg_numsteps.append(np.mean(num_steps[-10:]))
				print(f"episode: {episode}, length: {t}")
				break
		if min(num_steps[-100:])>195:
			print(f'Stopping early at episode {episode}')
			break 
		scores_deque.append(sum(rewards))
		scores.append(sum(rewards))

		discounts = [gamma**i for i in range(len(rewards) + 1)]
		R = sum([a*b for a,b in zip(discounts, rewards)])
		
		policy_loss = []
		for log_prob in saved_log_probs:
			policy_loss.append(- log_prob*R)

		policy_loss = torch.stack(policy_loss).sum()

		policy.optimizer.zero_grad()
		policy_loss.backward()
		policy.optimizer.step()

	plt.plot(num_steps)
	plt.plot(avg_numsteps)
	plt.xlabel('Episode')
	plt.show()

reinforce()