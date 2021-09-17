import gym
import numpy as np
# import time
from tqdm import tqdm
import plotly.express as px

env = gym.make('MountainCar-v0')

eta = 0.1
discount = 0.9
episodes = 25000
show_on = 500
epsilon = 0.5
epsilon_decay_start = 1
epsilon_decay_end = episodes//2
epsilon_decay_value = epsilon/(epsilon_decay_end - epsilon_decay_start)

DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
	return tuple(((state - env.observation_space.low)/discrete_os_win_size).astype(np.int))

times = []
ep_rewards = []
agg_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

for episode in tqdm(range(episodes)):
	episode_reward = 0
	render = False
	done = False
	timeloop = False
	discrete_state = get_discrete_state(env.reset())

	if episode%200 == 1:
		render = True
	if episode%200 == 0:
		timeloop = True
		tqdm.write(str(episode))

	while not done:
		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state])
		else:
			action = np.random.randint(0, env.action_space.n)
		new_state, reward, done, ob = env.step(action)
		episode_reward += reward
		new_discrete_state = get_discrete_state(new_state)
		if render:
			env.render()

		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action, )]
			new_q = (1 - eta)*current_q + eta*(reward + discount*max_future_q)
			q_table[discrete_state + (action,)] = new_q

		elif new_state[0] >= env.goal_position:
			q_table[discrete_state + (action,)] = 0

		discrete_state = new_discrete_state

	if timeloop:
		ep_rewards.append(episode_reward)
		average_reward = sum(ep_rewards[-200:])/len(ep_rewards[-200:])
		agg_ep_rewards['ep'].append(episode)
		agg_ep_rewards['avg'].append(average_reward)
		agg_ep_rewards['min'].append(min(ep_rewards[-200:]))
		agg_ep_rewards['max'].append(max(ep_rewards[-200:]))

	if epsilon_decay_end >= episode >= epsilon_decay_start:
		epsilon -= epsilon_decay_value

px.line(agg_ep_rewards, x= 'ep', y= ['avg', 'min', 'max']).show()


env.close()