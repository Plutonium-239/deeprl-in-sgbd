import gym
from gym import spaces
import numpy as np
import utils

class CustomEnv(gym.Env):
	"""docstring for CustomEnv"""
	def __init__(self, path=None, products=None, distr=None):
		super(CustomEnv, self).__init__()
		self.i = 1 # no of component tanks
		self.n = 1 # no of blenders
		self.j = 1 # no of product tanks
		self.p = len(products) # no of products
		self.o = 1 # no of orders

		self.scheduling_horizon = 192 # (hours) defined schedling horizon as used in the paper [1]
		self.deltaT = 6 # (hours) defined time periods to divide scheduling operations
		self.k = self.scheduling_horizon/self.deltaT # gives number of periods k

		self.px = spaces.Box(low=0, high=1, shape=(self.p, self.i)) # product p’s property index pi

		# # comp transfer amt dictionary, access using (i, n) === amt of component transferred from comp tank i to blender n at end of k
		# self.f_c = spaces.Box(low=0, high=1000, shape=(self.i, self.n))
		# # prod transfer amt dictionary, access using (j, o) === amt of product transferred from prod tank j to order o at end of k
		# self.f_p = spaces.Box(low=0, high=1000, shape=(self.j, self.o))
		# amt of comp i to blender n at end of k
		self.b = spaces.Box(low=0, high=1000, shape=(self.i, self.n))
		# flow rate from blender n to prod tank j at end of k
		self.x = spaces.Box(low=0, high=1000, shape=(self.i, self.n))
		# changeover amt in blender n at end of k, access using (n)
		self.ct_blenders = spaces.Box(low=0, high=1000, shape=(1,self.n))
		# changeover amt in product tank j at end of k, access using (j)
		self.ct_prodtank = spaces.Box(low=0, high=1000, shape=(self.j,))
		# tardiness amount of order o, access using o
		self.T = spaces.Box(low=0, high=self.k, shape=(self.o,))
		# inventory level in comp i after time period k, access using (i)
		self.cc = spaces.Box(low=0, high=1000, shape=(self.i,))
		# blending amt(per time period = rate) in blender n at end of k, access using (n)
		self.cb = spaces.Box(low=0, high=1000, shape=(self.n,))
		# inventory level in prod tank j after time period k, access using (j)
		self.cpt = spaces.Box(low=0, high=1000, shape=(self.j,))

		self.Od = spaces.Box(low=0, high=1000, shape=(self.o,)) # product delivered to order o at end of k, (o)

		self.C = np.array([x*10 for x in range(self.i)]) # unit costs of each component, access by i of component
		self.Cbr = np.array([x*5 for x in range(self.n)]) # unit costs of changeover for blender n, access by n of blender
		self.Cp = np.array([x*5 for x in range(self.j)]) # unit costs of changeover for product tank j, access by j of prod tank
		self.Cr = 100 # unit tardiness cost 

		# self.z = {} # binary, 1 if prod p is assigned to blender n during k, (p,n,k)
		# self.x_in = {} # binary, 1 if comp tank i is transferring to blender n during k, (i,n,k)
		# self.x_nj = {} # binary, 1 if blender n is transferring to prod tank j during k, (i,n,k)
		# self.x_jo = {} # binary, 1 if prod tank j transfers to order o during k
		
		# fixed params/constraints
		# self.cc_max = {} # maximum inventory levels for comp i
		# self.cc_min = {} # minimum inventory levels for comp i
		# self.cpt_max = {} # maximum inventory levels for prod tank j
		# self.cpt_min = {} # minimum inventory levels for prod tank j
		# self.cb_max = {} # maximum blending rate for blender n
		# self.cb_min = {} # minimum blending rate for blender n
		# self.f_c_max = {} # maximum flow rate from comp tank i
		# self.f_c_min = {} # minimum flow rate from comp tank i
		# self.f_p_max = {} # maximum flow rate from blender n
		# self.f_p_min = {} # minimum flow rate from blender n

		self.px_max, self.px_min = {}, {} # product p’s maximum and minimum property index pi
		self.y_min, self.y_max = {}, {} # product p’s maximum and minimum composition from each comp tank i
		
		'''	An action will be the 'schedule' for the next period
			basically the values of the binary variables z, x_in, x_nj, x_jo for period k
			A sample action can be:
			(assuming i=3, j=2, n=2, p=2, o=1)
			[]
			basically [0, 1, 1, 0]--i1
					  [1, 1, 0, 0]--i2
					   |  |  |  |
					   n1 n2 n3 n4 
		'''
		self.action_space = spaces.Tuple(
			spaces.MultiBinary((len(self.p),len(self.n))),
			spaces.MultiBinary((len(self.i),len(self.n))),
			spaces.MultiBinary((len(self.n),len(self.j))),
			spaces.MultiBinary((len(self.j),len(self.o)))
		)
		self.observation_space = spaces.Dict({
			'px': self.px,
			# 'fc': self.fc,
			# 'fp': self.fp,
			'b': self.b,
			'x': self.x,
			'ct_blenders': self.ct_blenders,
			'ct_prodtank': self.ct_prodtank,
			'T': self.T,
			'cc': self.cc,
			'cb': self.cb,
			'cpt': self.cpt,
			'Od': self.Od
		})

		# no longer assuming amount to be fixed, amount varies with every order
		self.amt = 100 # (only for development stages) assume a fixed amount is transferred
		self.action_memory = []
		self.order = self._recover_order(path, products, distr)

	def reset(self):
		pass

	def _recover_order(self, path, products, distr):
		if path:
			return utils.read_order(path)
		return utils.generate_order(self.k, products=products, distr=distr)

	def _reward(self):
		comp_costs = (self.C*self.b.sum(axis=1)).sum()
		# changeover
		chng_b = (self.last_action[0] - self.action[0]).reshape(self.p, self.n).sum(axis = 1)
		chng_b_costs = (chng_b*self.amt*self.Cbr).sum()
		chng_j = (self.last_action[3] - self.action[3]).reshape(self.j, self.o).sum(axis = 1)
		chng_j_costs = (chng_j*self.amt*self.Cp).sum()
		tardiness_costs = (self.T*self.Cr).sum()
		return -(comp_costs + chng_b_costs + chng_j_costs + tardiness_costs)

	def step(self, action):
		''' Returns
		-------
		state, reward, episode_over, info
			state : an environment-specific object representing your observation of
				the environment.
			reward : amount of reward achieved by the previous action. The scale
				varies between environments, but the goal is always to increase
				your total reward.
			episode_over : whether it's time to reset the environment again. Most (but not
				all) tasks are divided up into well-defined episodes, and done
				being True indicates the episode has terminated. (For example,
				perhaps the pole tipped too far, or you lost your last life.)
			info : diagnostic information useful for debugging. It can sometimes
				be useful for learning (for example, it might contain the raw
				probabilities behind the environment's last state change).
				However, official evaluations of your agent are not allowed to
				use this for learning.
		'''
		self.last_action = self.action_memory[-1] if len(self.action_memory) > 0 else None
		self.action = action
		self.action_memory.append(action)
		self._take_action(action)
		reward = self._reward()
		state, episode_over = self._get_state() # tbd
		return state, reward, episode_over, {}

	def _take_action(self, action):
		p_n, i_n, n_j, j_o = action # splitting action into 4 one-hot np arrays
		# i_n = i_n.reshape(self.i, self.n) # so that i_n[i] refers to comp tank i and i_n[:,n] refers to blender n
		self.b = self.amt*i_n
		self.cc -= self.amt*i_n.sum(axis=1)
		self.cb = self.b.sum(axis=0)
		self.x = self.amt*n_j
		self.cpt += self.x.sum(axis=0)
		# self.Od +=
		# 
		# 

	def render():
		pass
