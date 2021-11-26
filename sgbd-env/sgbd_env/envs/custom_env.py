import gym
from gym import spaces
import numpy as np
from sgbd_env.envs import utils

class SGBDEnv(gym.Env):
	"""docstring for CustomEnv"""
	def __init__(self, path=None, products=range(4), distr='uniform'):
		super(SGBDEnv, self).__init__()
		self.scheduling_horizon = 192 # (hours) defined schedling horizon as used in the paper [1]
		self.deltaT = 6 # (hours) defined time periods to divide scheduling operations
		self.k = self.scheduling_horizon/self.deltaT # gives number of periods k
		self.current_period = 0 # keep track of time

		self.i = 4 # no of component tanks
		self.n = 3 # no of blenders
		self.j = 2 # no of product tanks
		if products:
			self.p = len(products) # no of products
		else:
			self.p = 0
		
		self.o = int(self.k) # no of orders ; should be an integer anyways

		# product p’s property index pi
		# self.px =  {}

		# # comp transfer amt dictionary, access using (i, n) === amt of component transferred from comp tank i to blender n at end of k
		# self.f_c = spaces.Box(low=0, high=1000, shape=(self.i, self.n))
		# # prod transfer amt dictionary, access using (j, o) === amt of product transferred from prod tank j to order o at end of k
		# self.f_p = spaces.Box(low=0, high=1000, shape=(self.j, self.o))
		# amt of comp i to blender n at end of k
		self.b = np.zeros(shape=(self.i, self.n), dtype=np.float64)
		# flow rate from blender n to prod tank j at end of k
		self.x = np.zeros(shape=(self.n, self.j), dtype=np.float64)
		# changeover amt in blender n at end of k, access using (n)
		self.ct_blenders = np.zeros(shape=(self.n,), dtype=np.float64)
		# changeover amt in product tank j at end of k, access using (j)
		self.ct_prodtank = np.zeros(shape=(self.j,), dtype=np.float64)
		# tardiness amount of order o, access using o
		self.T = np.zeros(shape=(self.o,), dtype=np.float64)
		# inventory level in comp i after time period k, access using (i)
		self.cc = np.full(shape=(self.i,), fill_value=1000, dtype=np.float64)
		# blending amt(per time period = rate) in blender n at end of k, access using (n)
		self.cb = np.zeros(shape=(self.n,), dtype=np.float64)
		# inventory level in prod tank j after time period k, access using (j)
		self.cpt = np.zeros(shape=(self.j,), dtype=np.float64)

		self.Od = np.zeros(shape=(self.o,), dtype=np.float64) # product delivered to order o at end of k, (o)

		self.C = np.array([(x+1)*10.0 for x in range(self.i)]) # unit costs of each component, access by i of component
		self.Cbr = np.array([(x+1)*5.0 for x in range(self.n)]) # unit costs of changeover for blender n, access by n of blender
		self.Cp = np.array([(x+1)*5.0 for x in range(self.j)]) # unit costs of changeover for product tank j, access by j of prod tank
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

		# self.px_max, self.px_min = {}, {} # product p’s maximum and minimum property index pi
		# self.y_min, self.y_max = {}, {} # product p’s maximum and minimum composition from each comp tank i
		
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
		self.action_space = spaces.Tuple((
			spaces.MultiBinary((self.i,self.n)),
			spaces.MultiBinary((self.n,self.j)),
			spaces.MultiBinary((self.j,self.o))
		))
		self.observation_space = spaces.Dict({
			# 'px': spaces.Box(low=0, high=1, shape=(self.p, self.i)), # product p’s property index pi
			# 'fc': self.fc,
			# 'fp': self.fp,
			'b': spaces.Box(low=0, high=1000, shape=(self.i, self.n)),
			'x': spaces.Box(low=0, high=1000, shape=(self.n, self.j)),
			'ct_blenders': spaces.Box(low=0, high=1000, shape=(self.n,)),
			'ct_prodtank': spaces.Box(low=0, high=1000, shape=(self.j,)),
			'T': spaces.Box(low=0, high=self.k, shape=(self.o,)),
			'cc': spaces.Box(low=0, high=1000, shape=(self.i,)),
			'cb': spaces.Box(low=0, high=1000, shape=(self.n,)),
			'cpt': spaces.Box(low=0, high=1000, shape=(self.j,)),
			'Od': spaces.Box(low=0, high=1000, shape=(self.o,))
		})

		# no longer assuming amount to be fixed, amount varies with every order
		self.amt = 100 # (only for development stages) assume a fixed amount is transferred
		self.action_memory = []
		self.order = self._recover_order(path, products, distr)
		self.px = self._recover_composition(path, products, distr)
		if self.p == 0:
			self.p = len(self.order)
		self.order['completed'] = False*len(self.order)
		self.order['late'] = False*len(self.order)
		self.completed_old = np.zeros(self.o)
		self.completed = np.zeros(self.o)

	def reset(self):
		pass

	def _recover_order(self, path, products, distr):
		if path == 'new':
			return utils.generate_order(self.k, products=products, distr=distr)
		elif path:
			return utils.read_order(path)
		return utils.read_order('sgbd-env/sgbd_env/envs/generated_order_uniform.csv')

	def _recover_composition(self, path, products, distr):
		if path == 'new':
			return utils.generate_product_compositions(self.i, products=products, distr=distr)
		elif path:
			return utils.read_composition(path)
		return utils.read_composition('sgbd-env/sgbd_env/envs/product_compositions_uniform.csv')

	def _reward(self):
		comp_costs = (self.C*self.b.sum(axis=1)).sum()
		# changeover
		chng_b, chng_b_costs,chng_j, chng_j_costs = 0,0,0,0
		if self.current_period >= 1:
			chng_b = (self.last_action[0] - self.action[0]).reshape(self.p, self.n).sum(axis = 1)
			self.ct_blenders = chng_b
			chng_b_costs = (chng_b*self.amt*self.Cbr).sum()
			chng_j = (self.last_action[2] - self.action[2]).reshape(self.j, self.o).sum(axis = 1)
			self.ct_prodtank = chng_j
			chng_j_costs = (chng_j*self.amt*self.Cp).sum()
		tardiness_costs = (self.T*self.Cr).sum()
		earned_margins = ((self.completed - self.completed_old)*self.order['margin']).sum()
		self.completed_old = self.completed
		return earned_margins - (comp_costs + chng_b_costs + chng_j_costs + tardiness_costs)

	def step(self, action):
		''' Returns
		-------
		state, reward, episode_over, info
			state : an environment-specific object representing your observation of
				the environment.
			reward : amount of reward achieved by the previous action. The scale
				varies between environments, but the goal is always to increase
				the total reward.
			done : whether it's time to reset the environment again. Most (but not
				all) tasks are divided up into well-defined episodes, and done
				being True indicates the episode has terminated.
			info : diagnostic information useful for debugging. It can sometimes
				be useful for learning (for example, it might contain the raw
				probabilities behind the environment's last state change).
				However, official evaluations of the agent are not allowed to
				use this for learning.
		'''
		self.last_action = self.action_memory[-1] if len(self.action_memory) > 0 else None
		self.action = action
		self.action_memory.append(action)
		self._take_action(action)
		reward = self._reward()
		state, done = self._get_state() # tbd
		self.current_period += 1
		return state, reward, done, {}

	def _take_action(self, action):
		amt = 100.0
		i_n, n_j, j_o = action # splitting action into 4 one-hot np arrays
		# i_n = i_n.reshape(self.i, self.n) # so that i_n[i] refers to comp tank i and i_n[:,n] refers to blender n
		# self.b = amt*i_n # REMOVE
		self.cc -= amt*i_n.sum(axis=1) # component goes out of comp tanks
		self.cb += amt*i_n.sum(axis=0) # components go into blender
		# self.x = amt*n_j # REMOVE
		self.cb -= amt*n_j.sum(axis=1) # blended components leave blender
		self.cpt += amt*n_j.sum(axis=0) # blended components enter product tanks
		self.cpt -= amt*j_o.sum(axis=1) # products leave product tanks
		self.Od += amt*j_o.sum(axis=0) # products enter order tanks (wrt their order id)
		# Check complete orders - in j_o wherever column sum = 1 -> order has been completed 
		filled = self.Od
		completed = (self.order['amount'] == filled)
		# print(completed)
		# print(completed.any())
		if completed.any():
			self.order.loc[completed, 'completed'] = True
			self.completed[completed] = 1
		late = (self.order['due_date'] < self.current_period)
		# print(late)
		if late.any():
			self.order.loc[late, 'late'] = True
			self.T[late] += 1
		
	def _get_state(self):
		state = {
			'current_period': self.current_period,
		}
		if self.order['completed'].all():
			return state, 1
		elif self.current_period == self.k:
			return state, -1
		return state, 0

	def render():
		pass
