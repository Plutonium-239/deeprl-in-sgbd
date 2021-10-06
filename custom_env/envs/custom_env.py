import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
	"""docstring for CustomEnv"""
	def __init__(self):
		super(CustomEnv, self).__init__()
		self.i = 1 # no of component tanks
		self.n = 1 # no of blenders
		self.j = 1 # no of product tanks
		self.p = 1 # no of products
		self.o = 1 # no of orders

		self.scheduling_horizon = 192 # (hours) defined schedling horizon as used in the paper [1]
		self.deltaT = 6 # (hours) defined time periods to divide scheduling operations
		self.k = self.scheduling_horizon/self.deltaT # gives number of periods k

		self.px = spaces.Box(low=0, high=1, shape=(self.p, self.i)) # product p’s property index pi

		# comp transfer amt dictionary, access using (i, n) === amt of component transferred from comp tank i to blender n at end of k
		self.f_c = spaces.Box(low=0, high=1000, shape=(self.i, self.n))
		# prod transfer amt dictionary, access using (j, o) === amt of product transferred from prod tank j to order o at end of k
		self.f_p = spaces.Box(low=0, high=1000, shape=(self.j, self.o))
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
		# blending rate in blender n at end of k, access using (n)
		self.cb = spaces.Box(low=0, high=1000, shape=(self.n,))
		# inventory level in prod tank j after time period k, access using (j)
		self.cpt = spaces.Box(low=0, high=1000, shape=(self.j,))

		self.Od = spaces.Box(low=0, high=1000, shape=(self.o,)) # product delivered to order o at end of k, (o)

		self.C = {str(x): x*10 for x in range(self.i)} # unit costs of each component, access by i	of component
		self.cbr = {str(x): x*5 for x in range(self.n)} # unit costs of changeover for blender n, access by n of blender
		self.cp = {str(x): x*5 for x in range(self.j)} # unit costs of changeover for product tank j, access by j of prod tank
		self.cr = 100 # unit tardiness cost 

		self.z = {} # binary, 1 if prod p is assigned to blender n during k, (p,n,k)
		self.x_in = {} # binary, 1 if comp tank i is transferring to blender n during k, (i,n,k)
		self.x_nj = {} # binary, 1 if blender n is transferring to prod tank j during k, (i,n,k)
		self.x_jo = {} # binary, 1 if prod tank j transfers to order o during k
		
		# fixed params/constraints
		self.cc_max = {} # maximum inventory levels for comp i
		self.cc_min = {} # minimum inventory levels for comp i
		self.cpt_max = {} # maximum inventory levels for prod tank j
		self.cpt_min = {} # minimum inventory levels for prod tank j
		self.cb_max = {} # maximum blending rate for blender n
		self.cb_min = {} # minimum blending rate for blender n
		self.f_c_max = {} # maximum flow rate from comp tank i
		self.f_c_min = {} # minimum flow rate from comp tank i
		self.f_p_max = {} # maximum flow rate from blender n
		self.f_p_min = {} # minimum flow rate from blender n

		self.px_max, self.px_min = {}, {} # product p’s maximum and minimum property index pi
		self.y_min, self.y_max = {}, {} # product p’s maximum and minimum composition from each comp tank i
		
		'''	An action will be the 'schedule' for the next period
			basically the values of the binary variables z, x_in, x_nj, x_jo for period k
			A sample action can be:
			(assuming i=3, j=2, n=2, p=2, o=1)
			[]
		'''
		self.action_space = spaces.Tuple(
			spaces.MultiBinary(len(self.p)*len(self.n)),
			spaces.MultiBinary(len(self.i)*len(self.n)),
			spaces.MultiBinary(len(self.n)*len(self.j)),
			spaces.MultiBinary(len(self.j)*len(self.o))
		)
		self.observation_space = spaces.Dict({
			'px': self.px,
			'fc': self.fc,
			'fp': self.fp,
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

	def reset(self):
		pass

	def _reward(self, action):
		pass

	def step(self, action):
		pass

	def render():
		pass
