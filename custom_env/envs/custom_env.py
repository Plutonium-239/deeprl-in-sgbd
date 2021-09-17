import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
	"""docstring for CustomEnv"""
	def __init__(self):
		super(CustomEnv, self).__init__()
		self.i = [] # list of component tanks
		self.n = [] # list of blenders
		self.j = [] # list of product tanks
		self.o = [] # list of orders

		self.scheduling_horizon = 192 # (hours) defined schedling horizon as used in the paper [1]
		self.deltaT = 6 # (hours) defined time periods to divide scheduling operations
		self.k = self.scheduling_horizon/self.deltaT # gives number of periods k

		# comp transfer amt dictionary, access using (n, i, k) === amt of component transferred from comp tank i to blender n at end of k
		self.f_c = {}
		# prod transfer amt dictionary, access using (j, o, k) === amt of product transferred from prod tank j to order o at end of k
		self.f_p = {}
		self.ct_blenders = {} # changeover amt in blender n at end of k, access using (n, k)
		self.ct_prodtank = {} # changeover amt in product tank j at end of k, access using (j, k)
		self.T = {} # tardiness amount of order o, access using o

		self.C = {} # unit costs of each component, access by i	of component
		self.cbr = {} # unit costs of changeover for blender n, access by n of blender
		self.cp = {} # unit costs of changeover for product tank j, access by j of prod tank
		self.cr = 100 # unit tardiness cost 
		

		self.action_space = spaces.Box()

	def reset():
		pass

	def step():
		pass

	def render():
		pass
