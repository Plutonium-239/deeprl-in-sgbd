from sgbd_env.envs import SGBDEnv
import gym
from stable_baselines3 import A2C

env = SGBDEnv()
env.action_space = gym.spaces.MultiBinary(env.i*(env.n**2)*(env.j**2)*env.o)
model = A2C('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=env.k)

env = SGBDEnv()
env.action_space = gym.spaces.MultiBinary(env.i*(env.n**2)*(env.j**2)*env.o)
for i in range(100):
	action, _state = model.predict(env, deterministic=True)
	env, reward, done, _ = env.step(action)
	if done:
		env = SGBDEnv()
		env.action_space = gym.spaces.MultiBinary(env.i*(env.n**2)*(env.j**2)*env.o)