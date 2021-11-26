from gym.envs.registration import register

register(
	id = 'SGBD-v0',
	entry_point = 'cld771.envs:CustomEnv',
	max_episode_steps = 2000
)