import gym
import minerl
import logging
import time

logging.basicConfig(level=logging.DEBUG)
env = gym.make('MineRLBasaltFindCave-v0')


# Note that this command will launch the MineRL environment, which takes time.
# Be patient!
obs = env.reset()

done = False

while not done:
    time.sleep(1)
    # Take a random action
    action = env.action_space.sample()
    # In BASALT environments, sending ESC action will end the episode
    # Lets not do that
    action["ESC"] = 0
    logging.info(action)
    obs, reward, done, _ = env.step(action)
    env.render()