import gymnasium as gym
import pygame
import numpy as np

# This is the visualization of the Taxi problem

env = gym.make('Taxi-v3', render_mode="human", max_episode_steps=15)
starting_state = env.reset()

if_finished = None
if_truncated = None
info = None
while not if_finished and not if_truncated:
    env.render()
    if not info:
        action = np.random.choice((np.where(starting_state[1]['action_mask'] == 1))[0])
    else:
        action = np.random.choice((np.where(info['action_mask'] == 1))[0])

    next_state, reward, if_finished, if_truncated, info = env.step(action)
    print("Aktualnie można wykonać akcje: " + str(info['action_mask']))
    pygame.time.delay(2_000)

env.render()
env.close()
