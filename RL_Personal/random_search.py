# Pseudocode:
# For # of times I want to adjust the weights
#     new_weights = random
#     For # episodes I want to play to decide whether to update the weights
#         play episodes
#     if avg episode length > best so far:
#         weights = new_weights
# Play a final set of episodes to see how good my best weights do again


from __future__ import print_function, division
from builtins import range
from gym import wrappers

import gym
import numpy as np
import matplotlib.pyplot as plt

def get_action(s,w):
    return 1 if np.dot(s,w) > 0 else 0

def play_one_episode(env, params):
    # reset environment to start new episode
    observation = env.reset()
    done = False
    t = 0

    while not done and t < 10000:
        # environment #render# function allow to see video of the episode
        # env.render()
        t += 1
        action = get_action(observation, params)
        # the environment's #step# function returns
        # observation(object), reward(float), done(boolean), info(dict)[for debugging]
        observation, reward, done, info = env.step(action)
        if done:
            break

    return t

# the point of this function is to keep track of
# all the episode length for these parameters and then
# return average
def play_multiple_episodes(env, T, params):
    # numpy #empty# function return a new array of given shape and type, without initializing entries
    episode_lengths = np.empty(T)

    for i in range(T):
        episode_lengths[i] = play_one_episode(env,params)

    avg_length = episode_lengths.mean()
    print("avg length:", avg_length)
    return avg_length

def random_search(env):
    episode_lengths = []
    best = 0
    params = None
    for t in range(100):
        new_params = np.random.random(4)*2-1
        avg_length = play_multiple_episodes(env, 100, new_params)
        episode_lengths.append(avg_length)

        if avg_length > best:
            params = new_params
            best = avg_length
    return episode_lengths, params

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    episode_lengths, params = random_search(env)
    plt.plot(episode_lengths)
    plt.show()

    # play final set of episodes
    env = wrappers.Monitor(env, 'Videos')
    print("###Final run with final wrights@@@")
    play_multiple_episodes(env, 100, params)
