# CartPole state space is continuous / infinite
# But some states are more likely than others
# From the documentation, can infer that some are impossible,
# since if you go pass a certain angle/position, you lose
# Very high velocity is likely impossible as well
# Cut up relevant part of the state space into boxes
# Now we have a discrete state space
# We just name these 0,1, etc.
# Not we can use the tabular method

# ______Hidden Complexities______
# How to we choose upper and lower limits of the box?
#       .Naive approach: just try different numbers until it works
#       .More complex: play some episodes, plot hisstograms to see what limits are
# What if we land in a state outside the box?
#       .Extend the edge boxes to infinity

# ________Implementation______
# Convert state into bin
# Must be a unique number, so we can index a dictionary or array
# Not easy, take time to test out a few ideas

# _________Overwriting rewards______
# Default rewards is +1 for every step
# Doesn't work too well(but makes perfect sense)
# Better: give a large negative reward(-300) if the pole falls
# Incentivizes the agent to not reah that point
# Is modifying the default rewads a good idea or not?
# In the real-world, if an agent is builded to solve a novel task, you would be defining the rewads anyway
# The programmer must define an intelligent reward structure

# ########## CODE ############

import gym
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gym import wrappers
from datetime import datetime

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]


class FeatureTranformer:
    def __init__(self):
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velocity_bins = np.linspace(-2, 2, 9)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9)

    def transform(self, observation):
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        return build_state([
            to_bin(cart_pos, self.cart_position_bins),
            to_bin(cart_vel, self.cart_velocity_bins),
            to_bin(pole_angle, self.pole_angle_bins),
            to_bin(pole_vel, self.pole_velocity_bins)
        ])


class Model:
    def __init__(self, env, features_transform):
        self.env = env
        self.features_transform = features_transform

        self.num_spaces = 10**env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.Q = np.random.uniform(low=-1.0, high=1.0, size=(self.num_spaces, self.num_actions))

    def predict(self, s):
        x = self.features_transform.transform(s)
        return self.Q[x]

    def update(self, s, a, G):
        lr = 1e-02
        x = self.features_transform.transform(s)
        self.Q[x,a] += lr * (G - self.Q[x,a])

    def random_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            p = self.predict(s)
            return np.argmax(p)

def play_one(model, eps, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iter = 0
    while not done and iter < 10000:
        action = model.random_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        totalreward += reward

        if done and iter < 199:
            reward = -300

        G = reward + gamma * np.max(model.predict(observation))
        model.update(prev_observation, action, G)

        iter += 1

    return totalreward

def plot_running_avg(totalreward):
    N = len(totalreward)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalreward[max(0, t-100):(t+1)].mean()

    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    ft = FeatureTranformer()
    model = Model(env, ft)
    gamma = 0.9

    # if 'monitor' in sys.argv:
    #     filename = os.path.basename(__file__).split('.')[0]
    #     monitor_dir = './' + filename + '_' + str(dateime.now())
    #     env = wrappers.Monitor(env, monitor_dir)

    N = 10000
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        totalreward = play_one(model, eps, gamma)
        totalrewards[n] = totalreward

        if n % 100 == 0:
            print("Episode: %d, total reward: %d, eps: %f" % (n, totalreward, eps))
    env = wrappers.Monitor(env, "Videos")
    print("avg reward for last 100 episodes: ", totalrewards[-100].mean())
    print("total steps: ", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)
