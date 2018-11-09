#import packages
import gym
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

GAMMA = 0.8

class Model:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)

    def partial_fit(self, input, Q, lr):
        self.w = lr*(Q - input.dot(self.w))

    def predict(self,s):
        X = s.dot(self.w)
# Convert state to explicit state
# Easier to learn
class ObservationTransform:
    def __init__(self, env, n_components = 500):
        observations = np.array([env.observation_space.sample() for x in range(10000)])
        scale = StandardScaler()
        scale.fit(observations)

        featurizer = FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components = n_components)),
                ("rbf2", RBFSampler(gamma=2.0, n_components = n_components)),
                ("rbf3", RBFSampler(gamma=1.0, n_components = n_components)),
                ("rbf4", RBFSampler(gamma=0.5, n_components = n_components))
        ])

        self.example_feature = featurizer.fit_transform(scale.transform(observations))
        self.dimension = self.example_feature.shape[1]
        print(self.dimension)
        self.scaler = scale
        self.featurizer = featurizer


    def transform(self, observation):
        scaler = self.scaler.transform(observation)
        return self.featurizer.transform(scaler)

def max_dict(d):
  # returns the argmax (key) and max (value) from a dictionary
  # put this into a function since we are using it so often
  max_key = None
  max_val = float('-inf')
  for k, v in d.iteritems():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val

def play_one_ep(env, policy, OT):
    observation = env.reset()
    observation = OT.transform(observation.reshape(1,2))
    a = policy[tuple(observation.reshape(2000,))]

    done = False
    iters = 0
    states_actions_rewards = [(observation,a,0)]
    while not done and iters < 10000:

        observation, r, done, info = env.step(a)
        observation = tuple(OT.transform(observation.reshape(1,2)))
        a = policy[observation]

        if done:
            state_action_reward = (observation, None, r)
        else:
            state_action_reward = (observation, a, r)

        states_actions_rewards.append(state_action_reward)

    G = 0
    states_actions_returns = []
    first = True
    for s,a,r in reversed(states_actions_rewards):
        if first:
            first = False
        else:
            states_actions_returns.append((s,a,G))
        G = r + GAMMA * G
    states_actions_returns.reverse()
    return states_actions_returns

if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    OT = ObservationTransform(env)
    policy = {}
    for s in OT.example_feature:
        policy[tuple(s)] = env.action_space.sample()

    Q = {}
    returns = {}
    states = OT.example_feature
    for s in states:
        Q[tuple(s)] = {}
        for a in range(env.action_space.n):
            Q[tuple(s)][a] = 0
            returns[(tuple(s),a)] = []

    for t in range(2000):
        if t % 100:
            print("Timestep: ", t)

        states_actions_returns = play_one_ep(env, policy, OT)
        biggest_change = 0
        for s,a,G in states_actions_returns:
            s = tuple(s)
            old_q = Q[s][a]
            sa = (s,a)
            returns[sa].append(G)
            Q[s][a] = np.mean(returns[sa])
            biggest_change= max(biggest_change, np.abs(old_q - Q[s][a]))

        for s in states:
            s = tuple(s)
            policy[s] = max_dict(Q[s])[0]
