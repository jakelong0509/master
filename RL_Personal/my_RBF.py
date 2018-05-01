from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
import gym
import os
import sys


from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import SGDRegressor
from gym import wrappers

# Create Transform class s => x
class FeatureTransformer:
    def __init__(self, env, n_components = 500):
        examples = np.array([env.observation_space.sample() for x in range(10000)], dtype=np.float64)
        scaler = StandardScaler()
        scaler.fit(examples)
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components = n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components = n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components = n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components = n_components))
        ])

        example_features = featurizer.fit_transform(scaler.transform(examples))
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        scaler = self.scaler.transform(observations)
        return self.featurizer.transform(scaler)

class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.ft = feature_transformer
        self.models = []
        for m in range(env.action_space.n):
            model = SGDRegressor(learning_rate = learning_rate)
            model.partial_fit(feature_transformer.transform([env.reset()]), [0])
            self.models.append(model)

    def predict(self, observations):
        X = self.ft.transform([observations])
        result = np.stack([m.predict(X) for m in self.models]).T
        return result

    def update(self, observations, action, G):
        X = self.ft.transform([observations])
        self.models[action].partial_fit(X, [G])

    def random_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

def play_one(env, model, eps, gamma):
    observations = env.reset()
    done = False
    totalreward = 0
    iters = 0
    first = True
    while not done and iters < 10000:
        action = model.random_action(observations, eps)
        old_observations = observations
        observations, reward, done, info = env.step(action)

        Q2 = model.predict(observations)
        G = reward + gamma * np.max(Q2[0])
        model.update(old_observations, action, G)

        totalreward += reward
        iters += 1

    return totalreward

def main():
    env = gym.make("MountainCar-v0")
    ft = FeatureTransformer(env)
    model = Model(env, ft, "constant")

    env = wrappers.Monitor(env, "Videos")

    N = 300
    gamma = 0.99
    totalrewards = np.empty(N)
    for n in range(N):
        # Epsilon will go down in every episode
        # random at first, later pick highest action
        # explore frist, exploit later
        eps = 0.1*(0.97**n)
        totalreward = play_one(env, model, eps, gamma)
        totalrewards[n] = totalreward
        if n % 10 == 0:
            print("Eps: ", eps)
        if (n+1) % 100 == 0:
            print("episode: ", n, "total reward:", totalreward)

    plt.plot(totalrewards)
    plt.show()

if __name__ == "__main__":
    main()
