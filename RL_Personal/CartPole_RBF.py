# ________Modification________
#
#     ----SGDRegressor----
#         We are going to build our own
#         Practice building a linear gradient descent model
#         We'll use the same API as Sci-Kit Learn's SGDRegressor
#         Can't use Optimistic Initial Values
#
#     ----RBF Exemplars----
#         For Mountain Car, sampled from the state space to get exemplars
#         Mountain Car state space is bounded in a small region
#         CartPole state space is not (velocity min = -inf, max = inf)
#         Unfortunately the sample() method samples uniformly from all posible values, not proportional to how likely they are
#         Therefore, these would be poor exemplers
#         Instead, just "quess" a plausible range (or collect data to find it)
#         Use different scales too
#
#             No -> observation_examples = np.array([env.observation_space.sample() for _ in range(20000)])
#             Yes -> observation_examples = np.random.random((20000,4))*2-2


import numpy as np
import gym
import os
import sys
import matplotlib.pyplot as plt


from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from RBF import plot_running_avg
from gym import wrappers

class SGDRegressor:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 0.1

    def partial_fit(self, X, Y):
        # Shape of X = (1,4000)
        # Shape of Y = (1,)
        # Shape of w = (4000,) <=> (1,4000)
        self.w += self.lr*(Y - np.dot(X, self.w)).dot(X)

    def predict(self, X):
        return np.dot(X, self.w)

class FeatureTransformer:
    def __init__(self, n_components = 1000):
        # env.reset().shape = (1,4)
        examples = np.random.random((20000,4))*2 -2
        scaler = StandardScaler()
        scaler.fit(examples)
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=1.0, n_components = n_components)),
            ("rbf2", RBFSampler(gamma=0.5, n_components = n_components)),
            ("rbf3", RBFSampler(gamma=0.25, n_components = n_components)),
            ("rbf4", RBFSampler(gamma=0.1, n_components = n_components))
        ])
        feature_example = featurizer.fit_transform(scaler.transform(examples))
        self.dimension = feature_example.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, s):
        scaler = self.scaler.transform([s])
        return self.featurizer.transform(scaler)

class Model:
    def __init__(self, env, feature_transformer):
        self.models = []
        self.env = env
        self.ft = feature_transformer
        for m in range(env.action_space.n):
            model = SGDRegressor(feature_transformer.dimension)
            self.models.append(model)

    def predict(self, s):
        X = self.ft.transform(s)
        result = np.stack([m.predict(X) for m in self.models]).T
        return result

    def update(self, s, a, G):
        X = self.ft.transform(s)
        self.models[a].partial_fit(X, [G])

    def random_action(self, eps, s):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

def play_one(env, model, eps, gamma):
    observation = env.reset()
    iters = 0
    done = False
    totalreward = 0
    while not done and iters < 10000:
        action = model.random_action(eps, observation)
        old_observation = observation
        observation, reward, done, info = env.step(action)

        Q2 = model.predict(observation)
        G = reward + gamma * np.max(Q2[0])
        model.update(old_observation, action, G)

        totalreward += reward
        iters += 1

    return totalreward

def main():
    env = gym.make("CartPole-v0")
    ft = FeatureTransformer()
    model = Model(env, ft)

    gamma = 0.99
    env = wrappers.Monitor(env, "Videos")

    N = 500
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        totalreward = play_one(env, model, eps, gamma)
        totalrewards[n] = totalreward

        if n % 10 == 0:
            print("Eps: ", eps)
        if (n+1) % 100 == 0:
            print("episode: ", n, "total reward:", totalreward)

    plt.plot(totalrewards)
    plt.show()
    plot_running_avg(totalrewards)

if __name__ == "__main__":
    main()
