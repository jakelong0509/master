import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from gym import wrappers
from datetime import datetime



class SGDRegressor:
    def __init__(self, dimension):
        self.w = np.random.randn(dimension) / np.sqrt(dimension)
        self.lr = 1e-2

    def partial_fit(self, X, Y):
        self.w += self.lr * (Y - np.dot(X, self.w)).dot(X)

    def predict(self, X):
        return np.dot(X, self.w)

class FeatureTransformer:
    def __init__(self, env, n_components = 500):
        examples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(examples)

        featurizer = FeatureUnion([
            ("name1", RBFSampler(gamma=5.0, n_components = n_components)),
            ("name2", RBFSampler(gamma=2.0, n_components = n_components)),
            ("name3", RBFSampler(gamma=1.0, n_components = n_components)),
            ("name4", RBFSampler(gamma=0.5, n_components = n_components))
        ])

        fit_examples = featurizer.fit_transform(scaler.transform(examples))
        self.dimension = fit_examples.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, s):
        scaler = self.scaler.transform(np.atleast_2d(s))
        return self.featurizer.transform(scaler)

class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.ft = feature_transformer
        self.models = []
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

    def random_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


def play_one_episode(model, eps, gamma, n = 5):
    observation = env.reset()
    iters = 0
    done = False
    totalreward = 0
    rewards = []
    states = []
    actions = []
    multiplier = np.array([gamma]*n)**np.arange(n)
    while not done and iters < 200:
        action = model.random_action(observation, eps)

        states.append(observation)
        actions.append(action)

        old_observation = observation
        observation, reward, done, info = env.step(action)

        rewards.append(reward)

        if len(rewards) >= n:
            Q2 = model.predict(observation)
            G = np.dot(multiplier, rewards[-n:]) + (gamma**n) * np.max(Q2[0])
            model.update(states[-n], actions[-n], G)
        totalreward += reward
        iters += 1

    if n == 1: # Only look 1 step a head TD(0)
        rewards = []
        actions = []
        states = []
    else:
        rewards = rewards[-n+1:]
        actions = actions[-n+1:]
        states = states[-n+1:]

    # After the loop e.g. 10 steps => reached the goal
    # n = 5 => done at ithers = 9 is True => loop stop
    # only updated states that 5 [0,1,2,3,4] steps ago => [5,6,7,8,9] states were not updated
    if observation[0] >= 0.5:
        print("Reached the goal State")
        while len(rewards) > 0:
            G = np.dot(multiplier[:len(rewards)], rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            actions.pop(0)
            states.pop(0)
    else:
        print("No reached the goal State")
        while len(rewards) > 0:
            quess_rewards = rewards + [-1]*(n-len(rewards))
            G = np.dot(multiplier, quess_rewards)
            model.update(states[0], actions[0], G)
            rewards.pop(0)
            actions.pop(0)
            states.pop(0)

    return totalreward

if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    ft = FeatureTransformer(env)
    model = Model(env, ft)
    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = 'Videos/' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)
    gamma = 0.99
    N = 300
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 1.0/(0.1*n+1)
        totalreward = play_one_episode(model, eps, gamma)
        totalrewards[n] = totalreward
        print("episode:", n, "total reward:", totalreward, "eps: ", eps)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", -totalrewards.sum())

    plt.plot(totalrewards)
    plt.show()
