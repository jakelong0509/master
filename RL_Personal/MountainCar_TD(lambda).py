import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from datetime import datetime
from gym import wrappers

class TDLambda:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 1e-2

    def partial_fit(self, target, eligibility):
        self.w += self.lr * target * eligibility

    def predict(self, X):
        return np.dot(X, self.w)

class FeatureTransformer:
    def __init__(self, env, n_components = 500):
        examples = np.array([env.observation_space.sample() for _ in range(10000)])
        scaler = StandardScaler()
        scaler.fit(examples)

        featurizer = FeatureUnion([
            ("td1", RBFSampler(gamma=5.0, n_components = n_components)),
            ("td2", RBFSampler(gamma=2.0, n_components = n_components)),
            ("td3", RBFSampler(gamma=1.0, n_components = n_components)),
            ("td4", RBFSampler(gamma=0.5, n_components = n_components))
        ])

        obs_examples = featurizer.fit_transform(scaler.transform(examples))

        self.dimension = obs_examples.shape[1]
        self.featurizer = featurizer
        self.scaler = scaler

    def transform(self, s):
        scaled = self.scaler.transform(np.atleast_2d(s))
        return self.featurizer.transform(scaled)

class Model:
    def __init__(self, env, feature_transformer):
        dimension = feature_transformer.dimension
        self.models = []
        self.env = env
        self.ft = feature_transformer
        self.eligibilities = np.zeros((env.action_space.n, dimension))
        for m in range(env.action_space.n):
            model = TDLambda(dimension)
            self.models.append(model)

    def predict(self, s):
        X = self.ft.transform(s)
        result = np.stack([m.predict(X) for m in self.models]).T
        return result

    def update(self, s, action, target, gamma, lambda_):
        X = self.ft.transform(s)
        self.eligibilities[action] = gamma*lambda_*self.eligibilities[action] + X
        self.models[action].partial_fit(target, self.eligibilities[action])

    def random_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

def play_one_episode(env, model, gamma, eps, lambda_):
    observation = env.reset()
    totalreward = 0
    iters = 0
    done = False

    while not done and iters < 200:
        action = model.random_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        q2 = model.predict(observation)
        q1 = model.predict(prev_observation)

        target = reward + gamma * np.max(q2[0]) - np.max(q1[0])
        model.update(prev_observation, action, target, gamma, lambda_)

        totalreward += reward
        iters += 1

    return totalreward

if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    ft = FeatureTransformer(env)
    model = Model(env, ft)

    gamma = 0.999
    lambda_ = 0.7
    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = 'Videos/' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)
    N = 300
    totalrewards = np.empty(N)

    for n in range(N):
        eps = 1.0/(0.1+n+1)
        totalreward = play_one_episode(env, model, gamma, eps, lambda_)
        totalrewards[n] = totalreward
        print("episode:", n, "total reward:", totalreward)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", -totalrewards.sum())

    plt.plot(totalrewards)
    plt.show()
