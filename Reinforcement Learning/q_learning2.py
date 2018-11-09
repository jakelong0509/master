import gym
import os
import sys
import random
import numpy as np
from gym import wrappers
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

# Q-learning is off-policy therefore update Q by alternative action but still following eps-greedy action
# Q-learning is online update therefore update previous state from current state
min_GAMMA = 0

class FeatureTransform:
    def __init__(self, env, n_components = 500):
        observations = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(observations)

        features = FeatureUnion([
            ("rbf1", RBFSampler(gamma = 5.0, n_components = n_components)),
            ("rbf2", RBFSampler(gamma = 2.5, n_components = n_components)),
            ("rbf3", RBFSampler(gamma = 1.0, n_components = n_components)),
            ("rbf4", RBFSampler(gamma = 0.5, n_components = n_components))
        ])

        # shape = (10000,2000)
        example = features.fit_transform(scaler.transform(observations))
        self.dimension = example.shape[0]
        self.scaler = scaler
        self.features = features

    def transform(self, observation): # shape = (2000,)
        scaled = self.scaler.transform([observation]) # shape = (1,2000)
        return self.features.transform(scaled)

# create Model
# update Q
class Model:
    def __init__(self, env, ft, learning_rate):
        self.env = env
        self.ft = ft
        self.models = []
        for a in range(env.action_space.n):
            model = SGDRegressor(learning_rate = learning_rate)
            model.partial_fit(self.ft.transform(self.env.reset()), [0])
            self.models.append(model)

    def predict(self, observation): # shape = (2000,)
        X = self.ft.transform(observation) # shape = (1,2000)
        return np.array([m.predict(X)[0] for m in self.models])

    def update(self, s, a, G):
        X = self.ft.transform(s)
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps = 0.9):
        random = np.random.random()
        if random < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

def play_one_ep(env, model, eps, gamma):
    observation = env.reset()
    done = False
    rewards = []
    iters = 0
    totalrewards = 0

    while not done and iters < 10000:
        # choose action from eps-greedy
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, _ = env.step(action) # follow behavior action (greedy-action)
                                                        # observation = state after taking action (current_state)

        # update model (update Q)
        # update Q toward alternative action
        G = reward + gamma*np.max(model.predict(observation))
        model.update(prev_observation, action, G) # Q-learning is online update => update previous_state base on current_state

        iters += 1
        totalrewards += reward

    print(gamma)
    return totalrewards

if __name__ == "__main__":
    env = gym.make("Acrobot-v1")
    ft = FeatureTransform(env)
    model = Model(env, ft, "constant")
    gamma = 0.99
    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = 'Videos/' + "q_learning - " + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 1000
    for n in range(N):
        eps = 0.1*(0.97**n)
        totalreward = play_one_ep(env, model, eps, gamma)
        print("episode: ", n, "gamma: ", gamma, "total reward: ", totalreward)
