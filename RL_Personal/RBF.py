# _________RBF________
#     RBF = Radial Basis Function
#     Useful in RL
#     2 perspectives
#
#     1) Linear model with feature extraction, where the feature extraction is RBF kernel
#     2) 1-hidden layer neural network, with RBF kernel as activation Function
#
# _________Radial Basis Function_______
#     is a non-normalized Gaussian
#             gamma(x) = exp(-np.linalg.norm(x-c)**2/sigma**2)
#
#     x = input vector
#     c = center / exemplar vector
#     Only denpends on distance between x and c, not direction, hence the term "radial"
#     Max is 1, when x == c, approaches 0 as x goes further away from c
#
#     ? How to choose c ?
#     Number of centers / exemplars == number of hidden units in the RBF Network
#     Each unit will have a different center
#     A few different ways to choose them
#
#         --Support Vector Machines--
#             SVMs also use RBF kernels
#             number of exemplars == number of training points
#             In face, with SVMs the exemplars are the training points
#             This is why SVMs have fallen out of favor
#             Training becomes O(N**2), prediction is O(N), N = number of training samples
#             Important piece of deep learning history
#                 SVMs were once thought to be superior
#         --Another Method--
#             Just sample a few points from the state space
#             Can then choose the number of excemplars
#             env.observation_space.sample()
#             How many exemplars we choose is like how many hidden units in a neural network - it's a hyperparameter that must be tuned
#
# ________Implementation______
#     We'll make use of Sci-Kit learn
#     Our own direct-from-definition implementation would be unnecessarily slow
#     RBFSampler uses a Monte Carlo algorithm
#
#     from sklearn.kernel_approximation import RBFSampler
#
#     Standard interface
#
#     sampler = RBFSampler()
#     sampler.fit(raw_data)
#     features = sampler.transform(raw_data)
#
# ______Old perspective vs. New perspective_____
#     The other perspective: 1-hidden layer neural network
#     In general, this is a nonlinear transformation -> linear model at the final layer
#     Recall: dot product is just a cosine distance: np.dot(a.T, b) = np.abs(a)*np.abs(b)*np.cos(np.angle(a,b))
#
#     Feedforward net: zj = np.tanh(np.dot(Wi.T, x)) = f(cosine dist(Wj,x))
#     RBF net: zj = exp(-np.linalg.norm(x-ci)**2/sigma**2) = f(squared dist(ci, x))
#
# ________Implementation______
#         --Details--
#             Scale parameter (aka. variance)
#             We don't know what scale is good
#             Perhaps multiple scales are good
#             Sci-Kit Learn has facilities that allow us to use multiple RBF samplers simultaneously
#
#                 from sklearn.pipline import FeatureUnion
#
#             Can concatenate any features, not just those from RBFSampler
#             Standarlize our data too:
#
#                 from sklearn.preprocessing import StandardScaler
#
#                 from sklearn.linear_model import SGDRegressor
#
#                 # Functions:
#                 partial_fit(X, Y) # one step of gradient descent
#                 predict(X)
#
#             SGDRegressor behaves a little strangely
#             partial_fit() must be called at least once before we do any prediction
#             Prediction must come before any real fitting, b/c we are using Q_learning (where we need to max over Q(s,a))
#             So we'll start by calling partial_fit with dummy values
#
#                 input = transform(env.reset()), target = 0
#                 model.partial_fit(input, target)
#
#             After calling partial_fit with target 0, it will make all predictins 0 for awhile
#             This is weird - a lincear model shouldn't behave this way (it may not be a purely linear model)
#             This quirk is useful
#             For our next task, Mountain Car, all rewards are -1
#             Therefore, a Q prediction of 0 is higher than anything we can actually get
#             This is the optimistic initial values method
#             Technically don't need epsilon-greedy
#
#         --One model per action--
#             Another implementation detail used by Deep Q Learning too
#             Instead of x <- transform(s,a)
#             We'll use x <- tranform(s)
#             Since actions are discrete, we can have a different Q(s) for every a
#             For Mountain Car, 3 actions: left, right, nothing
#             Neural Netowrk with 3 output nodes
#
#         --Cost-To-Go function--
#             Is the negative of optimal value function V*(s)
#             What they call it in Sutton & Barto
#             2 state variables -> 3-D plot


import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

# SGDRegressor defaults:
# loss = 'squared_loss', penalty='l2', alpha = 0.0001,
# l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True,
# verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling',
# eta0=0.01, power_t=0.25, warm_start=False, average=False

class FeatureTransformer:
    def __init__(self, env, n_components=500):
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)], dtype = np.float64)
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        # concatenate => add collumns => return (10000, n_components*4)
        featurizer = FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
        ])
        example_features = featurizer.fit_transform(scaler.transform(observation_examples))

        self.dimension = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)

class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            model.partial_fit(feature_transformer.transform([env.reset()]), [0])
            self.models.append(model)

    def predict(self, s):
        X = self.feature_transformer.transform([s])
        result = np.stack([m.predict(X) for m in self.models]).T
        assert(len(result.shape) == 2)
        return result

    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

def play_one(model, env, eps, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    first = True
    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        old_observation = observation
        observation, reward, done, info = env.step(action)

        next = model.predict(observation)
        if first:
            print(next)
            first=False
        G = reward + gamma * np.max(next[0])
        model.update(old_observation, action, G)

        totalreward += reward
        iters += 1

    return totalreward

def plot_cost_to_go(env, estimator, num_tiles = 20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x,y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X,Y]))

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title("Cost-To-Go Function")
    fig.colorbar(surf)
    plt.show()

def plot_running_avg(totalreward):
    N = len(totalreward)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalreward[max(0, t-100):(t+1)].mean()

    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

def main(show_plots=True):
    env = gym.make("MountainCar-v0")
    ft = FeatureTransformer(env)
    model = Model(env, ft, "constant")
    gamma = 0.99


    # env = wrappers.Monitor(env, "Videos")


    N = 300
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 0.1*(0.97**n)
        if n == 199:
            print("eps: ", eps)

        totalreward = play_one(model, env, eps, gamma)
        totalrewards[n] = totalreward

        if (n+1) % 100 == 0:
            print("episode: ", n, "total reward:", totalreward)

    print("avg reward for last 100 episodes: ", totalrewards[-100].mean())
    print("total step:", -totalrewards.sum())

    if (show_plots):
        plt.plot(totalrewards)
        plt.title("Rewards")
        plt.show()

        plot_running_avg(totalrewards)
        plot_cost_to_go(env, model)

if __name__ == "__main__":
    main()
