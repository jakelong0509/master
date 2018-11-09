import gym
import os
import random
import tensorflow as tf
import numpy as np
from gym import wrappers
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.kernel_approximation import RBFSampler

GAMMA = 0.8
lr = 0.05

# Convert observation to feature for Neural Network
class FeatureTransform:
    def __init__(self, env, n_components = 500):
        observations = np.array([env.observation_space.sample() for x in range(10000)])
        actions = np.array([env.action_space.sample() for x in range(10000)])
        obs_act = np.append(observations, actions.reshape((10000,1)), axis=1)
        scaler = StandardScaler()
        scaler.fit(obs_act)

        featurize = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components = n_components)),
            ("rbf2", RBFSampler(gamma=2.5, n_components = n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components = n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components = n_components))
        ])

        example = featurize.fit_transform(scaler.transform(obs_act))
        print(example )
        self.dimension = example.shape[0]
        self.scale = scaler
        self.feature = featurize

    def transform(self, observation, action):
        scaled = self.scale.transform(np.append(observation, action, axis=1))
        return self.feature.transform(scaled)

class HiddenLayer:
    def __init__(self, M1, M2, function = tf.nn.relu):
        self.w = tf.Variable(tf.random_normal(shape=(M1,M2)))
        self.b = tf.Variable(np.zeros(M2))
        self.f = function

    def forward(self, X):
        a = tf.matmul(X, self.w) + self.b
        return self.f(a)

class Model:
    def __init__(self, ft, critic_number_layers, action_number_layers):
        self.feature = ft
        D = self.feature.dimension
        # Create Neural Network for critic
        critic_layers = []
        for i in range(length(critic_number_layers)-1):
            M1 = critic_number_layers[i]
            M2 = critic_number_layers[i+1]
            layer = HiddenLayer(M1,M2)
            critic_layers.append(layer)

        M1 = critic_number_layers[length(critic_number_layers)-1]
        layer = HiddenLayer(M1, 1, tf.nn.softplus) # Critic Last Layer
        critic_layers.append(layer)

        # Create Neural Network for action
        action_layers = []
        for i in range(length(action_number_layers)-1):
            M1 = action_number_layers[i]
            M2 = action_number_layers[i+1]
            layer = HiddenLayer(M1,M2)
            action_layers.append(layer)

        M1 = action_number_layers[length(action_number_layers)-1]
        layer = HiddenLayer(M1,1, tf.nn.softplus) # Action Last layer
        action_layers.append(layer)

        self.X = tf.placeholder(tf.float32, shape=(None,D), name="X")
        self.action = tf.placeholder(tf.float32, shape=(None,), name="action")

        def get_output(layers):
            Z = self.X
            for layer in layers:
                Z = layer.forward(Z)
                return tf.reshape(Z, [-1])

        self.Q_w =
        self.Q_theta =

if __name__ == "__main__":
    env = gym.make("MountainCar-v0")

    FeatureTransform(env)
