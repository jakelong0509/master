import os
import sys
import gym
import numpy as np
import tensorflow as tf
from gym import wrappers
from datetime import datetime


class Hiddenlayer:
    def __init__(self, M1, M2, function = tf.nn.tanh, bias = True):
        self.function = function
        self.w = tf.Variable(tf.random_normal(shape=(M1, M2)))
        self.params = [self.w]
        self.bias = bias
        if bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
            self.params.append(self.b)




    def forward(self, X):
        if self.bias:
            Z = tf.matmul(X, self.w) + self.b
        else:
            Z = tf.matmul(X, self.w)
        return self.function(Z)

class DQN:
    def __init__(self, D, K, layer_size, gamma, max_experience = 10000, min_experience = 100, history_length = 4, batch_size = 32):
        self.D = D
        self.K = K
        self.layers = []
        # Create hidden layer
        M1 = self.D
        for M2 in layer_size:
            layer = Hiddenlayer(M1, M2)
            self.layers.append(layer)
            M1 = M2
        # Create last layer



        last_layer = Hiddenlayer(M1, self.K, lambda x: x) # last layer will be calculated by tensorflow therefore no need to calculate AL
        self.layers.append(last_layer)

        self.params = []
        for layer in self.layers:
            self.params += layer.params

        self.X = tf.placeholder(tf.float32, shape=(None, self.D), name = "X")
        self.action = tf.placeholder(tf.int32, shape=(None,), name = "Action")
        self.target = tf.placeholder(tf.float32, shape=(None,), name="Target")



        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)

        Y = Z
        self.Y_hat = Y

        selected_action_values = tf.reduce_sum(Y * tf.one_hot(self.action, K), reduction_indices = [1])
        cost = tf.reduce_mean(tf.square(self.target - selected_action_values))
        self.cost = cost
        self.train_op = tf.train.RMSPropOptimizer(0.001, decay=0.9, momentum = 0.0, epsilon = 0.01).minimize(cost)

        self.experiences = {'s': [], 'a': [], 'r': [], 'next_s': [], 'done': []}
        self.max_experience = max_experience
        self.min_experience = min_experience
        self.batch_size = batch_size
        self.gamma = gamma

    def set_session(self, session):
        self.session = session

    def copy_from(self, other):
        # collect all the ops
        ops = []
        my_params = self.params
        other_params = other.params
        for p, q in zip(my_params, other_params):
          actual = self.session.run(q)
          op = p.assign(actual)
          ops.append(op)
        # now run them all
        self.session.run(ops)

    def predict(self, s):
        x = np.atleast_2d(s)
        return self.session.run(self.Y_hat, feed_dict={self.X:x})

    def train(self, nn_target):
        if len(self.experiences['s']) < self.min_experience:
            return

        indexes = np.random.choice(len(self.experiences['s']), size = self.batch_size, replace = False)
        states = [self.experiences['s'][i] for i in indexes]
        actions = [self.experiences['a'][i] for i in indexes]
        rewards = [self.experiences['r'][i] for i in indexes]
        dones = [self.experiences['done'][i] for i in indexes]
        next_state = [self.experiences['next_s'][i] for i in indexes]
        Q_next = np.amax(nn_target.predict(next_state), axis=1)
        targets = [r + self.gamma*q_next if not done else r for r, q_next, done in zip(rewards, Q_next, dones)]
        self.session.run(self.train_op, feed_dict={self.X: states, self.action: actions, self.target: targets})

    def add_experience(self, s, a, r, s2, done):
        if len(self.experiences['s']) >= self.max_experience:
            self.experiences['s'].pop(0)
            self.experiences['a'].pop(0)
            self.experiences['r'].pop(0)
            self.experiences['done'].pop(0)
            self.experiences['next_s'].pop(0)
        self.experiences['s'].append(s)
        self.experiences['a'].append(a)
        self.experiences['r'].append(r)
        self.experiences['done'].append(done)
        self.experiences['next_s'].append(s2)

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            X = np.atleast_2d(s)
            return np.argmax(self.predict(X)[0])

def play_one_epi(env, tmodel, qmodel, gamma, eps, iters, update_frequency = 100):
    totalreward = 0
    observation = env.reset()

    done = False
    count = 0

    while not done:



        action = qmodel.sample_action(observation, eps)

        prev_observation = observation
        observation, reward, done, info = env.step(action)

        qmodel.add_experience(prev_observation, action, reward, observation, done)
        qmodel.train(tmodel)

        iters += 1
        totalreward += reward

        if iters % update_frequency == 0:
            print("Copying parameters from qmodel to tmodel..........")
            tmodel.copy_from(qmodel)

    return totalreward, iters

def main():
    env = gym.make("Acrobot-v1")
    gamma = 0.99
    D = env.reset().shape[0]
    K = env.action_space.n
    sizes = [512]
    qmodel = DQN(D, K, sizes, gamma)
    tmodel = DQN(D, K, sizes, gamma)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    qmodel.set_session(session)
    tmodel.set_session(session)
    iters = 0


    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = 'Videos/' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    epsilon = 1.0
    epsilon_min = 0.1


    N = 100000
    totalrewards = np.empty(N)
    for t in range(N):
        epsilon_change = (epsilon - epsilon_min) / 100
        totalreward, iters = play_one_epi(env, tmodel, qmodel, gamma, epsilon, iters)
        totalrewards[t] = totalreward
        print("Episode: ", t, "eps: ", epsilon, "total reward: ", totalreward, "avg reward (last 100):", totalrewards[max(0, t-100):(t+1)].mean())
        epsilon = max(epsilon - epsilon_change, epsilon_min)


if __name__ == "__main__":
    main()
