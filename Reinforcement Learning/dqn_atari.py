 import gym
import sys
import os
import numpy as np
import tensorflow as tf
from gym import wrappers
from datetime import datetime

GAMMA = 0.99
history_length = 4

class Hiddenlayer:
    def __init__(self, M1, M2, f=tf.nn.relu, bias = True):
        self.W = tf.Variable(tf.random_normal(shape = (M1, M2))) # (516,512)
        self.params = [] # list
        self.params.append(self.W)
        self.f = f
        self.bias = bias
        if bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
            self.params.append(self.b)

    def forward(self, obs):
        if self.bias:
            z = tf.matmul(obs, self.W) + self.b # (32, 516) * (516, 512) = (32,512)
                                                # (32, 512) * (512, 4) = (32,4)
        else:
            z = tf.matmul(obs, self.W)  # (32,512)
                                        # (32,4)
        return self.f(z) # (32,512)
                         # (32,4)


class DQN:
    def __init__(self, layers_size, D, K, max_experience = 1000000, min_experience = 50000, batch_size = 32):
        # layers_size: number of hidden layer
        # D: dimension
        # K: number of action
        self.K = K  # 4
        self.D = D # 128
        self.experiences = {"s":[], "a":[], "r":[], "s2":[], "done":[]}
        self.min_experience = min_experience
        self.max_experience = max_experience
        self.history_length = history_length
        self.batch_size = batch_size
        self.layers = []
        self.flat_size = history_length * self.D  # s = x1, a1, x2, a2, ...., x_t, a_t

        # create layer
        M1 = self.flat_size
        for M2 in layers_size:
            layer = Hiddenlayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # create last layer
        layer = Hiddenlayer(M1, self.K, lambda x: x)
        self.layers.append(layer)

        self.params = []
        for layer in self.layers:
            self.params += layer.params

        self.X = tf.placeholder(tf.float32, shape=(None, history_length, self.D), name="X") # (32, 4, 129) = (32, 516)
        self.G = tf.placeholder(tf.float32, shape=(None,), name="G") # (32,1)
        self.action = tf.placeholder(tf.int32, shape=(None,), name="action")

        X = tf.reshape(self.X, [-1, self.flat_size])
        Z = X
        for layer in self.layers:
            Z = layer.forward(Z)
        self.Y_hat = Z # (32,4)
        self.selected_Q_value = tf.reduce_sum(self.Y_hat * tf.one_hot(self.action, self.K), axis = 1) # (1)

        cost = tf.reduce_mean(tf.square(self.G - self.selected_Q_value))
        self.cost = cost

        self.train = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(cost)

    def set_session(self, session):
        self.session = session

    def predict(self, obs):
        return self.session.run(self.Y_hat, feed_dict = {self.X: obs})

    def sample_action(self, eps, obs):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            obs = [obs]
            return np.argmax(self.predict(obs))

    def learn(self, target_model):
        if len(self.experiences["s"]) < self.min_experience:
            return

        index = np.random.choice(len(self.experiences["s"]), size = self.batch_size, replace = False)
        states = [self.experiences["s"][i] for i in index]
        actions = [self.experiences["a"][i] for i in index]
        rewards = [self.experiences["r"][i] for i in index]
        next_states = [self.experiences["s2"][i] for i in index]
        dones = [self.experiences["done"][i] for i in index]
        next_Qs = np.amax(target_model.predict(next_states), axis = 1)
        targets = [r + np.invert(done).astype(np.float32) * GAMMA * next_Q for r, done, next_Q in zip(rewards, dones, next_Qs)]

        cost, _ = self.session.run([self.cost, self.train], feed_dict = {self.X: states, self.G: targets, self.action: actions})

        return cost

    def add_experience(self, s, a, r, s2, done):
        if len(self.experiences["s"]) > self.max_experience:
            self.experiences["s"].pop(0)
            self.experiences["a"].pop(0)
            self.experiences["r"].pop(0)
            self.experiences["s2"].pop(0)
            self.experiences["done"].pop(0)
        self.experiences["s"].append(s)
        self.experiences["a"].append(a)
        self.experiences["r"].append(r)
        self.experiences["s2"].append(s2)
        self.experiences["done"].append(done)

    def copy_from(self, other):
        ops = []
        my_params = self.params
        other_params = other.params
        for p,q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.session.run(ops)

def run_one_epi(env, tmodel, qmodel, iters, eps, target_update_frequency = 10000):
    observation = env.reset()
    state = np.stack([observation] * history_length, axis=0)
    assert(state.shape == (4,128))
    cost = 0
    totalreward = 0
    update_frequency = 4
    count = 0
    done = False
    first = True
    while not done:
        if iters % target_update_frequency == 0:
            tmodel.copy_from(qmodel)

        if count % update_frequency == 0:
            action = qmodel.sample_action(eps, state)



        observation, reward, done, info = env.step(action)

        next_state = np.append(state[1:], [observation], axis=0)
        assert(next_state.shape == (4,128))


        if len(state) > history_length:
            print("pop")
            state.pop(0)
        qmodel.add_experience(state, action, reward, next_state, done)
        cost = qmodel.learn(tmodel)



        count += 1
        iters += 1
        totalreward += reward

    return totalreward, iters

def main():
    env = gym.make("Breakout-ram-v0")
    D = env.reset().shape[0]
    K = env.action_space.n
    layer_size = [512]
    qmodel = DQN(layer_size, D, K)
    tmodel = DQN(layer_size, D, K)
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
        print("Iter: ", iters)
        epsilon_change = (epsilon - epsilon_min) / 2500
        totalreward, iters = run_one_epi(env, tmodel, qmodel, iters, epsilon)
        totalrewards[t] = totalreward
        print("Episode: ", t, "eps: ", epsilon, "total reward: ", totalreward, "avg reward (last 100):", totalrewards[max(0, t-100):(t+1)].mean())
        epsilon = max(epsilon - epsilon_change, epsilon_min)

def test():
    env = gym.make("CartPole-v1")
    D = env.reset().shape[0]
    K = env.action_space.n
    layer_size = [512]
    qmodel = DQN(layer_size, D, K, max_experience = 10000, min_experience = 100, batch_size = 32)
    tmodel = DQN(layer_size, D, K, max_experience = 10000, min_experience = 100, batch_size = 32)
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
        print("Iter: ", iters)
        epsilon_change = (epsilon - epsilon_min) / 250
        totalreward, iters = run_one_epi(env, tmodel, qmodel, iters, epsilon)
        totalrewards[t] = totalreward
        print("Episode: ", t, "eps: ", epsilon, "total reward: ", totalreward, "avg reward (last 100):", totalrewards[max(0, t-100):(t+1)].mean())
        epsilon = max(epsilon - epsilon_change, epsilon_min)

if __name__ == "__main__":
    if 'test' in sys.argv:
        test()
    else:
        main()
