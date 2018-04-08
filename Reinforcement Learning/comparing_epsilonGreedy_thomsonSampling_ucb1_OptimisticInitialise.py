import numpy as np
import matplotlib.pyplot as plt

from comparing_epsilons import Bandit
from ucb1 import run_experiment as run_experiment_ucb1
from optimistic_initial_values import run_experiment as run_experiment_oiv

class BayesianBandit:
    def __init__(self, m):
        self.m = m #true mean
        self.m0 = 0
        self.lambda0 = 1
        self.sumX = 0
        self.tau = 1

    def pull(self):
        return np.random.randn() + self.m

    def sampling(self):
        return np.random.randn() / np.sqrt(self.lambda0) + self.m0

    def update(self, x):
        self.lambda0 += self.tau
        self.sumX += x
        self.m0 = self.tau * self.sumX / self.lambda0

def run_experiment_decaying_epsilon(m1, m2, m3, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    data = np.empty(N)

    for t in range(N):
        p = np.random.random()
        if p < 1.0/(t+1):
            j = np.random.choice(3)
        else:
            j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)
        data[t] = x

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    plt.plot(cumulative_average)
    plt.xscale('log')
    plt.show()

    return cumulative_average

def run_experiment_thomson_sampling(m1, m2, m3, N):
    bandits = [BayesianBandit(m1), BayesianBandit(m2), BayesianBandit(m3)]

    data = np.empty(N)

    for t in range(N):
        j = np.argmax([b.sampling() for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        data[t] = x
    cumulative_average = np.cumsum(data)/(np.arange(N) + 1)

    plt.plot(cumulative_average)
    plt.xscale('log')
    plt.show()

    return cumulative_average

if __name__ == '__main__':
    ts = run_experiment_thomson_sampling(1.0, 2.0, 3.0, 100000)
    oiv = run_experiment_oiv(1.0, 2.0, 3.0, 100000)
    ucb = run_experiment_ucb1(1.0, 2.0, 3.0, 100000)
    deps = run_experiment_decaying_epsilon(1.0, 2.0, 3.0, 100000)

    plt.plot(ts, label='Thompson Sampling')
    plt.plot(oiv, label="Optimistic Initial Values")
    plt.plot(ucb, label='Upper Confidence Bound')
    plt.plot(deps, label='Decaying Epsilon')
    plt.legend()
    plt.xscale('log')
    plt.show()
