from builtins import range
from comparing_epsilons import run_experiment as run_experiment_eps

import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, m):
        self.m = m #true mean
        self.mean = 0 #estimate mean
        self.N = 0 #number of times pulling bandit j

    def pull(self):
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        self.mean = self.mean + (1.0/self.N) * (x - self.mean)

def ucb(mean, t, Nt):
    if Nt == 0:
        return float('inf')
    return mean + np.sqrt((2*np.log(t))/Nt)

def run_experiment(m1, m2, m3, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    data = np.empty(N)

    for t in range(N):
        j = np.argmax([ucb(b.mean, t, b.N) for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        data[t] = x

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)



    plt.plot(cumulative_average)
    plt.xscale('log')
    plt.show()

    for b in bandits:
        print (b.mean)

    return cumulative_average

if __name__ == '__main__':
    c_1 = run_experiment_eps(1.0, 2.0, 3.0, 0.1, 100000)
    ucb = run_experiment(1.0, 2.0, 3.0, 100000)

    plt.plot(c_1, label='epsilon = 0.1')
    plt.plot(ucb, label='Upper Confidence Bounds')
    plt.legend()
    plt.xscale('log')
    plt.show()

    plt.plot(c_1, label='epsilon = 0.1')
    plt.plot(ucb, label='Upper Confidence Bounds')
    plt.legend()
    plt.show()
