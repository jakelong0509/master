from builtins import range

import numpy as np
import matplotlib.pyplot as plt
from comparing_epsilons import run_experiment as run_experiment_epsilon

class Bandit:
    def __init__(self, m, upper_limit):
        self.m = m #True mean
        self.mean = upper_limit #Initial estimate mean to be max (Highest value)
        self.N = 1

    def pull(self):
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        self.mean = self.mean + (1.0/self.N) * (x - self.mean)

def run_experiment(m1, m2, m3, N, upper_limit = 10):

    bandits = [Bandit(m1, upper_limit), Bandit(m2, upper_limit), Bandit(m3, upper_limit)]

    data = np.empty(N)

    for i in range(N):
        j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        data[i] = x
    accumulate_average = np.cumsum(data) / (np.arange(N) + 1)

    #plot
    plt.plot(accumulate_average)
    plt.xscale('log')
    plt.show()


    for b in bandits:
        print (b.mean)

    return accumulate_average

if __name__ == '__main__':
    c_1 = run_experiment_epsilon(1.0 , 2.0, 3.0, 0.1, 100000)
    opti = run_experiment(1.0, 2.0, 3.0, 100000)

    plt.plot(c_1, label='eps = 0.1')
    plt.plot(opti, label='Optimistic')
    plt.legend()
    plt.xscale('log')
    plt.show()

    plt.plot(c_1, label='eps = 0.1')
    plt.plot(opti, label='Optimistic')
    plt.legend()
    plt.show()
