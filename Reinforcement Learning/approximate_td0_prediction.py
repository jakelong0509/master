import numpy as np
import matplotlib.pyplot as plt

from policy_iterative_evaluation import print_policy, print_values
from grid_world import negative_grid, standard_grid

ALPHA = 0.1
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


class Model:
    def __init__(self):
        self.theta = np.random.randn(4) / 2

    def s2x(self,s):
        return np.array([s[0] - 1, s[1] - 1.5, s[0] * s[1] - 3, 1])

    def predict(self, s):
        x = self.s2x(s)
        return self.theta.dot(x)

    def grad(self, s):
        return self.s2x(s)

def random_action(a, eps = 0.1):
    p = np.random.random()

    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

def play_game(policy, grid):
    s = (2,0)
    grid.set_state(s)

    states_and_rewards = [(s,0)]

    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s,r))

    return states_and_rewards

if __name__ == '__main__':
    grid = standard_grid()

    print ("Rewards: ")
    print_values(grid.rewards, grid)

    states = grid.all_states()

    #initialize Policy
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',

    }

    model = Model()
    deltas = []

    k = 1.0
    for it in range(20000):
        if it % 10 == 0 :
            k +=  0.01
        alpha = ALPHA/k
        biggest_change = 0

        states_and_rewards = play_game(policy, grid)
        for t in range(len(states_and_rewards) - 1):
            s, _ = states_and_rewards[t]
            s2, r = states_and_rewards[t + 1]
            old_theta = model.theta.copy()
            if grid.is_terminal(s2):
                target = r
            else:
                target = r + GAMMA * model.predict(s2)
            #x = model.s2x(s)
            model.theta += alpha * (target - model.predict(s)) * model.grad(s)
            biggest_change = max(biggest_change, np.abs(old_theta - model.theta).sum())
        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    #Predict V
    V = {}
    for s in states:
        if s in grid.actions:
            V[s] = model.predict(s)
        else:
            V[s] = 0

    print("Values: ")
    print_values(V, grid)
    print("Policy: ")
    print_policy(policy, grid)
