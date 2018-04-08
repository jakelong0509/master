import numpy as np
import matplotlib.pyplot as plt

from policy_iterative_evaluation import print_values, print_policy
from grid_world import standard_grid, negative_grid

LEARNING_RATE = 0.1
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def random_action(a, eps = 0.5):
    p = np.random.random()

    if p < (1 - eps):
        return a
    else:
        tmp = list(ALL_POSSIBLE_ACTIONS)
        tmp.remove(a)
        return np.random.choice(tmp)

def play_game(grid, policy):
    s = (2,0)
    grid.set_state(s)


    states_and_rewards = [(s,0)]

    while not grid.game_over():
        a = random_action(policy[s])
        r = grid.move(a)
        s = grid.current_state()

        states_and_rewards.append((s,r))

    G = 0
    states_and_returns = []
    first = True
    for s,r in reversed(states_and_rewards):
        if first:
            first = False
        else:
            states_and_returns.append((s,G))
        G = r + GAMMA * G
    states_and_returns.reverse()
    return states_and_returns

if __name__ == '__main__':
    grid = negative_grid(step_cost = -0.1)

    print("Reward: ")
    print_values(grid.rewards, grid)

    states = grid.all_states()

    #Initialize Policy
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'L',
        (2, 2): 'U',
        (2, 3): 'L',

    }

    #Initialize Theta
    theta = np.random.randn(4) / 2
    def s2x(s):
        return np.array([s[0] - 1, s[1] - 1.5, s[0] * s[1] - 3, 1])

    t = 1.0
    deltas = []
    for it in range(20000):
        if it % 100 == 0:
            t += 0.01
        if it % 1000 == 0:
            print ("Iteration", it)
        alpha = LEARNING_RATE/t
        states_and_returns = play_game(grid, policy)
        biggest_change = 0
        seen_states = set()
        for s,G in states_and_returns:
            if s not in seen_states:
                old_theta = theta.copy()
                x = s2x(s)
                V_hat = theta.dot(x)
                theta += alpha * (G - V_hat) * x
                biggest_change = max(biggest_change, np.abs(old_theta - theta).sum())
                seen_states.add(s)
        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    V = {}
    for s in states:
        if s in grid.actions:
            V[s] = theta.dot(s2x(s))
        else:
            V[s] = 0

    print("Policy: ")
    print_policy(policy, grid)
    print("Value: ")
    print_values(V, grid)
