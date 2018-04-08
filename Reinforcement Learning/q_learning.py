import numpy as np
import matplotlib.pyplot as plt

from policy_iterative_evaluation import print_policy, print_values
from grid_world import standard_grid, negative_grid

ALPHA = 0.1
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def random_action(a, eps = 0.1):
    p = np.random.random()

    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

def max_dict(d):
    max_key = None
    max_value = float('-inf')
    for k,v in d.items():
        if v > max_value:
            max_value = v
            max_key = k
    return max_key, max_value

if __name__ == '__main__':
    grid = negative_grid(step_cost = -0.1)
    print('Rewards: ')
    print_values(grid.rewards, grid)

    states = grid.all_states()

    #initialize Q
    Q = {}
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0

    #initialize Count
    update_counts = {}
    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            update_counts_sa[s][a] = 1.0

    t = 1.0
    deltas = []
    for it in range(10000):
        if it % 100 == 0:
            t += 1e-2
        if it % 1000 == 0:
            print('Iteration: ', it)

        s = (2,0)
        grid.set_state(s)
        a, _ = max_dict(Q[s])
        biggest_change = 0
        while not grid.game_over():
            a = random_action(a, eps = 0.5/t)

            r = grid.move(a)
            s2 = grid.current_state()

            a2, max_q2s2 = max_dict(Q[s2])
            alpha = ALPHA / update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005

            old_qsa = Q[s][a]

            Q[s][a] = Q[s][a] + alpha * (r + GAMMA * max_q2s2 - Q[s][a])

            biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))
            update_counts[s] = update_counts.get(s,0) + 1

            s = s2

        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    #Find optimal Policy
    policy = {}
    V = {}
    for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    print('Update counts: ')
    total = np.sum(list(update_counts.values()))
    for k,v in update_counts.items():
        update_counts[k] = float(v) / total
    print_values(update_counts, grid)

    print('Values: ')
    print_values(V , grid)
    print('Policy: ')
    print_policy(policy, grid)
