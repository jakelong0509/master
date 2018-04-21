import numpy as np
from Grid_World import print_values, print_policy, negative_grid
import matplotlib.pyplot as plt

alpha = 0.1
GAMMA = 0.09
ALL_ACTIONS = ("D", "U", "L", "R")

def max_dict(d):
    max_value = float("-inf")
    max_key = None
    for k, v in d.items():
        if v > max_value:
            max_value = v
            max_key = k
    return max_value, max_key

def random_action(a, eps = 0.01):
    p = np.random.randn()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_ACTIONS)

if __name__ == "__main__":
    grid = negative_grid(step_cost=-0.24)
    print ("Reward: ")
    print_values(grid.rewards, grid)

    # Initialize Policy
    states = grid.all_states()
    Policy = {}
    for s in grid.actions.keys():
        Policy[s] = np.random.choice(ALL_ACTIONS)

    # Initialize Q
    Q = {}
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in ALL_ACTIONS:
                Q[s][a] = np.random.random()
        else:
            Q[s] = {}
            for a in ALL_ACTIONS:
                Q[s][a] = 0

    update_counts_sa = {}
    for s in states:
        update_counts_sa[s] = {}
        for a in ALL_ACTIONS:
            update_counts_sa[s][a] = 1.0

    t = 1.0
    deltas = []
    for it in range(10000):
        if it % 1000 == 0:
            t += 1e-2

        biggest_change = 0
        start_state = (2,0)
        grid.set_state(start_state)
        a = max_dict(Q[s])[1]
        while not grid.game_over():
            a = random_action(a)

            old_qsa = Q[s][a]
            r = grid.move(a)
            s_prime = grid.current_state()
            max_q_s2 = max_dict(Q[s_prime])[0]
            Q[s][a] = Q[s][a] + alpha * (r + GAMMA * max_q_s2 - Q[s][a])
            biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))
            s = s_prime
        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()
    for s in Policy:
        Policy[s] = max_dict(Q[s])[1]

    V = {}
    for s in states:
        if s in Policy:
            V[s] = max_dict(Q[s])[0]
        else:
            V[s] = 0

    print("Value: ")
    print_values(V, grid)

    print("Policy: ")
    print_policy(Policy, grid)
