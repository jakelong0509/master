import numpy as np
from policy_iterative_evaluation import print_policy, print_values
from grid_world import standard_grid, negative_grid

THRESHOLD = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

if __name__ == '__main__':
    grid = negative_grid()

    states = grid.all_states()
    #Initialize Value function
    V = {}
    for s in states:
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            V[s] = 0

    #Initialize policy
    Policy = {}
    for s in grid.actions.keys():
        Policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    while True:
        delta = 0
        for s in states:
            old_V = V[s]
            if s in Policy:
                new_v = float('-inf')
                for a in ALL_POSSIBLE_ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + GAMMA * V[grid.current_state()]
                    if v > new_v:
                        new_v = v
                V[s] = new_v
                delta = max(delta, np.abs(old_V - V[s]))
        if delta < THRESHOLD:
            break

    for s in Policy.keys():
        best_a = None
        best_value = float('-inf')
        for a in ALL_POSSIBLE_ACTIONS:
            grid.set_state(s)
            r = grid.move(a)
            v = r + GAMMA * V[grid.current_state()]
            if v > best_value:
                best_value = v
                best_a = a
        Policy[s] = best_a

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(Policy, grid)
