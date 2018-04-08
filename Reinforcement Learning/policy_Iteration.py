import numpy as np
from grid_world import standard_grid, negative_grid
from policy_iterative_evaluation import print_values, print_policy

threshold = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

if __name__ == '__main__':
    grid = negative_grid()
    print ("Rewards: ")
    print_values(grid.rewards, grid)

    #Initialize policy
    Policy = {}
    for s in grid.actions.keys():
        Policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    print ("Initial Policy: ")
    print_policy(Policy, grid)


    #Initialize Value function
    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            V[s] = 0



    while True:
        while True:
            delta = 0
            for s in states:
                old_v = V[s]
                if s in Policy:
                    a = Policy[s]
                    grid.set_state(s)
                    r = grid.move(a)
                    V[s] = r + GAMMA * V[grid.current_state()]
                    delta = max(delta, np.abs(old_v - V[s]))
            if delta < threshold:
                break

        is_policy_converged = True
        for s in states:
            if s in Policy:
                old_a = Policy[s]
                new_a = None
                best_value = float('-inf')
                for a in ALL_POSSIBLE_ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + GAMMA * V[grid.current_state()]
                    if v > best_value:
                        best_value = v
                        new_a = a
                Policy[s] = new_a
                if new_a != old_a:
                    is_policy_converged = False

        if is_policy_converged:
            break

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(Policy, grid)
