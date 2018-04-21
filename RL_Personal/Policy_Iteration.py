import numpy as np
from Grid_World import negative_grid, print_policy, print_values

threshold = 0.001
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ("U", "D", "L", "R")
if __name__ == "__main__":
    grid = negative_grid(step_cost = -0.24)
    states = grid.all_states()
    # Initialize Value Function
    V = {}
    for s in states:
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            V[s] = 0

    print("Rewards: ")
    print_values(grid.rewards, grid)

    # Initialize Policy
    Policy = {}
    for s in grid.actions.keys():
        Policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    print("Policy: ")
    print_policy(Policy, grid)


    while True:
        while True:
            biggest_change = 0
            for s in states:
                old_v = V[s]
                if s in Policy:
                    a = Policy[s]
                    grid.set_state(s)
                    r = grid.move(a)
                    V[s] = r + GAMMA * V[grid.current_state()]
                    biggest_change = max(biggest_change, np.abs(old_v - V[s]))

            if biggest_change < threshold:
                break
        is_converged = True
        for s in states:
            if s in Policy:
                old_a = Policy[s]
                new_a = None
                best_value = float("-inf")
                for a in ALL_POSSIBLE_ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + GAMMA * V[grid.current_state()]
                    if v > best_value:
                        best_value = v
                        new_a = a
                Policy[s] = new_a
                if new_a != old_a:
                    is_converged = False

        if is_converged:
            break

    print("Value: ")
    print_values(V, grid)

    print ("Policy: ")
    print_policy(Policy, grid)
