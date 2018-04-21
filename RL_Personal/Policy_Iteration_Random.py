import numpy as np
from Grid_World import print_policy, print_values, negative_grid

THRESHOLD = 0.001
ALL_POSSIBLE_ACTIONS = ("U", "D", "L", "R")
GAMMA = 0.9

if __name__ == "__main__":
    grid = negative_grid(step_cost = -0.1)

    states = grid.all_states()

    print ("Rewards: ")
    print_values(grid.rewards, grid)

    # Initialize Policy
    Policy = {}
    for s in grid.actions.keys():
        Policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    print ("Policy: ")
    print_policy(Policy, grid)

    # Initialize Value Function
    V = {}
    for s in states:
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            V[s] = 0

    while True:
        while True:
            biggest_change = 0
            for s in states:
                old_v = V[s]
                v = 0
                if s in Policy:
                    for a in ALL_POSSIBLE_ACTIONS:
                        if a == Policy[s]:
                            p = 0.5
                        else:
                            p = 0.5/3
                        grid.set_state(s)
                        r = grid.move(a)
                        v += p * (r + GAMMA * V[grid.current_state()])
                    V[s] = v
                    biggest_change = max(biggest_change, np.abs(old_v - V[s]))
            if biggest_change < THRESHOLD:
                break

        is_converged = True
        for s in states:
            if s in Policy:
                old_a = Policy[s]
                new_a = None
                biggest_value = float("-inf")
                for a in ALL_POSSIBLE_ACTIONS:
                    v = 0
                    for a2 in ALL_POSSIBLE_ACTIONS:
                        if a2 == a:
                            p = 0.5
                        else:
                            p = 0.5/3
                        grid.set_state(s)
                        r = grid.move(a)
                        v += p * ( r + GAMMA * V[grid.current_state()] )
                    if v > biggest_value:
                        biggest_value = v
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
