import numpy as np
import matplotlib.pyplot as plt

from policy_iterative_evaluation import print_values, print_policy
from grid_world import negative_grid, standard_grid


GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

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
    grid = negative_grid(step_cost = -0.24)

    print("Rewards: ")
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

    #initialize V
    V = {}
    for s in states:
        V[s] = 0

    for it in range(1000):
        states_and_rewards = play_game(policy, grid)

        for t in range(len(states_and_rewards) - 1):
            s, _ = states_and_rewards[t]
            s2, r = states_and_rewards[t + 1]

            V[s] = V[s] + ALPHA * (r + GAMMA * V[s2] - V[s])

    print ("Values: ")
    print_values(V, grid)

    print ("Policy: ")
    print_policy(policy, grid)
