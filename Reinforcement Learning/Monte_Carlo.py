"""import numpy as np
from iterative_policy_evaluation import print_policy, print_values
from grid_world import negative_grid, standard_grid

THRESHOLD = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def play_game(grid, policy):
    start_states = list(grid.actions.key())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    states_and_rewards = [(s,0)]
    while not grid.game_over():
        a = policy(s)
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
            states_and_returns.append((s, G))
        G = r + GAMMA * G
    states_and_returns.reverse()
    return states_and_returns

if __name__ == '__main__':
    grid = negative_grid()

    print('Rewards: ')
    print_values(grid.rewards, grid)

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

    #initialize value function and returns
    V = {}
    returns = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            returns[s] = []
        else:
            V[s] = 0

    for k in range(1000):
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        for s, G in states_and_returns:
            if s not in seen_states:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                seen_states.add(s)

    print("Values:")
    print_values(V, grid)
    print("Policy: ")
    print_policy(policy, grid)
"""

import numpy as np
from policy_iterative_evaluation import print_policy, print_values
from grid_world import standard_grid, negative_grid

THRESHOLD = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def play_game(grid, policy):
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    states_and_rewards = [(s, 0)]
    while not grid.game_over():
        a = policy[s]
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))

    G = 0
    states_and_returns = []
    first = True
    for s,r in reversed(states_and_rewards):
        if first:
            first = False
        else:
            states_and_returns.append((s, G))
        G = r + GAMMA * G
    states_and_returns.reverse()
    return states_and_returns

if __name__ == '__main__':
    grid = negative_grid()

    print('Rewards: ')
    print_values(grid.rewards, grid)

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

    #initialize value function and return
    V = {}
    returns = {}
    N = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            returns[s] = []
            N[s] = 0
            V[s] = 0
        else:
            V[s] = 0

    for k in range(100):
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        print("Iteration: ", k+1)
        for s,G in states_and_returns:
            if s not in seen_states:
                N[s] += 1
                returns[s].append(G)
                V[s] = V[s] + 1/N[s] * (G - V[s])
                seen_states.add(s)
    print("Values: ")
    print_values(V, grid)
    print("Policy: ")
    print_policy(policy, grid)
