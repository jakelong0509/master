import numpy as np
import matplotlib.pyplot as plt
from Grid_World import print_policy, print_values, negative_grid

GAMMA = 0.9
ALL_POSSIBLE_CHOICES = ("U", "D", "L", "R")

def max_dict(d):
    max_value = float("-inf")
    max_key = None
    for k,v in d.items():
        if v > max_value:
            max_value = v
            max_key = k
    return max_value, max_key


def play_game(policy, grid):
    start_states = list(grid.actions.keys())
    idx_start = np.random.choice(len(start_states))
    grid.set_state(start_states[idx_start])
    s = grid.current_state()
    a = np.random.choice(ALL_POSSIBLE_CHOICES)
    seen_state = set()
    state_action_reward = [(s,a,0)]
    while True:
        r = grid.move(a)
        s = grid.current_state()

        if s in seen_state:
            state_action_reward.append((s,None,-100))
            break
        elif grid.game_over():
            state_action_reward.append((s,None,r))
            break
        else:
            a = policy[s]
            state_action_reward.append((s,a,r))
        seen_state.add(s)

    first = True
    G = 0
    states_actions_returns = []
    for s,a,r in reversed(state_action_reward):
        if first:
            first=False
        else:
            states_actions_returns.append((s,a,G))
        G = r + GAMMA * G
    states_actions_returns.reverse()
    return states_actions_returns

if __name__ == "__main__":
    grid = negative_grid(step_cost=-0.1)
    states = grid.all_states()



    # initialize Q
    Q = {}
    Returns = {}
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in ALL_POSSIBLE_CHOICES:
                Q[s][a] = 0
                Returns[(s,a)] = []

    # Initialize Policy
    Policy = {}
    for s in grid.actions.keys():
        Policy[s] = np.random.choice(ALL_POSSIBLE_CHOICES)

    print ("Rewards: ")
    print_values(grid.rewards, grid)

    print ("Policy: ")
    print_policy(Policy, grid)

    deltas = []
    for t in range(5000):
        if t % 100 == 0:
            print ("Iteration %d" %(t))

        states_actions_returns = play_game(Policy, grid)
        biggest_change = 0
        seen_states_actions = set()
        for s,a,G in states_actions_returns:
            if (s,a) not in seen_states_actions:
                old_q = Q[s][a]
                Returns[(s,a)].append(G)
                Q[s][a] = np.mean(Returns[(s,a)])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_states_actions.add((s,a))
        deltas.append(biggest_change)

        for s in Policy.keys():
            Policy[s] = max_dict(Q[s])[1]

    plt.plot(deltas)
    plt.show()

    V = {}
    for s in states:
        if s in Policy:
            V[s] = max_dict(Q[s])[0]
        else:
            V[s] = 0

    print ("Value: ")
    print_values(V, grid)

    print ("Policy: ")
    print_policy(Policy, grid)
