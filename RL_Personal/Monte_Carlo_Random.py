import numpy as np
from Grid_World import print_policy, print_values, negative_grid

ALL_POSSIBLE_ACTIONS = ("U", "D", "L", "R")
GAMMA = 0.9

def random_action(a, eps = 0.1):
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        tmp = list(ALL_POSSIBLE_ACTIONS)
        tmp.remove(a)
        return np.random.choice(tmp)
def max_dict(v):
    max_value = float("-inf")
    max_key = None
    for k,v in v.items():
        if v > max_value:
            max_value = v
            max_key = k
    return max_value, max_key

def play_game(policy, grid):
    start_states = list(grid.actions.keys())
    state_idx = np.random.choice(len(start_states))
    current_state = start_states[state_idx]
    grid.set_state(current_state)
    s = grid.current_state()
    r = 0
    state_and_reward = [(s, r)]
    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()

        state_and_reward.append((s,r))

    G = 0
    state_and_return = []
    first = True
    for s,r in reversed(state_and_reward):
        if first:
            first = False
        else:
            state_and_return.append((s,G))
        G = r + GAMMA * G
    state_and_return.reverse()
    return state_and_return

if __name__ == "__main__":
    grid = negative_grid(step_cost=-0.24)
    states = grid.all_states()
    print ("Rewards: ")
    print_values(grid.rewards, grid)

    # Initialize Policy
    Policy = {}
    for s in states:
        if s in grid.actions:
            Policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    print ("Policy: ")
    print_policy(Policy, grid)

    # Initialize Value function and Return
    Returns = {}
    V = {}
    for s in states:
        if s in grid.actions:
            Returns[s] = []
            V[s] = np.random.random()
        else:
            V[s] = 0

    for t in range(5000):
        if t % 100 == 0:
            print ("Iteration: %d" %(t))
        seen_state = []
        state_and_return = play_game(Policy, grid)
        for s, G in state_and_return:
            if s not in seen_state:
                Returns[s].append(G)
                V[s] = np.mean(Returns[s])
                seen_state.append(s)




    print("Value: ")
    print_values(V, grid)

    print("Policy: ")
    print_policy(Policy, grid)

    print (grid.all_states())
