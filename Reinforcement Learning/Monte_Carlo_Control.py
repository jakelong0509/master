import numpy as np
import matplotlib.pyplot as plt
from policy_iterative_evaluation import print_policy, print_values
from grid_world import standard_grid, negative_grid

GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def play_game(grid, policy):
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    a = np.random.choice(ALL_POSSIBLE_ACTIONS)

    states_actions_rewards = [(s, a, 0)]
    seen_states = set()

    while True:
        old_s = grid.current_state()
        r = grid.move(a)
        s = grid.current_state()
        seen_states.add(old_s)
        if s in seen_states:
            states_actions_rewards.append((s , None, -10))
            break
        elif grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = policy[s]
            states_actions_rewards.append((s, a, r))




    G = 0
    states_actions_returns = []
    first = True
    for s,a,r in reversed(states_actions_rewards):
        if first:
            first = False
        else:
            states_actions_returns.append((s,a,G))
        G = r + GAMMA * G
    states_actions_returns.reverse()



    """print("States Action Reward:")
    print(states_actions_rewards)
    print("States Action Returns:")
    print(states_actions_returns)"""

    return states_actions_returns

def max_dict(d):
    max_key = None
    max_value = float('-inf')
    for k, v in d.items():
        if v > max_value:
            max_value = v
            max_key = k
    return max_key, max_value

def eps_Greedy(a, eps = 0.1):
    p = np.random.random()

    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)


if __name__ == '__main__':

    grid = negative_grid(step_cost = -0.5)

    print('Rewards:')
    print_values(grid.rewards, grid)

    #initialize Policy
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    #initialize Q and Returns
    Q = {}
    returns = {}
    N = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            N[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0
                N[s][a] = 0
                returns[(s,a)] = []
        else:
            pass

    deltas = []
    for t in range(10000):
        if t%1000 == 0:
            print(t)
        if t%100 == 0:
            t += 0.1
        biggest_change = 0
        states_actions_returns = play_game(grid,policy)
        seen_state_action_pairs = set()

        for s,a,G in states_actions_returns:
            sa = (s,a)
            if sa not in seen_state_action_pairs:
                old_q = Q[s][a]
                N[s][a] += 1
                returns[sa].append(G)
                Q[s][a] = Q[s][a] + 1/N[s][a] * (G - Q[s][a])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)
        deltas.append(biggest_change)
        eps = 1/t
        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]
            policy[s] = eps_Greedy(policy[s], eps)

    plt.plot(deltas)
    plt.show()

    print("Final policy:")
    print_policy(policy, grid)

    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Qs)[1]

    print("Final Values:")
    print_values(V, grid)
