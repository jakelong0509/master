import numpy as np
import matplotlib.pyplot as plt
from Grid_World import negative_grid, print_values, print_policy

ALL_ACTION = ("D", "U", "L", "R")
ALPHA = 0.1
GAMMA = 0.9
THRESHOLD = 0.01
class Model:
    def __init__(self):
        self.theta = np.random.randn(25)/np.sqrt(25)

    def sa2x(self,s,a):
        x = [
            s[0] - 1                    if a == "U" else 0,
            s[1] - 1.5                  if a == "U" else 0,
            (s[0]*s[1] - 3)/3           if a == "U" else 0,
            (s[0]*s[0] - 2)/2           if a == "U" else 0,
            (s[1] * s[1] - 4.5)/4.5     if a == "U" else 0,
            1                           if a == "U" else 0,
            s[0] - 1                    if a == "D" else 0,
            s[1] - 1.5                  if a == "D" else 0,
            (s[0]*s[1] - 3)/3           if a == "D" else 0,
            (s[0]*s[0] - 2)/2           if a == "D" else 0,
            (s[1] * s[1] - 4.5)/4.5     if a == "D" else 0,
            1                           if a == "D" else 0,
            s[0] - 1                    if a == "L" else 0,
            s[1] - 1.5                  if a == "L" else 0,
            (s[0]*s[1] - 3)/3           if a == "L" else 0,
            (s[0]*s[0] - 2)/2           if a == "L" else 0,
            (s[1] * s[1] - 4.5)/4.5     if a == "L" else 0,
            1                           if a == "L" else 0,
            s[0] - 1                    if a == "R" else 0,
            s[1] - 1.5                  if a == "R" else 0,
            (s[0]*s[1] - 3)/3           if a == "R" else 0,
            (s[0]*s[0] - 2)/2           if a == "R" else 0,
            (s[1] * s[1] - 4.5)/4.5     if a == "R" else 0,
            1                           if a == "R" else 0,
            1,
        ]

        return np.array(x)

    def predict(self, s, a):
        x = self.sa2x(s,a)
        q_sa = np.dot(x.T, self.theta)
        return q_sa

    def grad(self, s, a):
        return self.sa2x(s,a)

def random_action(a, eps=0.9):
    p = np.random.randn()
    if p > (1- eps):
        return a
    else:
        temp_actions = list(ALL_ACTION).remove(a)
        return np.random.choice(temp_actions)

def max_dict(q):
    max_key = None
    max_value = float("-inf")
    for k, v in q.items():
        if v > max_value:
            max_value = v
            max_key = k
    return max_key, max_value

def getQs(model, s):
    Qs = {}
    for a in ALL_ACTION:
        Qs[a] = model.predict(s,a)

    return Qs

def play_game(policy,grid):
    s = (2,0)
    grid.set_state(s)

    a = np.random.choice(ALL_ACTION)
    r = 0
    state_action_reward = [(s,a,r)]
    seen_states = []
    while True:
        r = grid.move(a)
        current_state = grid.current_state()
        if current_state in seen_states:
            state_action_reward.append((current_state, None, -100))
            break
        elif grid.game_over():
            state_action_reward.append((current_state, None, r))
            break
        elif current_state in grid.actions:
            a = policy[current_state]
            state_action_reward.append((current_state, a, r))
        seen_states.append(current_state)
    G = 0
    state_action_return = []
    first = True
    for s,a,r in reversed(state_action_reward):
        if first:
            first = False
        else:
            state_action_return.append((s,a,G))
        G = r + GAMMA * G
    state_action_return.reverse()
    return state_action_return

if __name__ == "__main__":
    grid = negative_grid(step_cost = -0.1)
    model = Model()
    # rewards
    print("Rewards: ")
    print_values(grid.rewards, grid)

    #Initialize Policy
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_ACTION)

    # #Initialize Q and Returns
    # Q = {}
    # Returns = {}
    # states = grid.all_states()
    # for s in states:
    #     if s in grid.actions:
    #         Q[s] = {}
    #         for a in ALL_ACTION:
    #             Q[s][a] = 0
    #             Returns[(s,a)] = []

    t = 1.0
    deltas = []

    for it in range(21000):
        if it % 1000 == 0:
            print("iteration: %d" % (it))

        if it % 100 == 0:
            t += 0.01


        lr = ALPHA/t
        biggest_change = 0
        state_action_return = play_game(policy, grid)
        for s,a,G in state_action_return:
            old_theta = model.theta.copy()
            # Returns[(s,a)].append(G)
            # Q[s][a] = np.mean(Returns[(s,a)])
            model.theta += lr * (G - model.predict(s,a)) * model.grad(s,a)
            biggest_change = max(biggest_change, np.abs(model.theta - old_theta).sum())
        deltas.append(biggest_change)

        for s in policy.keys():
            Qs = getQs(model, s)
            a = max_dict(Qs)[0]
            policy[s] = a

    plt.plot(deltas)
    plt.show()

    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            Qs = getQs(model, s)
            value = max_dict(Qs)[1]
            V[s] = value
        else:
            V[s] = 0 # Terminal State


    print ("Value: ")
    print_values(V, grid)

    print ("Policy: ")
    print_policy(policy, grid)
