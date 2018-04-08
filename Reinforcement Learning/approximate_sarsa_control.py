import numpy as np
import matplotlib.pyplot as plt

from policy_iterative_evaluation import print_policy, print_values
from grid_world import negative_grid, standard_grid

ALPHA = 0.1
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS= ('U', 'D', 'L', 'R')
SA2IDX = {}
IDX = 0

class Model:
    def __init__(self):
        self.theta = np.random.randn(25) / np.sqrt(25)

    def sa2x(self, s, a):
        return np.array([
            s[0] - 1              if a == 'U' else 0,
            s[1] - 1.5            if a == 'U' else 0,
            (s[0]*s[1] - 3)/3     if a == 'U' else 0,
            (s[0]*s[0] - 2)/2     if a == 'U' else 0,
            (s[1]*s[1] - 4.5)/4.5 if a == 'U' else 0,
            1                     if a == 'U' else 0,
            s[0] - 1              if a == 'D' else 0,
            s[1] - 1.5            if a == 'D' else 0,
            (s[0]*s[1] - 3)/3     if a == 'D' else 0,
            (s[0]*s[0] - 2)/2     if a == 'D' else 0,
            (s[1]*s[1] - 4.5)/4.5 if a == 'D' else 0,
            1                     if a == 'D' else 0,
            s[0] - 1              if a == 'L' else 0,
            s[1] - 1.5            if a == 'L' else 0,
            (s[0]*s[1] - 3)/3     if a == 'L' else 0,
            (s[0]*s[0] - 2)/2     if a == 'L' else 0,
            (s[1]*s[1] - 4.5)/4.5 if a == 'L' else 0,
            1                     if a == 'L' else 0,
            s[0] - 1              if a == 'R' else 0,
            s[1] - 1.5            if a == 'R' else 0,
            (s[0]*s[1] - 3)/3     if a == 'R' else 0,
            (s[0]*s[0] - 2)/2     if a == 'R' else 0,
            (s[1]*s[1] - 4.5)/4.5 if a == 'R' else 0,
            1                     if a == 'R' else 0,
            1
        ])

    def predict(self, s,a):
        x = self.sa2x(s,a)
        return self.theta.dot(x)

def getQs(model, s):
    Qs = {}
    for a in ALL_POSSIBLE_ACTIONS:
        q_sa = model.predict(s,a)
        Qs[a] = q_sa
    return Qs

def max_dict(d):
    max_k = None
    max_v = float('-inf')
    for k,v in d.items():
        if max_v < v:
            max_v = v
            max_k = k
    return max_k, max_v

def random_action(a, eps = 0.1):
    p = np.random.random()

    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

if __name__ == '__main__':
    grid = negative_grid(step_cost = -0.24)

    print ("Rewards: ")
    print_values(grid.rewards, grid)

    states = grid.all_states()

    for s in states:
        SA2IDX[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            SA2IDX[s][a] = IDX
            IDX += 1

    model = Model()
    t = 1.0
    t2 = 1.0
    deltas = []
    for it in range(20000):
        if it % 100 == 0:
            t += 0.01
            t2 += 0.01
        if it % 1000 == 0:
            print("Iteration: ", it)

        alpha = ALPHA/t2
        s = (2,0)
        grid.set_state(s)

        Qs = getQs(model, s)

        a = max_dict(Qs)[0]
        a = random_action(a, eps = 0.5/t)
        biggest_change = 0
        while not grid.game_over():
            r = grid.move(a)
            s2 = grid.current_state()
            old_theta = model.theta.copy()
            if grid.is_terminal(s2):
                model.theta += alpha * (r - model.predict(s,a)) * model.sa2x(s,a)
            else:
                Qs2 = getQs(model,s2)
                a2 = max_dict(Qs2)[0]
                a2 = random_action(a2, eps = 0.5/t)

                model.theta += alpha * (r + GAMMA * model.predict(s2,a2) - model.predict(s,a)) * model.sa2x(s,a)
                s = s2
                a = a2
            biggest_change = max(biggest_change, np.abs(old_theta - model.theta).sum())
        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    V = {}
    policy = {}
    Q = {}
    states = grid.all_states()
    for s in grid.actions.keys():
        Qs = getQs(model,s)
        Q[s] = Qs
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    print("Values: ")
    print_values(V, grid)
    print("Policy: ")
    print_policy(policy, grid)
