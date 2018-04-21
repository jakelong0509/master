import numpy as np
import matplotlib.pyplot as plt
from Grid_World import negative_grid, print_values, print_policy

ALL_ACTIONS = ("U", "D", "L", "R")
GAMMA = 0.9
lr = 0.1
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
        q_hat = np.dot(x.T, self.theta)
        return q_hat

    def grad(self, s, a):
        return self.sa2x(s,a)

def max_dict(qs):
    max_value = float("-inf")
    max_key = None
    for k,v in qs.items():
        if v > max_value:
            max_value = v
            max_key = k

    return max_value, max_key

def getQs(model,s):
    Qs = {}
    for a in ALL_ACTIONS:
        Q_sa = model.predict(s,a)
        Qs[a] = Q_sa

    return Qs

def random_action(a, eps=0.9):
    q = np.random.randn()
    if q < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_ACTIONS)

if __name__ == "__main__":
    grid = negative_grid(step_cost = -0.24)
    print("Rewards: ")
    print_values(grid.rewards, grid)
    model = Model()
    t = 1.0
    t2 = 1.0
    deltas = []
    loss = []
    for it in range(20000):
        if it % 1000 == 0:
            print("Iteration: %d" % (it))
        if it % 100 == 0:
            t += 0.01
            t2 += 0.01
        alpha = lr/t2
        s = (2,0)
        grid.set_state(s)
        Qs = getQs(model,s)
        a = max_dict(Qs)[1]
        a = random_action(a, eps=0.5/t)
        biggest_change = float("-inf")
        while not grid.game_over():
            r = grid.move(a)
            s2 = grid.current_state()

            old_theta = model.theta.copy()
            if grid.is_terminal(s2):
                model.theta += alpha * (r - model.predict(s,a)) * model.grad(s,a)
            else:
                Qs2 = getQs(model, s2)
                a2 = max_dict(Qs2)[1]
                a2 = random_action(a2, eps=0.5/t)
                model.theta += alpha * (r + GAMMA * model.predict(s2,a2) - model.predict(s,a)) * model.grad(s,a)

                s = s2
                a = a2

            biggest_change = max(biggest_change, np.abs(old_theta - model.theta).sum())

        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    policy = {}
    V = {}
    Q = {}
    for s in grid.actions.keys():
        Qs = getQs(model, s)
        Q[s] = Qs
        max_q, a = max_dict(Qs)
        policy[s] = a
        V[s] = max_q
    print(model.theta)
    print ("Value: ")
    print_values(V, grid)

    print ("Policy: ")
    print_policy(policy, grid)
