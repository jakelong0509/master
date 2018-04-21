import numpy as np


class Grid:
    def __init__(self, width, height, start):
        self.width = width
        self.height = height
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):
        self.rewards = rewards
        self.actions = actions

    def move(self, action):
        if action in self.actions[(self.i, self.j)]:
            if action == "U":
                self.i -= 1
            elif action == "D":
                self.i += 1
            elif action == "L":
                self.j -= 1
            elif action == "R":
                self.j += 1
        return self.rewards.get((self.i, self.j), 0)

    def undo_move(self, action):
        if action == "U":
            self.i += 1
        elif action == "D":
            self.i -= 1
        elif action == "L":
            self.j += 1
        elif action == "R":
            self.j -= 1
        assert(self.current_state() in self.all_states())

    def current_state(self):
        return (self.i, self.j)

    def is_terminal(self, s):
        return s not in self.actions

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def game_over(self):
        return (self.i, self.j) not in self.actions

    def all_states(self):
      # possibly buggy but simple way to get all states
      # either a position that has possible next actions
      # or a position that yields a reward
      return set(self.actions.keys()) | set(self.rewards.keys())

def standard_grid():
    g = Grid(3,4, (2,0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
      (0, 0): ('D', 'R'),
      (0, 1): ('L', 'R'),
      (0, 2): ('L', 'D', 'R'),
      (1, 0): ('U', 'D'),
      (1, 2): ('U', 'D', 'R'),
      (2, 0): ('U', 'R'),
      (2, 1): ('L', 'R'),
      (2, 2): ('L', 'R', 'U'),
      (2, 3): ('L', 'U'),
    }
    g.set(rewards, actions)
    return g

def negative_grid(step_cost = -1.0):
    g = standard_grid()
    g.rewards.update({
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost,

    })
    return g

def print_values(V, g):
  for i in range(g.width):
    print("---------------------------")
    for j in range(g.height):
      v = V.get((i,j), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
    print("")


def print_policy(P, g):
  for i in range(g.width):
    print("---------------------------")
    for j in range(g.height):
      a = P.get((i,j), ' ')
      print("  %s  |" % a, end="")
    print("")

if __name__ == "__main__":
    grid = negative_grid()
    states = grid.all_states()
    V = {}
    for s in states:
        V[s] = 0
