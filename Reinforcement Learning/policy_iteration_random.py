import numpy as np
import matplotlib.pyplot as plt

from grid_world import standard_grid, negative_grid
from policy_iterative_evaluation import print_policy, print_values

THRESHOLD = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

if __name__ == '__main__':

    grid = negative_grid(step_cost = -0.1)

    States = grid.all_states()

    print("Rewards:")
    print_values(grid.rewards, grid)

    #Intial Value Function
    V = {}
    for s in States:
        if s == grid.actions:
            V[s] = np.random.random()
        else:
            #Terminal States
            V[s] = 0



    #Initial policy
    Policy = {}
    for s in grid.actions.keys():
        Policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    print('Print Policy: ')
    print_policy(Policy, grid)

    while True:
        while True:# k : iteration
            #Setting delta to say the new Value
            #is the policy True Value or Not
            delta = 0
            for s in States:
                old_v = V[s]
                new_v = 0
                if s in Policy:
                    #Check all the Posible actions
                    for a in ALL_POSSIBLE_ACTIONS:
                        #If the action is the action that the agent
                        #want to take then the probability is 0.5 otherwise
                        #the probability is 0.5/3 for each ofther actions
                        if a == Policy[s]:
                            p_a = 0.5
                        else:
                            p_a = 0.5/3

                        #Set the state be the current_state
                        grid.set_state(s)

                        #Experience the action to update the value Function
                        #of particular state
                        r = grid.move(a)

                        #Update Value Function V(k+1)(s) base on the old V(k)(s')
                        new_v += p_a * (r + GAMMA * V[grid.current_state()])
                    #Assign the new Value to the current State S(k+1)
                    V[s] = new_v
                    #Set delta to be the V(k)(s) - V(k+1)(s)
                    delta = max(delta, np.abs(old_v - V[s]))
            #Check if the Value function is the True Value of the Policy or Not
            if delta < THRESHOLD:
                break

        is_policy_converged = True
        for s in States:
            if s in Policy:
                old_a = Policy[s]
                new_a = None
                best_value = float('-inf')
                #Test through all Actions and choose the one that have the highest
                #Value Function
                for a in ALL_POSSIBLE_ACTIONS:
                    v = 0
                    #The Probability that the wind might blow the agent to
                    #other states
                    for a2 in ALL_POSSIBLE_ACTIONS:
                        if a == a2:
                            p_a = 0.5
                        else:
                            p_a = 0.5/3

                        #Set the state to be the current state
                        grid.set_state(s)

                        #Experience the action
                        r = grid.move(a)

                        #Set Value Function for that action
                        v += p_a * (r + GAMMA * V[grid.current_state()])

                    #Check which action have the best value function
                    if v > best_value:
                        best_value = v
                        new_a = a

                #Set the action at particular state to be the highest action
                Policy[s] = new_a

                #Check if the new action and the old action is the same or not
                #If these actions are the same
                #then the MDP is solved (Found Optimal Policy)
                if new_a != old_a:
                    is_policy_converged = False
        if is_policy_converged:
            break
    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(Policy, grid)
