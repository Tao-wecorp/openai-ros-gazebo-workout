'''
Q-learning approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning
 
Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA
 
        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''

import random
import numpy as np

class TabularQLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.Q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions
    
    def init_q(self,env_size):
        self.env = env_size # For each z 100x100 x-y bin. There are 10 z bins.
        self.num_states = np.prod(self.env)
        for i in range(self.num_states):
            position = np.unravel_index(i,self.env)
            self.Q[position] = { a : 0 for a in self.actions}

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        self.Q[state][action] += self.alpha * (value - self.Q[state][action])

    def chooseAction(self, state, return_q=False):
        q = [self.Q[state][a] for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            return np.random.choice(self.actions)

        count = q.count(maxQ)
        # In case there're several state-action max values 
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]        
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.Q[state2][a] for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)