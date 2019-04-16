#!/usr/bin/python
import math
import random
from collections import deque

import numpy as np

import gym
import rospy
import time
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

# import training environment
from openai_ros.task_envs.cartpole_stay_up import stay_up

from openai_ros.msg import RLExperimentInfo
from std_msgs.msg import Bool

class DQNRobotSolver():
    def __init__(self, environment_name, n_states, n_actions, n_episodes=1000, n_win_ticks=195, min_episodes= 100, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=64, monitor=False, quiet=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make(environment_name)
        if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.min_episodes = min_episodes
        self.batch_size = batch_size
        self.quiet = quiet
        if max_env_steps is not None: 
            self.env._max_episode_steps = max_env_steps

        #init the model
        self.model = Sequential()
        self.model.add(Dense(24,input_dim=self.n_states,activation="tanh"))
        self.model.add(Dense(48,activation="tanh"))
        self.model.add(Dense(self.n_actions,activation="linear"))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        
        self.episode_num = 0
        self.cumulated_episode_reward = 0
        # Start Reward publishing
        self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)
        self.done_pub = rospy.Publisher('/openai/done', Bool, queue_size=1)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def choose_action(self,state,epsilon):
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))
    
    def get_epsilon(self,t):
        return max(self.epsilon_min,min(self.epsilon,1.0 - math.log10((t+1)*self.epsilon_decay)))
    
    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])
    
    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self):
        #rate = rospy.Rate(30)
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            while not done:
                action = self.choose_action(state,self.get_epsilon(e))
                next_state,reward,done,_ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                self.remember(state,action,reward,next_state,done)
                state = next_state
                i += 1
                self.cumulated_episode_reward += reward

            
            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                if not self.quiet: 
                    print('Ran {} episodes. Solved after {} trials'.format(e, e - 100))
                    done_msg = Bool()
                    done_msg.data = True
                    self.done_pub.publish(done_msg)
                return e - 100

            if not self.quiet:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

            self.replay(self.batch_size)
            self._update_episode()
        if not self.quiet: 
            print('Did not solve after {} episodes'.format(self.n_episodes))
        return self.n_episodes
    

    def _publish_reward_topic(self,reward,episode_number=1):
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        self.reward_pub.publish(reward_msg)
    
    def _update_episode(self):
        self._publish_reward_topic(
                                    self.cumulated_episode_reward,
                                    self.episode_num
                                    )
        self.episode_num += 1
        self.cumulated_episode_reward = 0


if __name__=='__main__':
    rospy.init_node('cartpole3D_n1try',anonymous=True,log_level=rospy.FATAL)
    environment_name = 'CartPoleStayUp-v0'
    n_states = rospy.get_param('/cartpole_v0/n_states')
    n_actions = rospy.get_param('/cartpole_v0/n_actions')

    n_episodes = rospy.get_param('/cartpole_v0/episodes')
    n_win_ticks = rospy.get_param('/cartpole_v0/n_win_ticks')
    min_episodes = rospy.get_param('/cartpole_v0/min_episodes')
    max_env_steps = None
    gamma = rospy.get_param('/cartpole_v0/gamma')
    epsilon = rospy.get_param('/cartpole_v0/epsilon')
    epsilon_min = rospy.get_param('/cartpole_v0/epsilon_min')
    epsilon_log_decay = rospy.get_param('/cartpole_v0/epsilon_decay')
    alpha = rospy.get_param('/cartpole_v0/alpha')
    alpha_decay = rospy.get_param('/cartpole_v0/alpha_decay')
    batch_size = rospy.get_param('/cartpole_v0/batch_size')
    monitor = rospy.get_param('/cartpole_v0/monitor')
    quiet = rospy.get_param('/cartpole_v0/quiet')

    agent = DQNRobotSolver(     environment_name,
                                n_states,
                                n_actions,
                                n_episodes,
                                n_win_ticks,
                                min_episodes,
                                max_env_steps,
                                gamma,
                                epsilon,
                                epsilon_min,
                                epsilon_log_decay,
                                alpha,
                                alpha_decay,
                                batch_size,
                                monitor,
                                quiet)
    
    start = time.time()
    episodes = agent.run()
    end = time.time()
    print('Took {} seconds and {} episodes'.format(end-start,episodes))
    agent.model.save_weights("/tmp/cartpole-v0.h5", overwrite=True)