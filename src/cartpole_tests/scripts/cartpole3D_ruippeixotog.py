#!/usr/bin/env python
import rospy
 
import os
 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np
import gym
from gym import wrappers
from openai_ros.task_envs.cartpole_stay_up import stay_up
from qlearning import QLearningAgent

from openai_ros.msg import RLExperimentInfo
from std_msgs.msg import Bool

class GymRunner:
    def __init__(self,environment,monitor_dir,max_timesteps=100000):
        self.monitor_dir = monitor_dir
        self.max_timesteps = max_timesteps

        self.env = gym.make(environment)
        self.env = wrappers.Monitor(self.env,monitor_dir,force=True)
        self.episode_num = 0
        self.cumulated_episode_reward = 0
        # Start Reward publishing
        self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)
        self.done_pub = rospy.Publisher('/openai/done', Bool, queue_size=1)

    def calc_reward(self,state,action,gym_reward,next_state,done):
        return gym_reward

    def train(self,agent,num_episodes):
        self.run(agent,num_episodes,do_train=True)

    def run(self,agent,num_episodes,do_train=False):
        scores = deque(maxlen=100)
        for episode in range(num_episodes):
            state = self.env.reset().reshape(1,self.env.observation_space.shape[0])
            total_reward = 0
            for t in range(self.max_timesteps):
                action = agent.select_action(state,do_train)

                next_state, reward, done, _ = self.env.step(action)
                next_state =  next_state.reshape(1,self.env.observation_space.shape[0])
                reward = self.calc_reward(state,action,reward,next_state,done)

                if do_train:
                    agent.record(state,action,reward,next_state,done)
                
                total_reward += reward
                state = next_state
                if done:
                    reward_msg = RLExperimentInfo()
                    reward_msg.episode_number = episode
                    reward_msg.episode_reward = total_reward
                    self.reward_pub.publish(reward_msg)
                    break
            scores.append(total_reward)
            mean_score = np.mean(scores)
            if do_train:
                agent.replay()
            
            print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(episode, mean_score))
        done_msg = Bool()
        done_msg.data = True
        self.done_pub.publish(done_msg)
    def close_and_upload(self,api_key):
        self.env.close()
        gym.upload(self.monitor_dir,api_key=api_key)

class CartPoleAgent(QLearningAgent):
    def __init__(self):
        self.state_size = rospy.get_param('/cartpole_v0/state_size')
        self.learning_rate = rospy.get_param('/cartpole_v0/learning_rate')
        action_size = rospy.get_param('/cartpole_v0/n_actions')
        gamma = rospy.get_param('/cartpole_v0/gamma')
        epsilon = rospy.get_param('/cartpole_v0/epsilon')
        epsilon_decay = rospy.get_param('/cartpole_v0/epsilon_decay')
        epsilon_min = rospy.get_param('/cartpole_v0/epsilon_min')
        batch_size = rospy.get_param('/cartpole_v0/batch_size')

        QLearningAgent.__init__(self,
                                state_size=self.state_size,
                                action_size=action_size,
                                gamma=gamma,
                                epsilon=epsilon,
                                epsilon_decay=epsilon_decay,
                                epsilon_min=epsilon_min,
                                batch_size=batch_size)
    def build_model(self):
        model = Sequential()
        model.add(Dense(12,input_shape=(self.state_size, ), activation='relu'))
        model.add(Dense(12,input_shape=(self.state_size, ), activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(Adam(lr=self.learning_rate), 'mse')
        
        model.load_weights("models/cartpole-v2.h5")
        return model



if __name__=='__main__':
    rospy.init_node('cartpole3D_ruippeixotog', anonymous=True, log_level=rospy.FATAL)
    episodes_training = rospy.get_param('/cartpole_v0/episodes')
    episodes_running = rospy.get_param('/cartpole_v0/episodes_running')
    max_timesteps = rospy.get_param('/cartpole_v0/max_timesteps', 10000)
    
    gym = GymRunner('CartPoleStayUp-v0', '/tmp/cartpole-experiment-2', max_timesteps)
    agent = CartPoleAgent()
    gym.train(agent, episodes_training)
    agent.model.save_weights("/tmp/cartpole-v2.h5", overwrite=True)
    gym.run(agent, episodes_running, do_train=False)

    

