#!/usr/bin/env python
import rospy

import gym
import keras
import numpy as np
import random
import time

from gym import wrappers
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from collections import deque

from openai_ros.task_envs.cartpole_stay_up import stay_up

from openai_ros.msg import RLExperimentInfo
from std_msgs.msg import Bool

class ReplayBuffer():
    def __init__(self,max_size):
        self.max_size = max_size
        self.transitions = deque()

    def add(self,obs,action,reward,obs2):
        if len(self.transitions) > self.max_size:
            self.transitions.popleft()
        self.transitions.append((obs,action,reward,obs2))

    def sample(self,count):
        return random.sample(self.transitions,count)
    
    def size(self):
        return len(self.transitions)


def get_q(model, observation, state_size):
    np_obs = np.reshape(observation, [-1,state_size])
    return model.predict(np_obs)


def train(model, observations, targets, actions_dim, state_size):
    np_obs = np.reshape(observations, [-1,state_size])
    np_targets = np.reshape(targets, [-1,actions_dim])

    model.fit(np_obs, np_targets, nb_epoch=1, verbose=0)


def predict(model, observation, state_size):
    np_obs = np.reshape(observation, [-1,state_size])
    return model.predict(np_obs)

def get_model(state_size, learning_rate):
    model = Sequential()
    model.add(Dense(16,input_shape=(state_size, ), activation='relu'))
    model.add(Dense(16,input_shape=(state_size, ), activation='relu'))
    model.add(Dense(2, activation='linear'))


    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='mse',
        metrics=[],
    )

    return model

def update_action(action_model, target_model, sample_transitions, actions_dim, state_size, gamma):
    random.shuffle(sample_transitions)
    batch_observations = []
    batch_targets = []

    for sample_transition in sample_transitions:
        old_observation, action, reward, observation = sample_transition

        targets = np.reshape(get_q(action_model,old_observation,state_size), actions_dim)
        targets[action] = reward

        if observation is not None:
            predictions = predict(target_model,observation,state_size)
            new_action = np.argmax(predictions)
            targets[action] += gamma * predictions[0,new_action]
        
        batch_observations.append(old_observation)
        batch_targets.append(targets)
    train(action_model, batch_observations, batch_targets, actions_dim, state_size)


def _publish_reward_topic(reward_pub,reward,episode_number=1):
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        reward_pub.publish(reward_msg)

if __name__=='__main__':
    rospy.init_node('cartpole_mbalunovic_algorithm',anonymous=True, log_level=rospy.FATAL)
    state_size = rospy.get_param('/cartpole_v0/state_size')
    action_size = rospy.get_param('/cartpole_v0/n_actions')
    gamma = rospy.get_param('/cartpole_v0/gamma')
    batch_size = rospy.get_param('/cartpole_v0/batch_size')
    target_update_freq = rospy.get_param('/cartpole_v0/target_update_freq')
    initial_random_action = rospy.get_param('/cartpole_v0/initial_random_action')
    replay_memory_size = rospy.get_param('/cartpole_v0/replay_memory_size')
    episodes = rospy.get_param('/cartpole_v0/episodes')
    max_iterations = rospy.get_param('/cartpole_v0/max_iterations')
    epsilon_decay = rospy.get_param('/cartpole_v0/epsilon_decay')
    done_episode_reward = rospy.get_param('/cartpole_v0/done_episode_reward')
    learning_rate = rospy.get_param('/cartpole_v0/learning_rate')
    steps_until_reset = target_update_freq
    random_action_probability = initial_random_action

    reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)
    done_pub = rospy.Publisher('/openai/done', Bool, queue_size=1)
    # Initialize replay memory D to capacity N
    replay = ReplayBuffer(replay_memory_size)
    action_model = get_model(state_size, learning_rate)

    env = gym.make('CartPoleStayUp-v0')
    env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
    start = time.time()
    scores = deque(maxlen=100)
    for episode in range(episodes):
        cumulated_episode_reward = 0
        observation = env.reset()
        for iteration in range(max_iterations):
            random_action_probability *= epsilon_decay
            random_action_probability = max(random_action_probability,0.1)
            old_observation = observation


            if np.random.random() < random_action_probability:
                action = np.random.choice(range(action_size))
            else: 
                q_values = get_q(action_model,observation,state_size)
                action = np.argmax(q_values)

            observation, reward, done, info = env.step(action)
            cumulated_episode_reward += reward
            if done:
                print 'Episode {}, iterations: {}'.format(
                    episode,
                    iteration
                )
                cumulated_episode_reward += done_episode_reward
                replay.add(old_observation, action, cumulated_episode_reward, None)
                break
            
            replay.add(old_observation, action, reward, observation)

            if replay.size() >= batch_size:
                sample_transitions = replay.sample(batch_size)
                update_action(action_model,action_model,sample_transitions, action_size,state_size,gamma)
        
        scores.append(iteration)
        mean_score = np.mean(scores)
        print 'Episode {}-> reward {}'.format(episode,mean_score)
        _publish_reward_topic(reward_pub,mean_score,episode)
        if mean_score >= 195.0:
            break
    done_msg = Bool()
    done_msg.data = True
    done_pub.publish(done_msg)
    end = time.time()
    print('Took {} seconds and {} episodes'.format(end-start,episode))
    action_model.save_weights("/tmp/cartpole-v2.h5", overwrite=True)