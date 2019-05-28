#!/usr/bin/python
import math
import random
from collections import deque

import numpy as np

import gym
import time

from tqdm import tqdm
import tensorflow as tf
import rospkg

class DQN():
    class Model():
        def __init__(self,graph,n_states,n_actions,min_clip_value,max_clip_value,learning_rate=0.01,is_agent=True):
            self.graph = graph
            with self.graph.as_default():
                self.x = tf.placeholder(tf.float32,[None,n_states])
                self.y = tf.placeholder(tf.float32,[None,n_actions])
                # now declare the weights connecting the input to the 1st hidden layer
                self.W1 = tf.Variable(tf.random_normal([n_states, 128], stddev=0.03), name='W1')
                self.b1 = tf.Variable(tf.random_normal([128]), name='b1')
                # and the weights connecting the 1st hidden layer to the 2nd hidden layer
                self.W2 = tf.Variable(tf.random_normal([128,32], stddev=0.03), name='W2')
                self.b2 = tf.Variable(tf.random_normal([32]), name='b2')
                # and finally the weights connecting the 2nd hidden layer to the output layer
                self.W3 = tf.Variable(tf.random_normal([32,n_actions],stddev=0.03),name="W3")
                self.b3 = tf.Variable(tf.random_normal([n_actions]),name='b3')
                # calculate the output of the 1st hidden layer
                self.hidden1_out = tf.add(tf.matmul(self.x, self.W1), self.b1)
                self.hidden1_out = tf.nn.relu(self.hidden1_out)
                # calculate the output of the 1st hidden layer
                self.hidden2_out = tf.add(tf.matmul(self.hidden1_out, self.W2), self.b2)
                self.hidden2_out = tf.nn.relu(self.hidden2_out)
                
                self.pred_q_value = tf.add(tf.matmul(self.hidden2_out, self.W3), self.b3)
                #self.pred_q_value = tf.clip_by_value(self.y_, min_clip_value, max_clip_value)
                self.pred_q_value_norm = tf.layers.batch_normalization(
                    tf.reshape(self.pred_q_value, (-1, 1)),
                    training=True,
                    trainable=False,
                )
                self.y_norm = tf.layers.batch_normalization(
                    tf.reshape(self.y,(-1,1)),
                    training=True,
                    trainable=False
                )
                self.avg_q_val = tf.reduce_mean(self.pred_q_value)
                self.mean_squared_error= tf.nn.l2_loss(self.y_norm -self.pred_q_value_norm)
                self.rewards = tf.placeholder(tf.float32,(None,1))
                self.avg_rew = tf.reduce_mean(self.rewards)

                self.optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.mean_squared_error)
                
                self.model_saver = tf.train.Saver()
                rospack = rospkg.RosPack()
                self.pkg_path = rospack.get_path('drone_training')
                self.is_agent = is_agent
                if self.is_agent:
                    with tf.name_scope("train"):
                        self.error_summary = tf.summary.scalar('mse_loss',self.mean_squared_error)
                        self.avg_q_val_summary = tf.summary.scalar('avg_q_val',self.avg_q_val)
                        self.avg_rew_summary = tf.summary.scalar('avg_rew',self.avg_rew)                

                    self.writer = tf.summary.FileWriter(self.pkg_path+'/logs/')
                self.init_op = tf.global_variables_initializer()
                self.rs = deque(maxlen=1000)
            self.sess = tf.Session(graph=self.graph)
        def compile(self):
            self.sess.run(self.init_op)
            
        
        def summarize_rewards(self,return_,ep):
            self.rs.append(return_)
            summary  = self.sess.run(self.avg_rew_summary, feed_dict={self.rewards: np.reshape(self.rs,(-1,1))})
            
            self.writer.add_summary(summary,ep)

        def predict(self,xs):
            return self.sess.run(self.pred_q_value,feed_dict={self.x: xs})
                
        def fit(self,x_batch,y_batch,ep):
            if self.is_agent:
                err_summary,q_summary,_,_ = self.sess.run([self.error_summary,self.avg_q_val_summary, self.optimiser, self.mean_squared_error], 
                            feed_dict={self.x: x_batch, self.y: y_batch})
                self.writer.add_summary(err_summary,ep)
                self.writer.add_summary(q_summary,ep)
            else:
                self.sess.run([self.optimiser, self.mean_squared_error], 
                            feed_dict={self.x: x_batch, self.y: y_batch})
        def save(self):
            self.model_saver.save(self.sess, self.pkg_path+"/checkpoints/dqn-final-model.ckpt")

        def load(self):
            self.model_saver.restore(self.sess,self.pkg_path+"/checkpoints/dqn-final-model.ckpt")

    def __init__(self, env, num_eps,load_model=False,gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=1.0, alpha=0.01, batch_size=128,tao=1,double_q=False):
        self.replay_memory = deque(maxlen=100000)
        self.env = env

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.n_episodes = num_eps
        self.batch_size = batch_size
        self.tao = tao
        self.double_q = double_q
        self.agent_graph = tf.Graph()
        self.agent_network = DQN.Model(self.agent_graph,self.n_states,self.n_actions,min_clip_value=-1000,max_clip_value=1000,learning_rate=self.alpha,is_agent=True)
        if load_model == False:
            # Init the model
            self.agent_network.compile()
        else:
            self.agent_network.load()
        
        self.target_graph = tf.Graph()
        self.target_network = DQN.Model(self.target_graph,self.n_states,self.n_actions,min_clip_value=-1000,max_clip_value=1000,learning_rate=self.alpha,is_agent=False)
        if self.double_q:
            self.target_network.compile()

        self.episode_num = 0
        self.target_network_restored = False
        
    def preprocess_state(self, state):
        return np.reshape(state, [1, self.env.observation_space.shape[0]])

    def choose_action(self,state,epsilon):
        if np.random.random() <= epsilon:
            to_be_rescued = np.where(self.env.rescued==False)[0]
            rtb_action = len(self.env.survivors)
            to_be_rescued = np.append(to_be_rescued,rtb_action)
            return np.random.choice(to_be_rescued)
        else:
            if self.double_q:
                action_values = self.agent_network.predict(state) + self.target_network.predict(state)
                effective_action_values = np.full(action_values.shape,float("-inf"))
                effective_action_values = np.reshape(effective_action_values,(action_values.shape[1]))
                action_values = np.reshape(action_values,(action_values.shape[1]))
                tmp = np.copy(self.env.rescued)
                tmp = np.append(tmp,False) # Treat rtb as always not-rescued.
                effective_action_values[np.where(tmp==False)] = action_values[tmp==False]
                return np.argmax(effective_action_values)
            else:
                action_values = self.agent_network.predict(state)
                effective_action_values = np.full(action_values.shape,float("-inf"))
                effective_action_values = np.reshape(effective_action_values,(action_values.shape[1]))
                action_values = np.reshape(action_values,(action_values.shape[1]))
                tmp = np.copy(self.env.rescued)
                tmp = np.append(tmp,False) # Treat rtb as always not-rescued.
                effective_action_values[np.where(tmp==False)] = action_values[tmp==False]
                return np.argmax(effective_action_values)

    def replay(self):
        if self.double_q:
            self.double_q_learn()
        else:
            self.q_learn()
    
    def q_learn(self):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.replay_memory, min(len(self.replay_memory), self.batch_size))
        for state, action, reward, next_state, done in minibatch:
            if self.target_network_restored == False:
                y_target = self.agent_network.predict(state)
            else:
                y_target = self.target_network.predict(state)
            if done:
                y_target[0][action] = reward
            else:
                if self.target_network_restored == False:
                    y_target[0][action] = reward + self.gamma * np.max(self.agent_network.predict(next_state)[0])
                else:
                    y_target[0][action] = reward + self.gamma * np.max(self.target_network.predict(next_state)[0])
            #y_target[0][action] = max(-500,min(500,y_target[0][action]))
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        self.agent_network.fit(x_batch,y_batch,self.episode_num)

    def double_q_learn(self,batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.replay_memory, min(len(self.replay_memory), self.batch_size))
        for state, action, reward, next_state, done in minibatch:
            if np.random.random() <= 0.5:
                q_state = self.agent_network.predict(state)
                q_next_state = self.agent_network.predict(next_state)
                q_target = self.target_network.predict(next_state)
                action_ = np.argmax(q_next_state)
                if done:
                    q_state[0][action] = reward
                else:
                    q_state[0][action] = reward + self.gamma * q_target[0][action_]
                #y_target[0][action] = max(-1000,min(1000,y_target[0][action]))
                x_batch.append(state[0])
                y_batch.append(q_state[0])
                self.agent_network.fit(x_batch,y_batch,self.episode_num)
            else:
                q_state = self.target_network.predict(state)
                q_next_state = self.target_network.predict(next_state)
                q_target = self.agent_network.predict(next_state)
                action_ = np.argmax(q_next_state)
                if done:
                    q_state[0][action] = reward
                else:
                    q_state[0][action] = reward + self.gamma * q_target[0][action_]
                #y_target[0][action] = max(-1000,min(1000,y_target[0][action]))
                x_batch.append(state[0])
                y_batch.append(q_state[0])
                self.target_network.fit(x_batch,y_batch,self.episode_num)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_epsilon(self,t):
        return max(self.epsilon_min,min(self.epsilon,1.0 - math.log10((t+1)*self.epsilon_decay)))