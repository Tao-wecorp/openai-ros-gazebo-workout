#!/usr/bin/env python

'''
    Training code made by Ricardo Tellez <rtellez@theconstructsim.com>
    Based on many other examples around Internet
    Visit our website at www.theconstruct.ai
'''
import gym
import time
import numpy as np
import random
import time
import DQN
import itertools
from gym import wrappers

# ROS packages required
import rospy
import rospkg

# import our training environment
import continuous_quadcopter_env
from tqdm import tqdm

from openai_ros.msg import RLExperimentInfo
from std_msgs.msg import Bool
    
if __name__ == '__main__':
    
    rospy.init_node('drone_gym', anonymous=True)

    # Create the Gym environment
    env = gym.make('QuadcopterLiveShow-v1')
    rospy.loginfo ( "Gym environment done")
        
    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('drone_training')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True) 
    rospy.loginfo ( "Monitor Wrapper started")

    last_time_steps = np.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/alpha")
    Epsilon = rospy.get_param("/epsilon")
    Gamma = rospy.get_param("/gamma")
    epsilon_discount = rospy.get_param("/epsilon_discount")
    nepisodes = rospy.get_param("/nepisodes")
    
    minx = rospy.get_param("/limits/min_x")
    maxx = rospy.get_param("/limits/max_x")
    miny = rospy.get_param("/limits/min_y")
    maxy = rospy.get_param("/limits/max_y")

    # Initialises the algorithm that we are going to use for learning
    agent = DQN.DQN(env,nepisodes,load_model=True,gamma=0.8,epsilon=0.05,epsilon_min=0.01,epsilon_log_decay=0.00001,alpha=0.001,batch_size=128,tao=5,double_q=False)
    t = 0
    # Starts the main training loop: the one about the episodes to do
    for e in tqdm(range(0,nepisodes),desc='DQN Learning'):
        rospy.loginfo ("STARTING Episode #"+str(e))
        state = agent.preprocess_state(env.reset())
        done = False
        returns = 0
        while done == False:
            action = agent.choose_action(state,agent.get_epsilon(e))
            next_state,reward,done,_ = env.step(action)
            next_state = agent.preprocess_state(next_state)
            
            agent.replay_memory.append((state,action,reward,next_state,done))
            state = next_state
            returns += reward
        agent.replay()
        agent.agent_network.summarize_rewards(returns,e)
        agent.episode_num += 1
        if agent.double_q:
            agent.agent_network.save()
        else:
            if t == agent.tao:
                agent.target_network_restored = True
                agent.agent_network.save() # Save current model with weights and bias
                agent.target_network.load() # Load current model with weights and bias for target network.
                t = 0
            else:
                t += 1

    env.close()

