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
import itertools
from gym import wrappers

# ROS packages required
import rospy
import rospkg

# import our training environment
from DQN_SAR import DQN_SAR
import safe_sar_env
from tqdm import tqdm

from openai_ros.msg import RLExperimentInfo
from std_msgs.msg import Bool
    
if __name__ == '__main__':
    
    rospy.init_node('drone_gym', anonymous=True)

    # Create the Gym environment
    env = gym.make('QuadSafeSAR-v1')
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
    agent = DQN_SAR(env,nepisodes,load_model=True,gamma=0.8,epsilon=0.05,epsilon_min=0.01,epsilon_log_decay=0.00001,alpha=0.001,batch_size=128,tao=5,double_q=False)
    ep_len = agent.run()
    env.close()

