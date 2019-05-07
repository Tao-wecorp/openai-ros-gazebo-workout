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
import qlearn
import itertools
from gym import wrappers

# ROS packages required
import rospy
import rospkg

# import our training environment
import quadcopter_env

from openai_ros.msg import RLExperimentInfo
from std_msgs.msg import Bool
if __name__ == '__main__':
    
    rospy.init_node('drone_gym', anonymous=True)

    # Create the Gym environment
    env = gym.make('QuadcopterLiveShow-v0')
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
    experiment = rospy.get_param("/experiment")
    
    minx = rospy.get_param("/limits/min_x")
    maxx = rospy.get_param("/limits/max_x")
    miny = rospy.get_param("/limits/min_y")
    maxy = rospy.get_param("/limits/max_y")
    minz = rospy.get_param("/limits/min_altitude")
    maxz = rospy.get_param("/limits/max_altitude")

    
    # Start Reward publishing
    reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)
    done_pub = rospy.Publisher('/openai/done', Bool, queue_size=1)
    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.TabularQLearn(actions=range(env.action_space.n),
                    alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon
    #qlearn.Q = np.load(outdir+'/q_value5.npy',Q)
    env_shape = (3,10,10)
    qlearn.init_q(env_shape)
    horizontal_bins = np.zeros((2,10))
    vertical_bin = np.zeros((3))
    
    horizontal_bins[0] = np.linspace(minx,maxx,10)
    horizontal_bins[1] = np.linspace(miny,maxy,10)
    vertical_bin = np.linspace(minz,maxz,3)

    start_time = time.time()

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.loginfo ("STARTING Episode #"+str(x))
        
        cumulated_reward = 0  
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount
        
        # Initialize the environment and get first state of the robot
        state = env.reset()
        
        # for each episode, we test the robot for nsteps
        for t in itertools.count():
            print "Step %i"%(t)
            state_ = np.zeros(3)
            state_[0] = int(np.digitize(state[2],vertical_bin))# z first
            state_[1] = int(np.digitize(state[0],horizontal_bins[0]))
            state_[2] = int(np.digitize(state[1],horizontal_bins[1]))
            #Clip the state
            for j in range(3):
                if state_[j] < 0:
                    state_[j] = 0
                elif state_[j] > env_shape[j]-1:
                    state_[j] = env_shape[j]-1

            # Pick an action based on the current state
            action = qlearn.chooseAction(tuple(state_))
            
            # Execute the action in the environment and get feedback
            next_state, reward, done, info = env.step(action)
            cumulated_reward += reward
            next_state_ = np.zeros(3)

            next_state_[0] = int(np.digitize(next_state[2],vertical_bin)) # z first
            next_state_[1] = int(np.digitize(next_state[0],horizontal_bins[0]))
            next_state_[2] = int(np.digitize(next_state[1],horizontal_bins[1]))
            for j in range(3):
                if next_state_[j] < 0:
                    next_state_[j] = 0
                elif next_state_[j] > env_shape[j]-1:
                    next_state_[j] = env_shape[j]-1

            # Make the algorithm learn based on the results
            qlearn.learn(tuple(state_), action, reward,tuple(next_state_))

            if not(done):
                state = next_state
            else:
                rospy.loginfo ("DONE")
                last_time_steps = np.append(last_time_steps, [int(t + 1)])
                reward_msg = RLExperimentInfo()
                reward_msg.episode_number = x
                reward_msg.episode_reward = cumulated_reward
                reward_pub.publish(reward_msg)
                break 

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.loginfo ( ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s)))

    done_msg = Bool()
    done_msg.data = True
    done_pub.publish(done_msg)
    
    l = last_time_steps.tolist()
    l.sort()
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
    # Save the q_value for latter usage.
    np.save(outdir+'/q_value'+str(experiment)+'.npy',qlearn.Q)

