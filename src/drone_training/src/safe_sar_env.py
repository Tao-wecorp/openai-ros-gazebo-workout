#!/usr/bin/env python

import gym
import rospy
import time
import numpy as np
import tf
import time
from gym import utils, spaces
from geometry_msgs.msg import Twist, Vector3Stamped, Pose, Point
from hector_uav_msgs.msg import Altimeter
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty as EmptyTopicMsg, Bool
from gym.utils import seeding
from gym.envs.registration import register
from gazebo_connection import GazeboConnection
from gazebo_msgs.msg import ModelStates

#register the training environment in the gym as an available one
reg = register(
    id='QuadSafeSAR-v1',
    entry_point='safe_sar_env:SafeSAREnv',
    max_episode_steps=100,
    )

class SafeSAREnv(gym.Env):

    def __init__(self):
        
        # We assume that a ROS node has already been created
        # before initialising the environment
        
        self.pos_pub = rospy.Publisher('/drone/cmd_pos',Point,queue_size=1)
        self.takeoff_pub = rospy.Publisher('/drone/takeoff', EmptyTopicMsg, queue_size=0)
        self.switch_pub = rospy.Publisher('/drone/posctrl',Bool,queue_size=1)
        
        self.running_step = rospy.get_param("/running_step")
        self.minx = rospy.get_param("/limits/min_x")
        self.maxx = rospy.get_param("/limits/max_x")
        self.miny = rospy.get_param("/limits/min_y")
        self.maxy = rospy.get_param("/limits/max_y")
        self.base = Point()
        self.base.x = rospy.get_param("/base/x")
        self.base.y = rospy.get_param("/base/y")
        self.base.z = rospy.get_param("/base/z")
        self.gazebo = GazeboConnection()
        # Get number of survivors
        self.survivors = []
        self.get_survivor_information()
        self.rescued = np.full((len(self.survivors)), False)
        # All of the survivors + return to base.
        self.action_space = spaces.Discrete(len(self.survivors)+1)
        self.num_actions = len(self.survivors)+1
        self.reward_range = (-np.inf, np.inf)
        
        self.battery = 100
        self.battery_depletion = rospy.get_param("/battery_depletion")
        self.battery_timer = rospy.Timer(rospy.Duration(2), self.battery_timer_callback)
        area_limits_high = np.array([
            self.maxx, #x
            self.maxy, #y
            100  #battery
            ])
        area_limits_low = np.array([
            self.minx,
            self.miny,
            0
            ])
        self.observation_space = spaces.Box(area_limits_low,area_limits_high)
        self._seed()

    # A function to initialize the random generator
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    # Resets the state of the environment and returns an initial observation.
    def _reset(self):
        
        # 1st: resets the simulation to initial values
        self.gazebo.resetSim()

        # 2nd: Unpauses simulation
        self.gazebo.unpauseSim()

        # 3rd: resets the robot to initial conditions
        self.check_topic_publishers_connection()
        self.reset_battery()
        self.rescued = np.full((len(self.survivors)),False)
        self.takeoff_sequence()
        self.switch_position_control()
        # 4th: takes an observation of the initial condition of the robot
        data_pose, data_imu = self.take_observation()
        observation = [data_pose.position.x,data_pose.position.y,self.battery]

        return observation

    def _step(self, action):

        # Action selection means the rescuing of the survivor.
        pos_cmd = Point()
        if action < len(self.survivors):
            rospy.loginfo("Chosen to rescue %dth survivor"%action)
            pos_cmd.z = 5.0
            pos_cmd.x = self.survivors[action][1].x
            pos_cmd.y = self.survivors[action][1].y
        else:
            rospy.loginfo("Chosen to return to base..")
            pos_cmd = self.base
        data_pose, _ =  self.take_observation()
        dist = self.calculate_dist_between_two_points(data_pose.position,pos_cmd)
        while dist > 0.2:
            self.pos_pub.publish(pos_cmd)
            rospy.sleep(self.running_step)
            data_pose, _ =  self.take_observation()
            dist = self.calculate_dist_between_two_points(data_pose.position,pos_cmd)
        if action < len(self.rescued):# Strictly small means it is indeed a survivor.
            self.rescued[action] = True

        # finally we get an evaluation based on what happened in the sim
        reward,done = self.process_data(data_pose)
        state = [data_pose.position.x,data_pose.position.y,self.battery]
        return state, reward, done, {}

    def _render(self, mode, close=True):
        pass
    
    def get_survivor_information(self):
        all_models = None
        while all_models is None:
            try:
                all_models = rospy.wait_for_message('/gazebo/model_states',ModelStates)
            except:
                pass
        for i in range(len(all_models.name)):
            if all_models.name[i].startswith('survivee'):
                self.survivors.append((all_models.name[i],all_models.pose[i].position))
        
    def take_observation (self):
        data_pose = None
        while data_pose is None:
            try:
                data_pose = rospy.wait_for_message('/drone/gt_pose', Pose, timeout=5)
            except:
                pass

        data_imu = None
        while data_imu is None:
            try:
                data_imu = rospy.wait_for_message('/drone/imu', Imu, timeout=5)
            except:
                pass
        
        return data_pose, data_imu

    def calculate_dist_between_two_points(self,p_init,p_end):
        a = np.array((p_init.x ,p_init.y, p_init.z))
        b = np.array((p_end.x ,p_end.y, p_end.z))
        
        dist = np.linalg.norm(a-b)
        
        return dist

    def check_topic_publishers_connection(self):
        
        rate = rospy.Rate(10) # 10hz
        while(self.takeoff_pub.get_num_connections() == 0):
            rospy.loginfo("No subscribers to Takeoff yet so we wait and try again")
            rate.sleep()
        rospy.loginfo("Takeoff Publisher Connected")

    def switch_position_control(self):
        msg = Bool()
        msg.data = True
        self.switch_pub.publish(msg)
        rospy.loginfo("Switched to position control")

    def takeoff_sequence(self, seconds_taking_off=2):
        
        takeoff_msg = EmptyTopicMsg()
        rospy.loginfo( "Taking-Off Start")
        pos_cmd = Point()
        pos_cmd.z = 5.0
        self.pos_pub.publish(pos_cmd)
        self.takeoff_pub.publish(takeoff_msg)
        rospy.sleep(seconds_taking_off)
        rospy.loginfo( "Taking-Off sequence completed")
        
    def reset_battery(self):
        self.battery_timer.shutdown()
        self.battery_timer = rospy.Timer(rospy.Duration(2), self.battery_timer_callback)
        self.battery = 100

    def battery_timer_callback(self,event):
        self.battery -= self.battery_depletion
        rospy.loginfo("Battery: %lf",self.battery)
        
    def process_data(self,pose):
        done = False
        
        if np.all(self.rescued) and self.calculate_dist_between_two_points(pose.position,self.base) < 0.5:
            done = True
            reward = self.battery
        elif self.battery < 5: # Not all of the survivors have been rescued and battery depleted too much. Punish harshly.
            reward = -1000
            done = True
        else:
            reward = -1
        return reward,done