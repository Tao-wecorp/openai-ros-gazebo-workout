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

#register the training environment in the gym as an available one
reg = register(
    id='QuadcopterLiveShow-v0',
    entry_point='discrete_quadcopter_env:DiscreteQuadCopterEnv',
    max_episode_steps=100,
    )

FORWARD=0
BACKWARD=1
LEFT=2
RIGHT=3
UP=4
DOWN=5

ACTIONS = [FORWARD,BACKWARD,LEFT,RIGHT,UP,DOWN]
class DiscreteQuadCopterEnv(gym.Env):

    def __init__(self):
        
        # We assume that a ROS node has already been created
        # before initialising the environment
        
        self.pos_pub = rospy.Publisher('/drone/cmd_pos',Point,queue_size=1)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.takeoff_pub = rospy.Publisher('/drone/takeoff', EmptyTopicMsg, queue_size=0)
        self.switch_pub = rospy.Publisher('/drone/posctrl',Bool,queue_size=1)
        # gets training parameters from param server
        self.desired_pose = Pose()
        self.desired_pose.position.z = rospy.get_param("/desired_pose/z")
        self.desired_pose.position.x = rospy.get_param("/desired_pose/x")
        self.desired_pose.position.y = rospy.get_param("/desired_pose/y")
        
        self.running_step = rospy.get_param("/running_step")
        self.max_incl = rospy.get_param("/max_incl")
        self.minx = rospy.get_param("/limits/min_x")
        self.maxx = rospy.get_param("/limits/max_x")
        self.miny = rospy.get_param("/limits/min_y")
        self.maxy = rospy.get_param("/limits/max_y")
        #self.max_altitude = rospy.get_param("/limits/max_altitude")
        #self.min_altitude = rospy.get_param("/limits/min_altitude")
        self.shape = (10,10)

        self.incx = (self.maxx - self.minx)/ (self.shape[0])
        self.incy = (self.maxy - self.miny)/ (self.shape[1])
        #self.incz = (self.max_altitude - self.min_altitude)/ self.shape[0]

        self.horizontal_bins = np.zeros((2,self.shape[1]))
        #self.vertical_bin = np.zeros((self.shape[0]))
        
        self.horizontal_bins[0] = np.linspace(self.minx,self.maxx,self.shape[1])
        self.horizontal_bins[1] = np.linspace(self.miny,self.maxy,self.shape[1])
        #self.vertical_bin = np.linspace(self.min_altitude,self.max_altitude,self.shape[0])
        self.goal = np.zeros(2)
        #self.goal[0] = int(np.digitize(self.desired_pose.position.z,self.vertical_bin))
        self.goal[0] = int(np.digitize(self.desired_pose.position.x,self.horizontal_bins[0]))
        self.goal[1] = int(np.digitize(self.desired_pose.position.y,self.horizontal_bins[1]))
        rospy.loginfo("Goal: %s"%(self.goal))
        # stablishes connection with simulator
        self.gazebo = GazeboConnection()
        
        self.action_space = spaces.Discrete(4) #Forward,Backward,Left,Right
        self.reward_range = (-np.inf, np.inf)

        self._seed()
        self.init_desired_pose()

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
        
        self.takeoff_sequence()
        self.switch_position_control()
        # 4th: takes an observation of the initial condition of the robot
        data_pose, data_imu = self.take_observation()
        observation = [data_pose.position.x,data_pose.position.y,data_pose.position.z]

        return observation

    def _step(self, action):

        # Given the action selected by the learning algorithm,
        # we perform the corresponding movement of the robot
        
        pose,_ = self.take_observation()
        rospy.loginfo("Observation has taken")
        pos_cmd = pose.position
        pos_cmd.z = 1.0
        if action == FORWARD:
            pos_cmd.x += self.incx
        elif action == BACKWARD:
            pos_cmd.x -= self.incx
        elif action == RIGHT:
            pos_cmd.y += self.incy
        elif action == LEFT: 
            pos_cmd.y -= self.incy
        #elif action == UP:
        #    pos_cmd.z += self.incz
        #elif action == DOWN:
        #    pos_cmd.z -= self.incz
        # Then we send the command to the robot and let it go
        # for running_step seconds
        #self.gazebo.unpauseSim()
        self.pos_pub.publish(pos_cmd)
        time.sleep(self.running_step)
        data_pose, data_imu = self.take_observation()
        #self.gazebo.pauseSim()

        # finally we get an evaluation based on what happened in the sim
        reward,done = self.process_data(data_pose, data_imu)
        state = [data_pose.position.x,data_pose.position.y]
        return state, reward, done, {}

    def _render(self, mode, close=True):
        pass
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


    def init_desired_pose(self):
        
        current_init_pose, imu = self.take_observation()
        
        self.best_dist = self.calculate_dist_between_two_points(current_init_pose.position, self.desired_pose.position)
    

    def check_topic_publishers_connection(self):
        
        rate = rospy.Rate(10) # 10hz
        while(self.takeoff_pub.get_num_connections() == 0):
            rospy.loginfo("No subscribers to Takeoff yet so we wait and try again")
            rate.sleep()
        rospy.loginfo("Takeoff Publisher Connected")

        while(self.vel_pub.get_num_connections() == 0):
            rospy.loginfo("No subscribers to Cmd_vel yet so we wait and try again")
            rate.sleep()
        rospy.loginfo("Cmd_vel Publisher Connected")
        

    def reset_cmd_vel_commands(self):
        # We send an empty null Twist
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.linear.y = 0.0
        vel_cmd.linear.z = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)

    def switch_position_control(self):
        msg = Bool()
        msg.data = True
        self.switch_pub.publish(msg)
        rospy.loginfo("Switched to position control")

    def takeoff_sequence(self, seconds_taking_off=2):
        # Before taking off be sure that cmd_vel value there is is null to avoid drifts
        self.reset_cmd_vel_commands()
        
        takeoff_msg = EmptyTopicMsg()
        rospy.loginfo( "Taking-Off Start")
        pos_cmd = Point()
        pos_cmd.z = 1.0
        self.pos_pub.publish(pos_cmd)
        self.takeoff_pub.publish(takeoff_msg)
        time.sleep(seconds_taking_off)
        rospy.loginfo( "Taking-Off sequence completed")
        
        
    def process_data(self, data_position, data_imu):
        done = False
        euler = tf.transformations.euler_from_quaternion([data_imu.orientation.x,data_imu.orientation.y,data_imu.orientation.z,data_imu.orientation.w])
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]
        
        state = np.zeros(2)
        #state[0] = int(np.digitize(data_position.position.z,self.vertical_bin))# z first
        state[0] = int(np.digitize(data_position.position.x,self.horizontal_bins[0]))
        state[1] = int(np.digitize(data_position.position.y,self.horizontal_bins[1]))
        #invalid_altitude = state[0] == 0 or state[0] == self.shape[0]-1
        if tuple(state) == tuple(self.goal):
            done = True
            reward = 0
        #elif invalid_altitude:
        #    reward = -50000 # Punish hardly
        #    done = True
        else:
            reward = -1
        return reward,done