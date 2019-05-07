#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import rospy
import rospkg
from openai_ros.msg import RLExperimentInfo
from std_msgs.msg import Bool
fig = plt.gcf()
fig.show()

fig.canvas.draw()

class RewardChart():
    def __init__(self,episodes):
        self.n_episodes = episodes
        self.rewards = np.zeros(episodes)
        self.reward_sub = rospy.Subscriber('/openai/reward',RLExperimentInfo,self.reward_callback)
        self.done_sub = rospy.Subscriber('/openai/done',Bool,self.done_callback)
        self.experiment = rospy.get_param('/experiment')
        self.current_episode = 0
        self.done = False

    def run(self):
        xs = np.arange(0,n_episodes,1)
        while not self.done:
            fig.clf()
            plt.plot(xs,self.rewards)
            plt.xlabel("Episodes")
            plt.ylabel("Rewards")
            fig.canvas.draw()
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('drone_training')
        outdir = pkg_path + '/training_results'
        plt.title('Experiment-'+str(self.experiment))
        plt.savefig(outdir+'/experiment-'+str(self.experiment)+'.png')

    def reward_callback(self,msg):
        self.rewards[msg.episode_number] = msg.episode_reward
        self.current_episode = msg.episode_number
    
    def done_callback(self,msg):
        self.done = msg.data



if __name__=='__main__':
    rospy.init_node('reward_chat',anonymous=True,log_level=rospy.FATAL)
    n_episodes = rospy.get_param('/nepisodes')
    reward_chart = RewardChart(n_episodes)
    reward_chart.run()
    