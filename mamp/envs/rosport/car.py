#!/usr/env/bin python
import numpy as np
import math
import rospy
from nav_msgs.msg import Odometry
from mrobot.msg import EnvInfo
from geometry_msgs.msg import Twist


class RosCarPort(object):

    """
    循迹机器人ros接口类
    """

    def __init__(self, agent):
        self.id   = agent.id
        self.name = agent.name
        self.group = agent.group # 0 表示 leader 1 表示 follower
        self.radianR = None
        self.radianP = None
        self.radianY = None
        self.angleR = None
        self.angleP = None
        self.angleY = None

        self.str_pub_twist = '/' + self.name + '/cmd_vel'
        self.pub_twist = rospy.Publisher(self.str_pub_twist,Twist,queue_size=1)
        self.str_odom = '/' + self.name + '/odom'
        self.sub_odom = rospy.Subscriber(self.str_odom, Odometry, self.pos_callback)
        self.agent = agent
        if self.group == 0:
            self.detect_sub = rospy.Subscriber("env_feedback", EnvInfo, \
                                                self.error_callback)

    def pos_callback(self, data):
        """require pos info"""

        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        dist = (x-self.agent.pose_global_frame[0])**2 + \
                                            (y-self.agent.pose_global_frame[1])**2
        self.agent.distance += math.sqrt(dist)

        self.agent.pose_list.append([x, y])
           
        self.agent.v_x = data.twist.twist.linear.x
        self.agent.w_z = data.twist.twist.angular.z
        self.agent.pose_global_frame = np.array([x,y])

        """
        require angle info
        """

        qx = data.pose.pose.orientation.x
        qy = data.pose.pose.orientation.y
        qz = data.pose.pose.orientation.z
        qw = data.pose.pose.orientation.w

        self.radianR = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        self.radianP = math.asin(2 *  (qw * qy - qz * qx))
        self.radianY = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qz * qz + qy * qy))

        self.angleR = self.radianR * 180 / math.pi # 横滚角
        self.angleP = self.radianP * 180 / math.pi # 俯仰角
        self.angleY = self.radianY * 180 / math.pi # 偏航角

        self.agent.heading_global_frame = self.radianY

        self.agent.vel_global_frame = [math.cos(self.radianY)*self.agent.v_x, 
                                        math.sin(self.radianY)* self.agent.v_x]


    def error_callback(self, data):
        """require error info"""

        e_x = data.error_x
        e_k = data.error_k
        self.agent.cx = data.cx
        self.agent.cy = data.cy
        self.agent.failed_flag = data.failed_flag
        self.agent.error.pop()
        self.agent.error.insert(0, e_x)
        self.agent.error_k.pop()
        self.agent.error_k.insert(0, e_k)
        self.agent.errorx_list.append(self.agent.error[0])
        self.agent.last_error = self.agent.error[0]
    
    def pubTwist(self, action, dt):
        twist = Twist()
        if self.group == 0 :
            twist.linear.x = action[0]
            twist.angular.z = action[1] 
        else:
            twist.linear.x  = action[0]
            twist.angular.z = action[1] / dt
            twist.angular.z = np.clip(twist.angular.z, -np.pi/2, np.pi/2)

        self.pub_twist.publish(twist)

    def stop_moving(self):
        twist = Twist()
        self.pub_twist.publish(twist)
    

        
       




