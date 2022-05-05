#!/usr/env/bin python

import math
import rospy
import numpy as np
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Twist, Vector3, Point, PoseWithCovarianceStamped, Pose


# from tf.transformations import euler_from_quaternion

class AgentPort(object):
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.taken_action = None
        self.pos_global_frame = None
        self.radianR = None
        self.radianP = None
        self.radianY = None
        self.angleR = None
        self.angleP = None
        self.angleY = None
        self.goal_global_frame = None
        self.new_goal_received = False

        self.str_pub_twist = '/' + self.name + '/cmd_vel'
        self.pub_twist = rospy.Publisher(self.str_pub_twist, Twist, queue_size=1)

        self.str_pub_twist_real = '/' + 'robot0' + '/cmd_vel'
        self.pub_twist_real = rospy.Publisher(self.str_pub_twist_real, Twist, queue_size=1)

        self.str_pub_path = '/' + self.name + '/trajectory'
        self.pub_path = rospy.Publisher(self.str_pub_path, Path, queue_size=1)

        self.str_pose = '/' + self.name + '/pose'
        self.str_odom = '/' + self.name + '/odom'
        self.str_goal = '/' + self.name + '/move_base_simple/goal'
        self.str_robot_pose = '/' + self.name + '/' + self.name + '/robot_pose'
        self.sub_odom = rospy.Subscriber(self.str_odom, Odometry, self.cbOdom)
        # self.sub_robot_pose = rospy.Subscriber(self.str_robot_pose, Pose, self.cbPose)

    def cbOdom(self, msg):
        self.odom = msg
        self.pos_global_frame = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        # (roll, pitch, yaw) = euler_from_quaternion([qx, qy, qz, qw])  # 将四元数转化为roll, pitch, yaw
        self.radianR = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        self.radianP = math.asin(2 * (qw * qy - qz * qx))
        self.radianY = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qz * qz + qy * qy))

        self.angleR = self.radianR * 180 / math.pi  # 横滚角
        self.angleP = self.radianP * 180 / math.pi  # 俯仰角
        self.angleY = self.radianY * 180 / math.pi  # 偏航角

        self.v_x = msg.twist.twist.linear.x
        self.w_z = msg.twist.twist.angular.z

        self.vel_global_frame = np.array([math.cos(self.angleY), math.sin(self.angleY)]) * self.v_x

    def cbPose(self, msg):
        self.pos_global_frame = np.array([msg.position.x, msg.position.y])

    def stop_moving(self):
        twist = Twist()
        self.pub_twist.publish(twist)

    def pubTwist(self, action, dt, agent_id):
        self.taken_action = action
        twist = Twist()
        pose = Path()
        twist.linear.x = round(1.0 * action[0], 5)
        twist.angular.z = round(0.22 * 0.5 * action[1] / dt, 5)

        # test
        # twist.linear.x = round(0.2, 5)
        # twist.angular.z = 0.0
        # twist.angular.z = round(0.22 * 0.5 * action[1] / dt, 5)
        self.pub_twist.publish(twist)

        self.pub_path.publish(pose)

        # self.pub_twist_real.publish(twist)

        # RVO Para
        # if agent_id == 0:       # Agent1
        #     twist.linear.x = round(1.1 * action[0], 5)
        #     twist.angular.z = round(0.22 * 0.5 * action[1] / dt, 5)
        #     self.pub_twist_real.publish(twist)
        #     if action[1] > 0:
        #         twist.angular.z = round(1.5*0.18 * 0.5 * action[1] / dt, 5)
        #     else:
        #         twist.angular.z = round(1.5*0.11 * 0.5 * action[1] / dt, 5)
        # elif agent_id == 1:       # Agent2
        #     if action[1] > 0:
        #         twist.angular.z = round(1.5*0.13 * 0.5 * action[1] / dt, 5)
        #     else:
        #         twist.angular.z = round(1.5*0.13 * 0.5 * action[1] / dt, 5)
        # elif agent_id == 2:       # Agent3
        #     if action[1] > 0:
        #         twist.angular.z = round(1.6*0.10 * 0.5 * action[1] / dt, 5)
        #     else:
        #         twist.angular.z = round(1.6*0.15 * 0.5 * action[1] / dt, 5)
        # elif agent_id == 3:       # Agent4
        #     if action[1] > 0:
        #         twist.angular.z = round(1.5*0.16 * 0.5 * action[1] / dt, 5)
        #     else:
        #         twist.angular.z = round(1.5*0.12 * 0.5 * action[1] / dt, 5)

        # SCA or Dubins Para
        # if agent_id == 0:       # Agent1
        #     if action[1] > 0:
        #         twist.angular.z = round(1.5*0.12 * 0.5 * action[1] / dt, 5)
        #     else:
        #         twist.angular.z = round(1.5*0.12 * 0.5 * action[1] / dt, 5)
        # elif agent_id == 1:       # Agent2
        #     if action[1] > 0:
        #         twist.angular.z = round(1.6*0.13 * 0.5 * action[1] / dt, 5)
        #     else:
        #         twist.angular.z = round(1.6*0.13 * 0.5 * action[1] / dt, 5)
        # elif agent_id == 2:       # Agent3
        #     if action[1] > 0:
        #         twist.angular.z = round(1.5*0.10 * 0.5 * action[1] / dt, 5)
        #     else:
        #         twist.angular.z = round(1.5*0.15 * 0.5 * action[1] / dt, 5)
        # elif agent_id == 3:       # Agent4
        #     if action[1] > 0:
        #         twist.angular.z = round(1.5*0.16 * 0.5 * action[1] / dt, 5)
        #     else:
        #         twist.angular.z = round(1.5*0.12 * 0.5 * action[1] / dt, 5)
        # self.pub_twist_real.publish(twist)

    #    def cbRobotPose1(self, msg):
    #        self.obstacle_list[0] = Obstacle(pos = np.array([msg.pose.position.x, msg.pose.position.y]), shape_dict = { 'shape' : "circle", 'feature' : 0.2})

    #    def cbRobotPose2(self, msg):
    #        self.obstacle_list[1] = Obstacle(pos = np.array([msg.pose.position.x, msg.pose.position.y]), shape_dict = { 'shape' : "circle", 'feature' : 0.2})

    def getRobotPose(self):
        return self.pos_global_frame

    def getEulerRadian(self):
        return self.radianR, self.radianP, self.radianY

    def getGoalPose(self):
        return self.goal_global_frame
