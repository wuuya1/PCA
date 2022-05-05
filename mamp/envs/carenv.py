#! /usr/bin/env python3
import gym
import rospy
import numpy as np
import time
from mamp.envs.rosport.car import RosCarPort
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
from math import sqrt, sin, cos, atan2
from copy import deepcopy

ERROR_MAX = 1


class CAREnv(gym.Env):
    """
    line follower robot environment
    多机器人编队环境:
    1位领导者:通过强化学习算法来学习PID系数完成循迹任务
    n位跟随者:跟随领导者前进,并且保持编队秩序
    """

    def __init__(self):
        super().__init__()
        # information for rl
        self.reward = 0
        self.state = np.zeros(13)
        self.action_space = np.zeros(6)
        self.state_space = np.zeros(13)
        # information recorded
        self.success_record = []
        self.game_over = False
        self.node = rospy.init_node('line_following_node', anonymous=False)
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.rate = rospy.Rate(10)
        self.count = 0
        # agent 
        self.agents = None

    def reset(self):
        # print('mean_v:', self.agents[0].distance/(time.time()-self.time))
        self.reset_world()
        for ag in self.agents:
            ag.stop()
        time.sleep(1)
        self.game_over = False
        # if self.agents[0].failed_flag:
        #     self.success_record.append(0)
        #     print("fail!")
        if self.agents[0].success_flag:
            self.success_record.append(1)
            print("success!")
        else:
            self.success_record.append(0)
        self.time = time.time()
        for ag in self.agents:
            ag.reset()
        self.agents[0].set_pos([0, 0])
        self.agents[1].set_pos([-1, 1])
        self.agents[2].set_pos([-1, -1])
        self.agents[3].set_pos([-2, 0])
        obs = self._get_obs()
        return obs

    def set_start(self, model_name, pos, heading):
        """
        设置gazebo中机器人的位置和姿态
        """
        x, y = pos[0], pos[1]
        state = ModelState()
        state.model_name = model_name
        state.reference_frame = 'world'  # ''ground_plane'
        # pose
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0
        #  quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)
        state.pose.orientation.x = 0
        state.pose.orientation.y = 0
        state.pose.orientation.z = np.sin(heading / 2)
        state.pose.orientation.w = np.cos(heading / 2)
        # twist
        state.twist.linear.x = 0
        state.twist.linear.y = 0
        state.twist.linear.z = 0
        state.twist.angular.x = 0
        state.twist.angular.y = 0
        state.twist.angular.z = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = self.set_state
            result = set_state(state)
            assert result.success is True
        except rospy.ServiceException:
            print("/gazebo/get_model_state service call failed")

    def set_agents(self, agents, neighbor_info):
        '''
        neighbor_info = [{1: [-1, 0], 3: [0, 1]},
                         {0: [1, 0],  2: [0, 1]},
                         {1: [0, -1], 3: [1, 0]},
                         {0: [0, -1], 2: [-1, 0]},]
        '''
        self.agents = agents
        self.neighbor_info = deepcopy(neighbor_info)  # complete formation matrix
        # 设置临接矩阵
        num_agent = len(agents)
        adj_matrix = np.zeros((num_agent, num_agent))
        for i, info in enumerate(neighbor_info):
            neighbor = info.keys()
            for j in neighbor:
                adj_matrix[i, j] = 1
                self.agents[i].neighbor.append(j)
            print(self.agents[i].name, self.agents[i].neighbor)
        self.adjMatrix = adj_matrix
        # 绑定智能体
        for ag in self.agents:
            ag.neighbor_info = neighbor_info[ag.id]
            ag.reset()
            ag.rosport = RosCarPort(ag)

    def _check_if_done(self):
        """
        检查是否回合结束
        """
        dones = []
        # check which agent is already done
        for ag in self.agents:
            if ag.done == True:
                ag.quit = True
        if abs(self.agents[0].pose_global_frame[0]) < 0.5 and \
                abs(self.agents[0].pose_global_frame[1]) < 0.5 and (time.time() - self.time) > 20:
            for ag in self.agents:
                ag.success_flag = True

        for ag in self.agents:
            if ag.group == 1:
                if abs(ag.error[0][0]) > ERROR_MAX or abs(ag.error[0][1]) > ERROR_MAX:
                    ag.failed_flag = True
            if ag.success_flag == True or ag.failed_flag == True:
                ag.done = True
            dones.append(ag.done)
        if self.agents[0].done == True:
            self.game_over = True
        if any([ag.done for ag in self.agents if ag.group == 1]):
            self.game_over = True
        return dones

    def _compute_rewards(self):
        rewards = []
        for ag in self.agents:
            if ag.quit:
                rewards.append(0)
            else:
                var_e = np.var(ag.error)
                dist = ag.distance - ag.last_distance
                reward = 2 * dist - var_e
                if ag.group == 0:
                    reward = 2 * dist - 10 * var_e
                    reward -= 0.1
                if ag.done:
                    if ag.success_flag:
                        reward = 5
                    else:
                        reward = -5
                rewards.append(reward)
                ag.last_distance = ag.distance
        return rewards

    def _get_obs(self):
        """
        获取机器人完整的状态观测值
        """
        ob = []
        self.agents[0].state = np.append(self.agents[0].cx, self.agents[0].cy)
        state2 = [self.agents[0].v_x, self.agents[0].w_z, self.agents[0].error_k[0],
                  self.agents[0].distance / 45.0]
        self.agents[0].state = np.append(self.agents[0].state, state2)
        ob.append(self.agents[0].state)
        for ag in self.agents:
            if ag.group == 1:
                ag.state = np.array([ag.error[0][0], ag.error[0][1], ag.v_x, ag.w_z,
                                     self.agents[ag.neighbor[0]].v_x, self.agents[ag.neighbor[0]].w_z,
                                     self.agents[ag.neighbor[1]].v_x, self.agents[ag.neighbor[1]].w_z,
                                     ag.distance / 45.0])  # 或许可以多加几个偏差

                ob.append(ag.state)
        return ob

    def _get_dict_comm(self):
        dict_comm = []
        for ag in self.agents:
            info = {'pose_global_frame': ag.pose_global_frame,
                    'vel_global_frame': ag.vel_global_frame}
            dict_comm.append(info)
        return dict_comm

    def step(self, actions, dt=None):
        """
        执行单次动作（ros控制周期循环3次）
        返回新的状态，奖励，回合结束标志位
        """
        time_step = 0
        all_command = []
        while time_step < 1:
            dones = self._check_if_done()
            dict_comm = self._get_dict_comm()
            if self.game_over:
                break
            for idx, ag in enumerate(self.agents):
                if ag.quit:
                    all_command.append([0, 0])
                else:
                    if ag.group == 1:
                        self._dynamic_update()
                    command = ag.generate_next_command(actions[idx], dict_comm)
                    all_command.append(command)
            self._control(all_command)
            time_step += 1
        obs = self._get_obs()
        rewards = self._compute_rewards()
        info = {}
        return obs, rewards, dones, info

    def _control(self, commands):
        """
        发送ROS控制指令
        """
        for idx, ag in enumerate(self.agents):
            ag.rosport.pubTwist(commands[idx], dt=1 / 10)
        self.rate.sleep()

    def stop(self):
        """
        停止机器人运行
        """
        for ag in self.agents:
            ag.stop()

    def _dynamic_update(self):
        """
        编队机器人动态矩阵信息的更新
        """
        # update formation matrix
        for ag in self.agents:
            if ag.group == 1:
                for key, value in ag.neighbor_info.items():
                    yaw = self.agents[key].heading_global_frame
                    Rz = np.array([[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]], dtype=float)
                    value = Rz @ np.array(self.neighbor_info[ag.id][key])
                    ag.neighbor_info[key] = value
