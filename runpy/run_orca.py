#!/usr/bin/env python3
import os
import sys
import json
import random
import numpy as np
import pandas as pd

# set path
sys.path.append('/home/wuuya/PycharmProjects/maros')
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os

os.environ['GYM_CONFIG_CLASS'] = 'Example'
os.environ['GYM_CONFIG_PATH'] = '../mamp/configs/config.py'
import gym

gym.logger.set_level(40)

from mamp.agents.agent import Agent, Obstacle
# Policies
from mamp.policies import policy_dict
# Dynamics
from mamp.dynamics.UnicycleDynamics import UnicycleDynamics
# Sensors
from mamp.sensors import Sensor
from mamp.policies.policy import safety_weight
from mamp.envs import Config


def mod2pi(theta):  # to 0-2*pi
    return theta - 2.0 * np.pi * np.floor(theta / 2.0 / np.pi)


def simulation_experiment1_in_circle(num_agents):
    center = [0.0, 0.0]
    rad = 40.0
    test_qi, test_qf = [], []
    k = 0  # np.random.uniform(0, 2)
    for n in range(num_agents):
        test_qi.append([center[0] + round(rad * np.cos(2 * n * np.pi / num_agents + k * np.pi / 4), 2),
                        center[1] + round(rad * np.sin(2 * n * np.pi / num_agents + k * np.pi / 4), 2),
                        mod2pi(2 * n * np.pi / num_agents + k * np.pi / 4 + np.pi)])
    for n in range(num_agents):
        test_qf.append([center[0] + round(rad * np.cos(2 * n * np.pi / num_agents + np.pi + k * np.pi / 4), 2),
                        center[1] + round(rad * np.sin(2 * n * np.pi / num_agents + np.pi + k * np.pi / 4), 2),
                        mod2pi(2 * n * np.pi / num_agents + k * np.pi / 4)])
    return test_qi, test_qf


def simulation_experiment2_in_circle(num_agents, rad):
    center = [0.0, 0.0]
    test_qi, test_qf = [], []
    k = 0  # np.random.uniform(0, 2)
    for n in range(num_agents):
        test_qi.append([center[0] + round(rad * np.cos(2 * n * np.pi / num_agents + k * np.pi / 4), 2),
                        center[1] + round(rad * np.sin(2 * n * np.pi / num_agents + k * np.pi / 4), 2),
                        mod2pi(2 * n * np.pi / num_agents + k * np.pi / 4 + np.pi)])
    for n in range(num_agents):
        test_qf.append([center[0] + round(rad * np.cos(2 * n * np.pi / num_agents + np.pi + k * np.pi / 4), 2),
                        center[1] + round(rad * np.sin(2 * n * np.pi / num_agents + np.pi + k * np.pi / 4), 2),
                        mod2pi(2 * n * np.pi / num_agents + k * np.pi / 4 + np.pi)])
    return test_qi, test_qf


def real_world_experiment_in_circle(num_agents):
    center = [1.5, 1.5]
    rad = 1.5
    test_qi, test_qf = [], []
    k = 0  # np.random.uniform(0, 2)
    for n in range(num_agents):
        test_qi.append([center[0] + round(rad * np.cos(2 * n * np.pi / num_agents + k * np.pi / 4), 2),
                        center[1] + round(rad * np.sin(2 * n * np.pi / num_agents + k * np.pi / 4), 2),
                        mod2pi(2 * n * np.pi / num_agents + k * np.pi / 4 + np.pi)])
    for n in range(num_agents):
        test_qf.append([center[0] + round(rad * np.cos(2 * n * np.pi / num_agents + np.pi + k * np.pi / 4), 2),
                        center[1] + round(rad * np.sin(2 * n * np.pi / num_agents + np.pi + k * np.pi / 4), 2),
                        mod2pi(2 * n * np.pi / num_agents + k * np.pi / 4)])
    return test_qi, test_qf


def set_random_pos(agent_num, r=15):
    """
    The square is 30 * 30
    The maximum number of agents is 961.
    """
    r = int(r)
    poses = []
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            pos = np.array([i, j, 0.0])
            poses.append(pos)
    agent_pos = random.sample(poses, agent_num)
    agent_goal = random.sample(poses, agent_num)

    return agent_pos, agent_goal


def build_agents():
    """
    simulation1: agents' num is 10-150, circle's radius is 15, random's radius is 15
    large simulation: agents' num is 200, circle's radius is 20, random's radius is 30
    """
    num_agents = 100
    test_qi, test_qf = simulation_experiment2_in_circle(num_agents, rad=15.0)
    # test_qi, test_qf = set_random_pos(num_agents, r=30)
    radius = 0.2
    pref_speed = 1.0  # turtleBot3's max speed is 0.22 or 0.26 m/s, and 2.84 or 1.82 rad/s
    agents = []
    for j in range(len(test_qi)):
        agents.append(Agent(start_pos=test_qi[j][:2],
                            goal_pos=test_qf[j][:2],
                            name='Agent' + str(j + 1),
                            radius=radius, pref_speed=pref_speed,
                            initial_heading=test_qi[j][2],
                            goal_heading=test_qf[j][2],
                            policy=policy_dict['orca'],
                            dynamics_model=UnicycleDynamics,
                            sensors=[Sensor],
                            id=j))
    return agents


def build_obstacles():
    """
        exp_2: r = 16.0, agents' num is 100, radius of circle is 40
    """
    obstacles = []
    r = 16.0
    # obstacles.append(Obstacle(pos=[0.0, 0.0], shape_dict={'shape': "rect", 'feature': (r, r)}, id=0))
    return obstacles


if __name__ == '__main__':
    # Set agent configuration (start/goal pos, radius, size, policy)
    agents = build_agents()
    obstacles = build_obstacles()
    [agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]

    agents_num = len(agents)

    # env = gym.make("MAGazebo-v0")
    env = gym.make("MultiAgentCollisionAvoidance-v0")
    env.set_agents(agents, obstacles=obstacles)

    epi_maximum = 1
    for epi in range(epi_maximum):
        env.reset()
        print("episode:", epi)
        game_over = False
        while not game_over:
            print("step is ", env.episode_step_number)
            actions = {}
            obs, rewards, game_over, which_agents_done = env.step(actions)
        print("All agents finished!", env.episode_step_number)
    print("Experiment over.")

    log_save_dir = os.path.dirname(os.path.realpath(__file__)) + '/../draw/orca/log/'
    os.makedirs(log_save_dir, exist_ok=True)

    # trajectory
    writer = pd.ExcelWriter(log_save_dir + '/trajs_'+str(agents_num)+'.xlsx')
    for agent in agents:
        agent.history_info.to_excel(writer, sheet_name='agent' + str(agent.id))
    writer.save()

    # scenario information
    info_dict_to_visualize = {
        'all_agent_info': [],
        'all_obstacle': [],
        'all_compute_time': 0.0,
        'all_straight_distance': 0.0,
        'all_distance': 0.0,
        'successful_num': 0,
        'all_desire_step_num': 0,
        'all_step_num': 0,
        'SuccessRate': 0.0,
        'ExtraTime': 0.0,
        'ExtraDistance': 0.0,
        'AverageSpeed': 0.0,
        'AverageCost': 0.0
    }
    all_straight_dist = 0.0
    all_agent_total_time = 0.0
    all_agent_total_dist = 0.0
    num_of_success = 0
    all_desire_step_num = 0
    all_step_num = 0

    SuccessRate = 0.0
    ExtraTime = 0.0
    ExtraDistance = 0.0
    AverageSpeed = 0.0
    AverageCost = 0.0
    for agent in agents:
        agent_info_dict = {'id': agent.id, 'gp': agent.group, 'radius': agent.radius,
                           'goal_pos': agent.goal_global_frame.tolist()}
        info_dict_to_visualize['all_agent_info'].append(agent_info_dict)
        if not agent.in_collision and not agent.ran_out_of_time:
            num_of_success += 1
            all_agent_total_time += agent.total_time
            all_straight_dist += agent.straight_path_length
            all_agent_total_dist += agent.total_dist
            all_desire_step_num += agent.desire_steps
            all_step_num += agent.step_num
    info_dict_to_visualize['all_compute_time'] = all_agent_total_time
    info_dict_to_visualize['all_straight_distance'] = all_straight_dist
    info_dict_to_visualize['all_distance'] = all_agent_total_dist
    info_dict_to_visualize['successful_num'] = num_of_success
    info_dict_to_visualize['all_desire_step_num'] = all_desire_step_num
    info_dict_to_visualize['all_step_num'] = all_step_num

    info_dict_to_visualize['SuccessRate'] = num_of_success / agents_num
    info_dict_to_visualize['ExtraTime'] = ((all_step_num - all_desire_step_num) * Config.DT) / num_of_success
    info_dict_to_visualize['ExtraDistance'] = (all_agent_total_dist - all_straight_dist) / num_of_success
    info_dict_to_visualize['AverageSpeed'] = all_agent_total_dist / all_step_num / Config.DT
    info_dict_to_visualize['AverageCost'] = 1000 * all_agent_total_time / all_step_num

    for obstacle in env.obstacles:
        obstacle_info_dict = {'position': obstacle.pos, 'shape': obstacle.shape, 'feature': obstacle.feature}
        info_dict_to_visualize['all_obstacle'].append(obstacle_info_dict)

    info_str = json.dumps(info_dict_to_visualize, indent=4)
    with open(log_save_dir + '/env_cfg_'+str(agents_num)+'.json', 'w') as json_file:
        json_file.write(info_str)
    json_file.close()
