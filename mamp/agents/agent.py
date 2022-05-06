import math
import copy
import numpy as np
import pandas as pd
from mamp.envs import Config
from mamp.util import wrap, l2norm, l2normsq, sqr, takeSecond
from mamp.agents.obstacle import Obstacle


class Agent(object):
    def __init__(self, name, radius, pref_speed, initial_heading, goal_heading, policy,
                 dynamics_model, sensors, id, start_pos=None, goal_pos=None, group=0):
        self.id = id
        self.group = group
        self.name = name
        self.radius = radius
        self.pref_speed = pref_speed
        self.initial_pos = np.array(start_pos, dtype='float64')
        self.start_global_frame = np.array(start_pos, dtype='float64')

        self.neighbors = []
        self.maxNeighbors = 20
        self.neighborDist = 5.0
        self.timeStep = Config.DT
        self.timeHorizon = 50.0
        self.timeHorizonObst = 50.0
        self.maxSpeed = 1.0
        self.maxAccel = 1.0
        self.safetyFactor = 7.5
        self.is_parallel_neighbor = []
        self.is_obstacle = False

        # for experiment data
        self.initial_goal_pos = np.array(goal_pos, dtype='float64')
        self.initial_heading = initial_heading
        self.goal_heading_frame = goal_heading
        self.vel_global_unicycle = np.array([0.0, 0.0], dtype='float64')
        self.is_back2start = False
        self.total_time = 0.0
        self.total_dist = 0.0
        self.straight_path_length = l2norm(start_pos[:2], goal_pos[:2]) - 0.5  # For computing Distance Rate.
        self.desire_steps = int(self.straight_path_length / (pref_speed * self.timeStep))  # For computing Time Rate.

        self.action = None
        self.obstacle_list = None
        self.pos_global_frame = None
        self.policy = policy()
        self.dynamics_model = dynamics_model(self)
        self.sensors = [sensor() for sensor in sensors]

        self.dist_to_goal = 0.0
        self.near_goal_threshold = Config.NEAR_GOAL_THRESHOLD
        self.dt_nominal = Config.DT

        range = getattr(Config, 'ENV_RANGE') if hasattr(Config, 'ENV_RANGE') else [[-200, 200], [-200, 200]]
        self.min_x = range[0][0]
        self.max_x = range[0][1]
        self.min_y = range[1][0]
        self.max_y = range[1][1]

        self.pitchlims = [-np.pi / 4, np.pi / 4]
        self.turning_radius = 0.5
        self.min_heading_change = self.pitchlims[0]
        self.max_heading_change = self.pitchlims[1]
        self.is_use_dubins = False
        self.dubins_now_goal = None
        self.dubins_last_goal = None
        self.v_pref = np.array([0.0, 0.0], dtype='float64')

        self.t_offset = None
        self.global_state_dim = 11
        self.ego_state_dim = 3
        self.action_dim = 2

        self.planner = None
        self.path = []
        self.speed = [0, 0]  # vx, vy

        self.is_at_goal = False
        self.was_at_goal_already = False
        self.was_in_collision_already = False
        self.in_collision = False
        self.ran_out_of_time = False

        self.obj = None  # attach object from simulator

        self.history_info = pd.DataFrame(columns=Config.ANIMATION_COLUMNS)

    def reset(self, pos=None, goal_pos=None, pref_speed=None, radius=None, heading=None, goal_heading=None):
        # Global Frame states
        if pos is not None:
            self.pos_global_frame = np.array(pos, dtype='float64')
        else:
            self.pos_global_frame = np.array(self.initial_pos, dtype='float64')
        if goal_pos is not None:
            self.goal_global_frame = np.array(goal_pos, dtype='float64')
        else:
            self.goal_global_frame = np.array(self.initial_goal_pos, dtype='float64')

        self.vel_global_frame = np.array([0.0, 0.0], dtype='float64')
        self.speed_global_frame = 0.0

        if self.initial_heading is None and heading is None and self.pos_global_frame is not None:
            vec_to_goal = self.goal_global_frame - self.pos_global_frame
            self.heading_global_frame = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        else:
            self.heading_global_frame = self.initial_heading
        if goal_heading is not None:
            self.goal_heading_frame = goal_heading
        self.delta_heading_global_frame = 0.0

        # Ego Frame states
        self.speed_ego_frame = 0.0
        self.heading_ego_frame = 0.0
        self.vel_ego_frame = np.array([0.0, 0.0])

        # Other parameters
        if radius is not None:
            self.radius = radius
        if pref_speed is not None:
            self.pref_speed = pref_speed

        self.straight_line_time_to_reach_goal = (np.linalg.norm(
            self.pos_global_frame - self.goal_global_frame) - self.near_goal_threshold) / self.pref_speed
        if Config.EVALUATE_MODE or Config.PLAY_MODE:
            self.time_remaining_to_reach_goal = Config.MAX_TIME_RATIO * self.straight_line_time_to_reach_goal
        else:
            self.time_remaining_to_reach_goal = Config.MAX_TIME_RATIO * self.straight_line_time_to_reach_goal
        self.time_remaining_to_reach_goal = max(self.time_remaining_to_reach_goal, self.dt_nominal)

        # self.time_remaining_to_reach_goal = 5000
        self.t = 0.0

        self.step_num = 0

        self.is_at_goal = False
        self.was_at_goal_already = False
        self.was_in_collision_already = False
        self.in_collision = False
        self.ran_out_of_time = False

        self.other_agent_states = np.zeros((Config.OTHER_AGENT_STATE_LENGTH,))
        self.other_agent_obs = np.zeros((Config.OTHER_AGENT_OBSERVATION_LENGTH,))

        self.dynamics_model.update_ego_frame()
        # self._update_state_history()
        # self._check_if_at_goal()
        self._to_vector()
        # self.take_action([0.0, 0.0])

        self.min_dist_to_other_agents = np.inf

        self.turning_dir = 0.0
        self.is_done = False

    def __deepcopy__(self, memo):
        """ Copy every attribute about the agent except its policy (since that may contain MBs of DNN weights) """
        cls = self.__class__
        obj = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k != 'policy':
                setattr(obj, k, v)
        return obj

    def set_planner(self, planner):
        self.planner = planner
        path = self.planner.path()
        if len(path) > 1: self.path = path
        self.reset_states_in_history()

    def reset_states_in_history(self):
        if len(self.path) > 1:
            straight_line = 0
            for i in range(len(self.path) - 1):
                straight_line += np.linalg.norm(
                    np.array(self.path[i]) - np.array(self.path[i + 1])) - self.near_goal_threshold
            self.straight_line_time_to_reach_goal = straight_line / self.pref_speed
        else:
            self.straight_line_time_to_reach_goal = (np.linalg.norm(
                self.pos_global_frame - self.goal_global_frame) - self.near_goal_threshold) / self.pref_speed
        if Config.EVALUATE_MODE or Config.PLAY_MODE:
            self.time_remaining_to_reach_goal = Config.MAX_TIME_RATIO * self.straight_line_time_to_reach_goal
        else:
            self.time_remaining_to_reach_goal = Config.MAX_TIME_RATIO * self.straight_line_time_to_reach_goal
        self.time_remaining_to_reach_goal = max(self.time_remaining_to_reach_goal, self.dt_nominal)

    def _check_if_at_goal(self):
        """ Set :code:`self.is_at_goal` if norm(pos_global_frame - goal_global_frame) <= near_goal_threshold """
        if self.goal_global_frame is not None:
            distance_to_goal = (self.pos_global_frame[0] - self.goal_global_frame[0]) ** 2 + \
                               (self.pos_global_frame[1] - self.goal_global_frame[1]) ** 2
        else:
            distance_to_goal = 0
        distance_to_goal = np.sqrt(distance_to_goal)
        is_near_goal = distance_to_goal <= self.near_goal_threshold
        self.is_at_goal = is_near_goal
        if is_near_goal:
            self.near_goal_reward = 0
        else:
            self.near_goal_reward = Config.REWARD_TO_GOAL_RATE * distance_to_goal  # -0.1*

    def set_state(self, px, py, vx=None, vy=None, heading=None):
        if vx is None or vy is None:
            if self.step_num == 0:
                # On first timestep, just set to zero
                self.vel_global_frame = np.array([0, 0])
            else:
                # Interpolate velocity from last pos
                self.vel_global_frame = (np.array([px, py]) - self.pos_global_frame) / self.dt_nominal
        else:
            self.vel_global_frame = np.array([vx, vy])

        if heading is None:
            # Estimate heading to be the direction of the velocity vector
            heading = np.arctan2(self.vel_global_frame[1], self.vel_global_frame[0])
            self.delta_heading_global_frame = wrap(heading - self.heading_global_frame)
        else:
            self.delta_heading_global_frame = wrap(heading - self.heading_global_frame)

        self.pos_global_frame = np.array([px, py])
        self.speed_global_frame = np.linalg.norm(self.vel_global_frame)
        self.heading_global_frame = heading

    def find_next_action(self, dict_obs, dict_comm, actions, kdTree):
        if not self.is_at_goal:
            if self.policy.type == "external":
                action = self.policy.external_action_to_action(self, actions[self.id])
            elif self.policy.type == "internal":
                action = self.policy.find_next_action(dict_obs, dict_comm, self, kdTree)
            elif self.policy.type == "mixed":
                action = self.policy.produce_next_action(dict_obs, self, actions)
            else:
                raise NotImplementedError
            if self.dynamics_model.action_type == "R_THETA":
                if action[0] > self.pref_speed: action[0] = self.pref_speed
                if action[1] > self.max_heading_change: action[1] = self.max_heading_change
                if action[1] < self.min_heading_change: action[1] = self.min_heading_change
        else:
            action = np.array([0, 0])
        self.action = action
        return action

    def take_action(self, action):
        # Agent is done if any of these conditions hold (at goal, out of time, in collision). Stop moving if so & ignore the action.
        if self.is_at_goal or self.ran_out_of_time or self.in_collision:
            if self.is_at_goal:
                self.was_at_goal_already = True
            else:
                self.was_at_goal_already = False
            if self.in_collision:
                print('agent'+str(self.id)+' was in collision already')
                self.was_in_collision_already = True
            else:
                self.was_in_collision_already = False

            self.vel_global_frame = np.array([0.0, 0.0])
            if self.obj is not None: self.obj.stop_moving()
            return

        # Store info about the TF btwn the ego frame and global frame before moving agent
        goal_direction = self.goal_global_frame - self.pos_global_frame
        theta = np.arctan2(goal_direction[1], goal_direction[0])
        self.T_global_ego = np.array([[np.cos(theta), -np.sin(theta), self.pos_global_frame[0]],
                                      [np.sin(theta), np.cos(theta), self.pos_global_frame[1]],
                                      [0, 0, 1]])
        self.ego_to_global_theta = theta

        if self.obj is None:  # update_from_cal
            self.dynamics_model.step(action, self.dt_nominal)
        else:  # update information from simulation env
            self.obj.pubTwist(action, self.dt_nominal, self.id)

    def update_after_action(self):
        if self.obj is not None: self.update_from_obj()
        self.dynamics_model.update_ego_frame()
        # Update time left so agent does not run around forever
        self.time_remaining_to_reach_goal -= self.dt_nominal
        self.t += self.dt_nominal
        self.step_num += 1
        if self.time_remaining_to_reach_goal <= 0.0:  # and Config.TRAIN_MODE:
            self.ran_out_of_time = True
            print('agent'+str(self.id)+'was ran out of time')

        self._check_if_at_goal()
        self._to_vector()

    def update_from_obj(self):
        self.pos_global_frame = self.obj.pos_global_frame
        self.heading_global_frame = self.obj.radianY
        self.dynamics_model.update_no_step(self.obj.taken_action)

    def sense(self, agents, agent_index):
        self.sensor_data = {}
        for sensor in self.sensors:
            sensor_data = sensor.sense(agents, agent_index)
            self.sensor_data[sensor.name] = sensor_data

    def print_agent_info(self):
        """ Print out a summary of the agent's current state. """
        print('----------')
        print('Global Frame:')
        print('(px,py):', self.pos_global_frame)
        print('(gx,gy):', self.goal_global_frame)
        print('(vx,vy):', self.vel_global_frame)
        print('speed:', self.speed_global_frame)
        print('heading:', self.heading_global_frame)
        print('Body Frame:')
        print('(vx,vy):', self.vel_ego_frame)
        print('heading:', self.heading_ego_frame)
        print('----------')

    def _to_vector(self):
        """ Convert the agent's attributes to a single global state vector. """
        global_state_dict = {
            't': self.t,
            'radius': self.radius,
            'pref_speed': self.pref_speed,
            'speed_global_frame': self.speed_global_frame,
            'pos_x': self.pos_global_frame[0],
            'pos_y': self.pos_global_frame[1],
            'goal_x': self.goal_global_frame[0],
            'goal_y': self.goal_global_frame[1],
            'vel_x': self.vel_global_frame[0],
            'vel_y': self.vel_global_frame[1],
            'alpha': self.heading_global_frame,
            'vel_linear': self.vel_global_unicycle[0],
            'vel_angular': self.vel_global_unicycle[1],
            'total_time': self.total_time
        }
        global_state = np.array([val for val in global_state_dict.values()])
        ego_state = np.array([self.t, self.dist_to_goal, self.heading_ego_frame])

        animation_columns_dict = {}
        for key in Config.ANIMATION_COLUMNS:
            animation_columns_dict.update({key: global_state_dict[key]})
        self.history_info = self.history_info.append([animation_columns_dict], ignore_index=True)
        return global_state, ego_state

    def get_sensor_data(self, sensor_name):
        if sensor_name in self.sensor_data:
            return self.sensor_data[sensor_name]

    def get_agent_data(self, attribute):
        return getattr(self, attribute)

    def get_agent_data_equiv(self, attribute, value):
        return eval("self." + attribute) == value

    def get_observation_dict(self):
        observation = {}
        for state in Config.STATES_IN_OBS_MULTI[self.group]:
            observation[state] = np.array(eval("self." + Config.STATE_INFO_DICT[state]['attr']))
        return observation

    def get_ref(self):
        if self.goal_global_frame is not None:
            goal_direction = self.goal_global_frame - self.pos_global_frame
        else:
            goal_direction = np.array([0, 0])
        self.goal_direction = goal_direction
        self.dist_to_goal = math.sqrt(goal_direction[0] ** 2 + goal_direction[1] ** 2)
        if self.dist_to_goal > 1e-8:
            ref_prll = goal_direction / self.dist_to_goal
        else:
            ref_prll = goal_direction
        ref_orth = np.array([-ref_prll[1], ref_prll[0]])  # rotate by 90 deg
        return ref_prll, ref_orth

    def ego_pos_to_global_pos(self, ego_pos):
        if ego_pos.ndim == 1:
            ego_pos_ = np.array([ego_pos[0], ego_pos[1], 1])
            global_pos = np.dot(self.T_global_ego, ego_pos_)
            return global_pos[:2]
        else:
            ego_pos_ = np.hstack([ego_pos, np.ones((ego_pos.shape[0], 1))])
            global_pos = np.dot(self.T_global_ego, ego_pos_.T).T
            return global_pos[:, :2]

    def global_pos_to_ego_pos(self, global_pos):
        ego_pos = np.dot(np.linalg.inv(self.T_global_ego), np.array([global_pos[0], global_pos[1], 1]))
        return ego_pos[:2]

    def insertAgentNeighbor(self, other_agent, rangeSq):
        if self.id != other_agent.id:
            distSq = l2normsq(self.pos_global_frame, other_agent.pos_global_frame)
            if distSq < sqr(self.radius + other_agent.radius) and distSq < rangeSq:     # COLLISION!
                if not self.in_collision:
                    self.in_collision = True
                    self.neighbors.clear()

                if len(self.neighbors) == self.maxNeighbors:
                    self.neighbors.pop()
                self.neighbors.append((other_agent, distSq))
                self.neighbors.sort(key=takeSecond)
                if len(self.neighbors) == self.maxNeighbors:
                    rangeSq = self.neighbors[-1][1]
            elif not self.in_collision and distSq < rangeSq:
                if len(self.neighbors) == self.maxNeighbors:
                    self.neighbors.pop()
                self.neighbors.append((other_agent, distSq))
                self.neighbors.sort(key=takeSecond)
                if len(self.neighbors) == self.maxNeighbors:
                    rangeSq = self.neighbors[-1][1]

    def insertObstacleNeighbor(self, obstacle, rangeSq):
        distSq1 = l2normsq(self.pos_global_frame, obstacle.pos_global_frame)
        # 适合半径接近或大于neighborDist的时候
        distSq = (l2norm(self.pos_global_frame, obstacle.pos_global_frame)-obstacle.radius)**2
        if distSq1 < sqr(self.radius + obstacle.radius) and distSq < rangeSq:  # COLLISION!
            if not self.in_collision:
                self.in_collision = True
                self.neighbors.clear()

            if len(self.neighbors) == self.maxNeighbors:
                self.neighbors.pop()
            self.neighbors.append((obstacle, distSq))
            self.neighbors.sort(key=takeSecond)
            if len(self.neighbors) == self.maxNeighbors:
                rangeSq = self.neighbors[-1][1]
        elif not self.in_collision and distSq < rangeSq:
            if len(self.neighbors) == self.maxNeighbors:
                self.neighbors.pop()
            self.neighbors.append((obstacle, distSq))
            self.neighbors.sort(key=takeSecond)
            if len(self.neighbors) == self.maxNeighbors:
                rangeSq = self.neighbors[-1][1]

    def set_other_agents(self, host_id, agents):
        host_agent = agents[host_id]
        self.obstacle_list = [None for _ in range(len(agents))]
        self.other_agent_list = []
        for i, other_agent in enumerate(agents):
            if i == host_id:
                continue
            self.other_agent_list.append(other_agent)
            self.obstacle_list[other_agent.id] = Obstacle(pos=other_agent.pos_global_frame,
                                                          shape_dict={'shape': 'circle', 'feature': other_agent.radius})
