'''
'''
import os
import gym
import copy
import itertools
import gym.spaces
import numpy as np

from mamp.envs import Config
from mamp.util import l2norm
from mamp.policies.kdTree import KDTree


class MACAEnv(gym.Env):
    def __init__(self):
        self.id = 0
        self.episode_number = 0
        self.episode_step_number = None

        # Initialize Rewards
        # self._initialize_rewards()

        # Simulation Parameters
        self.num_agents = Config.MAX_NUM_AGENTS_IN_ENVIRONMENT
        self.dt_nominal = Config.DT

        # Collision Parameters
        self.collision_dist = Config.COLLISION_DIST
        self.getting_close_range = Config.GETTING_CLOSE_RANGE
        self.evaluate = Config.EVALUATE_MODE

        ### The gym.spaces library doesn't support Python2.7 (syntax of Super().__init__())
        self.action_space_type = Config.ACTION_SPACE_TYPE
        # Upper/Lower bounds on Actions
        self.max_heading_change = np.pi / 4
        self.min_heading_change = -self.max_heading_change
        self.min_speed = 0.0
        self.max_speed = 1.0
        if self.action_space_type == Config.discrete:
            self.action_space = gym.spaces.Discrete(self.actions.num_actions, dtype=np.float32)
        elif self.action_space_type == Config.continuous:
            self.low_action = np.array([self.min_speed, self.min_heading_change])
            self.high_action = np.array([self.max_speed, self.max_heading_change])
            self.action_space = gym.spaces.Box(self.low_action, self.high_action, dtype=np.float32)

        # single agent dict obs
        self.observation = {}
        for agent in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT):
            self.observation[agent] = {}

        self.agents = None
        self.centralized_planner = None
        self.obstacles = []
        self.kdTree = None

    def set_agents(self, agents, obstacles=None):
        self.agents = agents
        self.obstacles = obstacles
        self.kdTree = KDTree(self.agents, self.obstacles)
        self.kdTree.buildObstacleTree()

    def reset(self):
        for ag in self.agents: ag.reset()
        if self.episode_step_number is not None and self.episode_step_number > 0:
            self.episode_number += 1
        self.begin_episode = True
        self.episode_step_number = 0

    def step(self, actions):
        self.episode_step_number += 1

        # Take action
        self._take_action(actions)

        self._update_after_action()

        collision_with_obstacle, collision_with_agent = self._check_for_collisions()

        # Take observation
        next_observations = self._get_obs()

        # Check which agents' games are finished (at goal/collided/out of time)
        which_agents_done, game_over = self._check_which_agents_done()

        which_agents_done_dict = {}
        which_agents_learning_dict = {}
        for i, agent in enumerate(self.agents):
            which_agents_done_dict[agent.id] = which_agents_done[i]
            which_agents_learning_dict[agent.id] = agent.policy.is_still_learning
        info = {
               'which_agents_done': which_agents_done_dict,
               'which_agents_learning': which_agents_learning_dict,
               }
        return next_observations, 0, game_over, info

    def _take_action(self, actions):
        self.kdTree.buildAgentTree()

        num_actions_per_agent = 2  # speed, delta heading angle
        all_actions = np.zeros((len(self.agents), num_actions_per_agent), dtype=np.float32)

        # Agents set their action (either from external or w/ find_next_action)
        for agent_index, agent in enumerate(self.agents):
            if agent.is_done:
                continue
            dict_obs = self.observation[agent_index]
            other_agents = copy.copy(self.agents)
            other_agents.remove(agent)
            dict_comm = {'other_agents': other_agents, 'obstacles': self.obstacles}
            all_actions[agent_index, :] = agent.find_next_action(dict_obs, dict_comm, actions, self.kdTree)

        # After all agents have selected actions, run one dynamics update
        for i, agent in enumerate(self.agents):
            agent.take_action(all_actions[i, :])

    def set_cbs_planner(self, planner):
        self.centralized_planner = planner
        solutions = self.centralized_planner.search()

        if not solutions:
            print("agent" + str(self.id) + "'s Solution not found", '\n')
            return
        for aid, solution in solutions.items():
            if len(solution) <= 1: continue
            for time_sol in solution[1:][::-1]:
                self.agents[aid].path.append(time_sol)
            print("agent" + str(aid) + "'s trajectory is:", solution)

    def _check_for_collisions(self):
        """ Check whether each agent has collided with another agent or a static obstacle in the map
        This method doesn't compute social zones currently!!!!!
        Returns:
            - collision_with_agent (list): for each agent, bool True if that agent is in collision with another agent
            - collision_with_wall (list): for each agent, bool True if that agent is in collision with object in map
            - entered_norm_zone (list): for each agent, bool True if that agent entered another agent's social zone
            - dist_btwn_nearest_agent (list): for each agent, float closest distance to another agent
        """
        collision_with_agent = [False for _ in self.agents]
        collision_with_wall = [False for _ in self.agents]
        entered_norm_zone = [False for _ in self.agents]
        dist_btwn_nearest_agent = [np.inf for _ in self.agents]
        agent_shapes = []
        agent_front_zones = []
        agent_inds = list(range(len(self.agents)))
        agent_pairs = list(itertools.combinations(agent_inds, 2))
        for i, j in agent_pairs:
            dist_btwn = l2norm(self.agents[i].pos_global_frame, self.agents[j].pos_global_frame)
            combined_radius = self.agents[i].radius + self.agents[j].radius
            dist_btwn_nearest_agent[i] = min(dist_btwn_nearest_agent[i], dist_btwn - combined_radius)
            if dist_btwn <= combined_radius:
                # Collision with another agent!
                collision_with_agent[i] = True
                collision_with_agent[j] = True
                self.agents[i].in_collision = True
                self.agents[j].in_collision = True
                print('collision:', 'agent' + str(i), 'and agent' + str(j))

        agent_inds = list(range(len(self.agents)))
        collision_with_obstacle = [False for _ in self.agents]
        dist_to_nearest_obstacle = [np.inf for _ in self.agents]
        for obstacle in self.obstacles:
            for j in agent_inds:
                dist_btwn = l2norm(obstacle.pos_global_frame, self.agents[j].pos_global_frame)
                combined_radius = obstacle.radius + self.agents[j].radius
                dist_to_nearest_obstacle[j] = min(dist_to_nearest_obstacle[j], dist_btwn - combined_radius)
                # diff_verct = self.agents[j].pos_global_frame - obstacle.pos_global_frame
                # norm = np.sqrt(diff_verct[0]**2 + diff_verct[1]**2)
                if dist_btwn <= combined_radius:
                    collision_with_obstacle[j] = True
                    self.agents[j].in_collision = True
                    print('collision:', 'agent' + str(j), 'and obstacle' + str(obstacle.id))

        return collision_with_obstacle, collision_with_agent

    def _check_which_agents_done(self):
        at_goal_condition = np.array([a.is_at_goal for a in self.agents])
        ran_out_of_time_condition = np.array([a.ran_out_of_time for a in self.agents])
        in_collision_condition = np.array([a.in_collision for a in self.agents])
        which_agents_done = np.logical_or.reduce((at_goal_condition, ran_out_of_time_condition, in_collision_condition))
        for agent_index, agent in enumerate(self.agents):
            agent.is_done = which_agents_done[agent_index]

        if Config.EVALUATE_MODE:
            # Episode ends when every agent is done
            game_over = np.all(which_agents_done)
        elif Config.TRAIN_SINGLE_AGENT:
            # Episode ends when ego agent is done
            game_over = which_agents_done[0]
        else:
            # Episode is done when all *learning* agents are done
            learning_agent_inds = [i for i in range(len(self.agents)) if self.agents[i].policy.is_still_learning]
            game_over = np.all(which_agents_done[learning_agent_inds])

        return which_agents_done, game_over

    def _update_after_action(self):
        for i, agent in enumerate(self.agents):
            if agent.is_at_goal: continue
            agent.update_after_action()

    def _get_obs(self):
        """ Update the map now that agents have moved, have each agent sense the world, and fill in their observations
        Returns:
            observation (list): for each agent, a dictionary observation.
        """
        # Agents collect a reading from their map-based sensors
        for i, agent in enumerate(self.agents):
            agent.sense(self.agents, i)
        # Agents fill in their element of the multiagent observation vector
        for i, agent in enumerate(self.agents):
            self.observation[i] = agent.get_observation_dict()

        return self.observation
