import numpy as np
'''
formation movement: from pos to goal_pos with other agent observation
'''
class Config(object):
    def __init__(self):
        #########################################################################
        # GENERAL PARAMETERS
        self.COLLISION_AVOIDANCE = True
        self.continuous, self.discrete = range(2) # Initialize game types as enum
        self.ACTION_SPACE_TYPE = self.continuous

        ### DISPLAY
        self.ANIMATE_EPISODES   = False
        self.SHOW_EPISODE_PLOTS = False
        self.SAVE_EPISODE_PLOTS = False
        if not hasattr(self, "PLOT_CIRCLES_ALONG_TRAJ"):
            self.PLOT_CIRCLES_ALONG_TRAJ = True
        self.ANIMATION_PERIOD_STEPS = 5 # plot every n-th DT step (if animate mode on)
        self.PLT_LIMITS = None
        self.PLT_FIG_SIZE = (10, 8)

        if not hasattr(self, "USE_STATIC_MAP"):
            self.USE_STATIC_MAP = False

        ### TRAIN / PLAY / EVALUATE
        self.TRAIN_MODE          = True # Enable to see the trained agent in action (for testing)
        self.PLAY_MODE           = False # Enable to see the trained agent in action (for testing)
        self.EVALUATE_MODE       = False # Enable to see the trained agent in action (for testing)

        ### REWARDS
        self.REWARD_MODE        = "no formation" # formation
        self.REWARD_AT_GOAL = 1.0 # reward given when agent reaches goal position
        self.REWARD_AT_FORMAT = 0.1 #
        self.REWARD_TO_GOAL_RATE = 0.0 #
        self.REWARD_COLLISION_WITH_AGENT = -0.25 # reward given when agent collides with another agent
        self.REWARD_COLLISION_WITH_WALL = -0.25 # reward given when agent collides with wall
        self.REWARD_GETTING_CLOSE   = -0.1 # reward when agent gets close to another agent (unused?)
        self.REWARD_ENTERED_NORM_ZONE   = -0.05 # reward when agent enters another agent's social zone
        self.REWARD_TIME_STEP   = 0.0 # default reward given if none of the others apply (encourage speed)
        self.REWARD_WIGGLY_BEHAVIOR = 0.0
        self.WIGGLY_BEHAVIOR_THRESHOLD = np.inf
        self.COLLISION_DIST = 0.0 # meters between agents' boundaries for collision
        self.GETTING_CLOSE_RANGE = 0.2 # meters between agents' boundaries for collision
        # self.SOCIAL_NORMS = "right"
        # self.SOCIAL_NORMS = "left"
        self.SOCIAL_NORMS = "none"

        ### SIMULATION
        self.DT                  = 0.2 # seconds between simulation time steps
        self.NEAR_GOAL_THRESHOLD = 0.2
        self.NEAR_FORMAT_THRESHOLD = 0.2
        self.MAX_TIME_RATIO = 10. # agent has this number times the straight-line-time to reach its goal before "timing out"

        ### PARAMETERS THAT OVERWRITE/IMPACT THE ENV'S PARAMETERS
        if not hasattr(self, "MAX_NUM_OBS_IN_ENVIRONMENT"):
            self.MAX_NUM_OBS_IN_ENVIRONMENT = 0
        if not hasattr(self, "MAX_NUM_AGENTS_IN_ENVIRONMENT"):
            self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 4
        if not hasattr(self, "MAX_NUM_AGENTS_TO_SIM"):
            self.MAX_NUM_AGENTS_TO_SIM = 4
        self.MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        if not hasattr(self, "MAX_NUM_OTHER_AGENTS_OBSERVED"):
            self.MAX_NUM_OTHER_AGENTS_OBSERVED = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1

        ### EXPERIMENTS
        self.PLOT_EVERY_N_EPISODES = 100 # for tensorboard visualization

        ### SENSORS
        self.SENSING_HORIZON  = np.inf
        # self.SENSING_HORIZON  = 3.0
        self.LASERSCAN_LENGTH = 512 # num range readings in one scan
        self.LASERSCAN_NUM_PAST = 3 # num range readings in one scan
        self.NUM_STEPS_IN_OBS_HISTORY = 1 # number of time steps to store in observation vector
        self.NUM_PAST_ACTIONS_IN_STATE = 0
        self.WITH_COMM = False

        ### RVO AGENTS
        self.RVO_TIME_HORIZON = 5.0
        self.RVO_COLLAB_COEFF = 0.5
        self.RVO_ANTI_COLLAB_T = 1.0

        ### OBSERVATION VECTOR
        self.TRAIN_SINGLE_AGENT = False
        self.STATE_INFO_DICT = {
            'dist_to_goal': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("dist_to_goal")',
                'std': np.array([5.], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
                },
            'radius': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("radius")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([0.5], dtype=np.float32)
                },
            'heading_ego_frame': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [-np.pi, np.pi],
                'attr': 'get_agent_data("heading_ego_frame")',
                'std': np.array([3.14], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
                },
            'pref_speed': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("pref_speed")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([1.0], dtype=np.float32)
                },
            'num_other_agents': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("num_other_agents_observed")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([1.0], dtype=np.float32)
                },
            'other_agent_states': {
                'dtype': np.float32,
                'size': 7,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("other_agent_states")',
                'std': np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32),
                'mean': np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32)
                },
            'other_agents_states': {
                'dtype': np.float32,
                'size': (self.MAX_NUM_OTHER_AGENTS_OBSERVED,7),
                'bounds': [-np.inf, np.inf],
                'attr': 'get_sensor_data("other_agents_states")',
                'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32), (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32), (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                },
            'laserscan': {
                'dtype': np.float32,
                'size': (self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH),
                'bounds': [0., 6.],
                'attr': 'get_sensor_data("laserscan")',
                'std': 5.*np.ones((self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH), dtype=np.float32),
                'mean': 5.*np.ones((self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH), dtype=np.float32)
                },
            'is_learning': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0., 1.],
                'attr': 'get_agent_data_equiv("policy.str", "learning")'
                },
            }

        self.ANIMATION_COLUMNS = ['pos_x', 'pos_y', 'alpha', 'vel_x', 'vel_y', 'vel_linear', 'vel_angular', 'total_time']
        if not hasattr(self, "STATES_IN_OBS_MULTI"):
            self.STATES_IN_OBS_MULTI = [
                ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius'],
            ]

        if not hasattr(self, "STATES_NOT_USED_IN_POLICY"):
            self.STATES_NOT_USED_IN_POLICY = ['is_learning']

        self.HOST_AGENT_OBSERVATION_LENGTH = 4 # dist to goal, heading to goal, pref speed, radius
        self.OTHER_AGENT_STATE_LENGTH = 7 # other px, other py, other vx, other vy, other radius, combined radius, distance between
        self.OTHER_AGENT_OBSERVATION_LENGTH = 7 # other px, other py, other vx, other vy, other radius, combined radius, distance between
        self.OTHER_AGENT_FULL_OBSERVATION_LENGTH = self.OTHER_AGENT_OBSERVATION_LENGTH
        self.HOST_AGENT_STATE_SIZE = self.HOST_AGENT_OBSERVATION_LENGTH

        # self.AGENT_SORTING_METHOD = "closest_last"
        self.AGENT_SORTING_METHOD = "closest_first"
        # self.AGENT_SORTING_METHOD = "time_to_impact"

class Example(Config):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 1024
        Config.__init__(self)
        self.EVALUATE_MODE = True
        self.TRAIN_MODE = False
        self.DT = 0.1
        self.SAVE_EPISODE_PLOTS = True
        self.PLOT_CIRCLES_ALONG_TRAJ = True
        self.ANIMATE_EPISODES = True
        # self.SENSING_HORIZON = 4
        # self.PLT_LIMITS = [[-20, 20], [-20, 20]]
        # self.PLT_FIG_SIZE = (10,10)

class AStar(Config):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 512
        Config.__init__(self)
        self.EVALUATE_MODE = True
        self.TRAIN_MODE = False
        self.DT = 0.1
        self.SAVE_EPISODE_PLOTS = True
        self.PLOT_CIRCLES_ALONG_TRAJ = True
        self.ANIMATE_EPISODES = True
        # self.SENSING_HORIZON = 4
        # self.PLT_LIMITS = [[-20, 20], [-20, 20]]
        # self.PLT_FIG_SIZE = (10,10)
