import numpy as np
from mamp.envs import Config

class Sensor(object):
    def __init__(self):
        self.name = 'empty_obs'

    def sense(self, agents, agent_index):
        """ Dummy method to be re-implemented by each Sensor subclass
        """
        host_agent = agents[agent_index]
        other_agents_states = np.zeros((Config.MAX_NUM_OTHER_AGENTS_OBSERVED, Config.OTHER_AGENT_OBSERVATION_LENGTH))

        other_agent_count = 0
        for other_agent in agents:
            if other_agent.id == host_agent.id: continue
            # rel_pos_to_other_global_frame = other_agent.pos_global_frame - host_agent.pos_global_frame
            # dist_2_other = np.linalg.norm(rel_pos_to_other_global_frame) - host_agent.radius - other_agent.radius
            # if dist_2_other > host_agent.view_radius: continue

            other_agent_count += 1

        host_agent.num_other_agents_observed = other_agent_count
        return other_agents_states

    def set_args(self, args):
        """ Update several class attributes (in dict format) of the Sensor object
        
        Args:
            args (dict): {'arg_name1': new_value1, ...} sets :code:`self.arg_name1 = new_value1`, etc. 

        """
        # Supply a dict of arg key value pairs
        for arg, value in args.items():
            # print("Setting self.{} to {}".format(arg, value))
            setattr(self, arg, value)
