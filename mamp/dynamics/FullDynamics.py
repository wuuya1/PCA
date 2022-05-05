import math
import numpy as np
from mamp.util import wrap
from mamp.dynamics.Dynamics import Dynamics


class FullDynamics(Dynamics):
    """ Convert a speed & heading to a new state according to Unicycle Kinematics model.

    """

    def __init__(self, agent):
        Dynamics.__init__(self, agent)
        self.action_type = "XY"

    def step(self, action, dt):
        """ 

        In the global frame, assume the agent instantaneously turns by :code:`heading`
        and moves forward at :code:`speed` for :code:`dt` seconds.  
        Add that offset to the current position. Update the velocity in the
        same way. Also update the agent's turning direction (only used by CADRL).

        Args:
            action (list): [vx, vy] command for this agent
            dt (float): time in seconds to execute :code:`action`
    
        """
        dx = action[0] * dt
        dy = action[1] * dt
        self.agent.pos_global_frame += np.array([dx, dy])

        selected_speed = math.sqrt(dx ** 2 + dy ** 2)
        selected_heading = np.arctan2(dy, dx)

        self.agent.vel_global_frame[0] = action[0]
        self.agent.vel_global_frame[1] = action[1]

        self.agent.speed_global_frame = selected_speed
        self.agent.delta_heading_global_frame = wrap(selected_heading - self.agent.heading_global_frame)
        self.agent.heading_global_frame = selected_heading

        # turning dir: needed for cadrl value fn
        if abs(self.agent.turning_dir) < 1e-5:
            self.agent.turning_dir = 0.11 * np.sign(selected_heading)
        elif self.agent.turning_dir * selected_heading < 0:
            self.agent.turning_dir = max(-np.pi, min(np.pi, -self.agent.turning_dir + selected_heading))
        else:
            self.agent.turning_dir = np.sign(self.agent.turning_dir) * max(0.0, abs(self.agent.turning_dir) - 0.1)
