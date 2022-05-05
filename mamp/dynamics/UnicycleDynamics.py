import numpy as np
from mamp.util import wrap
from mamp.dynamics.Dynamics import Dynamics


class UnicycleDynamics(Dynamics):
    """ Convert a speed & heading to a new state according to Unicycle Kinematics model.

    """

    def __init__(self, agent):
        Dynamics.__init__(self, agent)
        self.action_type = "R_THETA"

    def step(self, action, dt):
        """ 

        In the global frame, assume the agent instantaneously turns by :code:`heading`
        and moves forward at :code:`speed` for :code:`dt` seconds.  
        Add that offset to the current position. Update the velocity in the
        same way. Also update the agent's turning direction (only used by CADRL).

        Args:
            action (list): [delta heading angle, speed] command for this agent
            dt (float): time in seconds to execute :code:`action`
    
        """
        selected_speed = action[0]
        selected_heading = wrap(action[1] + self.agent.heading_global_frame)

        dx = selected_speed * np.cos(selected_heading) * dt
        dy = selected_speed * np.sin(selected_heading) * dt
        self.agent.pos_global_frame += np.array([dx, dy])
        self.agent.total_dist += np.sqrt(dx**2+dy**2)

        #   以下注释代码只是对环境范围做一下约束，当智能体在约束环境之外时，自动调为约束环境范围内
        # cmb_array = np.concatenate(
        #     [np.array([[self.agent.max_x, self.agent.max_y]]), self.agent.pos_global_frame[np.newaxis, :]])
        # self.agent.pos_global_frame = np.min(cmb_array, axis=0)
        # cmb_array = np.concatenate(
        #     [np.array([[self.agent.min_x, self.agent.min_y]]), self.agent.pos_global_frame[np.newaxis, :]])
        # self.agent.pos_global_frame = np.max(cmb_array, axis=0)

        self.agent.vel_global_frame[0] = round(selected_speed * np.cos(selected_heading), 5)
        self.agent.vel_global_frame[1] = round(selected_speed * np.sin(selected_heading), 5)
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

    def update_no_step(self, action):
        if action is None: return
        selected_speed = action[0]
        # selected_heading = wrap(action[1] + self.agent.heading_global_frame)
        selected_heading = self.agent.heading_global_frame

        self.agent.vel_global_frame[0] = selected_speed * np.cos(selected_heading)
        self.agent.vel_global_frame[1] = selected_speed * np.sin(selected_heading)
        self.agent.speed_global_frame = selected_speed
        # self.agent.delta_heading_global_frame = wrap(selected_heading - self.agent.heading_global_frame)
        # self.agent.heading_global_frame = selected_heading
