import rospy
import time
import threading
import numpy as np

from mamp.policies.kdTree import KDTree

from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

from .macaenv import MACAEnv
from .rosport.bot import AgentPort


def thread_job():
    rospy.spin()


class GazeboEnv(MACAEnv):
    def __init__(self):
        self.env_name = 'gazebo'
        super(GazeboEnv, self).__init__()
        self.node = rospy.init_node('macaenv_node', anonymous=True)
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.set_state   = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.rate = rospy.Rate(1 / self.dt_nominal)
        add_thread = threading.Thread(target=thread_job)
        add_thread.start()

    def set_start(self, model_name, pos, heading):
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
        state.pose.orientation.z = np.sin(heading/2)
        state.pose.orientation.w = np.cos(heading/2)
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

    def reset(self):
        super().reset()
        self.reset_world()
        # 将Agent和Obj进行绑定
        for agent in self.agents:
            agent.obj = AgentPort(agent.id, agent.name)
            self.set_start('Agent'+str(agent.id+1), agent.pos_global_frame, agent.heading_global_frame)
            agent.obj.stop_moving()
        time.sleep(0.5)
        for agent in self.agents:
            agent.update_from_obj()  # update agent's dynamic state

    def _take_action(self, actions):
        super()._take_action(actions)
        self.rate.sleep()

    def step(self, actions):
        next_observations, rewards, game_over, info = super().step(actions)
        if game_over: self._stop_robots()
        return next_observations, rewards, game_over, info

    def _stop_robots(self):
        for agent in self.agents:
            agent.obj.stop_moving()
