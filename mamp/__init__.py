import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='LineFollower-v0',
    entry_point='mamp.envs.carenv:CAREnv',
)

register(
    id='MAGazebo-v0',
    entry_point='mamp.envs.gazeboEnv:GazeboEnv',
)

register(
    id='MultiAgentCollisionAvoidance-v0',
    entry_point='mamp.envs.macaenv:MACAEnv',
)

__all__=['agent']
