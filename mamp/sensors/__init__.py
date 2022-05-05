from .OtherAgentsObsSensor import OtherAgentsObsSensor
from .OtherAgentsStatesSensor import OtherAgentsStatesSensor
from .Sensor import Sensor



sensor_dict = {
    'empty_obs':    Sensor,
    'other_agents_obs':    OtherAgentsObsSensor,
    'other_agents_states': OtherAgentsStatesSensor,
}
