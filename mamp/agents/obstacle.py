import numpy as np
import math


class Obstacle(object):
    def __init__(self, pos, shape_dict, id):
        self.shape = shape = shape_dict['shape']
        self.feature = feature = shape_dict['feature']
        if shape == 'rect':
            self.width, self.heigh = shape_dict['feature']
            self.radius = math.sqrt(self.width ** 2 + self.heigh ** 2) / 2
        elif shape == 'circle':
            self.radius = shape_dict['feature']
        else:
            raise NotImplementedError

        self.pos_global_frame = np.array(pos, dtype='float64')
        self.vel_global_frame = np.array([0.0, 0.0, 0.0])
        self.pos = pos
        self.is_obstacle = True
        self.id = id
        self.t = 0.0
        self.step_num = 0
        self.is_at_goal = True
        self.was_in_collision_already = False
        self.in_collision = False

        self.x = pos[0]
        self.y = pos[1]
        self.r = self.radius

