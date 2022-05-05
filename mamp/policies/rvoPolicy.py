import time
import numpy as np
from math import sqrt, sin, cos, atan2, asin, pi, floor, acos, asin
from itertools import product

from mamp.envs import Config
from mamp.util import sqr, l2norm, norm, normalize, absSq, mod2pi, pi_2_pi, eps, angle_2_vectors, is_intersect
from mamp.policies.policy import Policy


class RVOPolicy(Policy):
    def __init__(self):
        Policy.__init__(self, str="RVOPolicy")
        self.type = "internal"
        self.now_goal = None

    def find_next_action(self, obs, dict_comm, agent, kdTree):
        ts = time.time()
        self.now_goal = agent.goal_global_frame
        v_pref = compute_v_pref(agent)
        vA = agent.vel_global_frame
        pA = agent.pos_global_frame
        agent_rad = agent.radius + 0.05
        computeNeighbors(agent, kdTree)
        RVO_BA_all = []
        for obj in agent.neighbors:
            obj = obj[0]
            pB = obj.pos_global_frame
            if obj.is_at_goal:
                transl_vB_vA = pA
            else:
                vB = obj.vel_global_frame
                transl_vB_vA = pA + 0.5 * (vB + vA)  # Use RVO.
            obj_rad = obj.radius + 0.05

            RVO_BA = [transl_vB_vA, pA, pB, obj_rad + agent_rad]
            RVO_BA_all.append(RVO_BA)
        vA_post = intersect(v_pref, RVO_BA_all, agent)
        te = time.time()
        cost_step = te - ts
        agent.total_time += cost_step
        action = to_unicycle(vA_post, agent)
        theta = angle_2_vectors(vA, vA_post)
        agent.vel_global_unicycle[0] = int(1.0 * action[0]*eps)/eps
        agent.vel_global_unicycle[1] = int((0.22 * 0.5 * action[1] / Config.DT)*eps)/eps
        dist = l2norm(agent.pos_global_frame, agent.goal_global_frame)
        if theta > agent.max_heading_change:
            print('agent' + str(agent.id), len(agent.neighbors), action, '终点距离:', dist, 'theta:', theta)
        else:
            print('agent' + str(agent.id), len(agent.neighbors), action, '终点距离:', dist)

        return action


def compute_v_pref(agent):
    goal = agent.goal_global_frame
    diff = goal - agent.pos_global_frame
    v_pref = agent.pref_speed * normalize(diff)
    if l2norm(goal, agent.pos_global_frame) < Config.NEAR_GOAL_THRESHOLD:
        v_pref = np.zeros_like(v_pref)
    return v_pref


def intersect(v_pref, RVO_BA_all, agent):
    norm_v = np.linalg.norm(v_pref)
    suitable_V = []
    unsuitable_V = []

    Theta = np.arange(0, 2 * pi, step=pi / 32)
    RAD = np.linspace(0.02, 1.0, num=5)
    v_list = [v_pref[:]]
    for theta, rad in product(Theta, RAD):
        new_v = np.array([cos(theta), sin(theta)]) * rad * norm_v
        new_v = np.array([new_v[0], new_v[1]])
        v_list.append(new_v)

    for new_v in v_list:
        suit = True
        for RVO_BA in RVO_BA_all:
            p_0 = RVO_BA[0]
            pA = RVO_BA[1]
            pB = RVO_BA[2]
            combined_radius = RVO_BA[3]
            vAB = new_v + pA - p_0
            if is_intersect(pA, pB, combined_radius, vAB):
                suit = False
                break
        if suit:
            suitable_V.append(new_v)
        else:
            unsuitable_V.append(new_v)

    # ----------------------
    if suitable_V:
        suitable_V.sort(key=lambda v: l2norm(v, v_pref))  # sort begin at minimum and end at maximum
        vA_post = suitable_V[0]
    else:
        tc_V = dict()
        for unsuit_v in unsuitable_V:
            tc_V[tuple(unsuit_v)] = 0
            tc = []  # 时间
            for RVO_BA in RVO_BA_all:
                p_0 = RVO_BA[0]
                pA = RVO_BA[1]
                pB = RVO_BA[2]
                combined_radius = RVO_BA[3]
                vAB = unsuit_v + pA - p_0
                pApB = pB - pA
                if is_intersect(pA, pB, combined_radius, vAB):
                    discr = sqr(np.dot(vAB, pApB)) - absSq(vAB) * (absSq(pApB) - sqr(combined_radius))
                    if discr < 0.0:
                        print(discr)
                        continue
                    tc_v = (np.dot(vAB, pApB) - sqrt(discr)) / absSq(vAB)
                    if tc_v < 0:
                        tc_v = 0.0
                    tc.append(tc_v)
            if len(tc) == 0:
                tc = [0.0]
            tc_V[tuple(unsuit_v)] = min(tc) + 1e-5
        WT = agent.safetyFactor
        vA_post = min(unsuitable_V, key=lambda v: ((WT / tc_V[tuple(v)]) + l2norm(v, v_pref)))
    return vA_post


def computeNeighbors(agent, kdTree):
    agent.neighbors.clear()

    # Check obstacle neighbors.
    rangeSq = agent.neighborDist ** 2
    if len(agent.neighbors) != agent.maxNeighbors:
        rangeSq = 1.0 * agent.neighborDist ** 2
    kdTree.computeObstacleNeighbors(agent, rangeSq)

    if agent.in_collision:
        return

    # Check other agents.
    if len(agent.neighbors) != agent.maxNeighbors:
        rangeSq = agent.neighborDist ** 2
    kdTree.computeAgentNeighbors(agent, rangeSq)


def to_unicycle(vA_post, agent):
    vA_post = np.array(vA_post)
    norm_vA = norm(vA_post)
    yaw_next = mod2pi(atan2(vA_post[1], vA_post[0]))
    yaw_current = mod2pi(agent.heading_global_frame)
    delta_theta = yaw_next - yaw_current
    delta_theta = pi_2_pi(delta_theta)
    if delta_theta < -pi:
        delta_theta = delta_theta + 2 * pi
    if delta_theta > pi:
        delta_theta = delta_theta - 2 * pi
    if delta_theta >= 1.0:
        delta_theta = 1.0
    if delta_theta <= -1:
        delta_theta = -1.0
    action = np.array([norm_vA, delta_theta])
    return action
