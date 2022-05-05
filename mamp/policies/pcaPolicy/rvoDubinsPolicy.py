import time
import numpy as np
from math import sqrt, sin, cos, atan2, pi, acos
from itertools import product

from mamp.envs import Config
from mamp.policies.policy import Policy
from mamp.policies.pcaPolicy import dubinsmaneuver2d
from mamp.util import sqr, l2norm, norm, is_parallel, absSq, mod2pi, pi_2_pi, get_phi, angle_2_vectors, is_intersect


class RVODubinsPolicy(Policy):
    def __init__(self):
        Policy.__init__(self, str="RVODubinsPolicy")
        self.type = "internal"
        self.now_goal = None

    def find_next_action(self, obs, dict_comm, agent, kdTree):
        ts = time.time()
        self.now_goal = agent.goal_global_frame
        v_pref = compute_v_pref(agent)
        vA = agent.vel_global_frame
        pA = agent.pos_global_frame
        if l2norm(vA, [0, 0, 0]) <= 1e-5:
            vA_post = 0.2 * v_pref
            te = time.time()
            cost_step = te - ts
            agent.total_time += cost_step
            action = to_unicycle(vA_post, agent)
            theta = 0.0
        else:
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
        agent.vel_global_unicycle[0] = round(1.0 * action[0], 5)
        agent.vel_global_unicycle[1] = round(0.22 * 0.5 * action[1] / Config.DT, 5)
        dist = l2norm(agent.pos_global_frame, agent.goal_global_frame)
        if theta > agent.max_heading_change:
            print('agent' + str(agent.id), len(agent.neighbors), action, '终点距离:', dist, 'theta:', theta)
        else:
            print('agent' + str(agent.id), len(agent.neighbors), action, '终点距离:', dist)

        return action


def update_dubins(agent):
    dis = l2norm(agent.pos_global_frame, agent.dubins_now_goal)
    sampling_size = agent.dubins_sampling_size
    if dis <= sampling_size:
        if agent.dubins_path:
            agent.dubins_now_goal = np.array(agent.dubins_path.pop()[:2], dtype='float64')
        else:
            agent.dubins_now_goal = agent.goal_global_frame


def compute_dubins(agent):
    qi = np.hstack((agent.pos_global_frame, agent.heading_global_frame))
    qf = np.hstack((agent.goal_global_frame, agent.goal_heading_frame))
    rmin, pitchlims = agent.turning_radius, agent.pitchlims
    maneuver = dubinsmaneuver2d.dubins_path_planning(qi, qf, rmin)
    dubins_path = maneuver.path.copy()
    agent.dubins_sampling_size = maneuver.sampling_size
    desire_length = maneuver.length
    points_num = len(dubins_path)
    path = []
    for i in range(points_num):
        path.append((dubins_path.pop()))
    return path, desire_length, points_num


def is_parallel_neighbor(agent):
    range_para = 0.15
    max_parallel_steps = int((agent.radius + agent.neighbors[0][0].radius + 0.2) / (agent.pref_speed * agent.timeStep))
    if len(agent.is_parallel_neighbor) < max_parallel_steps and len(agent.neighbors) > 0:
        agent.is_parallel_neighbor.append((agent.neighbors[0][0].id, agent.neighbors[0][1]))
        return False
    else:
        agent.is_parallel_neighbor.pop(0)
        agent.is_parallel_neighbor.append((agent.neighbors[0][0].id, agent.neighbors[0][1]))
        nei = agent.is_parallel_neighbor
        is_same_id = True
        is_in_range = True
        for i in range(len(nei) - 1):
            if nei[0][0] != nei[i + 1][0]:
                is_same_id = False
            dis = abs(sqrt(nei[0][1]) - sqrt(nei[i + 1][1]))
            if dis > range_para:
                is_in_range = False
        if is_same_id and is_in_range:
            return True
        else:
            return False


def dubins_path_node_pop(agent):
    agent.dubins_last_goal = np.array(agent.dubins_path.pop()[:3], dtype='float64')
    agent.dubins_last_goal = np.array(agent.dubins_path.pop()[:3], dtype='float64')
    agent.dubins_last_goal = np.array(agent.dubins_path.pop()[:3], dtype='float64')
    agent.dubins_last_goal = np.array(agent.dubins_path.pop()[:3], dtype='float64')


def compute_v_pref(agent):
    """
        is_parallel(vA, v_pref): —— Whether to leave Dubins trajectory as collision avoidance.
        dis_goal <= k: —— Regard as obstacles-free when the distance is less than k.
        dis < 6 * sampling_size: —— Follow the current Dubins path when not moving away from the current Dubins path.
        theta >= np.deg2rad(100): —— Avoid the agent moving away from the goal position after update Dubins path.
    """
    pA = agent.pos_global_frame
    pG = agent.goal_global_frame
    dist_goal = l2norm(pA, pG)
    k = 3.0 * agent.turning_radius
    if not agent.is_use_dubins:  # first
        agent.is_use_dubins = True
        dubins_path, desire_length, points_num = compute_dubins(agent)
        agent.dubins_path = dubins_path
        dubins_path_node_pop(agent)
        agent.dubins_now_goal = np.array(agent.dubins_path.pop()[:2], dtype='float64')
        dif_x = agent.dubins_now_goal - pA
    else:
        update_dubins(agent)
        vA = np.array(agent.vel_global_frame)
        v_pref = agent.v_pref
        dis = l2norm(pA, agent.dubins_now_goal)
        max_size = round(6 * agent.dubins_sampling_size, 5)
        pApG = pG - pA
        theta = angle_2_vectors(vA, pApG)
        deg135 = np.deg2rad(135)
        if ((is_parallel(vA, v_pref) or dist_goal <= k) and dis < max_size) or (theta >= deg135):
            update_dubins(agent)
            if agent.dubins_path:
                dif_x = agent.dubins_now_goal - pA
            else:
                dif_x = pG - pA
        else:
            dubins_path, length, points_num = compute_dubins(agent)
            agent.dubins_path = dubins_path
            dubins_path_node_pop(agent)
            agent.dubins_now_goal = np.array(agent.dubins_path.pop()[:2], dtype='float64')
            dif_x = agent.dubins_now_goal - pA

    norm_dif_x = dif_x * agent.pref_speed / l2norm(dif_x, [0, 0])
    v_pref = np.array(norm_dif_x)
    if dist_goal < Config.NEAR_GOAL_THRESHOLD:
        v_pref[0] = 0.0
        v_pref[1] = 0.0
    agent.v_pref = v_pref
    v_pref = np.array([round(v_pref[0], 5), round(v_pref[1], 5)])
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
            v_dif = np.array(new_v + pA - p_0)  # new_v-0.5*(vA+vB) or new_v-vB
            if is_intersect(pA, pB, combined_radius, v_dif):
                suit = False
                break
        if suit:
            suitable_V.append(new_v)
        else:
            unsuitable_V.append(new_v)

    # ----------------------
    if suitable_V:
        suitable_V.sort(key=lambda v: l2norm(v, v_pref))  # Sort begin at minimum and end at maximum.
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
                v_dif = np.array(unsuit_v + pA - p_0)
                pApB = pB - pA
                if is_intersect(pA, pB, combined_radius, v_dif):
                    discr = sqr(np.dot(v_dif, pApB)) - absSq(v_dif) * (absSq(pApB) - sqr(combined_radius))
                    if discr < 0.0:
                        print(discr)
                        continue
                    tc_v = max((np.dot(v_dif, pApB) - sqrt(discr)) / absSq(v_dif), 0.0)
                    tc.append(tc_v)
            if len(tc) == 0:
                tc = [0.0]
            tc_V[tuple(unsuit_v)] = min(tc) + 1e-5
        WT = agent.safetyFactor
        vA_post = min(unsuitable_V, key=lambda v: ((WT / tc_V[tuple(v)]) + l2norm(v, v_pref)))

    return vA_post


def satisfied_constraint(agent, vCand):
    vA = agent.vel_global_frame
    costheta = np.dot(vA, vCand) / (np.linalg.norm(vA) * np.linalg.norm(vCand))
    if costheta > 1.0:
        costheta = 1.0
    elif costheta < -1.0:
        costheta = -1.0
    theta = acos(costheta)  # Rotational constraints.
    if theta <= agent.max_heading_change:
        return True
    else:
        return False


def select_right_side(v_list, vA):
    opt_v = [v_list[0]]
    i = 1
    while True:
        if abs(l2norm(v_list[0], vA) - l2norm(v_list[i], vA)) < 1e-1:
            opt_v.append(v_list[i])
            i += 1
            if i == len(v_list):
                break
        else:
            break
    vA_phi_min = min(opt_v, key=lambda v: get_phi(v))
    vA_phi_max = max(opt_v, key=lambda v: get_phi(v))
    phi_min = get_phi(vA_phi_min)
    phi_max = get_phi(vA_phi_max)
    if abs(phi_max - phi_min) <= pi:
        vA_opti = vA_phi_min
    else:
        vA_opti = vA_phi_max
    return vA_opti


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
