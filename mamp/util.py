import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, sqrt, atan2, acos, asin
eps = 1e5


def is_intersect(pA, pB, combined_radius, vAB):
    pApB = pB - pA
    dist_pAB = l2norm(pA, pB)
    if dist_pAB <= combined_radius:
        dist_pAB = combined_radius
    theta_pABBound = asin(combined_radius / dist_pAB)
    if (dist_pAB * np.linalg.norm(vAB)) == 0.0:   # The v_dif is zero vector
        return False

    theta_pABvAB = angle_2_vectors(pApB, vAB)
    if theta_pABBound < theta_pABvAB:  # No intersecting or tangent.
        return False
    else:
        return True


def angle_2_vectors(v1, v2):
    v1v2_norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if v1v2_norm == 0.0:
        v1v2_norm = 1e-5
    cosdv = np.dot(v1, v2) / v1v2_norm
    if cosdv > 1.0:
        cosdv = 1.0
    elif cosdv < -1.0:
        cosdv = -1.0
    else:
        cosdv = cosdv
    angle = acos(cosdv)
    return angle


def reached(p1, p2, bound=0.5):
    if l2norm(p1, p2) < bound:
        return True
    else:
        return False


def is_in_between(theta_right, vec_AB, theta_v, theta_left):
    """
    判断一条射线（速度）在不在一个角度范围内
    """
    b_left = [cos(theta_left), sin(theta_left)]
    b_right = [cos(theta_right), sin(theta_right)]
    v = [cos(theta_v), sin(theta_v)]
    norm_b1 = norm(b_left + vec_AB)
    norm_b2 = norm(b_right + vec_AB)
    norm_bV = norm(v + vec_AB)
    return norm_bV >= norm_b1 or norm_bV >= norm_b2


def absSq(vec):
    return np.dot(vec, vec)


def left_of_line(p, p1, p2):
    tmpx = (p1[0] - p2[0]) / (p1[1] - p2[1]) * (p[1] - p2[1]) + p2[0]
    if tmpx > p[0]:
        return True
    else:
        return False


def cross(p1, p2, p3):  # 跨立实验
    x1 = p2[0] - p1[0]
    y1 = p2[1] - p1[1]
    x2 = p3[0] - p1[0]
    y2 = p3[1] - p1[1]
    return x1 * y2 - x2 * y1


def IsIntersec(p1, p2, p3, p4):  # 判断两线段是否相交
    # 快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if (max(p1[0], p2[0]) >= min(p3[0], p4[0])  # 矩形1最右端大于矩形2最左端
            and max(p3[0], p4[0]) >= min(p1[0], p2[0])  # 矩形2最右端大于矩形最左端
            and max(p1[1], p2[1]) >= min(p3[1], p4[1])  # 矩形1最高端大于矩形最低端
            and max(p3[1], p4[1]) >= min(p1[1], p2[1])):  # 矩形2最高端大于矩形最低端

        # 若通过快速排斥则进行跨立实验
        if (cross(p1, p2, p3) * cross(p1, p2, p4) <= 0
                and cross(p3, p4, p1) * cross(p3, p4, p2) <= 0):
            D = True
        else:
            D = False
    else:
        D = False
    return D


"""
用于列表排序，指定每个数组第二个元素
Example:
    a=[(2, 2), (3, 4), (4, 1), (1, 3)]
    a.sort(key=takeSecond)
"""


def takeSecond(elem):
    return elem[1]


def sqr(a):
    return a ** 2


def leftOf(a, b, c):
    return det(a - c, b - a)        #


def det(p, q):
    return p[0] * q[1] - p[1] * q[0]


def is_parallel(vec1, vec2):
    """ 判断二个向量是否平行 """
    assert vec1.shape == vec2.shape, r'输入的参数 shape 必须相同'
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    vec1_normalized = vec1 / norm_vec1
    vec2_normalized = vec2 / norm_vec2
    if norm_vec1 <= 1e-3 or norm_vec2 <= 1e-3:
        return True
    elif 1.0 - abs(np.dot(vec1_normalized, vec2_normalized)) < 1e-3:
        return True
    else:
        return False


def get_phi(vec):  # 计算两个向量投影后的夹角, 返回值是0-2*pi
    phi = mod2pi(atan2(vec[1], vec[0]))
    return int(phi*eps)/eps


def pi_2_pi(angle):  # to -pi-pi
    return int(((angle + np.pi) % (2 * np.pi) - np.pi)*eps)/eps


def mod2pi(theta):  # to 0-2*pi
    return int((theta - 2.0 * np.pi * np.floor(theta / 2.0 / np.pi))*eps)/eps


def norm(vec):
    return int(np.linalg.norm(vec)*eps)/eps


def normalize(vec):
    return vec / np.linalg.norm(vec)


def l2norm(x, y):
    return int(sqrt(l2normsq(x, y))*eps)/eps


def l2normsq(x, y):
    return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2


def plot_learning_curve(x, scores, figure_file, time, title='average reward'):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - time):(i + 1)])
    plt.plot(x, running_avg)
    plt.title(title)
    plt.savefig(figure_file)


##################
# Utils
################

def compute_time_to_impact(host_pos, other_pos, host_vel, other_vel, combined_radius):
    # http://www.ambrsoft.com/TrigoCalc/Circles2/CirclePoint/CirclePointDistance.htm
    v_rel = host_vel - other_vel
    coll_cone_vec1, coll_cone_vec2 = tangent_vecs_from_external_pt(host_pos[0],
                                                                   host_pos[1],
                                                                   other_pos[0],
                                                                   other_pos[1],
                                                                   combined_radius)

    if coll_cone_vec1 is None:
        # collision already occurred ==> collision cone isn't meaningful anymore
        return 0.0
    else:
        # check if v_rel btwn coll_cone_vecs
        # (B btwn A, C): https://stackoverflow.com/questions/13640931/how-to-determine-if-a-vector-is-between-two-other-vectors)

        if (np.cross(coll_cone_vec1, v_rel) * np.cross(coll_cone_vec1, coll_cone_vec2) >= 0 and
                np.cross(coll_cone_vec2, v_rel) * np.cross(coll_cone_vec2, coll_cone_vec1) >= 0):
            # quadratic eqn for soln to line from host agent pos along v_rel vector to collision circle
            # circle: (x-a)**2 + (y-b)**2 = r**2
            # line: y = v1/v0 *(x-px) + py
            # solve for x: (x-a)**2 + ((v1/v0)*(x-px)+py-a)**2 = r**2
            v0, v1 = v_rel
            if abs(v0) < 1e-5 and abs(v1) < 1e-5:
                # agents aren't moving toward each other ==> inf TTC
                return np.inf

            px, py = host_pos
            a, b = other_pos
            r = combined_radius
            if abs(v0) < 1e-5:  # vertical v_rel (solve for y, x known)
                print("[warning] v0=0, and not yet handled")
                x1 = x2 = px
                A = 1
                B = -2 * b
                C = b ** 2 + (px - a) ** 2 - r ** 2
                y1 = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
                y2 = (-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
            else:  # non-vertical v_rel (solve for x)
                A = 1 + (v1 / v0) ** 2
                B = -2 * a + 2 * (v1 / v0) * (py - b - (v1 / v0) * px)
                C = a ** 2 - r ** 2 + ((v1 / v0) * px - (py - b)) ** 2

                det = B ** 2 - 4 * A * C
                if det == 0:
                    print("[warning] det == 0, so only one tangent pt")
                elif det < 0:
                    print("[warning] det < 0, so no tangent pts...")

                x1 = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
                x2 = (-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
                y1 = (v1 / v0) * (x1 - px) + py
                y2 = (v1 / v0) * (x2 - px) + py

            d1 = np.linalg.norm([x1 - px, y1 - py])
            d2 = np.linalg.norm([x2 - px, y2 - py])
            d = min(d1, d2)
            spd = np.linalg.norm(v_rel)
            return d / spd
        else:
            return np.inf


def tangent_vecs_from_external_pt(xp, yp, a, b, r):
    # http://www.ambrsoft.com/TrigoCalc/Circles2/CirclePoint/CirclePointDistance.htm
    # (xp, yp) is coords of pt outside of circle
    # (x-a)**2 + (y-b)**2 = r**2 is defn of circle

    sq_dist_to_perimeter = (xp - a) ** 2 + (yp - b) ** 2 - r ** 2
    if sq_dist_to_perimeter < 0:
        # print("sq_dist_to_perimeter < 0 ==> agent center is already within coll zone??")
        return None, None

    sqrt_term = np.sqrt((xp - a) ** 2 + (yp - b) ** 2 - r ** 2)
    xnum1 = r ** 2 * (xp - a)
    xnum2 = r * (yp - b) * sqrt_term

    ynum1 = r ** 2 * (yp - b)
    ynum2 = r * (xp - a) * sqrt_term

    den = (xp - a) ** 2 + (yp - b) ** 2

    # pt1, pt2 are the tangent pts on the circle perimeter
    pt1 = np.array([(xnum1 + xnum2) / den + a, (ynum1 - ynum2) / den + b])
    pt2 = np.array([(xnum1 - xnum2) / den + a, (ynum1 + ynum2) / den + b])

    # vec1, vec2 are the vecs from (xp,yp) to the tangent pts on the circle perimeter
    vec1 = pt1 - np.array([xp, yp])
    vec2 = pt2 - np.array([xp, yp])

    return vec1, vec2


def vec2_l2_norm(vec):
    # return np.linalg.norm(vec)
    return sqrt(vec2_l2_norm_squared(vec))


def vec2_l2_norm_squared(vec):
    return vec[0] ** 2 + vec[1] ** 2


# speed filter
# input: past velocity in (x,y) at time_past
# output: filtered velocity in (speed, theta)
def filter_vel(dt_vec, agent_past_vel_xy):
    average_x = np.sum(dt_vec * agent_past_vel_xy[:, 0]) / np.sum(dt_vec)
    average_y = np.sum(dt_vec * agent_past_vel_xy[:, 1]) / np.sum(dt_vec)
    speeds = np.linalg.norm(agent_past_vel_xy, axis=1)
    speed = np.linalg.norm(np.array([average_x, average_y]))
    angle = np.arctan2(average_y, average_x)

    return np.array([speed, angle])


# angle_1 - angle_2
# contains direction in range [-3.14, 3.14]
def find_angle_diff(angle_1, angle_2):
    angle_diff_raw = angle_1 - angle_2
    angle_diff = (angle_diff_raw + np.pi) % (2 * np.pi) - np.pi
    return angle_diff


# keep angle between [-pi, pi]
def wrap(angle):
    while angle >= np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def find_nearest(array, value):
    # array is a 1D np array
    # value is an scalar or 1D np array
    tiled_value = np.tile(np.expand_dims(value, axis=0).transpose(), (1, np.shape(array)[0]))
    idx = (np.abs(array - tiled_value)).argmin(axis=1)
    return array[idx], idx


def rad2deg(rad):
    return rad * 180 / np.pi


def rgba2rgb(rgba):
    # rgba is a list of 4 color elements btwn [0.0, 1.0]
    # or a 2d np array (num_colors, 4)
    # returns a list of rgb values between [0.0, 1.0] accounting for alpha and background color [1, 1, 1] == WHITE
    if isinstance(rgba, list):
        alpha = rgba[3]
        r = max(min((1 - alpha) * 1.0 + alpha * rgba[0], 1.0), 0.0)
        g = max(min((1 - alpha) * 1.0 + alpha * rgba[1], 1.0), 0.0)
        b = max(min((1 - alpha) * 1.0 + alpha * rgba[2], 1.0), 0.0)
        return [r, g, b]
    elif rgba.ndim == 2:
        alphas = rgba[:, 3]
        r = np.clip((1 - alphas) * 1.0 + alphas * rgba[:, 0], 0, 1)
        g = np.clip((1 - alphas) * 1.0 + alphas * rgba[:, 1], 0, 1)
        b = np.clip((1 - alphas) * 1.0 + alphas * rgba[:, 2], 0, 1)
        return np.vstack([r, g, b]).T


def yaw_to_quaternion(yaw):
    pitch = 0;
    roll = 0
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr
    return qx, qy, qz, qw


def run_episode(env, one_env):
    total_reward = 0
    step = 0
    done = False
    while not done:
        obs, rew, done, info = env.step([None])
        total_reward += rew[0]
        step += 1

    # After end of episode, store some statistics about the environment
    # Some stats apply to every gym env...
    generic_episode_stats = {'total_reward': total_reward, 'steps': step, }

    agents = one_env.agents
    # agents = one_env.prev_episode_agents
    time_to_goal = np.array([a.t for a in agents])
    extra_time_to_goal = np.array([a.t - a.straight_line_time_to_reach_goal for a in agents])
    collision = np.array(np.any([a.in_collision for a in agents])).tolist()
    all_at_goal = np.array(np.all([a.is_at_goal for a in agents])).tolist()
    any_stuck = np.array(np.any([not a.in_collision and not a.is_at_goal for a in agents])).tolist()
    outcome = "collision" if collision else "all_at_goal" if all_at_goal else "stuck"
    specific_episode_stats = {'num_agents': len(agents), 'time_to_goal': time_to_goal,
                              'total_time_to_goal': np.sum(time_to_goal),
                              'extra_time_to_goal': extra_time_to_goal, 'collision': collision,
                              'all_at_goal': all_at_goal, 'any_stuck': any_stuck, 'outcome': outcome,
                              'policies': [agent.policy.str for agent in agents], }

    # Merge all stats into a single dict
    episode_stats = {**generic_episode_stats, **specific_episode_stats}

    env.reset()

    return episode_stats, agents


def store_stats(df, hyperparameters, episode_stats):
    # Add a new row to the pandas DataFrame (a table of results, where each row is an episode)
    # that contains the hyperparams and stats from that episode, for logging purposes
    df_columns = {**hyperparameters, **episode_stats}
    df = df.append(df_columns, ignore_index=True)
    return df
