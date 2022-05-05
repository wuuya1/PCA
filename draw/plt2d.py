import time
import numpy as np
import pandas as pd
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from math import sin, cos, atan2, sqrt, pow

from vis_util import get_2d_car_model, get_2d_uav_model
from vis_util import rgba2rgb

simple_plot = True


# img = plt.imread('beijing.jpg', 100)


def convert_to_actual_model_2d(agent_model, pos_global_frame, heading_global_frame):
    alpha = heading_global_frame
    for point in agent_model:
        x = point[0]
        y = point[1]
        # 进行航向计算
        ll = sqrt(pow(x, 2) + pow(y, 2))
        alpha_model = atan2(y, x)
        alpha_ = alpha + alpha_model - np.pi / 2  # 改加 - np.pi / 2 因为画模型的时候UAV朝向就是正北方向，所以要减去90°
        point[0] = ll * cos(alpha_) + pos_global_frame[0]
        point[1] = ll * sin(alpha_) + pos_global_frame[1]


def draw_agent_2d(ax, pos_global_frame, heading_global_frame, my_agent_model, color='blue'):
    agent_model = my_agent_model
    convert_to_actual_model_2d(agent_model, pos_global_frame, heading_global_frame)

    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
             ]

    path = Path(agent_model, codes)
    # 第二步：创建一个patch，路径依然也是通过patch实现的，只不过叫做pathpatch
    col = [0.8, 0.8, 0.8]
    patch = patches.PathPatch(path, fc=col, ec=col, lw=1.5)

    ax.add_patch(patch)


def draw_traj_2d(ax, obstacles_info, agents_info, agents_traj_list, step_num_list, current_step):
    cmap = get_cmap(64)
    plt_colors = get_colors()
    for idx, agent_traj in enumerate(agents_traj_list):
        # ax.imshow(img, extent=[-1, 4, -1, 4])     # 添加仿真背景
        agent_id = agents_info[idx]['id']
        agent_gp = agents_info[idx]['gp']
        agent_rd = agents_info[idx]['radius']
        agent_goal = agents_info[idx]['goal_pos']
        info = agents_info[idx]
        group = info['gp']
        radius = info['radius'] / 1
        color_ind = idx % len(plt_colors)
        # p_color = p_colors[color_ind]
        plt_color = plt_colors[color_ind]

        ag_step_num = step_num_list[idx]
        if current_step > ag_step_num - 1:
            plot_step = ag_step_num - 1
        else:
            plot_step = current_step

        pos_x = agent_traj['pos_x']
        pos_y = agent_traj['pos_y']
        alpha = agent_traj['alpha']

        # # 绘制目标点
        plt.plot(agent_goal[0], agent_goal[1], color=[0.933, 0.933, 0.0], marker='*', markersize=4)

        # 绘制探测区域
        # rect = patches.Rectangle((agent_goal[0] - 0.5, agent_goal[1] - 0.5), 1.0, 1.0, linewidth=1,
        #                          edgecolor=plt_color, facecolor='none', alpha=0.5)
        # ax.add_patch(rect)
        # plt.text(agent_goal[0]-0.85, agent_goal[1]-0.65, 'UGV ' + str(agent_id+1) +
        # ' Exploration Region', fontsize=12, color=p_color)

        # 绘制实线
        # plt.plot(pos_x[:plot_step], pos_y[:plot_step], color=plt_color)

        # 绘制箭头
        plt.arrow(pos_x[plot_step], pos_y[plot_step], 0.6 * cos(alpha[plot_step]), 0.6 * sin(alpha[plot_step]),
                  fc=plt_color, ec=plt_color, head_width=0.3, head_length=0.4)

        # 绘制渐变线
        # pcolors = np.zeros((plot_step, 4))
        # pcolors[:, :3] = plt_color[:3]
        # pcolors[:, 3] = np.linspace(0.2, 1., plot_step)
        # pcolors = rgba2rgb(pcolors)
        endure_time = 320
        # if plot_step <= endure_time:
        ax.scatter(pos_x[:plot_step], pos_y[:plot_step], marker='.', color=plt_color, s=0.2, alpha=0.5)
        # else:
        #     nt = plot_step - endure_time
        #     ax.scatter(pos_x[nt:plot_step], pos_y[nt:plot_step], color=pcolors[nt:plot_step], s=0.1, alpha=0.5)

        if simple_plot:
            ax.add_patch(plt.Circle((pos_x[plot_step], pos_y[plot_step]), radius=agent_rd, fc=plt_color, ec=plt_color))
            text_offset = agent_rd
            ax.text(pos_x[plot_step] + text_offset, pos_y[plot_step] + text_offset, str(agent_id + 1), color=plt_color)
        else:
            if group == 0:
                my_model = get_2d_car_model(size=agent_rd)
            else:
                my_model = get_2d_uav_model(size=agent_rd)
            pos = [pos_x[plot_step], pos_y[plot_step]]
            heading = alpha[plot_step]
            draw_agent_2d(ax, pos, heading, my_model)
    for i in range(len(obstacles_info)):
        pos = obstacles_info[i]['position']
        heading = 0.0
        rd = obstacles_info[i]['feature']
        agent_rd = rd[0] / 2
        my_model = get_2d_car_model(size=agent_rd)
        draw_agent_2d(ax, pos, heading, my_model)


def plot_save_one_pic(obstacles_info, agents_info, agents_traj_list, step_num_list, filename, current_step):
    fig = plt.figure(0)
    fig_size = (10, 8)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='X',
           ylabel='Y',
           )
    ax.axis('equal')
    # plt.grid(alpha=0.2)
    draw_traj_2d(ax, obstacles_info, agents_info, agents_traj_list, step_num_list, current_step)
    fig.savefig(filename, bbox_inches="tight")
    if current_step == 0: plt.show()
    # fig.savefig(filename)
    plt.close()


def plot_episode(obstacles_info, agents_info, traj_list, step_num_list, plot_save_dir, base_fig_name, last_fig_name,
                 show=False):
    current_step = 0
    num_agents = len(step_num_list)
    total_step = max(step_num_list)
    print('num_agents:', num_agents, 'total_step:', total_step)
    while current_step < total_step:
        fig_name = base_fig_name + "_{:05}".format(current_step) + '.png'
        filename = plot_save_dir + fig_name
        plot_save_one_pic(obstacles_info, agents_info, traj_list, step_num_list, filename, current_step)
        print(filename)
        current_step += 3
    filename = plot_save_dir + last_fig_name
    plot_save_one_pic(obstacles_info, agents_info, traj_list, step_num_list, filename, total_step)


def get_cmap(N):
    """Returns a function that maps each index in 0, 1, ... N-1 to a distinct RGB color."""
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color


def get_colors():
    py_colors = np.array(
        [
            [255, 0, 0], [255, 69, 0], [255, 165, 0], [0, 255, 0], [152, 251, 152], [0, 0, 255], [160, 32, 240],
            [255, 99, 71], [132, 112, 255], [0, 255, 255], [255, 69, 0], [148, 0, 211], [255, 192, 203],
            [255, 127, 0], [0, 191, 255], [255, 0, 255],
        ]
    )
    # py_colors = np.array(
    #     [
    #         [255, 99, 71], [255, 69, 0], [255, 0, 0], [255, 105, 180], [255, 192, 203], [238, 130, 238],
    #         [78, 238, 148], [67, 205, 128], [46, 139, 87], [154, 255, 154], [144, 238, 144], [0, 255, 127],
    #         [0, 238, 118], [0, 205, 102], [0, 139, 69], [0, 255, 0], [0, 139, 0], [127, 255, 0], [102, 205, 0],
    #         [192, 255, 62], [202, 255, 112], [255, 255, 0], [255, 215, 0], [255, 193, 37], [255, 193, 193],
    #         [250, 128, 114], [255, 160, 122], [255, 165, 0], [255, 140, 0], [255, 127, 80], [240, 128, 128],
    #         [221, 160, 221], [218, 112, 214], [186, 85, 211], [148, 0, 211], [138, 43, 226], [160, 32, 240],
    #         [106, 90, 205], [123, 104, 238], [0, 0, 205], [0, 0, 255], [30, 144, 255], [0, 191, 255], [152, 251, 152],
    #         [0, 255, 127], [124, 252, 0], [255, 250, 240], [253, 245, 230], [250, 235, 215],
    #         [255, 235, 205], [255, 218, 185], [255, 228, 181], [255, 248, 220], [255, 255, 240], [240, 255, 240],
    #         [230, 230, 250], [0, 245, 255], [187, 255, 255], [0, 255, 255], [127, 255, 212], [118, 238, 198],
    #         [102, 205, 170], [193, 255, 193], [84, 255, 159],
    #     ]
    # )
    return py_colors / 255
