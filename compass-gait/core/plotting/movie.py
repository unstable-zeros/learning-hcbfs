"""Code needed to produce a movie of the compass gait walker
walking down his ramp."""

import os
import pickle
import numpy as np
from numpy import radians as rad
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arc, RegularPolygon
from collections import namedtuple

import cg_dynamics.compass_gait as compass_gait

TOLERANCE = 1e-7
PARAMS = namedtuple('CompassGaitParams',
        ['mass_hip', 'mass_leg', 'length_leg', 'center_of_mass_leg', 'gravity', 'slope'],
        defaults=[10.0, 5.0, 1.0, 0.5, 9.81, 0.0525])

def make_movie(traj, ctrl_name, args, downsample=False):
    """Create a movie from a given compass gait trajectory.
    
    Params:
        traj: trajectory from compass gait walker.
        ctrl_name: name of controller used to collect rollout.
        args: command line arguments for train.py.
        downsample: if True, downsamples frames of video.
    """

    print(f'Making movie with {ctrl_name} ctrl trajectory...', end='')

    frames = []
    for idx in range(args.horizon):

        state = (traj['true_dis_state'][idx, :], traj['x_cts'][idx, :],
            traj['obs'][idx], traj['u_seq'][idx])

        if downsample is True:
            if idx % 20 == 0:
                frames.append(state)
        else:
            frames.append(state)

    name = os.path.join(args.results_dir, ctrl_name)
    plot_trajectory(frames, PARAMS(), name, dt=args.dt)

    print('done.')
    
def plot_trajectory(frames, params, ctrl_name, dt):
    """Accumulate all frames into a movie of a single trajectory."""

    fig, ax = plt.subplots()
    ax.set_yticks([])

    left_leg, = plt.plot([], [], '#23d1de', linewidth=3)
    right_leg, = plt.plot([], [], 'r', linewidth=3)
    ground_x = np.linspace(-1, 7, num=40)
    ground_y = np.array([-toe * np.sin(params.slope) for toe in ground_x])
    plt.plot(ground_x, ground_y, 'k--')

    def get_positions(discrete_state, cg_state, obs, action):
        """Get positions of right toe, left_top and top of walker."""

        toe = discrete_state[compass_gait._TOE_POSITION]
        left_toe = np.array([toe * np.cos(params.slope), 
                                -toe * np.sin(params.slope)])
        top = np.array([obs[compass_gait._FB_X], obs[compass_gait._FB_Z]])

        direction = left_toe - top
        new_direction = rot2d(cg_state[compass_gait._STANCE] - \
            cg_state[compass_gait._SWING]).dot(direction)
        right_toe = top + new_direction

        if (discrete_state[compass_gait._TICKS] % 2) == 0:
            return left_toe, top, right_toe
        return right_toe, top, left_toe

    def init():
        """Initialize function for FuncAnimation."""

        ax.set_xlim(-0.5, 6)
        ax.set_ylim(-0.5, 1.25)
        ax.set_aspect('equal')
        return [left_leg, right_leg]

    def update(frame):
        """Update function for FuncAnimation."""

        left_toe, top, right_toe = get_positions(*frame)
        action = frame[3]
        discrete_state = frame[0]
        
        left_leg, = plt.plot([], [], '#23d1de', linewidth=2, alpha=0.4)
        right_leg, = plt.plot([], [], 'r', linewidth=2, alpha=0.4)

        left_leg.set_data([left_toe[0], top[0]], [left_toe[1], top[1]])
        right_leg.set_data([top[0], right_toe[0]], [top[1], right_toe[1]])
        circ = plt.Circle((top[0], top[1]), 0.05, color='k', alpha=0.4)
        ax.add_artist(circ)

        if action[0] > TOLERANCE:
            ax.plot([top[0]],[top[1]], marker=r'$\circlearrowleft$', 
                        ms=20, linewidth=0.5, color='green')

        elif action[1] > TOLERANCE:
            if (discrete_state[compass_gait._TICKS] % 2) == 0:
                ax.plot([left_toe[0]],[left_toe[1]], 
                            marker=r'$\circlearrowleft$', ms=20, 
                            linewidth=0.5, color='green')
            else:
                ax.plot([right_toe[0]],[right_toe[1]], 
                            marker=r'$\circlearrowleft$', 
                            ms=20, linewidth=0.5, color='green')

        return [left_leg, right_leg]

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 * dt, 
                        init_func=init, blit=True)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()

    # save video to file in results directory
    ani.save(f'{ctrl_name}-movie.mp4', fps=None, 
                extra_args=['-vcodec', 'libx264'])

    # save final frame of video
    # plt.savefig(f'{ctrl_name}-last-frame.png')

def angdiff(theta1, theta2):
    """Difference between two angles."""

    return np.mod((theta1 - theta2) + np.pi, 2 * np.pi) - np.pi

def rot2d(theta):
    """Return 2d rotation matrix with angle theta."""

    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]])

def drawCirc(ax, radius, centX, centY, angle_, theta2_, color_='black'):
    """Draw circular arrow at the current hip joint."""

    # create the arced curve for the arrow
    arc = Arc([centX,centY],radius,radius,angle=angle_,
          theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=10,color=color_)
    ax.add_patch(arc)

    # create the arrow head
    endX=centX+(radius/2)*np.cos(rad(theta2_+angle_)) #Do trig to determine end position
    endY=centY+(radius/2)*np.sin(rad(theta2_+angle_))

    ax.add_patch(                    #Create triangle as arrow head
        RegularPolygon(
            (endX, endY),            # (x,y)
            3,                       # number of vertices
            radius/9,                # radius
            rad(angle_+theta2_),     # orientation
            color=color_
        )
    )
