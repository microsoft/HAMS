import argparse
import csv
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
from scipy.signal import savgol_filter
from sklearn import decomposition

# from mapping_cli.utils import smooth
# from mapping_cli.utils import smoothen_tr


def smooth_halt_signal(x, window_size=31, poly_order=2):
    """ """
    print("Len: ", len(x))
    smooth_x = savgol_filter(x, window_size, poly_order)
    # for i in range(5):
    # 	smooth_x = savgol_filter(smooth_x, window_size, poly_order)
    return smooth_x


def debug_halt_visualize(
    traj, traj_1d, smooth_traj, velocity, smooth_vel, zero_frames, fname
):
    """ """
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(traj[:, 0], traj[:, 2], label="Original Camera Path")
    ax2.plot(traj_1d, label="1D Camera Path")
    ax2.plot(smooth_traj, label="Filtered Trajectory")
    ax3.plot(velocity, label="Velocity")
    ax3.plot(smooth_vel, label="Smooth Velocity")
    ax3.scatter(
        [i for i in range(zero_frames.shape[0]) if zero_frames[i]],
        [smooth_vel[i] for i in range(zero_frames.shape[0]) if zero_frames[i]],
        c="red",
    )
    ax2.legend()
    ax1.legend()
    ax3.legend()
    # plt.show()

    save_image_name = os.path.splitext(fname)[0] + "_halts.png"
    plt.savefig(save_image_name)


def get_halts_worker(tx, ty, tz, vel_delta):
    # if len(tx) < 199:
    #     # pass
    #     tx, ty, tz = smoothen_trajectory(tx, ty, tz, 21, 49, 2)
    # else:
    #     tx, ty, tz = smoothen_trajectory(tx, ty, tz, 99, 199, 2)

    traj = np.array([x for x in zip(tx, ty, tz)])
    pca = decomposition.PCA(n_components=1)
    traj_1d = traj
    traj_1d = pca.fit_transform(traj).reshape((-1,))

    # Smoothen trajectory
    try:
        smooth_traj = smooth_halt_signal(traj_1d)
    except ValueError as e:
        print("Trajectory length is too small!\n", e)

    # Get Velocity and smoothen it
    velocity = np.abs(smooth_traj[:-vel_delta] - smooth_traj[vel_delta:])
    traj_1d = traj_1d[:-vel_delta]
    smooth_traj = smooth_traj[:-vel_delta]
    traj = traj[:-vel_delta]

    smooth_vel = smooth_halt_signal(velocity)

    # Get frames with no motion
    zero_frames = np.zeros((traj_1d.shape[0]), dtype=bool)
    zero_frames[smooth_vel <= 1e-3] = True
    zero_frames[traj[:, 0] == 0] = False

    # Compensate for velocity delta
    zero_frames = np.array(zero_frames.tolist() + [False] * vel_delta)

    # debug_halt_visualize(traj, traj_1d, smooth_traj, velocity, smooth_vel, zero_frames)

    return zero_frames


def get_halts(tx, ty, tz, vel_delta=15):
    """ """
    return get_halts_worker(tx, ty, tz, vel_delta)
