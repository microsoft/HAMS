import itertools
import json
import logging
import math
import os
import subprocess
from typing import Dict

import cv2
import ffmpeg
import filelock
import numpy as np
import scipy
import shapely
import yaml


def yml_parser(map_file):
    ret = None
    with open(map_file, "r") as stream:
        stream.readline()  # YAML 1.0 bug
        try:
            ret = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)
            raise exc
    ## YAML 1.0 bug
    # YAML library handles YAML 1.1 and above,
    # to fix one of the bug in the generated YAML
    for x in ret["aruco_bc_markers"]:
        invalid_key = list(filter(lambda x: x.startswith("id"), x.keys()))[0]
        key, value = invalid_key.split(":")
        x[key] = value
        del x[invalid_key]
    marker_dict = {}
    for dct in ret["aruco_bc_markers"]:
        marker_dict[int(dct["id"])] = dct["corners"]
    ret["aruco_bc_markers"] = marker_dict
    return ret


def euclidean_distance(A, B):
    return np.linalg.norm(np.array(A) - np.array(B), 2)


def euclidean_distance_batch(A, B):
    dist_array = scipy.spatial.distance.cdist(A, B, metric="euclidean")
    return dist_array


def generate_trajectory(
    input_video: str,
    maneuver: str,
    map_file_path: str,
    out_folder: str,
    calibration: str,
    size_marker: str,
    aruco_test_exe: str,
    cwd: str,
):
    """
    Generate a trajectory from a video and a maneuver
    :param input_video: input video path
    :param maneuver: maneuver trajectory to be generated
    :param map_file_path: map file path
    :param out_folder: output folder
    :param calibration: calibration
    :param size_marker: size of the marker
    :param aruco_test_exe: aruco test executable
    :return:
    """
    print(os.getcwd())
    output_traj_path = os.path.join(out_folder, maneuver + "_trajectory.txt")
    exec_string = f"{aruco_test_exe} {input_video} {map_file_path} {calibration} -s {size_marker} {output_traj_path}"

    assert os.path.exists(input_video), f"{input_video} does not exist"
    assert os.path.exists(map_file_path), f"{map_file_path} does not exist"
    assert os.path.exists(calibration), f"{calibration} does not exist"
    # assert os.path.exists(aruco_test_exe), f"{aruco_test_exe} does not exist"

    try:
        # output = subprocess.check_output(
        #     exec_string, shell=True, stderr=subprocess.STDOUT
        # )

        my_env = os.environ.copy()
        my_env["PATH"] = out_folder + ";" + cwd + ";" + my_env["PATH"]

        p = subprocess.Popen(
            exec_string,
            shell=True,
            cwd=cwd,
            env=my_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = p.communicate()

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(
                e.cmd, e.returncode, e.output
            )
        )

    # popen = subprocess.Popen(exec_string, shell=True, stdout=subprocess.PIPE)
    # out, err = popen.communicate()

    # print(exec_string, out, err)

    output = stdout.decode("utf-8")
    output = (
        output.replace(";\r\n", "")
        .replace("]", "")
        .replace("[", "")
        .replace(",", "")
        .replace("\r\n", "\n")
    )

    with open(output_traj_path, "w+") as f:
        f.write(output)
        f.close()

    return read_aruco_traj_file(output_traj_path, {})


def get_marker_coord(marker_obj):
    return [marker_obj[0][0], marker_obj[0][2]]


def aggregate_direction(direction_vector, halt_len):
    """
    Input: vector containing 'forward', 'reverse'
    Output: # of forwards, # of reverse
    """
    direction_count = {"F": 0, "R": 0, "H": 0}
    # Group consecutive directions
    grouped_class = [
        list(l)
        for _, l in itertools.groupby(enumerate(direction_vector), key=lambda x: x[1])
    ]
    new_direction_vector = []
    for c_idx, c in enumerate(grouped_class):
        if c[0][1] == 0:
            direction_count["R"] += 1
            new_direction_vector += [0 for i in range(len(c))]
        elif c[0][1] == 1:
            direction_count["F"] += 1
            new_direction_vector += [1 for i in range(len(c))]
        elif c[0][1] == -1:
            if len(c) > halt_len:
                direction_count["H"] += 1
                new_direction_vector += [-1 for i in range(len(c))]
            else:
                # Start segment
                if c_idx > 0:
                    prev_direction = grouped_class[c_idx - 1][0][1]
                else:
                    next_direction = grouped_class[c_idx + 1][0][1]
                    prev_direction = next_direction

                # End segment
                if c_idx < len(grouped_class) - 1:
                    next_direction = grouped_class[c_idx + 1][0][1]
                else:
                    next_direction = prev_direction

                new_direction_vector += [prev_direction for i in range(len(c) // 2)]
                new_direction_vector += [next_direction for i in range(len(c) // 2)]

    return direction_count, new_direction_vector


def get_maneuver_box(box_path):
    val = None
    with open(box_path) as f:
        d = json.load(f)
        val = d["box"]
    val = list(
        zip(*shapely.geometry.Polygon(val).minimum_rotated_rectangle.exterior.coords.xy)
    )
    return np.array(val)


def get_C_coord(rvec, tvec):
    """
    Input: rotation vector, translation vector
    Returns: Camera coordinates
    """
    rvec = cv2.Rodrigues(np.array(rvec))[0]
    tvec = np.array(tvec)
    return -np.dot(np.transpose(rvec), tvec)


def read_aruco_traj_file(filename, ignoring_points_dict):
    f_id = []
    t_x = []
    t_y = []
    t_z = []
    camera_matrices = []
    with open(filename, "r") as f:
        line = f.readline()
        while line:
            if not line[0].isdigit():
                line = f.readline()
                continue
            else:  # rvec- rotational vector, tvec - translational vector
                row = line.split(" ")
                cur_id = int(row[0].strip())
                if not cur_id in ignoring_points_dict:
                    rvec = [
                        float(row[1].strip()),
                        float(row[2].strip()),
                        float(row[3].strip()),
                    ]
                    tvec = [
                        float(row[4].strip()),
                        float(row[5].strip()),
                        float(row[6].strip()),
                    ]

                    cvec = get_C_coord(rvec, tvec)
                    cvec = cvec.reshape(3, 1)

                    temp = np.hstack((cv2.Rodrigues(np.array(rvec))[0], cvec))
                    camera_matrix = np.vstack((temp, np.array([0.0, 0.0, 0.0, 1.0])))

                    f_id.append(int(row[0].strip()))
                    camera_matrices.append(camera_matrix)
                    t_x.append(cvec[0][0])
                    t_y.append(cvec[1][0])
                    t_z.append(cvec[2][0])

                line = f.readline()

    # if len(t_x) == 0:
    #     raise Exception("Trajectory length 0")

    return f_id, t_x, t_y, t_z, camera_matrices


def stitch(input_imgs_path: str, input_file_extension: str, output_file_name: str):
    ffmpeg.input(os.path.join(input_imgs_path, f"*.{input_file_extension}")).output(
        output_file_name
    ).run()
    return output_file_name


def detect_marker(frame, marker_dict):
    DICT_CV2_ARUCO_MAP = {
        "TAG16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    }
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(DICT_CV2_ARUCO_MAP[marker_dict])
    parameters = cv2.aruco.DetectorParameters_create()
    _, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    markers = []
    if ids is not None:
        for i in ids:
            markers.append(i[0])
    return markers


def trim_video(
    video_path: str,
    start_segment,
    end_segment,
    use_gpu: bool,
    out_folder: str,
    output_name: str,
):
    device_str = ""
    if use_gpu:
        device_str = " -gpu 0 "
    call_string = f"ffmpeg -y -hide_banner -loglevel panic -ss {start_segment} -i {video_path} -t {end_segment-start_segment} -b:v 4000K -vcodec h264_nvenc {device_str}{os.path.join(out_folder, output_name)}"
    print("call string: ", call_string)
    my_env = os.environ.copy()
    my_env["PATH"] = out_folder + ";" + my_env["PATH"]
    p = subprocess.Popen(call_string, shell=True, cwd=out_folder, env=my_env)
    stdout, stderr = p.communicate()
    print(stdout, stderr)
    return os.path.join(out_folder, output_name)


class Report:
    def __init__(self, textfile):
        self.textfile = textfile
        self.lockfile = textfile + ".lock"

    def open_file(self):
        if os.path.exists(self.textfile):
            with open(self.textfile, "r") as f:
                self.jsonf = json.load(f)
        else:
            self.jsonf = {}

    def add_report(self, key, val):
        with filelock.FileLock(self.lockfile):
            self.open_file()
            self.jsonf[key] = val
            with open(self.textfile, "w") as f:
                json.dump(self.jsonf, f, indent=2)


def stitch(input_imgs_path: str, input_file_extension: str, output_file_name: str):
    ffmpeg.input(os.path.join(input_imgs_path, f"*.{input_file_extension}")).output(
        output_file_name
    ).run()
    return output_file_name


def detect_marker(frame, marker_dict):
    DICT_CV2_ARUCO_MAP = {
        "TAG16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    }
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(DICT_CV2_ARUCO_MAP[marker_dict])
    parameters = cv2.aruco.DetectorParameters_create()
    _, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    markers = []
    if ids is not None:
        for i in ids:
            markers.append(i[0])
    return markers


def yml_parser(map_file):
    ret = None

    with open(map_file, "r") as stream:
        stream.readline()  # YAML 1.0 bug
        try:
            ret = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    ## YAML 1.0 bug
    # YAML library handles YAML 1.1 and above,
    # to fix one of the bug in the generated YAML
    for x in ret["aruco_bc_markers"]:
        invalid_key = list(filter(lambda x: x.startswith("id"), x.keys()))[0]
        key, value = invalid_key.split(":")
        x[key] = value
        del x[invalid_key]

    marker_dict = {}
    for dct in ret["aruco_bc_markers"]:
        marker_dict[int(dct["id"])] = dct["corners"]
    ret["aruco_bc_markers"] = marker_dict
    return ret


def get_marker_coord(marker_obj):
    return [marker_obj[0][0], marker_obj[0][2]]


def get_maneuver_box(box_path):
    val = None
    with open(box_path) as f:
        d = json.load(f)
        val = d["box"]
    val = list(
        zip(*shapely.geometry.Polygon(val).minimum_rotated_rectangle.exterior.coords.xy)
    )
    return np.array(val)


def get_C_coord(rvec, tvec):
    """
    Input: rotation vector, translation vector
    Returns: Camera coordinates
    """
    rvec = cv2.Rodrigues(np.array(rvec))[0]
    tvec = np.array(tvec)
    return -np.dot(np.transpose(rvec), tvec)


# def read_aruco_traj_file(filename):
# 	f_id = []
# 	t_x = []
# 	t_y = []
# 	t_z = []
# 	camera_matrices = []
# 	with open(filename, 'r') as f:
# 		line = f.readline()
# 		while line:
# 			if not line[0].isdigit():
# 				line = f.readline()
# 				continue
# 			else: #rvec- rotational vector, tvec - translational vector
# 				row = line.split(' ')
# 				rvec = [float(row[1].strip()), float(row[2].strip()), float(row[3].strip())]
# 				tvec = [float(row[4].strip()), float(row[5].strip()), float(row[6].strip())]

# 				cvec = get_C_coord(rvec, tvec)
# 				cvec = cvec.reshape(3, 1)

# 				temp = np.hstack(( cv2.Rodrigues( np.array(rvec) )[0], cvec))
# 				camera_matrix = np.vstack((temp, np.array([0.0, 0.0, 0.0, 1.0])))

# 				f_id.append(int(row[0].strip()))
# 				camera_matrices.append(camera_matrix)
# 				t_x.append( cvec[0][0] )
# 				t_y.append( cvec[1][0] )
# 				t_z.append( cvec[2][0] )
# 				line = f.readline()
# 	return f_id, t_x, t_y, t_z, camera_matrices


def majority_vote_smoothen(vector, window):
    """
    Input: vector of 1's and 0's
    Output: smoothing applied to window
    """
    vector_smooth = []
    for i in range(0, len(vector), window):
        forw = vector[i : i + window].count(1)
        rev = vector[i : i + window].count(0)
        halt = vector[i : i + window].count(-1)
        x = None
        if forw > 2 * rev and forw > halt:
            x = 1
            for i in range(window):
                vector_smooth.append(1)
        elif rev > 2 * forw and rev > halt:
            x = 2
            for i in range(window):
                vector_smooth.append(0)
        else:
            x = 3
            for i in range(window):
                vector_smooth.append(-1)
        # print(forw, rev, halt, x)
    return vector_smooth[: len(vector)]


def debug_directions_visualize(plt, tx, ty, tz, directions):
    """ """
    colors = ["red", "blue", "black"]
    le = min(min(len(tx) - 1, len(tz) - 1), len(directions) - 1)
    plt.scatter(
        [tx[i] for i in range(0, le)],
        [tz[i] for i in range(0, le)],
        c=[colors[directions[i]] for i in range(0, le)],
    )
    plt.show()


def rotate_point(x, y, rot_radian):
    return [
        x * math.cos(rot_radian) - y * math.sin(rot_radian),
        y * math.cos(rot_radian) + x * math.sin(rot_radian),
    ]


def rotate_rectangle(points, camera_center, rot_radian):
    # Displace to origin
    o_points = np.zeros(points.shape)
    for i in range(4):
        o_points[i, :] = points[i, :] - camera_center
    # Rotate points
    for i in range(4):
        o_points[i, 0], o_points[i, 1] = rotate_point(
            o_points[i, 0], o_points[i, 1], rot_radian
        )
    # Displace again
    for i in range(4):
        o_points[i, :] = o_points[i, :] + camera_center
    return o_points


def rotation_matrix_to_euler_angles(R):
    assert is_rotation_matrix(R)
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def is_rotation_matrix(R):
    Rt = np.transpose(R)
    should_be_identity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - should_be_identity)
    return n < 1e-6


def plot_line(plt, l_stop, x_lim, color):
    for x in np.arange(x_lim[0], x_lim[1], 0.1):
        y = l_stop[0] * x + l_stop[1]
        plt.scatter(x, y, color=color)


def smoothen_trajectory(tx, ty, tz, outlier_window, poly_fit_window, poly_fit_degree):
    # Applying median filter
    tx = scipy.signal.medfilt(tx, outlier_window)
    ty = scipy.signal.medfilt(ty, outlier_window)
    tz = scipy.signal.medfilt(tz, outlier_window)
    # Applying savitzky golay filter
    tx = scipy.signal.savgol_filter(tx, poly_fit_window, poly_fit_degree)
    ty = scipy.signal.savgol_filter(ty, poly_fit_window, poly_fit_degree)
    tz = scipy.signal.savgol_filter(tz, poly_fit_window, poly_fit_degree)
    return tx, ty, tz


def get_graph_box(box_path, label):
    with open(box_path) as f:
        d = json.load(f)
        val = d[label]
    val = list(
        zip(*shapely.geometry.Polygon(val).minimum_rotated_rectangle.exterior.coords.xy)
    )
    return np.array(val)


def get_plt_rotation_from_markers(origin_marker, axis_marker, is_vertical):
    try:
        marker_origin = origin_marker
        marker_axis = axis_marker
        m = (marker_axis[1] - marker_origin[1]) / (marker_axis[0] - marker_origin[0])
        deg = np.rad2deg(np.arctan(m))
        if is_vertical:
            deg = 90 - deg
        else:
            deg = -1 * deg  # make anti clockwise
        return deg
    except:
        return 0.0
