import os
import subprocess
import sys

import matplotlib
import shapely
import shapely.geometry

matplotlib.use("Agg")
import glob
import json

import cv2
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip
import numpy as np
from cv2 import aruco
from natsort import natsorted

from mapping_cli.utils import (get_marker_coord, read_aruco_traj_file,
                               smoothen_trajectory, yml_parser)

DICT_CV2_ARUCO_MAP = {
    "TAG16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
}


def generate_trajectory_from_photos(
    input_path: str,
    maneuver: str,
    map_file_path: str,
    out_folder: str,
    calibration: str,
    size_marker: str,
    aruco_test_exe: str,
    cwd: str,
    ignoring_points: str = "",
    delete_video: bool = False,
    input_extension: str = ".jpg",
    framerate: int = 30,
    box_plot: bool = True,
    annotate: bool = True,
):
    out_file = os.path.join(out_folder, "temp.mp4")

    image_files = [
        os.path.join(input_path, img)
        for img in os.listdir(input_path)
        if img.endswith(input_extension)
    ]
    image_files = natsorted(image_files)
    print(image_files)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
        image_files, fps=framerate
    )
    clip.write_videofile(out_file)

    return get_locations(
        out_file,
        maneuver,
        map_file_path,
        out_folder,
        calibration,
        size_marker,
        aruco_test_exe,
        cwd,
        ignoring_points=ignoring_points,
        box_plot=box_plot,
        annotate=annotate,
    )
    if delete_video:
        os.remove(out_file)


def parse_marker_set(marker_set_str):
    return [int(x.strip()) for x in marker_set_str.split(",")]


def black_out(image_dir, out_dir, marker_set_str, marker_dict_str):
    images_f = glob.glob(image_dir + "*.jpg")
    dict_type = DICT_CV2_ARUCO_MAP[marker_dict_str]
    aruco_dict = aruco.Dictionary_get(dict_type)
    marker_set = parse_marker_set(marker_set_str)
    flatten = lambda l: [item for sublist in l for item in sublist]
    for image_f in images_f:
        im = cv2.imread(image_f)
        marker_info = cv2.aruco.detectMarkers(im, aruco_dict)
        markers_in_image = flatten(marker_info[1].tolist())
        blacked_marker_ids = [
            i for i, m in enumerate(markers_in_image) if m not in marker_set
        ]
        if blacked_marker_ids != []:
            cv2.fillPoly(
                im,
                pts=[marker_info[0][i][0].astype(int) for i in blacked_marker_ids],
                color=(0, 0, 0),
            )
        cv2.imwrite(out_dir + os.path.split(image_f)[1], im)


def get_locations(
    input_video: str,
    maneuver: str,
    map_file: str,
    out_folder: str,
    calibration: str,
    size_marker: str,
    aruco_test_exe: str,
    cwd: str,
    ignoring_points: str = "",
    plot: bool = True,
    box_plot: bool = True,
    smoothen: bool = True,
    blacken: bool = False,
    return_read: bool = False,
    annotate: bool = True,
):
    output_traj_path = os.path.join(out_folder, maneuver + "_CameraTrajectory.txt")

    if not os.path.exists(map_file):
        map_file = os.path.join(cwd, map_file)
    if not os.path.exists(calibration):
        calibration = os.path.join(cwd, calibration)

    # aruco_test_exe = r"C:\Users\t-jj\Documents\Projects\electron-hams\data\aruco_binaries\aruco_test_markermap.exe"
    sys.path.append(os.path.dirname(aruco_test_exe))
    exec_string = (
        f"{aruco_test_exe} {input_video} {map_file} {calibration} -s {size_marker}"
    )

    print(exec_string, cwd)

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
    # print("Hellloooo traj")
    print("out: ", stdout, "\n", "err: ", stderr)

    # output = subprocess.check_output(exec_string, shell=True, cwd=out_folder)
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

    plt_path = ""
    json_path = ""

    if plot:
        if box_plot:
            plt_path, json_path = save_box_coords(
                map_file,
                output_traj_path,
                out_folder,
                maneuver,
                ignoring_points,
                smoothen,
                annotate,
            )
        else:
            plt_path, json_path = save_line_coords(
                map_file, output_traj_path, out_folder, maneuver, ignoring_points
            )

    if return_read:
        return read_aruco_traj_file(output_traj_path, {})

    return output_traj_path, plt_path, json_path


def plot_markers(markers):
    markers_array = []
    for key in markers["aruco_bc_markers"]:
        marker_obj = markers["aruco_bc_markers"][key]
        marker_coord = get_marker_coord(marker_obj)
        plt.scatter(
            marker_coord[0], marker_coord[1], c=[[0, 0, 0]], marker=".", s=100.0
        )
        plt.annotate(str(key), (marker_coord[0], marker_coord[1]))
        markers_array.append([marker_coord[0], marker_coord[1]])

    return markers_array


def save_box_coords(
    manu_map,
    box_trajectory,
    out_folder,
    manu,
    ignoring_points: str = "",
    smoothen: bool = False,
    annotate: bool = True,
):
    # inputs: mapfile, box_trajectory, out_json, manoueuvre
    # output: set containing 4 corners of the box to the out_json
    if len(ignoring_points) > 0:
        ignoring_points_array = ignoring_points.split(",")
        ignoring_points_array = list(map(int, ignoring_points_array))
        ignoring_points_dict = {i: True for i in ignoring_points_array}

    else:
        ignoring_points_dict = dict()

    t_x = t_y = t_z = f_id = None
    if smoothen:
        f_id, t_x, t_y, t_z, _ = read_aruco_traj_file(
            box_trajectory, ignoring_points_dict
        )
        for k in ignoring_points_dict.keys():
            assert k not in f_id
        # TODO Check window length
        if len(f_id) < 99:
            smooth_window = len(f_id) - 1 if len(f_id) % 2 == 0 else len(f_id) - 2
        else:
            smooth_window = 99
        # smooth_window = 15
        t_x, t_y, t_z = smoothen_trajectory(t_x, t_y, t_z, smooth_window, 199, 2)

    print(ignoring_points_dict)

    # try:
    #     t_x, t_y, t_z = smoothen_trajectory(t_x, t_y, t_z, 3, 3, 2)
    # except:
    #     pass
    camera_coords = list(zip(t_x, t_z))

    rect_align = shapely.geometry.MultiPoint(
        camera_coords
    ).minimum_rotated_rectangle  # .envelope #minimum_rotated_rectangle
    x, y = rect_align.exterior.coords.xy
    rect_align = list(zip(x, y))

    markers = plot_markers(yml_parser(manu_map))

    for i in range(len(t_x)):
        plt.scatter(t_x[i], t_z[i], color="green")
        if annotate:
            plt.annotate(str(f_id[i]), (t_x[i], t_z[i]))
    for i in range(len(rect_align)):
        plt.scatter(rect_align[i][0], rect_align[i][1], color="blue")
    plt.gca().invert_yaxis()

    trackId = manu_map.split("/maps/")[0].split("/")[-1]
    plt_path = os.path.join(out_folder, f"{manu}.png")
    plt.savefig(plt_path)

    points = [list(a) for a in zip(t_x, t_z)]
    value = {"box": rect_align, "markers": markers, "points": points}
    json_path = os.path.join(out_folder, f"{manu}.json")
    with open(json_path, "w") as f:
        json.dump(value, f)

    print(rect_align)

    return plt_path, json_path


def save_line_coords(
    manu_map, line_trajectory, out_folder, manu, ignoring_points: str = ""
):
    if len(ignoring_points) > 0:
        ignoring_points_array = ignoring_points.split(",")
        ignoring_points_array = list(map(int, ignoring_points_array))
        ignoring_points_dict = {i: True for i in ignoring_points_array}

    else:
        ignoring_points_dict = dict()

    f_id, t_x, _, t_z, _ = read_aruco_traj_file(line_trajectory, ignoring_points_dict)
    line_eq = np.polyfit(t_x, t_z, 1)

    plot_markers(yml_parser(manu_map))
    for i in range(len(t_x)):
        plt.scatter(t_x[i], t_z[i], color="green")
        plt.annotate(str(f_id[i]), (t_x[i], t_z[i]))

    x_lim = (2, 10)
    y_lim = (0, 5)
    for x in np.arange(x_lim[0], x_lim[1], 0.1):
        y = line_eq[0] * x + line_eq[1]
        plt.scatter(x, y, color="blue")
    plt.gca().invert_yaxis()
    plt_path = os.path.join(out_folder, f"{manu}.png")
    plt.savefig(plt_path)

    value = {"line_eq": line_eq.tolist(), "pts": {"tx": t_x, "tz": t_z}}
    json_path = os.path.join(out_folder, f"{manu}.json")
    with open(json_path, "w") as f:
        json.dump(value, f)

    return plt_path, json_path
