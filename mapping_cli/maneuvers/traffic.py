import json
import math
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import shapely

from mapping_cli.halts import get_halts
from mapping_cli.maneuvers.maneuver import Maneuver
from mapping_cli.utils import (aggregate_direction, debug_directions_visualize,
                               generate_trajectory, get_marker_coord,
                               majority_vote_smoothen, plot_line,
                               rotate_rectangle,
                               rotation_matrix_to_euler_angles,
                               smoothen_trajectory, yml_parser)


def get_side_of_marker_from_line(markers, l_stop):
    markers_side = []
    # In Standard form: -m*x + y - c = 0
    for key in markers["aruco_bc_markers"]:
        marker_obj = markers["aruco_bc_markers"][key]
        marker_coord = get_marker_coord(marker_obj)
        # placing marker coords into the line
        # to determine their side
        if (
            -1 * l_stop[0] * marker_coord[0] + marker_coord[1] - l_stop[1] > 0
        ):  # Might want to have a looser criterion
            markers_side.append(1)
        else:
            markers_side.append(-1)
    if not all(map(lambda x: x == markers_side[0], markers_side)):
        raise Exception("All markers for this maneuver should be behind the line!")
    else:
        return markers_side[0]


def select_traj_points_by_time(tx, tz, segment_json, fps):
    if not os.path.exists(segment_json):
        raise Exception("segment_json not part of input arguments!")
    with open(segment_json) as f:
        seg_vals = json.load(f)
        print("Seg vals: ", seg_vals.keys())
    frame_num_traffic_light_vid_start_abs = math.ceil(seg_vals["traffic"]["start"][0])

    frame_num_red_light_off_abs = fps
    if frame_num_red_light_off_abs == -1:
        print("WARNING!!!!! traffic Light hook is not implemented")
        return list(zip(tx, tz))

    if (
        frame_num_traffic_light_vid_start_abs > frame_num_red_light_off_abs
    ):  # my video somehow cut afterwards
        return []

    end_frame_red_light_check = (
        frame_num_red_light_off_abs - frame_num_traffic_light_vid_start_abs
    )
    if end_frame_red_light_check > len(tx):
        end_frame_red_light_check = len(tx)
    return list(zip(tx[:end_frame_red_light_check], tz[:end_frame_red_light_check]))


def percentage_of_points_behind_line(selected_pts, l_stop, markers_side):
    behind_lines_decision = []
    for x, z in selected_pts:
        val_halt = -1 * l_stop[0] * x + z - l_stop[1]
        if np.sign(val_halt) != np.sign(markers_side):
            behind_lines_decision.append(1)
        else:
            behind_lines_decision.append(0)

    percentage_obey = 0.0
    for x in behind_lines_decision:
        if x == 1:
            percentage_obey += 1
    if behind_lines_decision == []:
        percentage_obey = 0.0
    else:
        percentage_obey /= len(behind_lines_decision)
    return behind_lines_decision, percentage_obey


def post_process_direction_vector(stats, directions):
    # aggregate seq
    uniq_vals = []
    pivot_val = directions[0]
    pivot_len = 1
    i = 1
    while i < len(directions):
        while i < len(directions) and pivot_val == directions[i]:
            pivot_len += 1
            i += 1
        if i < len(directions):
            uniq_vals.append([pivot_val, pivot_len])
            pivot_val = directions[i]
            pivot_len = 1
            i += 1
        if i == len(directions):
            uniq_vals.append([pivot_val, pivot_len])
            break

    # delete from seq
    if len(uniq_vals) >= 3:
        del_elem = []
        for idx in range(0, len(uniq_vals) - 2):
            # R = 0, H = -1, F = 1
            if (
                uniq_vals[idx + 0][0] == 0
                and uniq_vals[idx + 1][0] == -1
                and uniq_vals[idx + 2][0] == 0
            ):  # RHR --> RR
                stats["R"] -= 1
                stats["H"] -= 1
                uniq_vals[idx][1] += uniq_vals[idx + 1][1]
                uniq_vals[idx][1] += uniq_vals[idx + 2][1]
                del_elem.append(idx + 1)
                del_elem.append(idx + 2)
        for idx in del_elem[::-1]:
            del uniq_vals[idx]

    # recreate seq
    directions = []
    for a, b in uniq_vals:
        directions += [a] * b  # pythonic?

    return stats, directions


def get_line(line_info_json_f):
    line_info = None
    with open(line_info_json_f) as f:
        line_info = json.load(f)
    l_stop = np.polyfit(line_info["pts"]["tx"], line_info["pts"]["tz"], 1)
    return l_stop


def get_maneuver_box(box_path):
    val = None
    with open(box_path) as f:
        d = json.load(f)
        val = d["box"]
    val = list(
        zip(*shapely.geometry.Polygon(val).minimum_rotated_rectangle.exterior.coords.xy)
    )
    return np.array(val)


def get_car_box(cam_x, cam_y, camera_matrix, manu, car_dims, site_config):
    car_top_left = [cam_x - car_dims["x_offset"], cam_y - car_dims["z_offset"]]
    car_top_right = [car_top_left[0] + car_dims["width"], car_top_left[1]]
    car_bottom_right = [
        car_top_left[0] + car_dims["width"],
        car_top_left[1] + car_dims["length"],
    ]
    car_bottom_left = [car_top_left[0], car_top_left[1] + car_dims["length"]]

    angle = rotation_matrix_to_euler_angles(camera_matrix)[1]

    if site_config.maneuvers_config["viz"][manu]["markers_vertical"]:
        angle *= -1.0

    car_pts = np.array([car_top_left, car_top_right, car_bottom_right, car_bottom_left])
    car_rotated_pts = rotate_rectangle(car_pts, np.array([cam_x, cam_y]), angle)

    return car_rotated_pts


def plot_traj(
    tx,
    tz,
    manu,
    car_dims,
    line_json,
    config,
    maneuver: Maneuver,
    markers=None,
    save_image_name=None,
    camera_matrices=None,
    max_iou_idx=None,
    segment_json=None,
):
    if markers is not None:
        plot_markers(markers)

    ax = plt.gca()

    # poly = mpatches.Polygon(box, alpha=1, fill=False, edgecolor='black')
    # ax.add_patch(poly)

    # Plot Car
    if max_iou_idx is not None:
        position = max_iou_idx
        car_pts = get_car_box(
            tx[position],
            tz[position],
            camera_matrices[position][:3, :3],
            manu,
            car_dims,
        )
        car_patch = mpatches.Polygon(car_pts, alpha=1, fill=False, edgecolor="red")
        ax.add_patch(car_patch)

    line_info_json_f = line_json

    if config["line_type"] == "stop":
        l_stop = get_line(line_info_json_f)

        stop_ped_offset = config["stop_ped_line_dist"]
        l_ped = [l_stop[0], l_stop[1] + stop_ped_offset]
    elif config["line_type"] == "ped":
        l_ped = get_line(line_info_json_f)
        stop_line_offset = config["stop_ped_line_dist"]
        l_stop = [l_ped[0], l_ped[1] + stop_line_offset]

    line_viz_x_lim = config["xlim"]

    plot_line(plt, l_stop, line_viz_x_lim, "blue")
    plot_line(plt, l_ped, line_viz_x_lim, "yellow")

    ## markers are always ahead of stop line,
    ## ped line is gonna be ahead of stop line
    ## X X * X $   |
    ##     *   $   |
    ##     *   $   |
    ##     *   $   |
    ## Legend: Marker : X, StopLine : |, PedLine1 : $, Pedline2 : *
    ## Define markers_side_ped such that both PL1 and PL2 are satisfied
    ##
    ## As we can assume that ped_line will always be ahead of stop_line,
    ## every point on ped_line will be sign(stop_line(point)) == sign(stop_line(marker))
    ##
    ## Let's find a point on stop_line, that will always be behind ped_line
    ## Use that point to find the side and then invert it
    markers_side = get_side_of_marker_from_line(markers, l_stop)  # say this is 1
    point_on_stop_line = (0, l_stop[0] * 0 + l_stop[1])
    markers_side_ped = -1 * np.sign(
        -1 * l_ped[0] * point_on_stop_line[0] + point_on_stop_line[1] - l_ped[1]
    )

    selected_pts = select_traj_points_by_time(tx, tz, segment_json, config["fps"])

    behind_lines_decision_ped, percentage_obey = percentage_of_points_behind_line(
        selected_pts, l_ped, markers_side_ped
    )
    behind_lines_decision_stop, percentage_stop = percentage_of_points_behind_line(
        selected_pts, l_stop, markers_side
    )
    label_decision = []
    for i in range(len(behind_lines_decision_ped)):
        if behind_lines_decision_ped[i] == 1 and behind_lines_decision_stop[i] == 1:
            label_decision.append(0)
        elif behind_lines_decision_ped[i] == 1 and behind_lines_decision_stop[i] == 0:
            label_decision.append(1)
        elif behind_lines_decision_ped[i] == 0 and behind_lines_decision_stop[i] == 0:
            label_decision.append(2)
        else:
            raise Exception("Error: Ped Line is not ahead of Stop Line in track!")

    obey_decision = "Fail"
    if percentage_obey >= config["behind_lines_obey_threshold"]:
        obey_decision = "Pass"

    stop_decision = "Fail"
    if percentage_stop > config["behind_lines_stop_threshold"]:
        stop_decision = "Pass"

    maneuver.report.add_report("trafficLight_obey_decision", obey_decision)
    maneuver.report.add_report("trafficLight_stop_decision", stop_decision)
    maneuver.report.add_report(
        "trafficLight_outcome",
        {
            "percentage_behind_both_lines": percentage_obey,
            "percentage_behind_stop_line": percentage_stop,
        },
    )

    print(
        "({}) Halt Behind Stop Line %: {} Decision: {}".format(
            manu, percentage_stop, stop_decision
        )
    )
    print(
        "({}) Halt Behind Ped Line %: {} Decision: {}".format(
            manu, percentage_obey, obey_decision
        )
    )

    color_f = ["green", "yellow", "brown"]
    size_f = [25, 25, 40]
    len_plot = len(selected_pts)
    print(len(selected_pts), len(label_decision))
    plt.scatter(
        [x[0] for x in selected_pts],
        [x[1] for x in selected_pts],
        c=[color_f[label_decision[i]] for i in range(0, len_plot)],
        s=[size_f[label_decision[i]] for i in range(0, len_plot)],
        marker="*",
    )

    plot_legend()
    plot_limits(config["xlim"], config["ylim"])

    # if markers is not None:
    #     rotate_plot(ax, markers, manu)

    plt.savefig(save_image_name, dpi=200)
    plt.close()


def plot_legend():
    red_patch = mpatches.Patch(color="red", label="Reverse")
    blue_patch = mpatches.Patch(color="blue", label="Forward")
    black_patch = mpatches.Patch(color="black", label="Halt")
    plt.legend(
        handles=[blue_patch, red_patch, black_patch],
        prop={"size": 10},
        loc="upper right",
    )


def plot_markers(markers):
    for key in markers["aruco_bc_markers"]:
        marker_obj = markers["aruco_bc_markers"][key]
        marker_coord = get_marker_coord(marker_obj)
        plt.scatter(
            marker_coord[0], marker_coord[1], c=[[0, 0, 0]], marker=".", s=100.0
        )


def plot_limits(x_limits, y_limits):
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.gca().set_aspect("equal", "box")


class Traffic(Maneuver):
    def run(self):
        map_path = self.config.get_config_value("map_file_path")
        if not os.path.exists(map_path):
            map_path = os.path.join(self.inputs["cwd"], map_path)

        calib_path = self.config["calibration_file_path"]
        if not os.path.exists(calib_path):
            calib_path = os.path.join(self.inputs["cwd"], calib_path)
        try:
            traj = generate_trajectory(
                self.inputs["back_video"],
                self.config.get_config_value("maneuver"),
                map_path,
                self.out_folder,
                calib_path,
                self.config["size_marker"],
                self.config["aruco_test_exe"],
                self.inputs["cwd"],
            )
        except Exception as e:
            print("Error generating traffic trajectory: ", e)
            raise e

        _, tx, ty, tz, camera_matrices = traj
        tx, ty, tz = smoothen_trajectory(tx, ty, tz, 25, 25, 2)
        markers = yml_parser(map_path)

        # get_direction_stats
        direction, stats = self.get_direction_stats(
            tx,
            ty,
            tz,
            camera_matrices,
            self.config["rev_fwd_halt_segment_min_frame_len"],
            self.config["min_frame_len"],
        )

        # fwd rev halts
        halts = get_halts(tx, ty, tz)

        # plot
        # plot_traj(
        #     tx, tz, "traffic", self.config['car_dims'], self.config['line_file_path'], self.config, "traffic", camera_matrices=camera_matrices, max_iou_idx=self.config['max_iou'], segment_json=os.path.join(self.out_folder, f"segment.json")
        # )

        line_path = self.config["line_file_path"]
        if not os.path.exists(line_path):
            line_path = os.path.join(self.inputs["cwd"], line_path)

        plot_traj(
            tx,
            tz,
            "traffic",
            self.config["car_dims"],
            line_path,
            self.config,
            self,
            markers,
            segment_json=os.path.join(self.out_folder, "manu_json_seg_int.json"),
            save_image_name=os.path.join(self.out_folder, "traffic.png"),
        )

        return (
            True,
            stats,
            {  # TODO: Add pass/fail
                "trajectory": f"{os.path.join(self.out_folder, 'traffic.png')}"
            },
        )

    def get_direction_stats(
        self,
        tx,
        ty,
        tz,
        camera_matrices,
        rev_fwd_halt_segment_min_frame_len,
        min_frame_len,
    ):
        """ """
        directions = [1]
        Xs = []
        for i in range(1, len(tx)):
            translation = (
                np.array([tx[i], ty[i], tz[i], 1.0]).reshape(4, 1).astype(np.float64)
            )
            X = camera_matrices[i - 1][:3, :3].dot(
                translation[:3, 0] - camera_matrices[i - 1][:3, -1]
            )
            # print(X)
            direction = X[2]

            if [tx[i], ty[i], tz[i]] == [tx[i - 1], ty[i - 1], tz[i - 1]]:
                directions.append(directions[-1])
            else:
                if direction >= 0:
                    directions.append(1)
                else:
                    directions.append(0)

        directions = directions[1:]

        # Get Forward-Reverse
        directions = np.array(directions + [-1])

        # Get halts
        halt_info = get_halts(tx, ty, tz)
        directions[halt_info == True] = -1
        directions = directions.tolist()
        directions = majority_vote_smoothen(
            directions, rev_fwd_halt_segment_min_frame_len
        )

        stats, directions = aggregate_direction(directions, min_frame_len)
        stats, _ = aggregate_direction(directions, min_frame_len)

        stats, directions = post_process_direction_vector(stats, directions)

        debug_directions_visualize(plt, tx, ty, tz, directions)

        return directions, stats
