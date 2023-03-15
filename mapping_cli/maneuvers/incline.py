import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from mapping_cli.halts import get_halts
from mapping_cli.maneuvers.maneuver import Maneuver
from mapping_cli.utils import (aggregate_direction, debug_directions_visualize,
                               generate_trajectory, get_marker_coord,
                               majority_vote_smoothen, plot_line,
                               smoothen_trajectory, yml_parser)


def select_traj_pts_by_direction(tx, tz, directions, label):
    selected_pts = []
    for x, z, d in zip(tx, tz, directions):
        if d == label:  # remember, this label is reverse
            selected_pts.append([x, z])
    return selected_pts


def incline_rollback_logic(tx, tz, directions, max_rollback_allowed_metres, rep):
    selected_pts = select_traj_pts_by_direction(
        tx, tz, directions, 0
    )  # remember, this label is reverse
    max_di = 0
    for p in selected_pts:
        for p_d in selected_pts:
            di = (p[0] - p_d[0]) ** 2 + (p[1] - p_d[1]) ** 2
            if max_di < di:
                max_di = di
    incline_rollback_decision = "Fail"
    if selected_pts == []:
        incline_rollback_decision = "Fail"
    elif selected_pts != [] and np.sqrt(max_di) <= max_rollback_allowed_metres:
        incline_rollback_decision = "Pass"
    print("(incline) Rollback (m): {}".format(np.sqrt(max_di)))
    print("(incline) Rollback Decision: {}".format(incline_rollback_decision))
    rep.add_report(
        "incline_rollback",
        {"value_in_m": np.sqrt(max_di), "decision": incline_rollback_decision},
    )
    return incline_rollback_decision


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


def get_line(line_info_json_f):
    line_info = None
    with open(line_info_json_f) as f:
        line_info = json.load(f)
    l_stop = np.polyfit(line_info["pts"]["tx"], line_info["pts"]["tz"], 1)
    return l_stop


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


def plot_traj(
    tx,
    tz,
    directions,
    line_json,
    x_limits,
    y_limits,
    rep,
    behind_lines_obey_threshold,
    markers,
    save_image_name=None,
):
    if markers is not None:
        plot_markers(markers)

    ax = plt.gca()

    line_info_json_f = line_json
    l_stop = get_line(line_info_json_f)
    line_viz_x_lim = x_limits

    plot_line(plt, l_stop, line_viz_x_lim, "green")

    markers_side = get_side_of_marker_from_line(markers, l_stop)

    selected_pts = select_traj_pts_by_direction(tx, tz, directions, -1)  ## halt is -1

    behind_lines_decision, percentage_obey = percentage_of_points_behind_line(
        selected_pts, l_stop, markers_side
    )

    decision = "Fail"
    if percentage_obey >= behind_lines_obey_threshold:
        decision = "Pass"
    rep.add_report(
        "{}_behind_line".format("incline"),
        {"value": percentage_obey, "decision": decision},
    )
    print(
        "({}) Halt Behind Line %: {} Decision: {}".format(
            "incline", percentage_obey, decision
        )
    )

    color_f = ["red", "blue", "black"]
    size_f = [25, 25, 40]
    len_plot = min(min(len(tx), len(tz)), len(directions))
    plt.scatter(
        tx[:len_plot],
        tz[:len_plot],
        c=[color_f[directions[i]] for i in range(0, len_plot)],
        s=[size_f[directions[i]] for i in range(0, len_plot)],
        marker=".",
    )

    plot_legend()
    plot_limits(x_limits, y_limits)

    # if markers is not None:
    #     rotate_plot(ax, markers, manu)

    plt.savefig(save_image_name, dpi=200)
    plt.close()


class Incline(Maneuver):
    def run(self) -> None:
        map_path = self.config.get_config_value("map_file_path")
        if not os.path.exists(map_path):
            map_path = os.path.join(self.inputs["cwd"], map_path)

        calib_path = self.config["calibration_file_path"]
        if not os.path.exists(calib_path):
            calib_path = os.path.join(self.inputs["cwd"], calib_path)

        traj = generate_trajectory(
            self.inputs["back_video"],
            "incline",
            map_path,
            self.out_folder,
            calib_path,
            self.config["size_marker"],
            self.config["aruco_test_exe"],
            self.inputs["cwd"],
        )

        _, tx, ty, tz, camera_matrices = traj
        direction, stats = self.get_direction_stats(
            traj,
            self.config["rev_fwd_halt_segment_min_frame_len"],
            self.config["min_frame_len"],
        )

        # smoothen trajectory
        tx, ty, tz = smoothen_trajectory(tx, ty, tz, 25, 25, 2)

        # get stats
        decision = incline_rollback_logic(
            tx, tz, direction, self.config["max_rollback_allowed_metres"], self.report
        )

        # rollback

        line_path = self.config["line_file_path"]
        if not os.path.exists(line_path):
            line_path = os.path.join(self.inputs["cwd"], line_path)

        markers = yml_parser(map_path)
        # plot
        plot_traj(
            tx,
            tz,
            direction,
            line_path,
            self.config["x_limits"],
            self.config["y_limits"],
            self.report,
            self.config["behind_lines_obey_threshold"],
            markers,
            os.path.join(self.out_folder, "incline.png"),
        )

        return (
            (stats, decision),
            (
                {
                    "back_video": self.inputs["back_video"],
                    "trajectory": os.path.join(self.out_folder, "incline.png"),
                }
            ),
        )

    def get_direction_stats(
        self, traj, rev_fwd_halt_segment_min_frame_len, min_frame_len
    ):
        """ """
        f_ids, tx, ty, tz, camera_matrices = traj
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

        # debug_directions_visualize(plt, tx, ty, tz, directions)

        return directions, stats
