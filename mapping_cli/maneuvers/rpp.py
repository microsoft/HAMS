import itertools
import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.collections import PathCollection
from matplotlib.transforms import Affine2D

from mapping_cli.halts import get_halts
from mapping_cli.locator import get_locations, read_aruco_traj_file
from mapping_cli.maneuvers.maneuver import Maneuver
from mapping_cli.utils import (get_graph_box, get_marker_coord,
                               get_plt_rotation_from_markers,
                               majority_vote_smoothen, rotate_rectangle,
                               rotation_matrix_to_euler_angles,
                               smoothen_trajectory, yml_parser)


class RPP(Maneuver):
    def run(self) -> None:
        # 1 Get trajectory
        traj = get_locations(
            self.inputs["back_video"],
            "rpp",
            self.config["map_file_path"],
            self.out_folder,
            self.config["calib_file_path"],
            self.config["marker_size"],
            self.config["aruco_test_exe"],
            cwd=self.inputs["cwd"],
            plot=False,
            return_read=True,
            annotate=False,
        )
        if traj is False:
            return False

        _, tx, ty, tz, camera_matrices = traj

        if len(tx) > 99:
            tx, ty, tz = smoothen_trajectory(tx, ty, tz, 99, 199, 2)
        else:
            tx, ty, tz = smoothen_trajectory(tx, ty, tz, 33, 199 // 3, 2)

        map_file = self.config["map_file_path"]
        if not os.path.exists(map_file):
            map_file = os.path.join(self.inputs["cwd"], map_file)
        markers = yml_parser(map_file)
        # 2 In box
        box_path = self.config["box_file_path"]
        if not os.path.exists(box_path):
            box_path = os.path.join(self.inputs["cwd"], box_path)
        is_inside, max_iou, _ = is_car_inside(
            tx,
            ty,
            tz,
            camera_matrices,
            self.config["car_dims"],
            markers_vertical=self.config["markers_vertical"],
            box_overlap_threshold=self.config["box_overlap_threshold"],
            box_path=box_path,
        )

        _, stats = get_direction_stats(
            tx,
            ty,
            tz,
            camera_matrices,
            self.config["rev_fwd_halt_segment_min_frame_len"],
            self.config["min_frame_len"],
            self.out_folder,
        )
        # 3 Standard path check

        _, std_path_decision = check_standard_path(
            tx,
            ty,
            tz,
            camera_matrices,
            markers,
            self.config,
            "rpp",
            self.out_folder,
            self.inputs["cwd"],
            True,
        )
        print(
            f"is inside: {is_inside}, stats: {stats}, std path decision: {std_path_decision}"
        )
        return (
            (is_inside and std_path_decision, stats),
            {
                "back_video": self.inputs["back_video"],
                "rpp_standard_path": f"{os.path.join(self.out_folder, 'rpp_trajectory.png')}",
                "rpp_trajectory_path": f"{os.path.join(self.out_folder, 'standard_path_debug_rpp.png')}",
            },
        )


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


def get_direction_stats(
    tx,
    ty,
    tz,
    camera_matrices,
    rev_fwd_halt_segment_min_frame_len,
    min_frame_len,
    out_folder,
    maneuver=None,
    args=None,
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
    directions = majority_vote_smoothen(directions, rev_fwd_halt_segment_min_frame_len)

    stats, directions = aggregate_direction(directions, min_frame_len)
    stats, _ = aggregate_direction(directions, min_frame_len)

    stats, directions = post_process_direction_vector(stats, directions)

    # if args and args.debug:
    debug_directions_visualize(tx, ty, tz, directions, out_folder)

    return directions, stats


def debug_directions_visualize(tx, ty, tz, directions, out_folder):
    """ """
    colors = ["red", "blue", "black"]
    le = min(min(len(tx) - 1, len(tz) - 1), len(directions) - 1)
    plt.scatter(
        [tx[i] for i in range(0, le)],
        [tz[i] for i in range(0, le)],
        c=[colors[directions[i]] for i in range(0, le)],
    )
    plt.show()
    plt.savefig(os.path.join(out_folder, "rpp_trajectory.png"))
    plt.clf()


def get_maneuver_box(box_path):
    val = None
    with open(box_path) as f:
        d = json.load(f)
        val = d["c"]
    val = list(
        zip(*shapely.geometry.Polygon(val).minimum_rotated_rectangle.exterior.coords.xy)
    )
    return np.array(val)


def get_car_box(cam_x, cam_y, camera_matrix, car_dims, markers_vertical):
    car_top_left = [cam_x - car_dims["x_offset"], cam_y - car_dims["z_offset"]]
    car_top_right = [car_top_left[0] + car_dims["width"], car_top_left[1]]
    car_bottom_right = [
        car_top_left[0] + car_dims["width"],
        car_top_left[1] + car_dims["length"],
    ]
    car_bottom_left = [car_top_left[0], car_top_left[1] + car_dims["length"]]

    angle = rotation_matrix_to_euler_angles(camera_matrix)[1]

    if markers_vertical:
        angle *= -1.0

    car_pts = np.array([car_top_left, car_top_right, car_bottom_right, car_bottom_left])
    car_rotated_pts = rotate_rectangle(car_pts, np.array([cam_x, cam_y]), angle)

    return car_rotated_pts


def get_car_box_iou(camera_coords, camera_matrix, car_dims, markers_vertical, box_path):
    box = get_maneuver_box(box_path)
    car_pts = get_car_box(
        camera_coords[0], camera_coords[1], camera_matrix, car_dims, markers_vertical
    )

    box_poly = shapely.geometry.Polygon(box.tolist())
    car_poly = shapely.geometry.Polygon(car_pts.tolist())

    iou = car_poly.intersection(box_poly).area / car_poly.union(box_poly).area
    norm_iou = car_poly.area / box_poly.area

    return iou / norm_iou


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


def is_car_inside(
    tx,
    ty,
    tz,
    camera_matrices,
    car_dims,
    markers_vertical,
    box_overlap_threshold,
    box_path,
):
    assert len(tx) == len(ty) == len(tz)
    max_iou = -1
    max_iou_idx = 0
    counts = 0
    for i in range(len(tx)):
        iou = get_car_box_iou(
            [tx[i], tz[i]],
            camera_matrices[i][:3, :3],
            car_dims,
            markers_vertical,
            box_path,
        )
        if iou > box_overlap_threshold:
            counts += 1
        if iou > max_iou:
            max_iou = iou
            max_iou_idx = i
    max_iou = float(np.round(max_iou, 2))
    if max_iou > box_overlap_threshold:
        return True, max_iou, max_iou_idx
    else:
        return False, max_iou, max_iou_idx


def rotate_plot(ax, config, origin_marker, axis_marker, is_vertical):
    rot_deg = get_plt_rotation_from_markers(origin_marker, axis_marker, is_vertical)
    if rot_deg == 0.0:
        return
    r = Affine2D().rotate_deg(rot_deg)

    for x in ax.images + ax.lines + ax.collections + ax.patches:
        trans = x.get_transform()
        x.set_transform(r + trans)
        if isinstance(x, PathCollection):
            transoff = x.get_offset_transform()
            x._transOffset = r + transoff

    if config["invert_y"]:
        ax.invert_yaxis()
    if config["invert_x"]:
        ax.invert_xaxis()


def plot_limits():
    plot_limits = {"xlim": [-20, 20], "ylim": [-20, 20]}
    plt.xlim(*plot_limits["xlim"])
    plt.ylim(*plot_limits["ylim"])
    plt.gca().set_aspect("equal", "box")


def debug_plot(
    tx,
    tz,
    manu,
    markers,
    box,
    labels,
    out_f,
    config,
    origin_marker,
    axis_marker,
    is_vertical,
):
    plt.scatter(tx[:-1], tz[:-1], marker=".", c=[[1, 0, 0]])

    for key in markers["aruco_bc_markers"]:
        marker_obj = markers["aruco_bc_markers"][key]
        marker_coord = get_marker_coord(marker_obj)
        plt.scatter(
            marker_coord[0], marker_coord[1], c=[[0, 0, 0]], marker=".", s=100.0
        )

    ax = plt.gca()

    for label in labels:
        poly = mpatches.Polygon(box[label], alpha=1, fill=False, edgecolor="blue")
        ax.add_patch(poly)

    plot_limits()

    rotate_plot(ax, config, origin_marker, axis_marker, is_vertical)
    plt.savefig(os.path.join(out_f, "standard_path_debug_{}.png".format(manu)))


def check_standard_path(
    tx, ty, tz, camera_matrices, markers, config, manu, out_f, cwd, debug=False
):
    # tx, ty, tz = smoothen_trajectory(tx, ty, tz, 99, 199, 2)
    assert len(tx) > 0
    plt.cla()
    plt.clf()
    labels = config["standard_node_visit_order"]
    box_path = config["box_file_path"]
    if not os.path.exists(box_path):
        box_path = os.path.join(cwd, box_path)
    boxes = [
        shapely.geometry.Polygon(get_graph_box(box_path, label).tolist())
        for label in labels
    ]
    values = []
    for i in range(len(tx)):
        pos = shapely.geometry.Point([tx[i], tz[i]])
        for j, box in enumerate(boxes):
            if box.contains(pos):
                # print(labels[j])
                values.append(labels[j])
                break
            else:
                print("Strange")

    assert len(tx) > 0
    manu_order = [values[0]]
    for i in range(1, len(values) - 1):
        if values[i] != values[i - 1]:
            manu_order.append(values[i])

    origin_marker = markers["aruco_bc_markers"][config["origin_marker"]]
    axis_marker = markers["aruco_bc_markers"][config["axis_marker"]]

    with open(box_path) as f:
        debug_plot(
            tx,
            tz,
            manu,
            markers,
            json.load(f),
            labels,
            out_f,
            config,
            origin_marker,
            axis_marker,
            config["is_vertical"],
        )

    # is the subsequence 'BC' in the visit order of the sequence or not?
    # C is the box we want to be in, B is the box ahead (from
    # where we want the car to reverse)
    # Acceptable:
    # ABC, ABCB, ACBC, ACBCB etc
    # Non Acceptable:
    # ABAC etc
    manu_order = "".join(manu_order)
    substrings = config["acceptable_order_substrings"]
    for substring in substrings:
        if substring in manu_order:
            return manu_order, True
    return manu_order, False
