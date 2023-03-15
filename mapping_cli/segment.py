"""Video segmentation module
"""

import json
import os
import time
from typing import Dict

import cv2
import numpy as np
from decord import VideoReader

from mapping_cli.utils import detect_marker, trim_video


def get_manu_frame_segments(back_video, out_fldr, configs):
    manus_frame_nums = {
        x: {"start": [], "end": []}
        for x in configs["maneuver_order"]  # manu : { start : [], end : [] }
    }
    marker_list_json = {}

    skip_frames = configs["skip_frames"]
    vidcap = cv2.VideoCapture(back_video)
    # success, image = vidcap.read()
    vid_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    maneuvers_in_order = configs["maneuver_order"]
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    reader = VideoReader(back_video)

    json_manu_f = os.path.join(out_fldr, "manu_json_seg_int.json")
    json_marker_f = os.path.join(out_fldr, "manu_json_marker_list.json")
    if os.path.exists(json_manu_f):
        # with open(json_manu_f) as f:
        #     manus_frame_nums = json.load(f)
        # with open(json_marker_f) as f:
        #     marker_list_json = json.load(f)
        pass
    else:
        marker_list_json = {}

    count = 0
    idx = 0
    for i in range(len(reader)):
        if idx % skip_frames == 0:
            image = reader.next().asnumpy()
            marker_list = detect_marker(image, configs["marker_type"])
            marker_list_json[idx] = [int(x) for x in marker_list]
            for manu in maneuvers_in_order:
                if (
                    list(
                        value in marker_list for value in configs[manu]["startMarkers"]
                    ).count(True)
                    >= configs[manu]["startMarkersLen"]
                ):
                    manus_frame_nums[manu]["start"].append(float(idx) / fps)
                    break

                elif (
                    list(
                        value in marker_list for value in configs[manu]["endMarkers"]
                    ).count(True)
                    >= configs[manu]["endMarkersLen"]
                ):
                    manus_frame_nums[manu]["end"].append(float(idx) / fps)
                    break
        # success, image = vidcap.read()
        idx += 1
        if idx > 1e5:
            raise Exception("Too much time")

    with open(json_marker_f, "w") as f:
        json.dump(marker_list_json, f)
    with open(json_manu_f, "w") as f:
        json.dump(manus_frame_nums, f)

    print("Manus: ", manus_frame_nums)
    for k, v in manus_frame_nums.items():
        assert v["start"], f"Maneuver {k} has empty start segmentation number"
        assert v["end"], f"Maneuver {k} has empty start segmentation number"

    # transform_manus_with_constraints(
    #     manus_frame_nums, configs, maneuvers_in_order, len(reader), configs["fps"]
    # )

    return manus_frame_nums, vid_length


def transform_manus_with_constraints(
    manus_frame_nums, manu_info, manus_in_order, vid_length, video_fps
):
    try:
        outlier_fn_dispatcher = {
            "outlier_from_std_dev": outlier_from_std_dev,
            "outlier_from_time_threshold": outlier_from_time_threshold,
            "pad_time_to_small_seq": pad_time_to_small_seq,
            "outlier_from_max_time_difference": outlier_from_max_time_difference,
            "ensure_arr_vals_greater_than_prev_manu": ensure_arr_vals_greater_than_prev_manu,
        }  # Can do eval(fn_string)(args) but unsafe
        for manu in manu_info.keys():
            # print(manu)
            if manu_info[manu]["outlier_fns_list_start"] is not None:
                for fn in manu_info[manu]["outlier_fns_list_start"]:
                    fn_name = fn[0]
                    args = fn[1]
                    if fn_name == "outlier_from_std_dev":
                        manus_frame_nums[manu]["start"] = outlier_fn_dispatcher[
                            fn_name
                        ](manus_frame_nums[manu]["start"], args["s_threshold"])
                    elif fn_name == "outlier_from_time_threshold":
                        manu_to_test, seg, seg_idx, time_slack = (
                            args["manu"],
                            args["seg_type"],
                            args["seg_idx"],
                            args["time_slack"],
                        )
                        manus_frame_nums[manu]["start"] = outlier_fn_dispatcher[
                            fn_name
                        ](
                            manus_frame_nums[manu]["start"],
                            manus_frame_nums[manu_to_test][seg][seg_idx],
                            time_slack,
                            False,
                            video_fps,
                        )
                    elif fn_name == "outlier_from_max_time_difference":
                        manus_frame_nums[manu]["start"] = outlier_fn_dispatcher[
                            fn_name
                        ](
                            manus_frame_nums[manu]["start"],
                            args["time_slack"],
                            video_fps,
                            args["take_first_seg"],
                        )
                    elif fn_name == "ensure_arr_vals_greater_than_prev_manu":
                        prev_manu, prev_seg, prev_seg_idx = (
                            args["prev_manu"],
                            args["prev_seg_type"],
                            args["prev_seg_idx"],
                        )
                        try:
                            prev_thresh = manus_frame_nums[prev_manu][prev_seg][
                                prev_seg_idx
                            ]
                        except:
                            prev_thresh = None
                        manus_frame_nums[manu]["start"] = outlier_fn_dispatcher[
                            fn_name
                        ](manus_frame_nums[manu]["start"], prev_thresh)
                    elif fn_name == "pad_time_to_small_seq":
                        manus_frame_nums[manu]["start"] = outlier_fn_dispatcher[
                            fn_name
                        ](
                            manus_frame_nums[manu]["start"],
                            args["min_size"],
                            args["time_slack"],
                            False,
                            video_fps,
                            None,
                        )
            if manu_info[manu]["outlier_fns_list_end"] is not None:
                for fn in manu_info[manu]["outlier_fns_list_end"]:
                    fn_name = fn[0]
                    args = fn[1]
                    if fn_name == "outlier_from_std_dev":
                        manus_frame_nums[manu]["end"] = outlier_fn_dispatcher[fn_name](
                            manus_frame_nums[manu]["end"], args["s_threshold"]
                        )
                    elif fn_name == "outlier_from_time_threshold":
                        manu_to_test, seg, seg_idx, time_slack = (
                            args["manu"],
                            args["seg"],
                            args["seg_idx"],
                            args["time_slack"],
                        )
                        manus_frame_nums[manu]["end"] = outlier_fn_dispatcher[fn_name](
                            manus_frame_nums[manu]["end"],
                            manus_frame_nums[manu_to_test][seg][seg_idx],
                            time_slack,
                            True,
                            video_fps,
                        )
                    elif fn_name == "outlier_from_max_time_difference":
                        manus_frame_nums[manu]["end"] = outlier_fn_dispatcher[fn_name](
                            manus_frame_nums[manu]["end"],
                            args["time_slack"],
                            video_fps,
                            args["take_first_seg"],
                        )
                    elif fn_name == "ensure_arr_vals_greater_than_prev_manu":
                        prev_manu, prev_seg, prev_seg_idx = (
                            args["prev_manu"],
                            args["prev_seg_type"],
                            args["prev_seg_idx"],
                        )
                        try:
                            prev_thresh = manus_frame_nums[prev_manu][prev_seg][
                                prev_seg_idx
                            ]
                        except:
                            prev_thresh = None
                        manus_frame_nums[manu]["end"] = outlier_fn_dispatcher[fn_name](
                            manus_frame_nums[manu]["end"], prev_thresh
                        )
                    elif fn_name == "pad_time_to_small_seq":
                        manus_frame_nums[manu]["end"] = outlier_fn_dispatcher[fn_name](
                            manus_frame_nums[manu]["end"],
                            args["min_size"],
                            args["time_slack"],
                            True,
                            video_fps,
                        )

        for manu in manus_in_order:
            if manu_info[manu]["constraint_for_start"] is not None:
                constraint = manu_info[manu]["constraint_for_start"]
                constraint_manu = constraint["constraint_manu"]
                constraint_type = constraint["constraint_type"]
                if constraint_type == "hard" or manus_frame_nums[manu]["start"] == []:
                    if constraint_manu == "vid_start":
                        value = 0
                    else:
                        seg_type = constraint["seg_type"]
                        seg_idx = constraint["seg_idx"]
                        value = manus_frame_nums[constraint_manu][seg_type][seg_idx]

                    manus_frame_nums[manu]["start"] = [value]

            if manu_info[manu]["constraint_for_end"] is not None:
                constraint = manu_info[manu]["constraint_for_end"]
                constraint_manu = constraint["constraint_manu"]
                constraint_type = constraint["constraint_type"]
                if constraint_type == "hard" or manus_frame_nums[manu]["end"] == []:
                    if constraint_manu == "vid_end":
                        value = vid_length - 1
                    else:
                        seg_type = constraint["seg_type"]
                        seg_idx = constraint["seg_idx"]
                        value = manus_frame_nums[constraint_manu][seg_type][seg_idx]

                    manus_frame_nums[manu]["end"] = [value]
    except Exception as e:
        print("Segmentation transformation exception: ", e)

    return manus_frame_nums


def outlier_from_std_dev(arr, s_threshold):
    m = np.median(arr)
    s = np.std(arr)
    arr = filter(lambda x: x < m + s_threshold * s, arr)
    arr = filter(lambda x: x > m - s_threshold * s, arr)
    return list(arr)


def outlier_from_time_threshold(arr, val, time_skip, after_manu, video_fps):
    if after_manu == True:
        time_skip = val + time_skip * video_fps
        return list(filter(lambda x: x < time_skip, arr))
    else:
        time_skip = val - time_skip * video_fps
        return list(filter(lambda x: x > time_skip, arr))


def pad_time_to_small_seq(arr_prev, min_size, time_slack, flag, video_fps):
    if arr_prev == []:
        return []
    if len(arr_prev) <= min_size:
        if flag == True:
            for i in range(arr_prev[0] + 1, arr_prev[0] + time_slack * video_fps):
                arr_prev.append(i)
        else:
            new_arr = []
            for i in range(arr_prev[0] - time_slack * video_fps, arr_prev[0] - 1):
                new_arr = new_arr.append(i)
            new_arr.extend(arr_prev)
            return new_arr
    return arr_prev


def ensure_arr_vals_greater_than_prev_manu(arr, thresh):
    if thresh is None:
        return arr
    return list(filter(lambda x: x > thresh, arr))


def outlier_from_max_time_difference(arr, time_slack, video_fps, take_first_seg):
    seg_dist = time_slack * video_fps
    idx = None

    for i in range(0, len(arr) - 1):
        if arr[i + 1] - arr[i] >= seg_dist:
            idx = i
            break

    if idx is None:
        return arr
    if take_first_seg:
        return arr[: idx + 1]
    return arr[idx + 1 :]


def segment(
    front_video_path, back_video_path: str, output_path: str, configs: Dict
) -> None:
    """Segment video into frames"""
    # Create output folder if it doesn't exist
    t = time.time()
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    assert (
        "maneuver_order" in configs.keys()
    ), f"Missing maneuver_order in configs keys: {configs.keys()}"

    maneuver_order = configs["maneuver_order"]

    maneuver_frame_numbers, _ = get_manu_frame_segments(
        back_video_path, output_path, configs
    )
    segment_warnings = []
    segment_paths = {}
    for i in range(len(maneuver_order)):
        if i != len(maneuver_order) - 1:
            if (
                maneuver_frame_numbers[maneuver_order[i]]["end"]
                > maneuver_frame_numbers[maneuver_order[i + 1]]["start"]
            ):
                segment_warnings.append(
                    f"{maneuver_order[i]} crossing {maneuver_order[i+1]} limits"
                )
        print(
            f"Start;;;;; {maneuver_frame_numbers[maneuver_order[i]]['start']} || {maneuver_frame_numbers[maneuver_order[i]]['end']}"
        )
        path = trim_video(
            back_video_path,
            maneuver_frame_numbers[maneuver_order[i]]["start"][0],
            maneuver_frame_numbers[maneuver_order[i]]["end"][-1],
            configs["use_gpu"],
            output_path,
            maneuver_order[i] + ".mp4",
        )
        segment_paths[str(maneuver_order[i])] = path

    print("Time: ", time.time() - t)
    return segment_paths, segment_warnings
