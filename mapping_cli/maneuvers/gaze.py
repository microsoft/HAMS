import csv
import logging
import math
import os
import time

import cv2
import ffmpeg
import numpy as np
import pandas as pd

from mapping_cli.maneuvers.maneuver import Maneuver
from mapping_cli.utils import euclidean_distance_batch


def get_eucledian_distances(
    left_gaze_angle_centroid,
    right_gaze_angle_centroid,
    centre_gaze_angle_centroid,
    test_gaze_angle=[0, 0],
):
    eu_dist_left = math.sqrt(
        (test_gaze_angle[0] - left_gaze_angle_centroid) ** 2
        + (test_gaze_angle[1] - left_gaze_angle_centroid) ** 2
    )
    eu_dist_right = math.sqrt(
        (test_gaze_angle[0] - right_gaze_angle_centroid) ** 2
        + (test_gaze_angle[1] - right_gaze_angle_centroid) ** 2
    )
    eu_dist_centre = math.sqrt(
        (test_gaze_angle[0] - centre_gaze_angle_centroid) ** 2
        + (test_gaze_angle[1] - centre_gaze_angle_centroid) ** 2
    )
    eu_dist_arr = [eu_dist_left, eu_dist_right, eu_dist_centre]
    return eu_dist_arr


def start_calib(
    calib_vid_path: str,
    output_path: str,
    name: str,
    face_landmark_exe_path: str,
    left_gaze_angle_centroid,
    right_gaze_angle_centroid,
    centre_gaze_angle_centroid,
):
    start_time = time.time()
    run_openface(
        calib_vid_path, output_path, f"{name}_calib.csv", face_landmark_exe_path
    )
    logging.info("Finished recording calibration video...")
    (
        left_gaze_angle_centroid,
        centre_gaze_angle_centroid,
        right_gaze_angle_centroid,
    ) = kmeans_clustering(
        left_gaze_angle_centroid,
        right_gaze_angle_centroid,
        centre_gaze_angle_centroid,
        os.path.join(output_path, f"{name}_calib.csv"),
        calib_vid_path,
    )
    logging.info("calib_time = " + str(time.time() - start_time))
    return (
        left_gaze_angle_centroid,
        centre_gaze_angle_centroid,
        right_gaze_angle_centroid,
    )


def run_openface(
    input_vid_path, saveto_folder, output_filename, face_landmark_exe_path: str
):
    logging.info("OpenFace: Processing video...")
    call_string = "{} -f {} -out_dir {} -of {}".format(
        face_landmark_exe_path, input_vid_path, saveto_folder, output_filename
    )
    logging.info(call_string)
    # print
    # subprocess.Popen(call_string, cwd=working_dir, shell=True)
    os.system(call_string)
    # logging.info(x)


def kmeans_clustering(
    left_gaze_angle_centroid,
    right_gaze_angle_centroid,
    centre_gaze_angle_centroid,
    csvfile,
    calib_vid,
):
    # get array in form [[gaze_x0, gaze_y0], [gaze_x1, gaze_y1]]
    video = cv2.VideoCapture(calib_vid)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    f = open(csvfile, "r")
    reader = csv.reader(f)
    next(reader)
    gaze_angles = []
    row_count = 0
    for row in reader:
        row_count += 1
        if float(row[299]) > float(video_frame_width / 2):
            # print("not_appending:condition1: ", row[0], row[1], row[299])
            continue
        if float(row[11]) == 0.0 and float(row[12]) == 0:
            continue
        if row_count > 1 and not (len(gaze_angles) == 0):
            if row[0] == prev_row[0]:
                if float(row[299]) > float(prev_row[299]):
                    # print("not_appending:condition2: ", row[0], row[1], row[299])
                    prev_row = row
                    continue
        # print("appending", row[11], row[12])
        gaze_angles.append([float(row[11]), float(row[12])])
        prev_row = row
    gaze_angles = np.array(gaze_angles)
    # logging.info(gaze_angles.shape, total_frames)
    if gaze_angles.shape[0] > (int(total_frames * 3 / 4)):
        # if gaze_angles.shape[0] > 0:
        dim = gaze_angles.shape[-1]  # find the dimensionality of given points
        k = 3
        indices = np.random.choice(gaze_angles.shape[0], k, replace=False)
        centroids_curr = np.array(
            [gaze_angles[i] for i in indices]
        )  # randomly select any data points from the input file as current centroids
        centroids_old = np.zeros(centroids_curr.shape)
        error = euclidean_distance_batch(centroids_curr, centroids_old)
        cumulative_error = np.sum([error[i][i] for i in range(k)])
        # Iterate until the error between centroids_old and centroids_curr converges
        while not (cumulative_error == 0):
            # assign cluster
            distance_array = euclidean_distance_batch(gaze_angles, centroids_curr)
            cluster_array = np.argmin(distance_array, axis=1)
            # find new centroid
            centroids_old = centroids_curr
            for i in range(k):
                cluster_i = np.array(
                    [
                        gaze_angles[j]
                        for j in range(len(cluster_array))
                        if cluster_array[j] == i
                    ]
                )
                centroid_i = np.mean(cluster_i, axis=0)
                if i == 0:
                    temp_centroids_curr = np.array([centroid_i])
                else:
                    temp_centroids_curr = np.append(
                        temp_centroids_curr, [centroid_i], axis=0
                    )
            centroids_curr = temp_centroids_curr
            # find error
            error = euclidean_distance_batch(centroids_curr, centroids_old)
            cumulative_error = np.sum([error[i][i] for i in range(k)])
        list_centroids_curr = list(centroids_curr)
        sorted_coord_centroids = sorted(
            list_centroids_curr, key=lambda list_centroids_curr: list_centroids_curr[0]
        )
        right_gaze_angle_centroid = (
            sorted_coord_centroids[0][0],
            sorted_coord_centroids[0][1],
        )
        centre_gaze_angle_centroid = (
            sorted_coord_centroids[1][0],
            sorted_coord_centroids[1][1],
        )
        left_gaze_angle_centroid = (
            sorted_coord_centroids[2][0],
            sorted_coord_centroids[2][1],
        )
        print("Changing: ", left_gaze_angle_centroid)
    logging.info("\n\n\n\n")
    logging.info(
        f"{left_gaze_angle_centroid}, {centre_gaze_angle_centroid}, {right_gaze_angle_centroid}"
    )
    logging.info("\n\n\n\n")
    return (
        left_gaze_angle_centroid,
        right_gaze_angle_centroid,
        centre_gaze_angle_centroid,
    )


def classify_gaze(
    vid_path,
    gazeangles_csv_path,
    gazeclassified_csv_path,
    gazeclassified_vid_path,
    left_threshold,
    right_threshold,
    centre_threshold,
    maneuver,
    output_folder,
    rep,
):
    # classify gaze by reading from csv generated by run_openface() and overlay this info on the video generated by run_openface()
    direction_map = {"0": "left", "1": "right", "2": "centre"}
    cap = cv2.VideoCapture(vid_path)
    width = int(cap.get(3))  # float
    height = int(cap.get(4))
    logging.info(gazeclassified_vid_path)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(gazeclassified_vid_path, fourcc, 25, (width, height))
    output_csv = open(gazeclassified_csv_path, mode="w")
    csv_writer = csv.writer(output_csv)
    csv_writer.writerow(["frame_no", "gaze_x", "gaze_y", "classified direction"])
    df = pd.read_csv(gazeangles_csv_path,)
    frame_count = 0
    right_count = 0
    left_count = 0
    centre_count = 0
    ret = True
    while ret:
        ret, frame = cap.read()
        if frame is None:
            break
        found_face = False
        if frame_count <= len(df):
            frame_count += 1
            # print("Frame count: ", frame_count)
            try:
                if frame_count == 5:
                    cv2.imwrite(os.path.join(output_folder, "face.jpg"), frame)
                try:
                    curr_gaze_angle = df.loc[
                        df["frame"] == frame_count,
                        ["gaze_angle_x", "gaze_angle_y", "face_id"],
                    ]
                except:
                    curr_gaze_angle = df.loc[
                        df["frame"] == frame_count,
                        [" gaze_angle_x", " gaze_angle_y", " face_id"],
                    ]
                # print("Gaze Angle: ", curr_gaze_angle)
                curr_gaze_angle = curr_gaze_angle.values.tolist()
                # print("Gaze Angle: ", curr_gaze_angle)
                for i in curr_gaze_angle:
                    found_face = True
                    if not (float(i[0]) == 0.0) and not (float(i[1]) == 0.0):
                        print("gaze: ", i)
                        found_face = True
                        break
                curr_gaze_angle = i
            except Exception as e:
                print("Exception: ", e)
                # exit(0)
                cv2.putText(
                    frame,
                    "No Face Found",
                    (int(frame.shape[1] / 2), frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                out.write(frame)
                continue
            if found_face == False:
                cv2.putText(
                    frame,
                    "No Face Found",
                    (int(frame.shape[1] / 2), frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                out.write(frame)
                continue
            else:
                curr_gaze_angle[1] = 0.0  # added
                eu_dist_arr = get_eucledian_distances(
                    curr_gaze_angle[0], curr_gaze_angle[1], curr_gaze_angle[2]
                )
                looking_dir = direction_map[str(eu_dist_arr.index(min(eu_dist_arr)))]
                print(
                    f"curr_gaze_angle: {curr_gaze_angle} looking direction: {looking_dir}"
                )
                cv2.rectangle(
                    frame,
                    (10, frame.shape[0] - 100),
                    (10 + 300, frame.shape[0] - 10),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    "gaze_x: %.3f" % (curr_gaze_angle[0]),
                    (15, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    frame,
                    "gaze_y: %.3f" % (curr_gaze_angle[1]),
                    (15, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    frame,
                    "gaze_dir: " + looking_dir,
                    (int(frame.shape[1] / 2), frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                if looking_dir == "right":
                    right_count += 1
                elif looking_dir == "left":
                    left_count += 1
                elif looking_dir == "centre":
                    centre_count += 1
            if frame_count == 50:
                cv2.imwrite(os.path.join(output_folder, "face.jpg"), frame)
            # disp_frame = cv2.resize(frame, (1080, 720))
            # cv2.imwrite('face.jpg', frame)
            out.write(frame)
            csv_writer.writerow(
                [frame_count, curr_gaze_angle[0], curr_gaze_angle[1], looking_dir]
            )
            # cv2.imshow("disp", disp_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break
    # if rep:
    stats = {"right": right_count, "left": left_count, "centre": centre_count}
    decision = "Fail"
    if (
        right_count > right_threshold
        and left_count > left_threshold
        and centre_count > centre_threshold
    ):
        decision = "Pass"
    rep.add_report("{}_gaze".format(maneuver), {"stats": stats, "decision": decision})
    with open(gazeclassified_csv_path.replace(".csv", ".txt"), mode="w") as f:
        logging.info("\n\n")
        logging.info("right_count: " + str(right_count))
        logging.info("left_count: " + str(left_count))
        logging.info("centre_count: " + str())
        logging.info("right_count: " + str(right_count))
        logging.info("left_count: " + str(left_count))
        logging.info("centre_count: " + str(centre_count))
        logging.info("\n\n")
    cap.release()
    cv2.destroyAllWindows()
    # call_string = "ffmpeg -i {} -c:v copy -c:a copy -y {}".format(gazeclassified_vid_path, os.path.splitext(gazeclassified_vid_path)[0] + ".mp4")
    ffmpeg.input(gazeclassified_vid_path).output(
        os.path.splitext(gazeclassified_vid_path)[0] + ".mp4"
    ).run()
    # os.system(call_string)
    return decision, stats


class Gaze(Maneuver):
    def run(self) -> None:
        left_gaze_angle_centroid = (0.6074, 0.0)
        right_gaze_angle_centroid = (-0.0820, 0.0)
        centre_gaze_angle_centroid = (0.1472, 0.0)
        (
            left_gaze_angle_centroid,
            right_gaze_angle_centroid,
            centre_gaze_angle_centroid,
        ) = start_calib(
            self.inputs["calib_video"],
            self.out_folder,
            "gaze",
            self.config["face_landmark_exe_path"],
            left_gaze_angle_centroid,
            right_gaze_angle_centroid,
            centre_gaze_angle_centroid,
        )
        print(
            f"Left: {left_gaze_angle_centroid} Right: {right_gaze_angle_centroid} Centre: {centre_gaze_angle_centroid}"
        )
        run_openface(
            self.inputs["fpath"],
            self.out_folder,
            "front_gaze",
            self.config["face_landmark_exe_path"],
        )
        decision, stats = classify_gaze(
            self.inputs["fpath"],
            os.path.join(self.out_folder, "front_gaze.csv"),
            os.path.join(self.out_folder, "front_gaze_output.csv"),
            os.path.join(self.out_folder, "front_gaze.avi"),
            self.config["left_threshold"],
            self.config["right_threshold"],
            self.config["centre_threshold"],
            self.config["maneuver"],
            self.out_folder,
            self.report,
        )
        return decision, stats
