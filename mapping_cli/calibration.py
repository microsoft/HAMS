"""
This code assumes that images used for calibration are of the same arUco marker board provided with code i.e. `DICT_6X6_1000`

Credit: https://github.com/abakisita/camera_calibration
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
from cv2 import aruco
from tqdm import tqdm


def camera_calibration(
    phone_model: str,
    calib_path: str,
    marker_length: str,
    marker_separation: str,
    output_folder: str,
):
    filename = os.path.join(output_folder, "calib_{}.yml".format(phone_model))
    markerLength = float(marker_length)
    markerSeparation = float(marker_separation)
    yaml_string = "%YAML:1.0 \n\
    --- \n\
    image_width: {} \n\
    image_height: {} \n\
    camera_matrix: !!opencv-matrix \n\
    rows: 3 \n\
    cols: 3 \n\
    dt: d \n\
    data: [ {}, {}, {}, {}, \n\
        {}, {}, {}, {}, {} ] \n\
    distortion_coefficients: !!opencv-matrix \n\
    rows: 1 \n\
    cols: 5 \n\
    dt: d \n\
    data: [ {}, {}, \n\
        {}, {}, \n\
        {} ]"

    # root directory of repo for relative path specification.
    root = Path(__file__).parent.absolute()

    # Set this flsg True for calibrating camera and False for validating results real time
    calibrate_camera = True

    # Set path to the images
    calib_imgs_path = Path(calib_path)
    # For validating results, show aruco board to camera.
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)

    # create arUco board
    board = aruco.GridBoard_create(4, 5, markerLength, markerSeparation, aruco_dict)

    """uncomment following block to draw and show the board"""
    # img = board.draw((432,540))
    # cv2.imshow("aruco", img)
    # cv2.waitKey()

    arucoParams = aruco.DetectorParameters_create()

    if calibrate_camera == True:
        img_list = []
        calib_fnms = calib_imgs_path.glob("*.jpg")
        logging.info("Using ...")
        for idx, fn in enumerate(calib_fnms):
            logging.info(f"{idx}, {fn}")
            print("Reading: ", r"{}".format(str(os.path.join(calib_imgs_path.parent, fn))))
            img = cv2.imread(r"{}".format(str(os.path.join(calib_imgs_path.parent, fn))))
            assert img is not None
            img_list.append(img)
            h, w, c = img.shape
            logging.info("Shape of the images is : {} , {}, {} ".format(h, w, c))
        logging.info("Calibration images")

        counter, corners_list, id_list = [], [], []
        first = True
        for im in tqdm(img_list):
            logging.info("In loop")
            img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                img_gray, aruco_dict, parameters=arucoParams
            )
            cv2.aruco.drawDetectedMarkers(im, corners, ids)
            # cv2.imshow("img", cv2.resize(im, None, fx=0.25,fy=0.25))
            # cv2.waitKey()
            if first == True:
                corners_list = corners
                id_list = ids
                first = False
            else:
                corners_list = np.vstack((corners_list, corners))
                id_list = np.vstack((id_list, ids))
            counter.append(len(ids))
        assert len(id_list) > 0
        print("Found {} unique markers".format(np.unique(id_list)))

        counter = np.array(counter)
        logging.info("Calibrating camera .... Please wait...")
        # mat = np.zeros((3,3), float)
        ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
            corners_list, id_list, counter, board, img_gray.shape, None, None
        )

        logging.info(
            "Camera matrix is \n",
            mtx,
            "\n And is stored in calibration.yaml file along with distortion coefficients : \n",
            dist,
        )

        mtx = np.asarray(mtx).tolist()
        dist = np.asarray(dist).tolist()[0]
        logging.info("printing ", dist)
        # logging.info("Printing w, h for confirmation : {}".format(w, h))
        yaml_string = yaml_string.format(
            w,
            h,
            mtx[0][0],
            mtx[0][1],
            mtx[0][2],
            mtx[1][0],
            mtx[1][1],
            mtx[1][2],
            mtx[2][0],
            mtx[2][1],
            mtx[2][2],
            dist[0],
            dist[1],
            dist[2],
            dist[3],
            dist[4],
        )

        with open(filename, "w") as text_file:
            text_file.write(yaml_string)

    else:
        camera = cv2.VideoCapture(0)
        ret, img = camera.read()

        with open("calibration.yaml") as f:
            loadeddict = yaml.load(f)
        mtx = loadeddict.get("camera_matrix")
        dist = loadeddict.get("dist_coeff")
        mtx = np.array(mtx)
        dist = np.array(dist)

        ret, img = camera.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = img_gray.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        pose_r, pose_t = [], []
        while True:
            ret, img = camera.read()
            img_aruco = img
            im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            h, w = im_gray.shape[:2]
            dst = cv2.undistort(im_gray, mtx, dist, None, newcameramtx)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                dst, aruco_dict, parameters=arucoParams
            )
            if corners == None:
                logging.info("pass")
            else:
                ret, rvec, tvec = aruco.estimatePoseBoard(
                    corners, ids, board, newcameramtx, dist
                )  # For a board
                logging.info("Rotation ", rvec, "Translation", tvec)
                if ret != 0:
                    img_aruco = aruco.drawDetectedMarkers(
                        img, corners, ids, (0, 255, 0)
                    )
                    # axis length 100 can be changed according to your requirement
                    img_aruco = aruco.drawAxis(
                        img_aruco, newcameramtx, dist, rvec, tvec, 10
                    )

                if cv2.waitKey(0) & 0xFF == ord("q"):
                    break
            # cv2.imshow("World co-ordinate frame axes", img_aruco)

    # cv2.destroyAllWindows()
