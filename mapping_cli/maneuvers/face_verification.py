import logging
import os

import cognitive_face as CF
import cv2
from tqdm import tqdm

from mapping_cli.maneuvers.maneuver import Maneuver

site_config = None


class Person:
    def __init__(
        self,
        vid_file,
        base_url,
        subscription_key,
        calib_frame_period,
        test_frame_period,
        recog_confidence_threshold,
        video_acceptance_threshold,
        debug=False,
        rep=None,
    ):
        SUBSCRIPTION_KEY = subscription_key
        BASE_URL = base_url

        CF.BaseUrl.set(BASE_URL)
        CF.Key.set(SUBSCRIPTION_KEY)

        self.vid_file = vid_file

        self.person_group_id = "test-persons"
        self.create_group(self.person_group_id)
        self.person_id = None
        self.debug = debug
        self.calib_frame_period = calib_frame_period
        self.test_frame_period = test_frame_period
        self.recog_confidence_threshold = recog_confidence_threshold
        self.video_acceptance_threshold = video_acceptance_threshold
        if vid_file:
            self.add_video(vid_file)

        self.rep = rep

    def create_group(self, person_group_id):
        exists = False
        for p in CF.person_group.lists():
            if p["personGroupId"] == person_group_id:
                exists = True

        if not exists:
            CF.person_group.create(self.person_group_id)

    def add_face(self, im_file, persist=False):
        # Persist if limit exceeded
        # Ret val: 0 if success, 1 if no face, 2 if server issue
        try:
            if not self.person_id:
                create_response = CF.person.create(self.person_group_id, "x")
                self.person_id = create_response["personId"]
            detect_result = CF.face.detect(im_file)
            if len(detect_result) < 1:
                if self.debug:
                    logging.info("No face detected!")
                return 1, None
            # idx = 0
            idx = -1
            min_left = 10000
            for i, r in enumerate(detect_result):
                if r["faceRectangle"]["left"] < min_left:
                    idx = i
                    min_left = r["faceRectangle"]["left"]
            # if > 1, take left most
            target_str = "{},{},{},{}".format(
                detect_result[idx]["faceRectangle"]["left"],
                detect_result[idx]["faceRectangle"]["top"],
                detect_result[idx]["faceRectangle"]["width"],
                detect_result[idx]["faceRectangle"]["height"],
            )
            CF.person.add_face(
                im_file, self.person_group_id, self.person_id, target_face=target_str
            )
            return 0, detect_result[idx]["faceRectangle"]
        except Exception as e:
            if self.debug:
                logging.info(e)
            if persist:
                self.add_face(im_file, persist)
            else:
                return 2, None
        return 2, None

    def add_video(self, vid_file):
        vc = cv2.VideoCapture(vid_file)
        vid_length = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        self.reg_face = None

        if self.debug:
            pbar = tqdm(total=vid_length)
            pbar.set_description("Adding calib video")

        i = 0
        while True and i < vid_length:
            ret, frame = vc.read()
            if not ret:
                return
            if i % self.calib_frame_period == 0:
                # TODO: Add face detection check before adding a new face
                im_file = "temp_{}.jpg".format(i)

                # crop the driver because there could be people at the back who might peep
                h, w, c = frame.shape
                frame = frame[:, : int(0.7 * w), :]
                cv2.imwrite(im_file, frame)
                ret, result = self.add_face(im_file, True)

                if result is not None:
                    cv2.rectangle(
                        frame,
                        (result["left"], result["top"]),
                        (
                            result["left"] + result["width"],
                            result["top"] + result["height"],
                        ),
                        (0, 255, 0),
                        3,
                    )
                    cv2.imwrite(im_file, frame)

                if ret == 0:
                    self.reg_face = frame
                if os.path.exists(im_file):
                    os.remove(im_file)
            i += 1
            if self.debug:
                pbar.update(1)

        if self.debug:
            pbar.close()

    def verify_face(self, im_file, persist=False):
        """
        Not detected = 2
        Detected & Verified = 1
        Detected but not verified = 0
        """

        person_id = self.person_id

        try:
            detect_result = CF.face.detect(im_file)

            if len(detect_result) < 1:
                return 2, -1, None

            idx = -1
            min_left = 10000
            for i, r in enumerate(detect_result):
                if r["faceRectangle"]["left"] < min_left:
                    idx = i
                    min_left = r["faceRectangle"]["left"]

            face_id = detect_result[idx]["faceId"]
            face_rect = detect_result[idx]["faceRectangle"]

            verify_result = CF.face.verify(
                face_id, person_group_id=self.person_group_id, person_id=person_id
            )

            if (
                verify_result["isIdentical"]
                and float(verify_result["confidence"]) > self.recog_confidence_threshold
            ):
                return 1, float(verify_result["confidence"]), face_rect
            else:
                return 0, float(verify_result["confidence"]), face_rect

        except Exception as e:
            if self.debug:
                logging.info(e)
            if persist:
                return self.verify_face(im_file, persist)
            else:
                return 2, -1, None

    def verify_video(self, vid_file, person_id=None):
        self.ver_face_true = None
        self.ver_face_true_conf = 0

        self.ver_face_false = None
        self.ver_face_false_conf = 1

        self.video_log = {}

        vc = cv2.VideoCapture(vid_file)
        vid_length = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.debug:
            pbar = tqdm(total=vid_length)
            pbar.set_description("Verifying test video")
        verify_count = 0
        total_count = 0
        frame_no = 0

        while True and frame_no < vid_length:
            ret, frame = vc.read()
            if not ret:
                break
            if frame_no % self.test_frame_period == 0:
                im_file = "V_temp_{}.jpg".format(frame_no)

                # crop the driver because there could be people at the back who might peep
                h, w, c = frame.shape
                frame = frame[:, : int(0.7 * w), :]
                cv2.imwrite(im_file, frame)

                verify_result, verify_confidence, face_rect = self.verify_face(
                    im_file, False
                )

                if face_rect is not None:
                    cv2.rectangle(
                        frame,
                        (face_rect["left"], face_rect["top"]),
                        (
                            face_rect["left"] + face_rect["width"],
                            face_rect["top"] + face_rect["height"],
                        ),
                        (0, 255, 0),
                        3,
                    )
                    cv2.imwrite(im_file, frame)

                self.video_log[frame_no] = [verify_result, verify_confidence]

                if verify_result == 1:
                    if (
                        verify_confidence > self.ver_face_true_conf
                    ):  # get the most confident face
                        self.ver_face_true = frame
                        self.ver_face_true_conf = verify_confidence
                    verify_count += 1
                    total_count += 1
                elif verify_result == 0:
                    if (
                        verify_confidence < self.ver_face_false_conf
                    ):  # get the least confident face
                        self.ver_face_false = frame
                        self.ver_face_false_conf = verify_confidence
                    total_count += 1

                if os.path.exists(im_file):
                    os.remove(im_file)
            frame_no += 1
            if self.debug:
                pbar.update(1)

        if self.debug:
            pbar.close()
            logging.info(
                "{} faces verified out of {}".format(verify_count, total_count)
            )

        self.rep.add_report("face_verify_log", self.video_log)

        if verify_count > self.video_acceptance_threshold * total_count:
            self.verified = True
            return True
        else:
            self.verified = False
            return False

    def save_faces(self, out_folder):
        if self.reg_face is not None:
            cv2.imwrite(
                os.path.join(out_folder, "face_registration.png"), self.reg_face
            )
        if self.verified:
            cv2.imwrite(
                os.path.join(out_folder, "face_validated.png"), self.ver_face_true
            )
        elif not self.verified and self.ver_face_false is not None:
            cv2.imwrite(
                os.path.join(out_folder, "face_validated.png"), self.ver_face_false
            )


def verify_two_videos(
    vid_file_1,
    vid_file_2,
    out_folder,
    base_url,
    subscription_key,
    calib_frame_period,
    test_frame_period,
    recog_confidence_threshold,
    video_acceptance_threshold,
    debug=True,
    rep=None,
):
    person = Person(
        vid_file_1,
        base_url,
        subscription_key,
        calib_frame_period,
        test_frame_period,
        recog_confidence_threshold,
        video_acceptance_threshold,
        debug=debug,
        rep=rep,
    )
    verified = person.verify_video(vid_file_2)
    person.save_faces(out_folder)
    return verified


def main(
    calib_file,
    test_file,
    out_folder,
    base_url,
    subscription_key,
    calib_frame_period,
    test_frame_period,
    recog_confidence_threshold,
    video_acceptance_threshold,
    rep,
):
    if verify_two_videos(
        calib_file,
        test_file,
        out_folder,
        base_url,
        subscription_key,
        calib_frame_period,
        test_frame_period,
        recog_confidence_threshold,
        video_acceptance_threshold,
        debug=True,
        rep=rep,
    ):
        out_str = "Face Verification: Pass"
        if rep:
            rep.add_report("face_verify", "Pass")
        logging.info(out_str)
        return True
    else:
        out_str = "Face Verification: Fail"
        if rep:
            rep.add_report("face_verify", "Fail")
        logging.info(out_str)
        return False


class FaceVerification(Maneuver):
    def run(self):
        return main(
            self.inputs["calib_video"],
            self.inputs["fpath"],
            self.out_folder,
            self.config["base_url"],
            self.config["subscription_key"],
            self.config["calib_frame_period"],
            self.config["test_frame_period"],
            self.config["recog_confidence_threshold"],
            self.config["video_acceptance_threshold"],
            rep=self.report,
        )
