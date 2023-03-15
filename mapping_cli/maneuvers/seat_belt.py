import os
import subprocess
from distutils.sysconfig import get_python_lib
from pathlib import Path
from typing import Dict

import cv2
import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from tqdm import tqdm

from mapping_cli.config.config import Config
from mapping_cli.maneuvers.maneuver import Maneuver

torchvision.ops.misc.ConvTranspose2d = torch.nn.ConvTranspose2d

# if os.path.isfile(get_python_lib() + "/plateDetect"):
#   BASE_DIR = get_python_lib() + "/plateDetect"
# else:
#   BASE_DIR = os.path.dirname(__file__)


class SeatBelt(Maneuver):
    def save(self):
        return super().save()

    def run(self):
        out_folder: str = self.out_folder
        use_gpu = False
        if "gpu" in str(self.config.get_config_value("device")) or "cuda" in str(
            self.config.get_config_value("device")
        ):
            use_gpu = True

        exitcode, new_fpath = frame_dropper(self.inputs["fpath"], out_folder, use_gpu)
        if exitcode != 0:
            raise Exception(f"Check input video file path! Received code {exitcode}")

        device = torch.device(self.config.get_config_value("device"))
        model = get_model_instance_segmentation(2)
        BASE_DIR = Path(__file__).parent.parent
        model.load_state_dict(
            torch.load(
                os.path.join(BASE_DIR, *self.config["model_path"]),
                map_location=torch.device(self.config.get_config_value("device")),
            )
        )
        model.to(device)
        model.eval()

        input_file = new_fpath
        output_file = os.path.join(
            out_folder, os.path.splitext(os.path.split(input_file)[1])[0] + "_out.mp4"
        )

        vid_iter = cv2.VideoCapture(input_file)
        W = int(vid_iter.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(vid_iter.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_length = int(vid_iter.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        skip_frames = self.config.get_config_value("skip_frames")
        out_vid_iter = cv2.VideoWriter(output_file, fourcc, 25 / skip_frames, (W, H),)

        where = []
        num_seatbelt = 0
        image_false = None
        image_true = None

        pbar = tqdm(total=vid_length)
        pbar.set_description("Checking for seatbelt:")
        idx = 0

        while True:
            ret, frame = vid_iter.read()
            pbar.update(1)
            if not ret:
                break

            if idx % skip_frames == 0:
                image = frame[int(2 * (H / 3)) : H, 0 : int(W / 2)]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image, target = get_transform()(image, {})

                outputs = model([image.to(device)])
                boxes = outputs[0]["boxes"].to(torch.device("cpu")).detach().numpy()
                confs = outputs[0]["scores"].to(torch.device("cpu")).detach().numpy()
                good_boxes = boxes[
                    confs
                    > self.config.get_config_value("classifier_confidence_threshold")
                ]

                if len(good_boxes) > 0:
                    num_seatbelt += 1
                    where.append(1)
                    image_true = frame
                else:
                    where.append(0)
                    image_false = frame

                for i, box in enumerate(good_boxes):
                    # I cropped the image, so the box coords are w.r.t. to cropped. Convert to original
                    box[0] += 0
                    box[1] += int(2 * (H / 3))
                    box[2] += 0
                    box[3] += int(2 * (H / 3))
                    cv2.rectangle(
                        frame,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (255, 0, 0),
                        2,
                    )
                out_vid_iter.write(frame)
            idx += 1
        out_vid_iter.release()
        vid_iter.release()
        pbar.close()
        stats = {}
        stats["vid_length"] = vid_length // skip_frames
        stats["num_seatbelt"] = num_seatbelt
        yes = 0
        for x in where:
            if x == 1:
                yes += 1
        if where != []:
            percentage_detections = (yes * 1.0) / len(where)
        else:
            percentage_detections = 0.0
        wearing_all_the_time = True
        if percentage_detections < self.config.get_config_value("detection_percentage"):
            wearing_all_the_time = False
        if wearing_all_the_time and image_true is not None:
            cv2.imwrite(os.path.join(out_folder, "seatbelt_image.jpg"), image_true)
        elif not wearing_all_the_time and image_false is not None:
            cv2.imwrite(os.path.join(out_folder, "seatbelt_image.jpg"), image_false)

        self.report.add_report(
            "Seatbelt", f"{percentage_detections}, {wearing_all_the_time}, {stats}"
        )
        return percentage_detections, wearing_all_the_time, stats


def frame_dropper(fpath: str, out_folder: str, gpu_id: bool = False):
    new_fpath = os.path.join(os.path.split(fpath)[0], "seatbelt_temp.mp4")

    ffmpeg_exec = "ffmpeg.exe" if os.name == "nt" else "ffmeg"

    if gpu_id:
        call_string = '{} -y -i {} -filter:v "setpts=1/25 * PTS" -an -b:v 4000K -vcodec h264_nvenc -gpu {} {}'.format(
            ffmpeg_exec, fpath, int(gpu_id), new_fpath
        )
    else:
        call_string = '{} -y -i {} -filter:v "setpts=1/25 * PTS" -an -b:v 4000K "{}"'.format(
            ffmpeg_exec, fpath, new_fpath
        )
    my_env = os.environ.copy()
    my_env["PATH"] = out_folder + ";" + my_env["PATH"]

    if os.name == "nt":
        exitcode = subprocess.call(call_string, shell=True, cwd=out_folder, env=my_env)
    elif os.name == "posix":
        exitcode = subprocess.call(call_string, shell=True)

    return exitcode, new_fpath


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def get_transform():
    transforms = []
    transforms.append(ToTensor())
    return Compose(transforms)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
