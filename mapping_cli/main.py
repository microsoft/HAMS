import imp
import os

import typer

from mapping_cli import mapper
from mapping_cli.calibration import camera_calibration
from mapping_cli.config.config import Config
from mapping_cli.locator import generate_trajectory_from_photos, get_locations
from mapping_cli.maneuvers.face_verification import FaceVerification
from mapping_cli.maneuvers.forward_eight import ForwardEight
from mapping_cli.maneuvers.gaze import Gaze
from mapping_cli.maneuvers.incline import Incline
from mapping_cli.maneuvers.marker_sequence import MarkerSequence
from mapping_cli.maneuvers.pedestrian import Pedestrian
from mapping_cli.maneuvers.perp import PERP
from mapping_cli.maneuvers.rpp import RPP
from mapping_cli.maneuvers.seat_belt import SeatBelt
from mapping_cli.maneuvers.traffic import Traffic
from mapping_cli.segment import segment as segment_test

app = typer.Typer(name="HAMS")


@app.callback()
def callback():
    """
    HAMS CLI
    """


@app.command()
def map(
    mapper_exe_path: str,
    images_directory: str,
    camera_params_path: str,
    dictionary: str,
    marker_size: str,
    output_path: str,
    cwd: str = None,
):
    """Command to build a Map using the mapper exe and images

    Args:
        mapper_exe_path (str): Mapper exe path.
        images_directory (str): Image Directory Path.
        camera_params_path (str): Camera config/param yml file path.
        dictionary (str): Type of Dictionary.
        marker_size (str): Size of the marker.
        output_path (str): Output file name.
        cwd (str): Working Directry.
    """
    mapper.run(
        mapper_exe_path,
        images_directory,
        camera_params_path,
        dictionary,
        marker_size,
        output_path,
        cwd,
    )


@app.command()
def error(map_file: str, dist_file: str):
    """Command to get the error of map generated

    Args:
        map_file (str): Map YML File Path
        dist_file (str): Dist Text File Path
    """
    mapper.distance_error(map_file, dist_file)


@app.command()
def get_trajectory_from_video(
    input_path: str,
    maneuver: str,
    map_file: str,
    out_folder: str,
    calibration: str,
    size_marker: str,
    aruco_test_exe: str,
    cwd: str = "",
    ignoring_points: str = "",
    box_plot: bool = True,
):
    return get_locations(
        input_path,
        maneuver,
        map_file,
        out_folder,
        calibration,
        size_marker,
        aruco_test_exe,
        ignoring_points,
        cwd=cwd if len(cwd) > 0 else out_folder,
        box_plot=box_plot,
    )


@app.command()
def get_trajectory_from_photos(
    input_path: str,
    maneuver: str,
    map_file: str,
    out_folder: str,
    calibration: str,
    size_marker: str,
    aruco_test_exe: str,
    cwd: str = "",
    ignoring_points: str = "",
    box_plot: bool = True,
):
    return generate_trajectory_from_photos(
        input_path,
        maneuver,
        map_file,
        out_folder,
        calibration,
        size_marker,
        aruco_test_exe,
        ignoring_points,
        cwd=cwd if len(cwd) > 0 else out_folder,
        box_plot=box_plot,
    )


@app.command()
def generate_calib(
    phone_model: str,
    calib_path: str,
    marker_length: str,
    marker_separation: str,
    output_folder: str,
):
    return camera_calibration(
        phone_model, calib_path, marker_length, marker_separation, output_folder
    )


@app.command()
def seat_belt(
    front_video: str = None,
    back_video: str = None,
    calib_video: str = None,
    inertial_data: str = None,
    output_path: str = None,
    config_path="./mapping_cli/config/seatbelt_config.yaml",
):
    assert front_video is not None, typer.echo("Front Video Path is required")
    inputs = {"fpath": front_video}

    sb = SeatBelt(inputs, None, Config(config_path), output_path)
    percentage_detections, wearing_all_the_time, stats = sb.run()

    typer.echo(f"{percentage_detections}, {wearing_all_the_time}, {stats}")

    media = [
        {
            "title": "Seatbelt Image",
            "path": os.path.join(output_path, "seatbelt_image.jpg"),
        },
    ]

    return percentage_detections, wearing_all_the_time, stats, media


@app.command()
def face_verify(
    front_video: str = None,
    back_video: str = None,
    calib_video: str = None,
    inertial_data: str = None,
    output_path: str = None,
    config_path="./mapping_cli/config/face_verification.yaml",
):
    assert front_video is not None, typer.echo("Front Video Path is required")
    assert calib_video is not None, typer.echo("Calib Video Path is required")
    inputs = {
        "fpath": front_video,
        "calib_video": calib_video,
    }

    face_verify = FaceVerification(inputs, None, Config(config_path), output_path)
    result = face_verify.run()

    media = [
        {"title": "Front Video", "path": front_video},
        {"title": "Calib Video", "path": calib_video},
        {
            "title": "Face Registration",
            "path": os.path.join(output_path, "face_registration.png"),
        },
        {
            "title": "Face Validated",
            "path": os.path.join(output_path, "face_validated.png"),
        },
    ]

    return result, media


@app.command()
def gaze(
    front_video: str = None,
    back_video: str = None,
    calib_video: str = None,
    inertial_data: str = None,
    output_path: str = "test_output",
    config_path="./mapping_cli/config/gaze.yaml",
):
    assert front_video is not None, typer.echo("Front Video Path is required")
    assert calib_video is not None, typer.echo("Calib Video Path is required")
    inputs = {
        "fpath": front_video,
        "calib_video": calib_video,
    }

    gaze = Gaze(inputs, None, Config(config_path), output_path)
    decision, stats = gaze.run()

    media = [
        {"title": "Front Video", "path": front_video},
        {"title": "Calib Video", "path": calib_video},
        {"title": "Front Gaze", "path": os.path.join(output_path, "front_gaze.mp4"),},
    ]

    return decision, stats, media


@app.command()
def rpp(
    front_video: str = None,
    back_video: str = None,
    calib_video: str = None,
    inertial_data: str = None,
    output_path: str = "test_output",
    config_path="./mapping_cli/config/rpp.yaml",
    cwd: str = "",
):
    assert back_video is not None, typer.echo("Back Video Path is required")
    try:
        inputs = {
            "back_video": back_video,
        }
        inputs["cwd"] = cwd if len(cwd) > 0 else output_path

        rpp = RPP(inputs, None, Config(config_path), output_path)
        (decision, stats), media = rpp.run()
        return decision, stats, media
    except Exception as e:
        print(e)
        raise e


@app.command()
def perp(
    front_video: str = None,
    back_video: str = None,
    calib_video: str = None,
    inertial_data: str = None,
    output_path: str = "test_output",
    config_path="./mapping_cli/config/perp.yaml",
    cwd: str = "",
):
    assert back_video is not None, typer.echo("Back Video Path is required")
    try:
        inputs = {
            "back_video": back_video,
        }
        inputs["cwd"] = cwd if len(cwd) > 0 else output_path

        perp = PERP(inputs, None, Config(config_path), output_path)
        (decision, stats), media = perp.run()
        return decision, stats, media
    except Exception as e:
        print(e)
        raise e


@app.command()
def incline(
    front_video: str = None,
    back_video: str = None,
    calib_video: str = None,
    inertial_data: str = None,
    output_path: str = "test_output",
    config_path="./mapping_cli/config/incline.yaml",
    cwd: str = "",
):
    assert back_video is not None, typer.echo("Back Video Path is required")
    try:
        inputs = {
            "back_video": back_video,
        }
        inputs["cwd"] = cwd if len(cwd) > 0 else output_path

        incline = Incline(inputs, None, Config(config_path), output_path)
        (stats, decision), (media) = incline.run()
        return decision, stats, media

    except Exception as e:
        print(e)
        raise e


@app.command()
def traffic(
    front_video: str = None,
    back_video: str = None,
    calib_video: str = None,
    inertial_data: str = None,
    output_path: str = "test_output",
    config_path="./mapping_cli/config/traffic.yaml",
    cwd: str = "",
):
    assert back_video is not None, typer.echo("Back Video Path is required")
    try:
        inputs = {
            "back_video": back_video,
        }
        inputs["cwd"] = cwd if len(cwd) > 0 else output_path

        traffic = Traffic(inputs, None, Config(config_path), output_path)
        decision, stats, media = traffic.run()
        return decision, stats, media
    except Exception as e:
        print(e)
        raise e


@app.command()
def pedestrian(
    front_video: str = None,
    back_video: str = None,
    calib_video: str = None,
    inertial_data: str = None,
    output_path: str = "test_output",
    config_path="./mapping_cli/config/pedestrian.yaml",
    cwd: str = "",
):
    assert back_video is not None, typer.echo("Back Video Path is required")
    try:
        inputs = {
            "back_video": back_video,
        }
        inputs["cwd"] = cwd if len(cwd) > 0 else output_path

        pedestrian = Pedestrian(inputs, None, Config(config_path), output_path)
        (decision, stats), (media) = pedestrian.run()
        return decision, stats, media
    except Exception as e:
        print(e)
        raise e


@app.command()
def marker_sequence(
    front_video: str = None,
    back_video: str = None,
    calib_video: str = None,
    inertial_data: str = None,
    output_path: str = "test_output",
    config_path="./mapping_cli/config/marker_sequence.yaml",
    cwd: str = "",
):
    assert back_video is not None, typer.echo("Back Video Path is required")
    try:
        inputs = {
            "back_video": back_video,
        }
        inputs["cwd"] = cwd if len(cwd) > 0 else output_path

        marker_sequence = MarkerSequence(inputs, None, Config(config_path), output_path)
        (decision, stats), (media) = marker_sequence.run()
        return decision, stats, media
    except Exception as e:
        print(e)
        raise e


@app.command()
def forward_eight(
    front_video: str = None,
    back_video: str = None,
    calib_video: str = None,
    inertial_data: str = None,
    output_path: str = "test_output",
    config_path="./mapping_cli/config/forward_eight.yaml",
    cwd: str = "",
):
    assert back_video is not None, typer.echo("Back Video Path is required")
    try:
        inputs = {
            "back_video": back_video,
        }
        inputs["cwd"] = cwd if len(cwd) > 0 else output_path

        forward_eight = ForwardEight(inputs, None, Config(config_path), output_path)
        (decision, stats), (media) = forward_eight.run()
        return decision, stats, media
    except Exception as e:
        print(e)
        raise e


@app.command()
def segment(
    front_video: str = None,
    back_video: str = None,
    calib_video: str = None,
    inertial_data: str = None,
    output_path: str = "test_output",
    config_path="./mapping_cli/config/site.yaml",
):
    assert back_video is not None, typer.echo("Back video is required")
    inputs = {
        "back_video": back_video,
    }
    segment_paths, segment_warnings = segment_test(
        None, back_video, output_path, Config(config_path)
    )
    print(segment_paths, segment_warnings)

    return segment_paths, segment_warnings
