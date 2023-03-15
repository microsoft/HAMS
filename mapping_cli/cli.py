import os

import typer

from mapping_cli import mapper
from mapping_cli.config.config import Config
from mapping_cli.maneuvers.face_verification import FaceVerification
from mapping_cli.maneuvers.seat_belt import SeatBelt

app = typer.Typer("HAMS CLI")


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


# @app.command()
# def find(key: str):
#     typer.echo(conf.get_config_value(key))


@app.command()
def seat_belt(
    front_video: str = None,
    back_video: str = None,
    calib_video: str = None,
    inertial_data: str = None,
    output_path: str = None,
    config=".",
    config_file_name="seatbelt_config.yaml",
):
    assert front_video is not None, typer.echo("Front Video Path is required")
    inputs = {"fpath": front_video}

    sb = SeatBelt(
        inputs, None, Config(os.path.join(config, config_file_name)), output_path
    )
    percentage_detections, wearing_all_the_time, stats = sb.run()

    typer.echo(f"{percentage_detections}, {wearing_all_the_time}, {stats}")


@app.command()
def face_verify(
    front_video: str = None,
    back_video: str = None,
    calib_video: str = None,
    inertial_data: str = None,
    output_path: str = None,
    config=".",
    config_file_name="face_verification.yaml",
):
    assert front_video is not None, typer.echo("Front Video Path is required")
    assert calib_video is not None, typer.echo("Calib Video Path is required")
    inputs = {
        "fpath": front_video,
        "calib_video": calib_video,
    }

    face_verify = FaceVerification(
        inputs=inputs,
        inertial_data=None,
        config=Config(os.path.join(config, config_file_name)),
        out_folder=output_path,
    )
    face_verify.run()


if __name__ == "__main__":
    app()
