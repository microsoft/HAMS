import logging
import os
import signal
import subprocess
from threading import Timer
from time import sleep

import typer
from click import command

from mapping_cli import utils
from mapping_cli.validation import *

LOG_FILENAME = "log.out"
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)


def run(
    mapper_exe_path: str,
    images_directory: str,
    camera_params_path: str,
    dictionary: str,
    marker_size: str,
    output_path: str,
    cwd: str = None,
):
    """Function to build a Map using the mapper exe and images

    Args:
        mapper_exe_path (str): Mapper exe path.
        images_directory (str): Image Directory Path.
        camera_params_path (str): Camera config/param yml file path.
        dictionary (str): Type of Dictionary.
        marker_size (str): Size of the marker.
        output_path (str): Output file name.
    """
    try:
        check_if_mapper_exe_is_valid(mapper_exe_path)
        check_if_images_dir_exists(images_directory)
        check_if_camera_config_is_valid(camera_params_path)

        typer.echo("Hey There! Starting to run the mapper!")

        num_of_images = len(
            [
                name
                for name in os.listdir(images_directory)
                if os.path.isfile(os.path.join(images_directory, name))
            ]
        )
        typer.echo(f"{num_of_images} num of images")

        def killFunc():
            os.kill(p.pid, signal.CTRL_C_EVENT)

        try:
            call_string = '"{}" "{}" "{}" {} {} {}'.format(
                mapper_exe_path,
                images_directory,
                camera_params_path,
                marker_size,
                dictionary,
                output_path,
            )

            if cwd is None:
                cwd = os.path.curdir

            my_env = os.environ.copy()
            my_env["PATH"] = cwd + ";" + my_env["PATH"]

            p = subprocess.Popen(
                call_string,
                cwd=cwd,
                env=my_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            timer = Timer(num_of_images, p.kill)

            timer.start()
            stdout, stderr = p.communicate()
            retcode = p.returncode
            typer.echo(f"{retcode} {stdout} {stderr} {cwd}")

            if retcode == 0:
                decoded_error = stderr.decode(encoding="utf-8")
                typer.echo(f"Error! {decoded_error}")

                if "Dictionary::loadFromFile" in decoded_error:
                    typer.Exit(code=1)
                    return "Invalid Doctionary"
                elif "CameraParameters::readFromXML" in decoded_error:
                    typer.Exit(code=2)
                    return "Invalid Camera Calibration File"

                return decoded_error
        except Exception as e:
            typer.echo(f"Error: {e}")
            timer.cancel()
        finally:
            typer.echo("Cancelling Timer")
            timer.cancel()
            try:
                call_string = "{} {} {} {} {} {}".format(
                    mapper_exe_path,
                    images_directory,
                    camera_params_path,
                    marker_size,
                    dictionary,
                    output_path,
                )

                if cwd is None:
                    cwd = os.path.curdir

                my_env = os.environ.copy()
                my_env["PATH"] = cwd + ";" + my_env["PATH"]

                p = subprocess.Popen(
                    call_string,
                    cwd=cwd,
                    env=my_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                timer = Timer(num_of_images, p.kill)

                timer.start()
                stdout, stderr = p.communicate()
                retcode = p.returncode
                typer.echo(f"{retcode} {stdout} {stderr} {cwd}")

                if retcode == 0:
                    decoded_error = stderr.decode(encoding="utf-8")
                    typer.echo(f"Error! {decoded_error}")

                    if "Dictionary::loadFromFile" in decoded_error:
                        typer.Exit(code=1)
                        return "Invalid Doctionary"
                    elif "CameraParameters::readFromXML" in decoded_error:
                        typer.Exit(code=2)
                        return "Invalid Camera Calibration File"

                    return decoded_error
            except Exception as e:
                typer.echo(f"Error: {e}")
                timer.cancel()
            finally:
                typer.echo("Cancelling Timer")
                timer.cancel()
                exit(0)

        typer.echo(f"Done! Generated the map and saved as {output_path}")

    except Exception as e:
        typer.echo(f"Error : {e}")


def distance_error(map_file, dist_file):
    try:
        aruco_map = utils.yml_parser(map_file)["aruco_bc_markers"]
        markers = []
        gt_distances = []
        measured_distances = []
        errors = []

        with open(dist_file) as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                content = line.split(" ")
                assert (
                    len(content) == 3
                ), f"Check your file again, {line_idx} has {len(content)}: {line} wrong formatting of input"
                a, corner_a = content[0].split("_")  # read the values
                b, corner_b = content[1].split("_")
                gt = float(content[2])
                markers.append([a + "_" + corner_a, b + "_" + corner_b])
                gt_distances.append(gt)
                if int(a) not in aruco_map.keys():
                    raise KeyError(
                        f"Marker {a} in distance file not found in the map. Expected marker list: {aruco_map.keys()}"
                    )
                if int(b) not in aruco_map.keys():
                    raise KeyError(
                        f"Marker {b} in distance file not found in the map. Expected marker list: {aruco_map.keys()}"
                    )

                measured_distance = utils.euclidean_distance(
                    aruco_map[int(a)][int(corner_a) - 1],
                    aruco_map[int(b)][int(corner_b) - 1],
                )
                measured_distances.append(measured_distance)
                err = abs(gt - measured_distance)
                errors.append(err)
                logging.info(
                    "Ground Truth: {} Measured Dist: {} Err: {}".format(
                        gt, measured_distance, err
                    )
                )
        avg_err = sum(errors) / len(errors)

        typer.echo(f"The Calculated average error is {avg_err}")
        return avg_err, None

    except Exception as e:
        logging.exception(e)
        typer.echo(f"Error : {e}")
        return None, str(e)
