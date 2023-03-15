from os.path import exists


def check_if_mapper_exe_is_valid(mapper_exe_path: str):
    """Validate mapper exe file path and check if file exists.

    Args:
        mapper_exe_path (str): Mapper exe path

    Raises:
        ValueError: Mapper exe File Path cannot be empty
        ValueError: Unsuported Mapper exe File Type. Only supports .exe
        FileNotFoundError: Mapper exe File does not exist
    """

    if len(mapper_exe_path) == 0:
        raise ValueError("Mapper exe File Path cannot be empty")

    if not mapper_exe_path.endswith(".exe"):
        raise ValueError("Unsuported Mapper exe File Type. Only supports .exe")

    if not exists(mapper_exe_path):
        raise FileNotFoundError("Mapper exe File does not exist")


def check_if_images_dir_exists(images_directory: str):
    """Validate output file path.

    Args:
        images_directory (str): Images Directory path

    Raises:
        ValueError: Images Dir Path cannot be empty
        FileNotFoundError: Images Dir Path not found or does not exist
    """

    if len(images_directory) == 0:
        raise ValueError("Images Dir Path cannot be empty")

    if not exists(images_directory):
        raise FileNotFoundError("Images Dir Path not found or does not exist")


def check_if_camera_config_is_valid(camera_params_path: str):
    """Validate camera config params path.

    Args:
        camera_params_path (str): Camera params path
    """
    if len(camera_params_path) == 0:
        raise ValueError("Camera Params path cannot be empty")

    if not camera_params_path.endswith(".yml"):
        raise ValueError("Camera params file not .yml type")

    if not exists(camera_params_path):
        raise FileNotFoundError("Camera params path not found or does not exist")


def check_if_map_yml_is_valid(map_yml_path: str):
    """Validate map yml path.

    Args:
        map_yml_path (str): Map YML File Path
    """
    if len(map_yml_path) == 0:
        raise ValueError("Map YML File Path cannot be empty")

    if not map_yml_path.endswith(".yml"):
        raise ValueError("Map YML File not .yml type")

    if not exists(map_yml_path):
        raise FileNotFoundError("Map YML File Path not found or does not exist")


def check_if_txt_file_is_valid(txt_path: str):
    """Validate txt path.

    Args:
        txt_path (str): Txt File Path
    """
    if len(txt_path) == 0:
        raise ValueError("Txt File Path cannot be empty")

    if not txt_path.endswith(".txt"):
        raise ValueError("Txt File not .yml type")

    if not exists(txt_path):
        raise FileNotFoundError("Txt File Path not found or does not exist")
