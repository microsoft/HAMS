import logging
import os

from mapping_cli.config.config import Config
from mapping_cli.utils import Report


class Maneuver:
    def __init__(
        self,
        inputs=None,
        inertial_data=None,
        config: Config = None,
        out_folder: str = None,
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.inertial_data = inertial_data
        self.config = config
        self.out_folder = out_folder

        self.report = Report(os.path.join(self.out_folder, "report.txt"))
        self.log(f"{self.__class__} Started")

    def run(self) -> None:
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def log(self, *args, **kwargs):
        if getattr(self, "logger", None) is not None:
            logging.info(*args, **kwargs)
        else:
            logging.basicConfig(
                filename=os.path.join(self.out_folder, f"hams_alt.log"),
                filemode="w",
                level=logging.INFO,
            )
            logging.info(*args, **kwargs)
            self.logger = {}
