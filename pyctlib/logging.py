import logging
from .filemanager import path

class Logger:

    def __init__(self, name: str, stream_log_level=logging.INFO, file_log_level=None, file_path=None):

        self.stream_log_level = stream_log_level
        self.file_log_level = file_log_level
        if file_path is not None:
            self.file_path = path(file_path)
        else:
            self.file_path = None
        self._logger = None

        # Logger = logging.getLogger(name)

        # if steam_log_level is not None:
        #     c_handler = logging.StreamHandler()
        #     c_handler.setLevel(stream_log_level)

        #     c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    @property
    def logger():
        if self._logger is not None:
            return self._logger
        if 
