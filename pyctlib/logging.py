import logging

class Logger:

    def __init__(name: str, stream_log_level=logging.INFO, file_log_level=None, file_path=None):
        Logger = logging.getLogger(name)

        if steam_log_level is not None:
            c_handler = logging.StreamHandler()
            c_handler.setLevel(stream_log_level)

            c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    def 
