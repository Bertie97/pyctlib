import logging
from .filemanager import path
from .touch import touch
import time
from datetime import timedelta
from datetime import datetime
import atexit
import sys
from functools import wraps
from typing import Callable
import random
import string
import argparse

__all__ = ["DEBUG", "INFO", "WARNING", "CRITICAL", "ERROR", "NOTSET", "Logger"]

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
NOTSET = logging.NOTSET

level_to_name = {DEBUG:     "DEBUG", INFO:     "INFO", WARNING:     "WARNING", CRITICAL:     "CRITICAL", ERROR:     "ERROR", NOTSET:     "NOTSET"}

class EmptyClass:

    def __init__(self, name="EmptyClass"):
        self.name = name
        pass

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

NoDefault = EmptyClass("No Default Value")
OutBoundary = EmptyClass("Out of Boundary")
UnDefined = EmptyClass("Not Defined")

def empty_func(*args, **kwargs):
    return

class Logger:

    def __init__(self, stream_log_level=logging.DEBUG, file_log_level=None, name: str="logger", c_format=None, file_path=None, file_name=None, f_format=None):
        self.name = name
        if stream_log_level is True:
            self.stream_log_level = logging.DEBUG
        else:
            self.stream_log_level = stream_log_level
        if file_log_level is True:
            self.file_log_level = logging.DEBUG
        else:
            self.file_log_level = file_log_level
        self.f_path = file_path
        self.f_name = file_name
        self.c_format = c_format
        self.f_format = f_format
        self.__disabled = False
        self.start_time = time.time()
        _parser = argparse.ArgumentParser(add_help=False)
        _parser.add_argument("--disable-logging", dest="disabled", action="store_true")
        self.sysargv = _parser.parse_known_args(sys.argv)[0]
        atexit.register(self.record_elapsed)

    def enable(self):
        self.__disabled = False

    def disable(self):
        self.__disabled = True

    @property
    def disabled(self):
        if self.sysargv.disabled:
            return True
        return self.__disabled

    @property
    def logger(self):
        if hasattr(self, "_Logger__logger"):
            return self.__logger
        else:
            self.__logger = logging.getLogger(self.name)
            self.__logger.setLevel(logging.DEBUG)
            if self.c_handler is not None:
                self.__logger.addHandler(self.c_handler)
            if self.f_handler is not None:
                self.__logger.addHandler(self.f_handler)
            formatters = [x.formatter for x in self.__logger.handlers]

            for handler in self.__logger.handlers:
                handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            self.__logger.info("start logging")
            self.__logger.info("sys.argv: {}".format(sys.argv))

            for handler, formatter in zip(self.__logger.handlers, formatters):
                handler.setFormatter(formatter)
            return self.__logger

    @property
    def c_handler(self):
        if touch(lambda: self._c_handler, UnDefined) is not UnDefined:
            return self._c_handler
        if self.stream_log_level is None:
            self._c_handler = None
            return None
        self._c_handler = logging.StreamHandler()
        self._c_handler.setLevel(self.stream_log_level)
        self._c_handler.setFormatter(logging.Formatter(self.c_format))
        return self._c_handler

    @property
    def f_handler(self):
        if touch(lambda: self._f_handler, UnDefined) is not UnDefined:
            return self._f_handler
        if self.file_log_level is None:
            self._f_handler = None
            return None
        self._f_handler = logging.FileHandler(self.get_f_fullpath(), "w")
        self._f_handler.setLevel(self.file_log_level)
        self._f_handler.setFormatter(logging.Formatter(self.f_format))
        return self._f_handler

    @property
    def c_format(self):
        if touch(lambda: self._c_format, None) is not None:
            return self._c_format
        # self._c_format = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
        self._c_format = "%(asctime)s - %(message)s"
        return self._c_format

    @c_format.setter
    def c_format(self, value):
        if value is None:
            return
        self._c_format = value

    @property
    def f_format(self):
        if touch(lambda: self._f_format, None) is not None:
            return self._f_format
        self._f_format = "%(asctime)s - %(message)s"
        return self._f_format

    @f_format.setter
    def f_format(self, value):
        if value is None:
            return
        if self.file_log_level is None:
            self.file_log_level = logging.DEBUG
        self._f_format = value

    @property
    def f_path(self):
        if touch(lambda: self._f_path, None) is not None:
            return self._f_path
        self._f_path = path("Log").mkdir()
        return self._f_path

    @f_path.setter
    def f_path(self, value):
        if value is None:
            return
        if self.file_log_level is None:
            self.file_log_level = logging.DEBUG
        if value.endswith(".log"):
            self.f_name = path(value).fullname
            self.f_path = path(value).parent
            return
        self._f_path = path(value).mkdir()
        if not self._f_path.isdir():
            raise RuntimeError("cannot make directory: {}".format(value))

    @property
    def f_name(self):
        if touch(lambda: self._f_name, None) is not None:
            return self._f_name
        self._f_name = time.strftime("%Y-%m%d-%H", time.localtime(time.time())) + ".log"
        return self._f_name

    @f_name.setter
    def f_name(self, value: str):
        if value is None:
            return
        if self.file_log_level is None:
            self.file_log_level = logging.DEBUG
        if value.endswith(".log"):
            self._f_name = value
        self._f_name = value + ".log"
        self._f_name = self._f_name.replace("{time}", "%Y-%m%d-%H")
        self._f_name = datetime.now().strftime(self._f_name)

    def get_f_fullpath(self):
        if not (self.f_path / self.f_name).isfile():
            return self.f_path / self.f_name
        index = 1
        while True:
            temp_path = self.f_path / (self.f_name[:-4] + "-{}".format(index) + ".log")
            if not temp_path.isfile():
                return temp_path
            index += 1

    def from_level(self, logging_level):
        if logging_level == DEBUG:
            return self.logger.debug
        if logging_level == INFO:
            return self.logger.info
        if logging_level == WARNING:
            return self.logger.warning
        if logging_level == ERROR:
            return self.logger.error
        if logging_level == CRITICAL:
            return self.logger.critical

    def debug(self, *msgs, sep=" "):
        if self.disabled:
            return
        try:
            raise Exception
        except:
            f = sys.exc_info()[2].tb_frame.f_back
        if sep == "\n":
            for msg in msgs:
                self.logger.debug("{}[line:{}] - DEBUG: {}".format(f.f_code.co_filename, f.f_lineno, msg))
        else:
            self.logger.debug("{}[line:{}] - DEBUG: {}".format(f.f_code.co_filename, f.f_lineno, sep.join(str(x) for x in msgs)))

    def info(self, *msgs, sep=" "):
        if self.disabled:
            return
        try:
            raise Exception
        except:
            f = sys.exc_info()[2].tb_frame.f_back
        if sep == "\n":
            for msg in msgs:
                self.logger.info("{}[line:{}] - INFO: {}".format(f.f_code.co_filename, f.f_lineno, msg))
        else:
            self.logger.info("{}[line:{}] - INFO: {}".format(f.f_code.co_filename, f.f_lineno, sep.join(str(x) for x in msgs)))

    def warning(self, *msgs, sep=" "):
        if self.disabled:
            return
        try:
            raise Exception
        except:
            f = sys.exc_info()[2].tb_frame.f_back
        if sep == "\n":
            for msg in msgs:
                self.logger.warning("{}[line:{}] - WARNING: {}".format(f.f_code.co_filename, f.f_lineno, msg))
        else:
            self.logger.warning("{}[line:{}] - WARNING: {}".format(f.f_code.co_filename, f.f_lineno, sep.join(str(x) for x in msgs)))

    def critical(self, *msgs, sep=" "):
        if self.disabled:
            return
        try:
            raise Exception
        except:
            f = sys.exc_info()[2].tb_frame.f_back
        if sep == "\n":
            for msg in msgs:
                self.logger.critical("{}[line:{}] - CRITICAL: {}".format(f.f_code.co_filename, f.f_lineno, msg))
        else:
            self.logger.critical("{}[line:{}] - CRITICAL: {}".format(f.f_code.co_filename, f.f_lineno, sep.join(str(x) for x in msgs)))

    def error(self, *msgs, sep=" "):
        if self.disabled:
            return
        try:
            raise Exception
        except:
            f = sys.exc_info()[2].tb_frame.f_back
        if sep == "\n":
            for msg in msgs:
                self.logger.error("{}[line:{}] - ERROR: {}".format(f.f_code.co_filename, f.f_lineno, msg))
        else:
            self.logger.error("{}[line:{}] - ERROR: {}".format(f.f_code.co_filename, f.f_lineno, sep.join(str(x) for x in msgs)))

    def exception(self, *msgs, sep=" "):
        if self.disabled:
            return
        try:
            raise Exception
        except:
            f = sys.exc_info()[2].tb_frame.f_back
        if sep == "\n":
            for msg in msgs:
                self.logger.exception("{}[line:{}] - EXCEPTION: {}".format(f.f_code.co_filename, f.f_lineno, msg))
        else:
            self.logger.exception("{}[line:{}] - EXCEPTION: {}".format(f.f_code.co_filename, f.f_lineno, sep.join(str(x) for x in msgs)))

    def wrapper_function_input_output(self, *args, logging_level=INFO):
        if len(args) == 1:
            func = args[0]
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.disabled:
                    return func(*args, **kwargs)
                try:
                    raise Exception
                except:
                    f = sys.exc_info()[2].tb_frame.f_back
                logging_func = self.from_level(logging_level)
                random_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
                logging_func("{}[line:{}] - {}: function [{}] start execution function {}".format(f.f_code.co_filename, f.f_lineno, level_to_name[logging_level], random_id, func.__name__))
                logging_func("{}[line:{}] - {}: function [{}] args: {}".format(f.f_code.co_filename, f.f_lineno, level_to_name[logging_level], random_id, args))
                if len(kwargs):
                    logging_func("{}[line:{}] - {}: function [{}] kargs: {}".format(f.f_code.co_filename, f.f_lineno, level_to_name[logging_level], random_id, kwargs))
                ret = func(*args, **kwargs)
                logging_func("{}[line:{}] - {}: function [{}] return of {}: {}".format(f.f_code.co_filename, f.f_lineno, level_to_name[logging_level], random_id, func.__name__, ret))
            return wrapper
        elif len(args) == 0:
            def temp_wrapper_function_input_output(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    if self.disabled:
                        return func(*args, **kwargs)
                    try:
                        raise Exception
                    except:
                        f = sys.exc_info()[2].tb_frame.f_back
                    logging_func = self.from_level(logging_level)
                    random_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
                    logging_func("{}[line:{}] - {}: function [{}] start execution function {}".format(f.f_code.co_filename, f.f_lineno, level_to_name[logging_level], random_id, func.__name__))
                    logging_func("{}[line:{}] - {}: function [{}] args: {}".format(f.f_code.co_filename, f.f_lineno, level_to_name[logging_level], random_id, args))
                    if len(kwargs):
                        logging_func("{}[line:{}] - {}: function [{}] kargs: {}".format(f.f_code.co_filename, f.f_lineno, level_to_name[logging_level], random_id, kwargs))
                    ret = func(*args, **kwargs)
                    logging_func("{}[line:{}] - {}: function [{}] return of {}: {}".format(f.f_code.co_filename, f.f_lineno, level_to_name[logging_level], random_id, func.__name__, ret))
                return wrapper
            return temp_wrapper_function_input_output
        else:
            raise TypeError

    def elapsed_time(self):
        end = time.time()
        elapsed = end - self.start_time
        hours, rem = divmod(elapsed, 3600)
        days, hours = divmod(hours, 24)
        minutes, seconds = divmod(rem, 60)
        seconds = int(seconds * 10000) / 10000
        return (days, hours, minutes, seconds)

    def pop_all_formatter(self):
        ret = list()
        for handler in self.logging.handlers:
            ret.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(messages)s"))
        return ret

    def restore_all_formatter(self, formatters):
        for handler, formatter in zip(self.logger.handlers, formatters):
            handler.setFormatter(formatter)

    def set_all_formatter(self, formatter):
        if isinstance(formatter, str):
            formatter = logging.Formatter(formatter)
        for handler in self.logger.handlers:
            handler.setFormatter(formatter)

    @property
    def already_logging(self):
        return hasattr(self, "_Logger__logger")

    def record_elapsed(self):
        if self.disabled:
            return
        if not self.already_logging:
            return

        for handler in self.logger.handlers:
            handler.setFormatter(logging.Formatter("%(message)s"))

        self.logger.info("-" * 30)
        days, hours, minutes, seconds = self.elapsed_time()
        str_time = ""
        if days != 0:
            str_time = "{}day{}, ".format(days, "s" if days > 1 else "")
        if days != 0 or hours != 0:
            str_time += "{}hour{}, ".format(hours, "s" if hours > 1 else "")
        if days != 0 or hours != 0 or minutes != 0:
            str_time += "{}minute{}, ".format(minutes, "s" if minutes > 1 else "")
        str_time += "{}seconds{}".format(seconds, "s" if seconds > 1 else "")
        self.logger.info("Elapsed time: " + str_time)
