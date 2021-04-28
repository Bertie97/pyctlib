import logging
from .filemanager import path
import time
from datetime import timedelta
import atexit

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
NOTSET = logging.NOTSET

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
        self._disabled = False
        self.start_time = time.time()
        atexit.register(self.record_elapsed)

    def enable(self):
        self._disabled = False

    def disable(self):
        self._disabled = True

    @property
    def logger(self):
        if touch(lambda: self._logger, UnDefined) is not UnDefined:
            return self._logger
        else:
            self._logger = logging.getLogger(self.name)
            self._logger.setLevel(logging.DEBUG)
            if self.c_handler is not None:
                self._logger.addHandler(self.c_handler)
            if self.f_handler is not None:
                self._logger.addHandler(self.f_handler)
            formatters = [x.formatter for x in self._logger.handlers]

            for handler in self._logger.handlers:
                handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            self._logger.info("start logging")
            for handler in self._logger.handlers:
                handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s: %(pathname)s"))
            self._logger.info("pathname")

            for handler, formatter in zip(self._logger.handlers, formatters):
                handler.setFormatter(formatter)
            return self._logger

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
        self._c_format = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
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
        self._f_format = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
        return self._f_format

    @f_format.setter
    def f_format(self, value):
        if value is None:
            return
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
        if value.endswith(".log"):
            self._f_name = value
        self._f_name = value + ".log"

    def get_f_fullpath(self):
        if not (self.f_path / self.f_name).isfile():
            return self.f_path / self.f_name
        index = 1
        while True:
            temp_path = self.f_path / (self.f_name[:-4] + "-{}".format(index) + ".log")
            if not temp_path.isfile():
                return temp_path
            index += 1

    def debug(self, msg, *args, **kwargs):
        if self._disabled:
            return
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        if self._disabled:
            return
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if self._disabled:
            return
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if self._disabled:
            return
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        if self._disabled:
            return
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        if self._disabled:
            return
        self.logger.exception(msg, *args, **kwargs)

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

    def record_elapsed(self):
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
            str_time += "{}minute{}, ".format(minute, "s" if minute > 1 else "")
        str_time += "{}seconds{}".format(seconds, "s" if seconds > 1 else "")
        self.logger.info("Elapsed time: " + str_time)
