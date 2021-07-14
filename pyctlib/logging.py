import logging
from .filemanager import path
from .touch import touch
from .vector import vector
import time
from datetime import timedelta, datetime
import atexit
import sys
from functools import wraps
from typing import Callable, Dict, Union
import random
import string
import argparse
# from wrapt_timeout_decorator import timeout
from .basicwrapper import timeout, TimeoutException
import re
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import notion
from notion.client import NotionClient
"""
from pyctlib import vector, touch
from pyctlib.basicwrapper import timeout, TimeoutException
"""

__all__ = ["DEBUG", "INFO", "WARNING", "CRITICAL", "ERROR", "NOTSET", "Logger", "TimeoutException"]

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
NOTSET = logging.NOTSET

level_to_name = {DEBUG:     "DEBUG", INFO:     "INFO", WARNING:     "WARNING", CRITICAL:     "CRITICAL", ERROR:     "ERROR", NOTSET:     "NOTSET"}

color_dict = {c: "\033[1;%dm" % (30+index) for index, c in enumerate(["BLACK", "RED", "GREEN", "YELLOW", "BLUE", "MAGENTA", "CYAN", "WHITE"])}

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

class ElapsedFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = record.created - self.start_time
        #using timedelta here for convenient default formatting
        elapsed = timedelta(seconds = elapsed_seconds)
        return "{} - {}".format(elapsed, record.getMessage())

class ColoredFormatter:

    def __init__(self, use_color: bool=True, deltatime: bool=False):
        self.use_color = use_color
        self.deltatime = deltatime
        self.start_time = time.time()

    @staticmethod
    def formatter_message(message: str, use_color: bool=True):
        if use_color:
            for color, seq in color_dict.items():
                message = message.replace("(%{})".format(color), seq)
            message = message.replace("(%RESET)", "\033[0m")
        else:
            for color, seq in color_dict.items():
                message = message.replace("(%{})".format(color), "")
            message = message.replace("(%RESET)", "")
        return message

    def format(self, record):
        if self.deltatime:
            elapsed_seconds = record.created - self.start_time
            displaied_time = timedelta(seconds= elapsed_seconds)
        else:
            displaied_time = str(datetime.now())[:-3]
        ret = "(%BLUE){}(%RESET) - {}".format(displaied_time, record.getMessage())
        ret = self.formatter_message(ret, self.use_color)
        return ret

class Logger:

    def __init__(self, stream_log_level=logging.DEBUG, file_log_level=None, deltatime: bool=False, name: str="logger", c_format=None, file_path=None, file_name=None, f_format=None, disable=False, autoplot_variable=False, notion_page_link=None):
        self.name = name
        if stream_log_level is True:
            self.stream_log_level = logging.DEBUG
        else:
            self.stream_log_level = stream_log_level
        if file_log_level is True:
            self.file_log_level = logging.DEBUG
        else:
            self.file_log_level = file_log_level
        self.__disabled = disable
        self._deltatime = deltatime
        self.f_path = file_path
        self.f_name = file_name
        self.c_format = c_format
        self.f_format = f_format
        self.start_time = time.time()
        self._parser = argparse.ArgumentParser(add_help=False)
        self._parser.add_argument("--disable-logging", dest="disabled", action="store_true")
        self.sysargv = self._parser.parse_known_args(sys.argv)[0]
        self.variable_dict = {}
        self.autoplot_variable = autoplot_variable
        self.notion_page_link = notion_page_link
        self.__update_notion_buffer = dict()
        atexit.register(self.record_elapsed)

    @property
    def parser(self):
        return self._parser

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
            self.__logger.propagate = False
            for handler in self.__logger.handlers[:]:
                self.__logger.removeHandler(handler)
            for f in self.__logger.filters[:]:
                self.__logger.removeFilter(f)
            self.__logger.setLevel(logging.DEBUG)
            if self.c_handler is not None:
                self.__logger.addHandler(self.c_handler)
            if self.f_handler is not None:
                self.__logger.addHandler(self.f_handler)
            formatters = [x.formatter for x in self.__logger.handlers]

            for handler in self.__logger.handlers:
                handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            self.__logger.info("start logging")
            if self.f_handler is not None:
                self.__logger.info("logging file: {}".format(self.get_f_fullpath()))
            self.__logger.info("sys.argv: {}".format(sys.argv))
            self.sysargv = self.parser.parse_known_args(sys.argv)[0]
            self.log_parser(self.parser)

            for handler, formatter in zip(self.__logger.handlers, formatters):
                handler.setFormatter(formatter)
            return self.__logger

    def log_parser(self, parser):
        default_vars = vars(parser.parse_args([]))
        virtaul_vars = vars(parser.parse_known_args()[0])
        same_vars = list()
        different_vars = list()
        for item in default_vars:
            if default_vars[item] == virtaul_vars[item]:
                same_vars.append(item)
            else:
                different_vars.append(item)
        def format_result(s):
            if isinstance(s, str):
                return '"{}"'.format(s)
            else:
                return str(s)
        self.logger
        self.__logger.info("Argument Parser:")
        self.__logger.info("-" * 30)
        self.__logger.info("> Specified Vars:")
        for item in different_vars:
            self.__logger.info(" " * 4 + item + "=" + format_result(virtaul_vars[item]))
        self.__logger.info("> Defalut Vars:")
        for item in same_vars:
            self.__logger.info(" " * 4 + item + "=" + format_result(virtaul_vars[item]))
        self.__logger.info("-" * 30)

    @staticmethod
    def get_parser_result(sysargv):
        ret = list()
        for t in dir(sysargv):
            if t.startswith("_"):
                continue
            def format_result(s):
                if isinstance(s, str):
                    return '"{}"'.format(s)
                else:
                    return str(s)
            ret.append(t + "=" + "{}".format(format_result(sysargv.__getattribute__(t))))
        return ret

    @property
    def c_handler(self):
        if touch(lambda: self._c_handler, UnDefined) is not UnDefined:
            return self._c_handler
        if self.stream_log_level is None:
            self._c_handler = None
            return None
        self._c_handler = logging.StreamHandler()
        self._c_handler.setLevel(self.stream_log_level)
        self._c_handler.setFormatter(self.c_format)
        return self._c_handler

    @property
    def f_handler(self):
        if touch(lambda: self._f_handler, UnDefined) is not UnDefined:
            return self._f_handler
        if self.file_log_level is False:
            self._f_handler = None
            return None
        if self.file_log_level is None:
            self._f_handler = None
            return None
        self._f_handler = logging.FileHandler(self.get_f_fullpath(), "w")
        self._f_handler.setLevel(self.file_log_level)
        self._f_handler.setFormatter(self.f_format)
        return self._f_handler

    @property
    def c_format(self):
        if touch(lambda: self._c_format, None) is not None:
            return self._c_format
        self._c_format = ColoredFormatter(True, self._deltatime)
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
        self._f_format = ColoredFormatter(False, self._deltatime)
        return self._f_format

    @f_format.setter
    def f_format(self, value):
        if value is None:
            return None
        if self.file_log_level is False:
            return None
        if self.file_log_level is None:
            self.file_log_level = logging.DEBUG
        self._f_format = value

    @property
    def f_path(self):
        if touch(lambda: self._f_path, None) is not None:
            return self._f_path
        if self.file_log_level is False:
            return None
        self._f_path = path("Log").mkdir()
        return self._f_path

    @f_path.setter
    def f_path(self, value):
        if value is None:
            return None
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
        if touch(lambda: self.__f_real_name, None) is not None:
            return self.__f_real_name
        if self.file_log_level is False:
            return None
        self.__f_real_name = self.get_f_fullpath().fullname
        return self.__f_real_name

    @property
    def _f_name(self):
        if touch(lambda: self.__f_name, None) is not None:
            return self.__f_name
        self.__f_name = time.strftime("%Y-%m%d-%H", time.localtime(time.time())) + ".log"
        return self.__f_name

    @f_name.setter
    def f_name(self, value: str):
        if value is None:
            return
        if self.file_log_level is None:
            self.file_log_level = logging.DEBUG
        if value.endswith(".log"):
            self.__f_name = value
        self.__f_name = value + ".log"
        self.__f_name = self.__f_name.replace("{time}", "%Y-%m%d-%H")
        self.__f_name = datetime.now().strftime(self.__f_name)

    def get_f_fullpath(self) -> Union[path, None]:
        if self.file_log_level is None:
            return None
        if self.file_log_level is False:
            return None
        if hasattr(self, "_Logger__f_fullpath"):
            return self.__f_fullpath
        if not (self.f_path / self._f_name).isfile():
            self.__f_fullpath: path = self.f_path / self._f_name
            return self.__f_fullpath
        index = 1
        while True:
            temp_path = self.f_path / (self._f_name[:-4] + "-{}".format(index) + ".log")
            if not temp_path.isfile():
                self.__f_fullpath: path = temp_path
                return self.__f_fullpath
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

    def debug(self, *msgs, sep=" ", loc_bias=0):
        if self.disabled:
            return
        try:
            raise Exception
        except:
            f = sys.exc_info()[2].tb_frame.f_back
            while loc_bias > 0:
                f = f.f_back
                loc_bias -= 1
        if sep == "\n":
            for msg in msgs:
                self.logger.debug("{}[line:{}] - DEBUG: {}".format(f.f_code.co_filename, f.f_lineno, msg))
        else:
            self.logger.debug("{}[line:{}] - DEBUG: {}".format(f.f_code.co_filename, f.f_lineno, sep.join(str(x) for x in msgs)))

    def info(self, *msgs, sep=" ", loc_bias=0):
        if self.disabled:
            return
        try:
            raise Exception
        except:
            f = sys.exc_info()[2].tb_frame.f_back
            while loc_bias > 0:
                f = f.f_back
                loc_bias -= 1
        if sep == "\n":
            for msg in msgs:
                self.logger.info("{}[line:{}] - (%WHITE)INFO(%RESET): {}".format(f.f_code.co_filename, f.f_lineno, msg))
        else:
            self.logger.info("{}[line:{}] - (%WHITE)INFO(%RESET): {}".format(f.f_code.co_filename, f.f_lineno, sep.join(str(x) for x in msgs)))

    def warning(self, *msgs, sep=" ", loc_bias=0):
        if self.disabled:
            return
        try:
            raise Exception
        except:
            f = sys.exc_info()[2].tb_frame.f_back
            while loc_bias > 0:
                f = f.f_back
                loc_bias -= 1
        if sep == "\n":
            for msg in msgs:
                self.logger.warning("{}[line:{}] - (%YELLOW)WARNING(%RESET): {}".format(f.f_code.co_filename, f.f_lineno, msg))
        else:
            self.logger.warning("{}[line:{}] - (%YELLOW)WARNING(%RESET): {}".format(f.f_code.co_filename, f.f_lineno, sep.join(str(x) for x in msgs)))

    def critical(self, *msgs, sep=" ", loc_bias=0):
        if self.disabled:
            return
        try:
            raise Exception
        except:
            f = sys.exc_info()[2].tb_frame.f_back
            while loc_bias > 0:
                f = f.f_back
                loc_bias -= 1
        if sep == "\n":
            for msg in msgs:
                self.logger.critical("{}[line:{}] - CRITICAL: {}".format(f.f_code.co_filename, f.f_lineno, msg))
        else:
            self.logger.critical("{}[line:{}] - CRITICAL: {}".format(f.f_code.co_filename, f.f_lineno, sep.join(str(x) for x in msgs)))

    def error(self, *msgs, sep=" ", loc_bias=0):
        if self.disabled:
            return
        try:
            raise Exception
        except:
            f = sys.exc_info()[2].tb_frame.f_back
            while loc_bias > 0:
                f = f.f_back
                loc_bias -= 1
        if sep == "\n":
            for msg in msgs:
                self.logger.error("{}[line:{}] - ERROR: {}".format(f.f_code.co_filename, f.f_lineno, msg))
        else:
            self.logger.error("{}[line:{}] - ERROR: {}".format(f.f_code.co_filename, f.f_lineno, sep.join(str(x) for x in msgs)))

    def exception(self, *msgs, sep=" ", loc_bias=0):
        if self.disabled:
            return
        try:
            raise Exception
        except:
            f = sys.exc_info()[2].tb_frame.f_back
            while loc_bias > 0:
                f = f.f_back
                loc_bias -= 1
        if sep == "\n":
            for msg in msgs:
                self.logger.exception("{}[line:{}] - (%RED)EXCEPTION(%RESET): {}".format(f.f_code.co_filename, f.f_lineno, msg))
        else:
            self.logger.exception("{}[line:{}] - (%RED)EXCEPTION(%RESET): {}".format(f.f_code.co_filename, f.f_lineno, sep.join(str(x) for x in msgs)))

    @staticmethod
    def _update_variable_dict(variable_dict, variable_name, variable):
        if variable_name not in variable_dict:
            if not isinstance(variable, list):
                variable_dict[variable_name] = variable
            else:
                variable_dict[variable_name] = vector([vector(variable)])
        else:
            if isinstance(variable_dict[variable_name], vector):
                variable_dict[variable_name].append(variable)
            else:
                variable_dict[variable_name] = vector([variable_dict[variable_name], variable])

    def variable(self, variable_name: str, variable):
        if self.disabled:
            return
        try:
            raise Exception
        except:
            f = sys.exc_info()[2].tb_frame.f_back
        self.logger.info("{}[line:{}] - VARIABLE<(%MAGENTA){}(%RESET)>: (%CYAN){}(%RESET)".format(f.f_code.co_filename, f.f_lineno, variable_name, variable))
        regex = re.compile(r"([^\[\]]+)\[([^\[\]]+)\]")
        m = regex.match(variable_name)
        if not m:
            Logger._update_variable_dict(self.variable_dict, variable_name, variable)
        else:
            group_name = m.group(1)
            variable_name = m.group(2)
            if group_name not in self.variable_dict:
                self.variable_dict[group_name] = dict()
            Logger._update_variable_dict(self.variable_dict[group_name], variable_name, variable)

    def notion_buffer_flush(self):
        if len(self.__update_notion_buffer) == 0:
            return
        if "NOTION_TOKEN_V2" not in os.environ:
            self.warning("there is no $NOTION_TOKEN_V2 in system path. Please check it")
        if self.notion_page_link is None or self.notion_page_link == "":
            self.warning("plz provide notion_page_link for logger object to use notion_update")
            return
        flag = False
        for _ in range(3):
            try:
                flag = self.__get_notion_client_and_page()
            except TimeoutException:
                pass
            if flag:
                break
        if not flag:
            self.warning("failed to get notion client and page, exit")
            return
        for _ in range(10):
            try:
                self.__update_notion()
            except TimeoutException:
                self.warning("update notion database Timeout", self.__update_notion_buffer)
            except Exception as e:
                self.exception(variable_name, variable, " cause exception: ", e)
            else:
                return
        return

    def update_notion(self, variable_name: str, variable):
        if "NOTION_TOKEN_V2" not in os.environ:
            self.warning("there is no $notion_token_v2 in system path. please check it")
            return
        if self.notion_page_link is None or self.notion_page_link == "":
            self.warning("plz provide notion_page_link for logger object to use notion_update")
            return
        self.__update_notion_buffer[variable_name] = variable
        flag = False
        self.info("start update notion", variable, variable_name)
        for _ in range(3):
            try:
                flag = self.__get_notion_client_and_page()
            except TimeoutException:
                self.warning("Timeout, try again")
                pass
            if flag:
                break
        if not flag:
            return
        try:
            self.__update_notion()
        except TimeoutException:
            self.warning("update notion database Timeout", self.__update_notion_buffer)
        except Exception as e:
            self.exception(variable_name, variable, " cause exception: ", e)
        else:
            self.__update_notion_buffer = dict()
        self.info("finish update notion")
        return

    @timeout(10)
    def __get_notion_client_and_page(self):
        if not hasattr(self, "_notion_client") or not hasattr(self, "_notion_page"):
            try:
                self.debug("trying to connect to notion client")
                self._notion_client = NotionClient(token_v2=os.environ["NOTION_TOKEN_V2"])
                self.debug("successfully connect to notion client", self._notion_client)
                self._notion_page = self._notion_client.get_block(self.notion_page_link)
                self.debug("successfully get notion page")
            except TimeoutException:
                self.debug("get notion page Timeout")
                return False
            except Exception as e:
                self.debug("cannot get notion page", type(e), e)
                return False
            else:
                return True
        else:
            return True

    @timeout(5)
    def __update_notion(self):
        for variable_name, variable in self.__update_notion_buffer.items():
            self.info("start update notion page, property name (%MAGENTA){}(%RESET), to {}".format(variable_name, variable))
            old_variable = getattr(self._notion_page, variable_name)
            setattr(self._notion_page, variable_name, variable)
            self.info("update notion page, property name (%MAGENTA){}(%RESET), from {} to {}".format(variable_name, old_variable, variable))

    @staticmethod
    def variable_from_logging_file(f_name):
        variable_dict = dict()
        with open(f_name, "r") as finput:
            for line in finput.readlines():
                if "VARIABLE<" in line:
                    match = re.search(r"VARIABLE<(.+)>: (.+)", line.rstrip())
                    if match:
                        variable_name = match.group(1)
                        variable_str = match.group(2)
                        if not variable_name or not variable_str:
                            continue
                        if variable_str[0].isdigit() or variable_str[0] == "-":
                            variable = float(variable_str)
                        elif variable_str[0] == "[" and variable_str[-1] == "]":
                            variable = vector([float(x.strip()) for x in variable_str[1:-1].split(",")])
                        else:
                            print("unknown variable", variable_str)
                            continue
                        regex = re.compile(r"([^\[\]]+)\[([^\[\]]+)\]")
                        m = regex.match(variable_name)
                        if not m:
                            Logger._update_variable_dict(variable_dict, variable_name, variable)
                        else:
                            group_name = m.group(1)
                            variable_name = m.group(2)
                            if group_name not in variable_dict:
                                variable_dict[group_name] = dict()
                            Logger._update_variable_dict(variable_dict[group_name], variable_name, variable)
        return variable_dict

    @staticmethod
    def plot_variable_dict(variable_dict: Dict[str, vector], saved_path=None, title=None, smooth=0, ignore=None, multi_vector=None, tight_layout=False, hline=None):
        float_variable = vector()
        for key, value in variable_dict.items():
            if ignore is not None and key in ignore:
                continue
            if isinstance(value, vector) and (value.check_type(float) or value.check_type(int)):
                float_variable.append(key)
            elif isinstance(value, dict) and len(value) > 0 and touch(lambda: vector(value.values()).map(len).all_equal(), False):
                float_variable.append(key)
            elif multi_vector is not None and key in multi_vector and isinstance(value, vector) and len(value) > 0 and touch(lambda: value.map(len).all_equal(), False) and all(value.map(lambda x: x.check_type(int) or x.check_type(float))):
                float_variable.append(key)
        float_variable.sort_()
        n = len(float_variable)
        if n == 0:
            return False
        if n <= 2:
            cols = 1
        elif n <= 4:
            cols = 2
        elif n <= 15:
            cols = 3
        else:
            cols = 4
        rows = (n + cols - 1) // cols
        fig = plt.figure(figsize=(8 * cols, (rows) * 4))
        if title is not None:
            fig.suptitle(title)
        for index in range(n):
            ax = plt.subplot(rows, cols, index + 1)
            temp = variable_dict[float_variable[index]]
            if isinstance(temp, vector):
                temp.plot(ax, title=float_variable[index], smooth=smooth, hline=hline)
            elif isinstance(temp, dict):
                x = vector.zip(temp.values()).map(lambda x: vector(x))
                legend = vector(temp.keys())
                x.plot(ax, title=float_variable[index], smooth=smooth, legend=legend, hline=hline)
        if tight_layout:
            plt.tight_layout()
        if saved_path is not None:
            if saved_path.endswith("pdf"):
                with PdfPages(saved_path, "w") as f:
                    plt.savefig(f, format="pdf")
            else:
                plt.savefig(saved_path, dpi=300)
        else:
            plt.show()
        return True

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

        self.notion_buffer_flush()

        if self.autoplot_variable:
            saved_plot_path = self.get_f_fullpath().with_ext("pdf")
            is_plot = Logger.plot_variable_dict(self.variable_dict, saved_plot_path)
            if is_plot:
                self.info("plot figure of variable_dict, with variable:", vector(self.variable_dict.keys()))

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
        self.logger.info("Finish Logging")
        if self.f_handler is not None:
            self.__logger.info("logging file: {}".format(self.get_f_fullpath()))
        self.logger.info("Elapsed time: " + str_time)
        for handler in self.__logger.handlers[:]:
            self.__logger.removeHandler(handler)
        for f in self.__logger.filters[:]:
            self.__logger.removeFilter(f)
