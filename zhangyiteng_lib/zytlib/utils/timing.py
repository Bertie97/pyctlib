#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package <main>
##############################

__all__ = """
    timethis
    timer
    scope
    jump
    JUMP
    Process
    periodic
""".split()

import time
from functools import wraps
from threading import Timer

def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        if hasattr(getattr(func, '__wrapped__', func), '__name__'):
            print("[%s takes %lfs]"%(func.__name__, end-start))
        else:
            print("[%s takes %lfs]"%(func.__class__.__name__, end-start))
        return result
    return wrapper

class timer(object):
    def __init__(self, name='', off=None):
        if off: name = ''
        self.name = name
        self.nround = 0
    def __enter__(self):
        self.start = time.time()
        self.prevtime = self.start
        return self
    def round(self, name = ''):
        self.nround += 1
        self.end = time.time()
        if self.name:
            if not name: name = "%s(round%d)"%(self.name, self.nround)
            print("[%s takes %lfs]"%(name, self.end - self.prevtime))
        self.prevtime = self.end
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type == RuntimeError and str(exc_value) == "JUMP": return True
        if self.name:
            print("[%s%s takes %lfs]"%
                  (self.name, '' if self.nround == 0 else "(all)", time.time() - self.start))
            
class JUMP(object):
    def __init__(self, jump=None): self.jump = True if jump is None else jump
    def __enter__(self):
        def dojump(): raise RuntimeError("JUMP")
        if self.jump: dojump()
        else: return dojump
    def __exit__(self, *args): pass
    def __call__(self, condition): return JUMP(condition)
    
def scope(name, timing=True):
    return timer(name, not timing)
jump = JUMP()
class Process:
    def __init__(self, *args): self.process = args
    def __getattr__(self, key): self.key=key; return timer(key)
    @property
    def j(self): return JUMP(self.key not in self.process)
    @property
    def jump(self): return JUMP(self.key not in self.process)

class TimerCtrl(Timer):

    def __init__(self, seconds, function):
        Timer.__init__(self, seconds, function)
        self.isCanceled = False
        self.seconds = seconds
        self.function = function
        self.funcname = function.__name__
        self.startTime = time.time()

    def cancel(self):
        Timer.cancel(self)
        self.isCanceled = True

    def is_canceled(self): return self.isCanceled

    def setFunctionName(self, funcname): self.funcname = funcname

    def __str__(self):
        return "%5.3fs to run next "%(self.seconds + self.startTime -
                                      time.time()) + self.funcname


def periodic(period, maxiter=float('Inf')):
    def wrap(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global count, timer_ctrl
            try:
                if timer_ctrl.is_canceled(): return
            except NameError: pass
            timer_ctrl = TimerCtrl(period, lambda : wrapper(*args, **kwargs))
            timer_ctrl.setFunctionName(func.__name__)
            timer_ctrl.start()
            ret = func(timer_ctrl, *args, **kwargs)
            try: count += 1
            except Exception: count = 1
            if count >= maxiter: return
            return ret
        return wrapper
    return wrap
