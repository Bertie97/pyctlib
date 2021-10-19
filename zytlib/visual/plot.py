#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package visual
##############################

__all__ = """
    plot_ratemaps
    plot_multiple_figure
""".split()

try:
    from matplotlib import pyplot as plt
except ImportError:
    raise ImportError("'pyctlib.watch.debugger' cannot be used without dependency 'matplotlib'. ")

from ..vector import vector
from ..touch import touch
"""
from zytlib import vector
from zytlib import touch
"""
import math
from matplotlib.backends.backend_pdf import PdfPages

def plot_ratemaps(ratemaps, cols=None, titles=None, interpolation="spline36", cmap="jet", saved_path=None, align_range=True):
    ratemaps = vector(ratemaps)
    vmin = ratemaps.map(lambda x: float(x.min())).min()
    vmax = ratemaps.map(lambda x: float(x.max())).max()
    N = len(ratemaps)
    assert N >= 1
    if cols is None:
        cols = math.ceil(math.sqrt(N))
    rows = math.ceil(N / cols)
    fig = plt.figure(figsize=(cols * 2, rows * 2))
    for index in range(N):
        ax = plt.subplot(rows, cols, index + 1)
        if align_range:
            pos = ax.imshow(ratemaps[index], interpolation=interpolation, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            pos = ax.imshow(ratemaps[index], interpolation=interpolation, cmap=cmap)
        ax.set_title(touch(lambda: titles[index], default=str(index + 1)))
        ax.axis('off')
    plt.tight_layout()
    if saved_path is not None:
        if saved_path.endswith("pdf"):
            with PdfPages(saved_path, "w") as f:
                plt.savefig(f, format="pdf")
        else:
            plt.savefig(saved_path, dpi=300)
    else:
        plt.show()

def plot_multiple_figure(figure_names: vector, plot_handler, tight_layout=False, saved_path=None, max_cols=-1):
    # plt.gca()
    N = figure_names.length
    if N == 0:
        return False
    if max_cols == -1:
        cols = math.floor(math.sqrt(N))
    else:
        cols = min(math.floor(math.sqrt(N)), max_cols)
    rows = (N + cols - 1) // cols
    plt.clear()
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    for index in range(N):
        ax = plt.subplot(rows, cols, index + 1)
        plot_handler[figure_names[index]](ax)
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
