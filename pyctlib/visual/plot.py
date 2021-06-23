#! python3.8 -u
#  -*- coding: utf-8 -*-

##############################
## Project PyCTLib
## Package visual
##############################

__all__ = """
    plot_ratemaps
""".split()

try:
    from matplotlib import pyplot as plt
except ImportError:
    raise ImportError("'pyctlib.watch.debugger' cannot be used without dependency 'matplotlib'. ")

from ..vector import vector
from ..touch import touch
"""
from pyctlib import vector
from pyctlib import touch
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
