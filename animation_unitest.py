import sys
import os
from sys import getsizeof

sys.path.append(os.path.abspath("."))
import zytlib
import pathlib
import numpy as np
from zytlib import vector, IndexMapping, scope, vhelp
from zytlib.vector import chain_function
from zytlib.filemanager import path, get_relative_path, file
from zytlib import touch
from zytlib.wrapper import generate_typehint_wrapper
import argparse
from time import sleep
from zytlib import totuple
from zytlib.touch import once
import seaborn as sns
import matplotlib.pyplot as plt
from zytlib.visual.animation import TimeStamp, ScatterAnimation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import torch

%matplotlib notebook

fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
# lnn_vector = vector()

ttl = fig.suptitle("PCA @ ")

ax_vector = vector()
sa_vector = vector()

for rank_colored in range(3):

    s_ax = plt.subplot(2, 2, rank_colored+1)
    ax_vector.append(s_ax)
    sa = ScatterAnimation(s_ax, 126)
    for item in range(6):
        sa.register(torch.randn(126, 64, 2))
    sa.set_xlim(-1, 1)
    sa.set_ylim(-1, 1)
    sa_vector.append(sa)

ax_t = plt.subplot(2, 2, 4)

ts = TimeStamp(ax_t, 126)
ts.register(vector.rand(126))
ts.register(vector.rand(126))

def init():
    ret = tuple()
    for sa in sa_vector:
        ret = ret + sa.init()
    ret = ret + ts.init()
    return ret

def update(frame):
    ret = tuple()
    for sa in sa_vector:
        ret = ret + sa.update(frame)
    ret = ret + ts.update(frame)
    ttl.set_text("PCA @ {}".format(frame))
    return ret

ani = FuncAnimation(fig, update, frames=np.arange(126),
                    init_func=init, blit=True)
plt.show()
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=18000)

# ani.save(f'PCA colored by different rank.mp4', writer="ffmpeg", fps=5, dpi=600)

