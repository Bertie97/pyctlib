from matplotlib import pyplot as plt
from ..vector import vector
from ..table import table
import torch

class animation_content:

    def __init__(self, ax, max_frame):
        self.ax = ax
        self.max_frame = max_frame
        self.default_colors = [u'b', u'g', u'r', u'c', u'm', u'y', u'k']
        self.title_func = None

    def set_titlefunc(self, title_func):
        """
        title_func can be either string or function
        """
        self.title_func = title_func

    def set_xlim(self, x_low, x_max):
        self.ax.set_xlim(x_low, x_max)

    def set_ylim(self, y_low, y_max):
        self.ax.set_ylim(y_low, y_max)

    def set_xticks(self, content, label):
        self.ax.set_xticks(content)
        self.ax.set_xticklabels(label)

    def set_yticks(self, content, label):
        self.ax.set_yticks(content)
        self.ax.set_yticklabels(label)

    def init(self):
        ret = tuple()
        if self.title_func:
            if isinstance(self.title_func, str):
                self.title = self.ax.set_title(self.title_func)
            else:
                self.title = self.ax.set_title(self.title_func(0))
            ret = ret + tuple([self.title])
        return ret

    def update(self, frame):
        ret = tuple()
        if self.title_func and not isinstance(self.title_func, str):
            self.title.set_text(self.title_func(frame))
            ret = ret + tuple([self.title])
        return ret

class ScatterAnimation(animation_content):

    def __init__(self, ax, max_frame):
        super().__init__(ax, max_frame)
        self.scatter_dots = vector()

    def register(self, content, **kwargs):
        """
        content.shape: [T, N, 2]
        """
        if isinstance(content, vector) and isinstance(content[0], torch.Tensor):
            content = torch.stack(content)
        if isinstance(content, torch.Tensor):
            content = content.detach().cpu()
        dots = table(content=content, color=self.default_colors[len(self.scatter_dots)], label=None, linewidth=None, alpha=None)
        dots.update_exist(kwargs)
        self.scatter_dots.append(dots)

    def init(self):
        self.lines = vector()
        need_legend = self.scatter_dots.map(lambda dots: dots["label"]).any(lambda x: x is not None)
        for dots in self.scatter_dots:
            ln, = self.ax.plot([], [], 'o', color=dots["color"], label=dots["label"], linewidth=dots["linewidth"], alpha=dots["alpha"])
            self.lines.append(ln)
        if need_legend:
            self.ax.legend()
        return tuple(self.lines) + super().init()

    def update(self, frame):
        for line, dots in vector.zip(self.lines, self.scatter_dots):
            line.set_data(dots["content"][frame, :, 0], dots["content"][frame, :, 1])
        return tuple(self.lines) + super().update(frame)

class TimeStamp(animation_content):

    def __init__(self, ax, max_frame, vertical_line=True):
        super().__init__(ax, max_frame)
        self.curves = vector()
        self.vertical_line = vertical_line

    def register(self, content, **kwargs):
        assert isinstance(content, list)
        content = vector(content)
        assert content.length >= self.max_frame
        content = content[:self.max_frame]
        if isinstance(content[0], torch.Tensor):
            content = content.map(lambda x: x.detach().cpu())
        curve = table(content=content, color=self.default_colors[len(self.curves)], linewidth=1, label=None)
        curve.update_exist(kwargs)
        self.curves.append(curve)

    def init(self):
        self.N = len(self.curves)
        lines = vector()
        need_legend = self.curves.map(lambda curve: curve["label"]).any(lambda x: x is not None)
        for index, curve in enumerate(self.curves):
            ln, = self.ax.plot(range(self.max_frame), curve["content"], color=curve["color"], linewidth=curve["linewidth"], label=curve["label"])
            lines.append(ln)
        if need_legend:
            self.ax.legend()
        self.dots = self.ax.scatter(vector.zeros(self.N), self.curves.map(lambda x: x["content"][0]), color=self.curves.map(lambda x: x["color"]))
        ret = tuple(lines.append(self.dots)) + super().init()
        if self.vertical_line:
            self.vl,  = self.ax.plot([0, 0], [0, self.curves.map(lambda x: x["content"][0]).max()], linewidth=0.3, ls="--", color=(0.5, 0.5, 0.5))
            ret = ret + tuple([self.vl])
        return ret

    def update(self, frame):
        ret = tuple()
        self.dots.set_offsets(vector.zip(vector.constant_vector(frame, self.N), self.curves.map(lambda x: x["content"][frame])))
        ret = ret + tuple([self.dots])
        if self.vertical_line:
            self.vl.set_data([frame, frame], [0, self.curves.map(lambda x: x["content"][frame]).max()])
            ret = ret + tuple([self.vl])
        return ret + super().update(frame)
