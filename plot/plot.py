import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
from typing import Union, List, Optional
import numpy as np
from pathlib import Path


def configurate_plot(x_title: str, y_title: str, title: str,
                     *, window_title: str = "Figure", label_fontsize: int = 12,
                     title_fontsize: int = 14) -> None:
    plt.xlabel(x_title, fontsize=label_fontsize)
    plt.ylabel(y_title, fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.style.use("fivethirtyeight")
    plt.grid(True)
    plt.get_current_fig_manager().set_window_title(window_title)


def make_plot(x_values: Union[list, np.array], y_values: Union[list, np.array],
              *, hex_color: str, line_width: int,
              label: Optional[str] = None) -> None:
    plt.plot(x_values,
             y_values,
             label=label, color=hex_color,
             linewidth=line_width)


def _make_annotations_above_bars(plt_bar: BarContainer,
                                 annotations_above_bars: List[str],
                                 *, fontsize: int = 10) -> None:
    for i, rect in enumerate(plt_bar):
        height = rect.get_height()
        plt.annotate(annotations_above_bars[i],
                     (rect.get_x() + rect.get_width() / 2, height + 0.05),
                     ha="center", va="bottom", fontsize=fontsize)


def make_bar_plot(x_values: Union[list, np.array], y_values: Union[list, np.array],
                  annotations_above_bars: Optional[List[str]] = None, *,
                  color: str, annotation_fontsize: int = 10,
                  label: Optional[str] = None) -> None:
    plt_bar = plt.bar(x_values, y_values, zorder=2, color=color, label=label)
    plt.xticks(ticks=x_values)
    plt.ylim(0, 1.25 * max(y_values))

    if not annotations_above_bars or len(x_values) != len(y_values) \
            or len(x_values) != len(annotations_above_bars):
        return None

    _make_annotations_above_bars(plt_bar, annotations_above_bars,
                                 fontsize=annotation_fontsize)


def show_legend() -> None:
    plt.legend()


def show_plot() -> None:
    plt.show()


def save_plot(path: Union[Path, str]) -> None:
    plt.savefig(path)


def clear_after_making_plot() -> None:
    plt.cla()
    plt.style.use('default')


if __name__ == "__main__":
    pass
