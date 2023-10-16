import sys
import os
sys.path.append(os.getcwd())
import plot


# THESE TEST SHOUD BE RUN SEPARATELY


def test_make_plot():
    plot.configurate_plot("X title", "Y title", "Title of the plot",
                          window_title="Test")
    plot.make_plot([1, 2, 3], [1, 34, 57], hex_color=plot.PLOT_CYAN,
                   line_width=plot.LINE_WIDTH_BIG, label="Some label")
    plot.show_legend()
    plot.show_plot()
    plot.clear_after_making_plot()


def test_make_bar_plot():
    plot.configurate_plot("X title", "Y title", "Title of the plot")
    plot.make_bar_plot([1, 2, 3], [1, 34, 57], annotation_fontsize=['a', 'b', 'c'],
                       color=plot.PLOT_GREEN)
    plot.show_plot()
    plot.clear_after_making_plot()
