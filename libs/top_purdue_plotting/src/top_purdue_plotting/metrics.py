import numpy as np
from matplotlib import pyplot as plt
import mplhep as hep
from typing import Dict, Optional

# Extra matplotlib imports for RMSE/bias plotting function
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from . import colorschemes
from . import labels

def plot_binned_metric(
    metric_values: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    x_label: Optional[str] = "UNKNOWN",
    y_label: Optional[str] = "UNKNOWN",
    x_scale: Optional[str] = "linear",
    y_scale: Optional[str] = "linear",
    ratio_key: Optional[str] = None,
    colors: Optional[dict] = colorschemes.reconstruction_method_colors,
    save_filename: Optional[str] = None,
    cms_text: Optional[str] = "Work in Progress",
    cms_year: Optional[str] = "2022",
    hline: Optional[float] = None,
    vline: Optional[float] = None,
    img_type: Optional[str] = "png"
):
    """
    A generic function to plot any quantity computed in bins of some other quantity.
    """
    use_ratio = ratio_key is not None
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_half_widths = np.diff(bin_edges) / 2

    # Get figure and axes objects
    if use_ratio:
        fig, (ax_main, ax_ratio) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1}, sharex=True, dpi=100)
        ax_ratio.axhline(y=1.0, color="black", linestyle="--", lw=2, alpha=0.8)
        ax_ratio.set_ylabel(f"All/{ratio_key}")
        ax_ratio.set_ylim(0.5, 1.5)
    else:
        fig, ax_main = plt.subplots(dpi=100)
        ax_ratio = ax_main

    for method, values in metric_values.items():
        current_color = colors.get(method, "black")

        ax_main.errorbar(bin_centers, values, xerr=bin_half_widths,
            fmt="o", color=current_color, label=labels.get_method_label(method)
        )
    
        if (use_ratio and (method != ratio_key)):
            ratio_values = values / metric_values[ratio_key]
            ax_ratio.errorbar(bin_centers, ratio_values, xerr=bin_half_widths,
                fmt="o", color=current_color
            )

    if (hline is not None):
        ax_main.axhline(y=hline, color="gray", linestyle="--")
    if (vline is not None):
        ax_main.axvline(x=vline, color="gray", linestyle="--")

    ax_main.set_ylabel(labels.get_var_label(y_label))
    ax_ratio.set_xlabel(labels.get_var_label(x_label))

    ax_main.set_xscale(x_scale)
    ax_main.set_yscale(y_scale)

    ax_main.legend()

    # CMS labelling
    try:
        com_energy = labels.get_com_energy(cms_year)
    except ValueError:
        com_energy = None
    hep.cms.label(cms_text, loc=0, data=False, year=cms_year, com=com_energy, ax=ax_main)

    if (save_filename is not None):
        plt.savefig(f"{save_filename}.{img_type}", dpi="figure")
    else:
        plt.show()
    
    plt.close()

def plot_binned_rmse_bias(
    bias: Dict[str, np.ndarray],
    rmse: Dict[str, np.ndarray], 
    bin_edges: np.ndarray,
    x_label: Optional[str] = "",
    y_label: Optional[str] = "",
    # ymax: Optional[float] = None,
    colors: Optional[dict] = colorschemes.reconstruction_method_colors,
    save_filename: Optional[str] = None,
    cms_text: Optional[str] = "Work in Progress",
    cms_year: Optional[str] = "2022",
    img_type: Optional[str] = "png"
):
    """
        Plots RMSE, bias and |bias| for all methods on the same plot.
        Main inputs:
            a dict of labels and numpy arrays of each method's bias
            a dict of labels and numpy arrays of each method's rmse,
            a numpy array of bin edges,
            x_label: title for x_axis,
            y_label: title for y_axis,
            save_filename: path to store output plot
    """
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # bin_widths = np.diff(bin_edges)

    ################ Plot ################
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    # max_height = float("-inf")
    # min_height = float("inf")

    # |biases|
    for key, weight in bias.items():
        abs_bias_needed = weight < 0

        value = bin_centers[abs_bias_needed]
        weight = np.abs(weight[abs_bias_needed])
        
        counts, _, patches = ax.hist(value, bins=bin_edges, weights=weight, # absolute value of bias
            histtype="step", edgecolor=colors.get(key, "black"), lw=2,
            label=labels.get_method_label(key),
            linestyle="--", color=colors.get(key, "black")
        )

    # line at y = 0
    plt.axhline(y=0, color="black", linestyle="-", linewidth=3.0)
    
    # biases
    for key, weight in bias.items():
        plt.plot(bin_centers, weight,
            marker="^", markersize=4,
            color=colors.get(key, "black"),
            linestyle="None",
            label=labels.get_method_label(key)
        )

        # if (weight.min() < min_height):
        #     min_height = weight.min()
        
    # RMSE
    for key, weight in rmse.items():
        counts, _, patches = ax.hist(bin_centers, bins=bin_edges, weights=weight,
            histtype="step", edgecolor=colors.get(key, "black"), lw=2.5, facecolor="none", 
            label=labels.get_method_label(key)
        )
        
        # if (counts.max() > max_height):
        #     max_height = counts.max()

    ################## Customize Legend ####################
    # custom_patch_blank = Patch(facecolor="none", edgecolor="none", label="")
    custom_line_rmse = Line2D([0], [0], color="black", linestyle="-", label="RMSE")
    custom_line_absbias = Line2D([0], [0], color="black", linestyle="--", label="|Bias|")
    custom_line_bias = Line2D([0], [0], marker="^", color="black",
        markerfacecolor="black", markeredgecolor="black", markersize=9, linestyle="None", label="Bias"
    )
    
    all_legend_handles = [custom_line_rmse, custom_line_bias, custom_line_absbias]
    for method in rmse.keys():
        method_color = colors.get(method, "black")
        new_patch = Patch(facecolor=method_color, edgecolor=method_color, label=labels.get_method_label(method), alpha=0.7)
        all_legend_handles.append(new_patch)

    # enough columns for each to be size 3, plus a column for the shapes
    num_cols = len(all_legend_handles) // 3
    if (len(all_legend_handles) % 3 != 0):
        num_cols += 1

    # Yao also used `bbox_to_anchor` argument for some of her cases for some reason...
    ax.legend(handles=all_legend_handles, loc="upper left", ncol=num_cols, fontsize=24, columnspacing=0.5)

    ax.set_xlim(left=bin_edges[0], right=bin_edges[-1])
    ax.set_xlabel(labels.get_var_label(x_label), fontsize=30)
    ax.tick_params(axis="both", labelsize=28)
    ax.xaxis.set_tick_params(pad=10)

    ax.set_ylabel(y_label, fontsize=30)
    # if (ymax is None):
    #     ymax = 1.9 * max_height
    # ax.set_ylim(bottom=1.2*min_height, top=ymax)
    ax = hep.utils.yscale_legend(ax) # we'll see how this guy works

    # CMS labelling
    try:
        com_energy = labels.get_com_energy(cms_year)
    except ValueError:
        com_energy = None
    hep.cms.label(cms_text, loc=0, data=False, year=cms_year, com=com_energy, ax=ax, fontsize=22)

    if (save_filename is not None):
        plt.savefig(f"{save_filename}.{img_type}", dpi="figure", bbox_inches="tight")
    else:
        plt.show()

    plt.close()
