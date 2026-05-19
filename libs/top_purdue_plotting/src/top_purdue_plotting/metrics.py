import numpy as np
from matplotlib import pyplot as plt
import mplhep as hep
from typing import Dict, Optional

# Extra matplotlib imports that Yao used
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

    if (save_filename is not None):
        plt.savefig(f"{save_filename}.{img_type}", dpi="figure")
    else:
        plt.show()
    
    plt.close()

######## NOTES ########
# from matplotlib.patches import Patch # where Patch comes from
# from matplotlib.lines import Line2D # where Line2D comes from
# AXIS_LABELS is variable labels, use labels.get_var_label
# LEGEND_LABELS is method labels, still need to add this I guess
#               it's really just a nice label for mlb_weighting

def plot_binned_rmse_bias(
    bias: Dict[str, np.ndarray],
    rmse: Dict[str, np.ndarray], 
    bin_edges: np.ndarray,
    x_label: Optional[str] = "",
    y_label: Optional[str] = "",
    ymax: Optional[float] = None,
    colors: Optional[dict] = colorschemes.reconstruction_method_colors,
    save_filename: Optional[str] = None,
    era: Optional[str] = "2017",
    legend_loc: Optional[int] = 0,
    img_type: Optional[str] = "png"
):
    """
        It plots RMSE, bias and |bias| for all methods on the same plot.
        Input: a list of numpy arrays of each method's bias
               a list of numpy arrays of each method's rmse,
               a list of labels for each method,
               xlabel: title for x_axis,
               ylabel: title for y_axis,
               ymax: Manually set y axis upper limit
               save_folder: path to store output plots
               era: '2016preVFP', '2016postVFP', '2017', '2018' 
    """
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # bin_widths = np.diff(bin_edges)

    # Identify center-of-mass energy from `era`
    com = labels.get_com_energy(era)

    ################ Plot ################

    # Set up canvas
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    hep.cms.label(label='Preliminary', data=False, year=era, com=com, loc=0, ax=ax, fontsize=22)

    max_height = -999
    min_height = 999

    # |biases|
    for key, weight in bias.items(): 
        value = bin_centers[weight<0]
        weight = weight[weight<0]
        
        counts, _, patches = ax.hist(value, bins=bin_edges, 
            weights=np.abs(weight), # absolute value of bias
            histtype='step', edgecolor=colors.get(key, "black"), lw=2,
            label=labels.get_method_label(key),
            linestyle='--', color=colors.get(key, "black")
        )

    # line at y = 0
    plt.axhline(y=0, color='black', linestyle='-', linewidth=3.0) 
    
    # biases
    for key, weight in bias.items():
        plt.plot(bin_centers, weight, 
            marker='^', markersize=4, 
            color=colors.get(key, "black"), 
            linestyle='None', 
            label=labels.get_method_label(key)
        )
        if weight.min() < min_height:
            min_height = weight.min()
        
    # RMSE
    for key, weight in rmse.items():
        counts, _, patches = ax.hist(bin_centers, bins=bin_edges, weights=weight,
            histtype='step', edgecolor=colors.get(key, "black"), lw=2.5, facecolor="none", 
            label=labels.get_method_label(key)
        )
        
        if counts.max() > max_height:
            max_height = counts.max()

    ################## Customize Legend ####################
    # Create legend entries
    custom_line_rmse = Line2D([0], [0], color='black', linestyle='-', 
                              label='RMSE')
    custom_line_bias = Line2D([0], [0], marker='^', color='black', markerfacecolor='black',
                              markeredgecolor='black', markersize=9, linestyle='None', 
                              label='Bias')
    custom_line_absbias = Line2D([0], [0], color='black', linestyle='--', 
                                 label='|Bias|')
    custom_patch_blank = Patch(facecolor='none', edgecolor='none', 
                               label='')
    custom_patch_tf_only = Patch(facecolor='none', edgecolor='none', 
                                 label='Transformer reconstruction')

    custom_patch_mlp = Patch(facecolor='#ffa90e', alpha = 0.7, edgecolor='#ffa90e', 
                             label='MLP')
    custom_patch_mlb = Patch(facecolor='#bd1f01', alpha = 0.7, edgecolor='#bd1f01', 
                             label=labels.get_method_label("mlb_weighting"))
    custom_patch_tf = Patch(facecolor='#3f90da', alpha = 0.7, edgecolor='#3f90da', 
                            label='Transformer')

    custom_patch_tf_mlb_fail = Patch(facecolor='#bd1f01', alpha = 0.7, edgecolor='#bd1f01', 
                                     label='mlb-weighting fail')
    custom_patch_tf_mlb_succeed = Patch(facecolor='#3f90da', alpha = 0.7, edgecolor='#3f90da', 
                                        label='mlb-weighting succeed')

    # Assemble legends
    legend_fontsize=24
    if legend_loc == 0: # for results when mlb-weighting, mlp and transformer exist, for note publication
        ax.legend(handles=[custom_line_rmse, custom_line_bias, custom_line_absbias, 
                           custom_patch_mlb, custom_patch_tf, custom_patch_mlp], 
                  loc='upper right', ncol=2,  fontsize=legend_fontsize, columnspacing=0.5) 
    elif legend_loc == 1: # special for pz, insert a empty column in the middle, for note publication
        ax.legend(handles=[custom_patch_mlb, custom_patch_tf, custom_patch_mlp, 
                           custom_patch_blank, custom_patch_blank, custom_patch_blank, 
                           custom_line_rmse, custom_line_bias, custom_line_absbias], 
                  loc='upper left', ncol=3, fontsize=legend_fontsize, columnspacing=1.5, bbox_to_anchor=(-0.02, 1.02))
    elif legend_loc == 2: # for 2016 events fail mlb-weighting method, for note publication
        ax.legend(handles=[custom_line_rmse, custom_line_bias, custom_line_absbias, 
                           custom_patch_tf_mlb_fail, custom_patch_tf_mlb_succeed], 
                  loc='upper center', ncol=2,  fontsize=legend_fontsize, columnspacing=1.0, bbox_to_anchor=(0.5, 1.02))
        ax.text(0.5, 0.68, "Transformer Reconstruction", transform=ax.transAxes, fontsize=legend_fontsize, 
                ha='center', va='top', bbox=dict(facecolor='none', edgecolor='none'))
    elif legend_loc == 3: # for results when only transformer results exist, temporary for Run 3
        ax.legend(handles=[custom_line_rmse, custom_patch_tf, 
                           custom_line_bias, custom_patch_blank, 
                           custom_line_absbias, custom_patch_blank], 
                  loc='upper center', ncol=3, fontsize=legend_fontsize, columnspacing=1.0, bbox_to_anchor=(0.5, 1.02))
    elif legend_loc == 4: # for results when only mlb-weighting and transformer exist, for Run 2
        ax.legend(handles=[custom_line_rmse, custom_line_bias, custom_line_absbias, 
                           custom_patch_mlb, custom_patch_tf], 
                  loc='upper right', ncol=2, fontsize=legend_fontsize, columnspacing=0.5) 

    # Set x axis
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    xlabel_tot = x_label.split()
    xlabel_tot = [labels.get_var_label(x) for x in xlabel_tot]
    xlabel_tot = ' '.join(xlabel_tot)
    ax.set_xlabel(xlabel_tot, fontsize=30)
    ax.tick_params(axis='both', labelsize=28)
    ax.xaxis.set_tick_params(pad=10)

    # Set y axis
    ax.set_ylabel(y_label, fontsize=30)
    ax.set_ylim(top=1.9 * max_height)
    if legend_loc == 1:
        ax.set_ylim(top=1.75*max_height)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    ax.set_ylim(bottom=1.2 * min_height)

    # Save plot
    x_label = x_label.replace(' ', '_')
    
    if (save_filename is not None):
        plt.savefig(f"{save_filename}.{img_type}", dpi="figure", bbox_inches="tight")
    else:
        plt.show()

    plt.close()
