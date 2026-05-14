import numpy as np
import hist
from hist import Hist
from matplotlib import pyplot as plt
import mplhep as hep
from typing import Dict, Optional

from . import colorschemes
from . import labels

plt.style.use(hep.style.CMS)

method_histplot_types = {
    "solver": "step",
    "feedforward": "step",
    "mlp": "step",
    "transformer": "step",
    "hybrid": "step",
    "nu2flows": "step",
    "mlb_weighting": "step",
    
    "gen": "fill",
    "true": "fill",
    "truth": "fill",
    "target": "fill",

    "Original": "fill",
    "Resampled": "step",
}

run2_eras = set(["2016preVFP", "2016postVFP", "2016", "2017", "2018", "Run 2", "Full Run 2"])
run3_eras = set(["2022preEE", "2022postEE", "2022", "2023preBPix", "2023postBPix", "2023"]) # TODO: Add 2024, 2025, full eras

def plot_1d_hists_overlay(
    hists: Dict[str, Hist],
    x_label: Optional[str] = None,
    save_filename: Optional[str] = None,
    density: Optional[bool] = False,
    binwnorm: Optional[float] = None,
    yscale: Optional[str] = "linear",
    ratio_key: Optional[str] = None,
    colors: Optional[dict] = colorschemes.reconstruction_method_colors,
    cms_text: Optional[str] = "Work in Progress",
    cms_year: Optional[str] = "2022",
    x_units: Optional[str] = "",
    img_type: Optional[str] = "png"
):
    """
    Plots multiple 1D histograms overlayed on the same axes, optionally with a ratio to one of them underneath.
    Takes (at least) a dictionary of histograms, where keys are names for the plot legend and values are Hist objects.
    If one of the keys is \"gen\" or \"truth\", will be plotted as a \"filled\" histogram. Others will be
    step (skyline) style plots.
    """
    use_ratio = ratio_key is not None

    y_label = "Density" if density else "Events"
    if (binwnorm is not None):
        y_label += f" / {binwnorm} {x_units}".rstrip()

    an_axis = hists[list(hists.keys())[0]].axes[0]
    bin_edges = an_axis.edges

    if (x_label is None):
        x_label = an_axis.label if (an_axis.label != "") else an_axis.name

    if use_ratio:
        fig, (ax_main, ax_ratio) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1}, sharex=True, dpi=100)
        ax_ratio.axhline(y=1.0, color="black", linestyle="--", lw=2, alpha=0.8)
        ax_ratio.set_ylabel(f"All/{ratio_key}")
        ax_ratio.set_ylim(0.5, 1.5)
    else:
        fig, ax_main = plt.subplots(dpi=100)
        ax_ratio = ax_main

    for method, histogram in hists.items():
        current_color = colors.get(method, "black")
        
        hep.histplot(histogram, ax=ax_main, stack=False, histtype=method_histplot_types.get(method, "step"), density=density, binwnorm=binwnorm, color=current_color, label=method)

        if (use_ratio and (method != ratio_key)):
            ratio_values = histogram.values() / hists[ratio_key].values()
            
            # if we're plotting an overlay, probably don't care about ratio uncertainties
            ax_ratio.stairs(ratio_values, bin_edges, color=current_color)

    ax_main.set_xlim(bin_edges[0], bin_edges[-1])

    ax_main.set_yscale(yscale)
    if (yscale == "log"):
        ax_main.set_ylim(bottom=1.0e-5)
    elif (yscale == "linear"):
        ax_main.set_ylim(bottom=0.0)

    ax_ratio.set_xlabel(labels.get_var_label(x_label))
    ax_main.set_ylabel(y_label)

    ax_main.legend()

    # CMS labelling
    com_energy = None
    if cms_year in run2_eras:
        com_energy = 13
    elif cms_year in run3_eras:
        com_energy = 13.6
    hep.cms.label(cms_text, loc=0, data=False, year=cms_year, com=com_energy, ax=ax_main)

    if (save_filename is not None):
        plt.savefig(f"{save_filename}.{img_type}", dpi="figure")
    else:
        plt.show()

    plt.close()

def plot_2d_hist(
    histogram: Hist,
    weighted: Optional[bool] = False,
    plot_unity: Optional[bool] = False,
    scale: Optional[str] = "linear",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    save_filename: Optional[str] = None,
    density: Optional[bool] = False,
    show_cbar: Optional[bool] = True,
    cbar_min: Optional[float] = None,
    cbar_max: Optional[float] = None,
    cms_text: Optional[str] = "Work in Progress",
    cms_year: Optional[str] = "2022",
    img_type: Optional[str] = "png"
):
    """
    Plots a 2D histogram using the \"Viridis\" colormap (recommended by CMS).
    Can optionally:
    - Normalize to density
    - Plot a unity line (y=x) for reference
    - Use logarithmic color scaling
    """
    if density:
        real_histogram = histogram.copy()

        # Get the counts and bins to normalize to density
        counts = real_histogram.values()
        first_edges = real_histogram.axes[0].edges
        second_edges = real_histogram.axes[1].edges

        first_bin_widths = np.diff(first_edges)
        second_bin_widths = np.diff(second_edges)
        bin_areas = np.outer(first_bin_widths, second_bin_widths)

        total_count = np.sum(counts)
        normalization_factor = total_count * bin_areas
        normalized_counts = counts / normalization_factor # Normalize to density

        if weighted:
            normalized_variances = real_histogram.variances() / np.square(normalization_factor) # Normalize variances as well
            
            dtype = np.dtype([('value', '<f8'), ('variance', '<f8')])
            stuff_to_fill = np.core.records.fromarrays([normalized_counts, normalized_variances], dtype=dtype)

            real_histogram[...] = stuff_to_fill
        else:
            real_histogram[...] = normalized_counts
    else:
        real_histogram = histogram
    
    xmin = real_histogram.axes[0].edges[0]
    xmax = real_histogram.axes[0].edges[-1]
    ymin = real_histogram.axes[1].edges[0]
    ymax = real_histogram.axes[1].edges[-1]

    fig, ax = plt.subplots(dpi=100)

    hep.hist2dplot(real_histogram, ax=ax, norm=scale, cmap="viridis", cbar=show_cbar, cmin=cbar_min, cmax=cbar_max)

    if plot_unity:
        limits = [xmin, xmax]
        ax.plot(limits, limits, color="fuchsia", alpha=1.0)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if (x_label is None):
        x_label = histogram.axes[0].label if (histogram.axes[0].label != "") else histogram.axes[0].name
    ax.set_xlabel(labels.get_var_label(x_label))
    
    if (y_label is None):
        y_label = histogram.axes[1].label if (histogram.axes[1].label != "") else histogram.axes[1].name
    ax.set_ylabel(labels.get_var_label(y_label))

    # CMS labelling
    com_energy = None
    if cms_year in run2_eras:
        com_energy = 13
    elif cms_year in run3_eras:
        com_energy = 13.6
    hep.cms.label(cms_text, loc=0, data=False, year=cms_year, com=com_energy, ax=ax)

    if (save_filename is not None):
        plt.savefig(f"{save_filename}.{img_type}", dpi="figure")
    else:
        plt.show()
    
    plt.close()

def plot_1d_hists_stacked(
    hists: Dict[str, Hist],
    data_hist: Optional[Hist] = None,
    syst_uncertainties_arrs: Optional[Dict[str, tuple]] = None,
    x_label: Optional[str] = None,
    save_filename: Optional[str] = None,
    binwnorm: Optional[float] = None,
    yscale: Optional[str] = "linear",
    colors: Optional[dict] = colorschemes.process_colors,
    cms_text: Optional[str] = "Work in Progress",
    cms_lumi: Optional[float] = 41.48,
    cms_com_energy: Optional[float] = 13.6,
    x_units: Optional[str] = "",
    xlim: Optional[tuple] = None,
    ratio_ylim: Optional[tuple] = (0.75, 1.25),
    img_type: Optional[str] = "png"
):
    """
    Plots a stacked histogram of many MC samples, optionally with data. Ratio plot is included if data histogram is provided.
    This function has many options, to document the most important ones:
    - hists: dictionary that should ONLY contain MC histograms. Keys are process names for the legend.
    - data_hist: a single Hist object for data.
    - syst_uncertainties_arrs: a dict containing names of systematic uncertainties as keys, tuples that are pairs of NumPy arrays
        as values. The arrays should contain the UNCERTAINTY MAGNITUDES (not histogram counts), and should be ordered as (UP, DOWN).
    """
    data_used = (data_hist is not None)

    # Make ratio plot if there's data
    if data_used:
        fig, (ax_main, ax_ratio) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), sharex=True)
    else:
        fig, ax_main = plt.subplots(dpi=100)
        ax_ratio = ax_main

    # Pulling out some general info
    histograms_list = list(hists.values())
    legend_entries = list(hists.keys())
    legend_labels = [labels.get_process_label(label) for label in legend_entries]
    colors_list = [colors.get(label, None) for label in legend_entries]
    bin_edges = histograms_list[0].axes[0].edges

    # Plot stacked MC hists
    hep.histplot(histograms_list, ax=ax_main,
        stack=True, histtype="fill", sort="yield",
        density=False, binwnorm=binwnorm,
        color=colors_list, label=legend_labels,
        edgecolor="black", linewidth=0.5
    )

    # Sum MC hists
    total_mc_hist = sum(histograms_list)

    # Plot MC uncertainties as bands
    if (syst_uncertainties_arrs is None):
        total_uncertainty_up = np.sqrt(total_mc_hist.variances())
        total_uncertainty_down = total_uncertainty_up
    else:
        syst_variance_up = np.sum(np.square(np.stack([x[0] for x in syst_uncertainties_arrs.values()], axis=0)), axis=0)
        syst_variance_down = np.sum(np.square(np.stack([x[1] for x in syst_uncertainties_arrs.values()], axis=0)), axis=0)

        total_uncertainty_up = np.sqrt(total_mc_hist.variances() + syst_variance_up)
        total_uncertainty_down = np.sqrt(total_mc_hist.variances() + syst_variance_down)
    
    unc_upper_values = total_mc_hist.values() + total_uncertainty_up
    unc_lower_values = total_mc_hist.values() - total_uncertainty_down

    if (binwnorm is None):
        binwnorm_divisor = 1.0
    else:
        binwnorm_divisor = np.diff(total_mc_hist.axes[0].edges) / binwnorm

    ax_main.stairs(values=unc_upper_values/binwnorm_divisor, baseline=unc_lower_values/binwnorm_divisor, edges=total_mc_hist.axes[0].edges,
        label=r"Stat $\oplus$ Syst",color="black", alpha=0.75, hatch="////", lw=0, facecolor="none"
    )
    
    if data_used: 
        hep.histplot(data_hist, ax=ax_main,
            yerr=True, xerr=True, stack=False, histtype="errorbar", density=False,
            binwnorm=binwnorm, color=colors.get("data", None), label="Data"
        )
        
        total_mc_hist_denom = np.where(total_mc_hist.values() < 0.01, 0.01, total_mc_hist.values()) # avoid division by zero
        data_relative_variance = data_hist.variances() / np.square(np.where(data_hist.values() < 0.01, 0.01, data_hist.values()))
        # data_relative_error = np.sqrt(data_hist.variances()) / np.where(data_hist.values() < 0.01, 0.01, data_hist.values())

        ratio = data_hist.values() / total_mc_hist_denom
        
        ratio_err_up = ratio * np.sqrt(data_relative_variance + np.square(total_uncertainty_up / total_mc_hist_denom))
        ratio_err_down = ratio * np.sqrt(data_relative_variance + np.square(total_uncertainty_down / total_mc_hist_denom))

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Use "xerr" to show bin width in ratio plot
        ax_ratio.errorbar(bin_centers, ratio, xerr=(np.diff(bin_edges)/2), yerr=np.stack([ratio_err_down, ratio_err_up]),
            fmt="o", color=colors.get("data", None), label="Data/MC"
        )
        # ax_ratio.errorbar(bin_centers, ratio, xerr=(np.diff(bin_edges)/2), yerr=data_relative_error,
        #     fmt="o", color=colors.get("data", None))#, label="Data/MC"
        # )
        
        # highlight unity line
        ax_ratio.axhline(y=1.0, color='black', linestyle='--', lw=2, alpha=0.8)
        
        ax_ratio.set_ylabel("Data/MC")
        ax_ratio.set_ylim(bottom=ratio_ylim[0], top=ratio_ylim[1])
        
        # # Add uncertainties manually as a block on MCs in the ratio plot.
        # ax_ratio.stairs(
        #     values=1 + (total_uncertainty_up / total_mc_hist_denom),
        #     baseline=1 - (total_uncertainty_down / total_mc_hist_denom),
        #     edges=total_mc_hist.axes[0].edges, 
        #     color='silver', 
        #     fill=True, 
        #     label=r'MC Stat $\oplus$ Syst'
        # )
        
        # ax_ratio.legend(loc='best', ncol=2) # ncol=2

    # CLEAR main axis x-label since it's shared with ratio plot
    ax_main.set_xlabel("") 

    if (x_label is None):
        x_label = histograms_list[0].axes[0].label if (histograms_list[0].axes[0].label != "") else histograms_list[0].axes[0].name
    ax_ratio.set_xlabel(labels.get_var_label(x_label))

    if (binwnorm is not None):
        ax_main.set_ylabel(f"Events / {binwnorm} {x_units}".rstrip())
    else:
        ax_main.set_ylabel("Events")
    ax_main.set_yscale(yscale)
    
    # # This all just sets y-axis limits to leave space for the legend,
    # # which should be done automatically by mplhep below
    # if binwnorm is None:
    #     tmp_binwnorm = np.diff(total_mc_hist.axes[0].edges)
    # else:
    #     tmp_binwnorm = binwnorm * np.ones(np.diff(total_mc_hist.axes[0].edges).shape)
    # if yscale == 'linear':
    #     ax_main.set_ylim(0, np.max(total_mc_hist.values()/np.diff(total_mc_hist.axes[0].edges)*tmp_binwnorm)*1.5) # leave space for the legend
    # elif yscale == 'log':
    #     ax_main.set_ylim(0.1, np.max(total_mc_hist.values()/np.diff(total_mc_hist.axes[0].edges)*tmp_binwnorm)*50) # leave space for the legend

    if (xlim is None):
        xlim = (bin_edges[0], bin_edges[-1])
    ax_main.set_xlim(left=xlim[0], right=xlim[1])

    # If limits are multiples of pi, set ticks and tick labels to multiples of pi/2
    if (xlim[1] % np.pi == 0):
        x_ticks = np.arange(xlim[0], xlim[1] + 0.1, np.pi/2)
        ax_main.set_xticks(x_ticks)
        ax_main.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])

    ax_main.legend(loc="upper left", ncol=3)
    ax_main = hep.utils.yscale_legend(ax_main) # rescale y-axis limits to make room for legend

    # CMS labelling
    hep.cms.label(cms_text, loc=0, data=data_used, lumi=cms_lumi, com=cms_com_energy, ax=ax_main)

    if (save_filename is not None):
        plt.savefig(f"{save_filename}.{img_type}", dpi='figure')  
    else:
        plt.show()  
    
    plt.close()
