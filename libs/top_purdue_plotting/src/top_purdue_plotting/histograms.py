import numpy as np
import hist
from hist import Hist
from matplotlib import pyplot as plt
import mplhep as hep
from typing import Dict, Optional

from . import colorschemes

plt.style.use(hep.style.CMS)

method_histplot_types = {
    "Solver": "step",
    "Feedforward": "step",
    "Nu2Flows": "step",
    "mlb_weighting": "step",
    "gen": "fill",

    "Original": "fill",
    "Resampled": "step",
}

def plot_1d_hists_overlay(
    hists: Dict[str, Hist],
    x_label: Optional[str] = None,
    save_filename: Optional[str] = None,
    density: Optional[bool] = False,
    binwnorm: Optional[float] = None,
    yscale: Optional[str] = "linear",
    ratio_key: Optional[str] = None,
    colors: Optional[dict] = colorschemes.reconstruction_method_colors,
    # cms_text: Optional[str] = "Work in Progress",
    # lumi_text: Optional[str] = "2017UL",
    x_units: Optional[str] = "",
    img_type: Optional[str] = "png"
):
    """

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
        ax_main.set_ylim(bottom=0.1)
    elif (yscale == "linear"):
        ax_main.set_ylim(bottom=0.0)

    ax_ratio.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)

    ax_main.legend()

    # TODO: Convert to new mplhep API
    # hep.cms.text(cms_text, ax=ax_main)
    # hep.cms.lumitext(lumi_text, ax=ax_main)

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
    cms_text: Optional[str] = "Work in Progress",
    lumi_text: Optional[str] = "2017UL",
    img_type: Optional[str] = "png"
):
    """

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

    hep.hist2dplot(real_histogram, ax=ax, norm=scale, cmap="viridis")

    if plot_unity:
        limits = [xmin, xmax]
        ax.plot(limits, limits, color="fuchsia", alpha=1.0)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if (x_label is None):
        x_label = histogram.axes[0].label
    else:
        ax.set_xlabel(x_label)
    
    if (y_label is None):
        y_label = histogram.axes[1].label
    else:
        ax.set_ylabel(y_label)

    # TODO: Convert to new mplhep API
    # hep.cms.text(cms_text, ax=ax)
    # hep.cms.lumitext(lumi_text, ax=ax)

    if (save_filename is not None):
        plt.savefig(f"{save_filename}.{img_type}", dpi="figure")
    else:
        plt.show()
    
    plt.close()
