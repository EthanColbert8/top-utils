__version__ = "0.1.3"

# Set up plots to use CMS styling
from matplotlib import pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)

from .histograms import plot_1d_hists_overlay, plot_1d_hists_stacked, plot_2d_hist
from .metrics import plot_binned_metric, plot_binned_rmse_bias
