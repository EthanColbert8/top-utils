import os
import unittest
import numpy as np
import hist
from hist import Hist
from datetime import datetime

from src import top_purdue_plotting as plotting

class TestHistOverlayPlotting(unittest.TestCase):
    def setUp(self):
        self.datetimestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        file_loc = os.path.dirname(os.path.abspath(__file__))
        self.test_images_dir = os.path.join(file_loc, "test_images")
        os.makedirs(self.test_images_dir, exist_ok=True)

        self.colors = {
            "sample1": "blue",
            "sample2": "orange",
            "gen": "gray",
        }

        rand_gen = np.random.default_rng(seed=58008)

        binning = [-10.0, -8.0, -6.0, -5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        self.hist1 = Hist(hist.axis.Variable(binning, name="x"), storage=hist.storage.Weight())
        self.hist2 = Hist(hist.axis.Variable(binning, name="x"), storage=hist.storage.Weight())
        
        data1 = rand_gen.normal(loc=0.0, scale=6.0, size=6000)
        weights1 = rand_gen.uniform(low=0.4, high=0.8, size=6000)
        data2 = rand_gen.normal(loc=0.0, scale=6.0, size=10000)
        weights2 = rand_gen.uniform(low=0.4, high=0.8, size=10000)

        self.hist1.fill(x=data1, weight=weights1)
        self.hist2.fill(x=data2, weight=weights2)

    def test_hist_overlay_steps(self):
        histograms = {
            "sample1": self.hist1,
            "sample2": self.hist2,
        }
        save_filename = os.path.join(self.test_images_dir, f"hist_overlay_steps_{self.datetimestamp}")

        plotting.plot_1d_hists_overlay(
            histograms, "x", save_filename,
            density=True, ratio_key=None, colors=self.colors,
            cms_text="Test Plot", cms_year="2022"
        )

    def test_hist_overlay_filled(self):
        histograms = {
            "sample1": self.hist1,
            "gen": self.hist2,
        }
        save_filename = os.path.join(self.test_images_dir, f"hist_overlay_filled_{self.datetimestamp}")

        plotting.plot_1d_hists_overlay(
            histograms, "x", save_filename,
            density=True, ratio_key=None, colors=self.colors,
            cms_text="Test Plot", cms_year="2022"
        )
    
    def test_hist_overlay_steps_ratio(self):
        histograms = {
            "sample1": self.hist1,
            "sample2": self.hist2,
        }
        save_filename = os.path.join(self.test_images_dir, f"hist_overlay_steps_ratio_{self.datetimestamp}")

        plotting.plot_1d_hists_overlay(
            histograms, "x", save_filename,
            density=True, ratio_key="sample2", colors=self.colors,
            cms_text="Test Plot", cms_year="2022"
        )
