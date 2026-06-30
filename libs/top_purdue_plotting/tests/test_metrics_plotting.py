import os
import unittest
import numpy as np
from datetime import datetime

from src import top_purdue_plotting as plotting

class TestRMSEBiasPlotting(unittest.TestCase):
    def setUp(self):
        self.rand_gen = np.random.default_rng(seed=58008)
        self.datetimestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        file_loc = os.path.dirname(os.path.abspath(__file__))
        self.test_images_dir = os.path.join(file_loc, "test_images")
        os.makedirs(self.test_images_dir, exist_ok=True)

        self.available_colors = [
            "blue", "orange", "green", "purple",
            "red", "pink", "darkgray", "lightgray", "brown",
            "cyan", "magenta", "fuchsia",
        ]

    def test_rmse_bias_plotting_big_legend(self):
        colors = {}
        biases = {}
        rmses = {}
        bins = np.linspace(-5, 5, num=21)
        for i in range(30):
            sample_name = f"sample_{i+1}"
            colors[sample_name] = self.available_colors[i % len(self.available_colors)]
            
            biases[sample_name] = self.rand_gen.uniform(low=-1.0, high=1.0, size=20)
            rmses[sample_name] = self.rand_gen.uniform(low=2.0, high=4.0, size=20)

            # # make one be huge
            # rmses[sample_name][1] += 6000000.0
        
        save_filename = os.path.join(self.test_images_dir, f"rmse_bias_big_legend_{self.datetimestamp}")

        plotting.plot_binned_rmse_bias(
            biases, rmses, bins,
            x_label="Something random", y_label="Error on the thing",
            colors=colors, save_filename=save_filename
        )
