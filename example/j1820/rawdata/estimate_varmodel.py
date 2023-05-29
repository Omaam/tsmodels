"""
"""
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
from statsmodels.tsa.api import VAR


def decide_best_order_by_aic(curves, maxlag=30):
    var = VAR(curves)
    aics = []
    for lag in range(1, maxlag+1, 1):
        res = var.fit(lag)
        aics.append([lag, res.aic])
    aics = np.array(aics)
    plt.plot(aics[:, 0], aics[:, 1])
    plt.ylim(*np.percentile(aics[:, 1], [0, 80]))
    plt.show()


def main():
    url_base = "https://raw.githubusercontent.com/Omaam/" \
               "xray_data_samples/main/sample/nicer/j1820/" \
               "curve_1200120106_{0}t{1}keV_dt1e-01.csv"

    band_edge_names = [["05", "1"],
                       ["1", "2"],
                       ["2", "3"],
                       ["3", "5"],
                       ["5", "10"]]

    os.makedirs(".cache", exist_ok=True)
    df_list = []
    for i, band_edge_name in enumerate(band_edge_names):
        curve_name = f".cache/curve_{i}.csv"
        if os.path.exists(curve_name):
            print(f"Found csv file {i}")
            df = pd.read_csv(curve_name)
        else:
            url = copy.copy(url_base)
            url = url.replace("{0}", band_edge_name[0])
            url = url.replace("{1}", band_edge_name[1])
            print(f"downloading csv file {i} >>>", end=" ")
            df = pd.read_csv(url)
            print("Done")
            df.to_csv(f".cache/curve_{i}.csv")
        df_list.append(df)

    curves = np.array([df["RATE"] for df in df_list]).T

    curve_band1 = curves[:, 0]
    curve_band2 = np.sum(curves[:, 1:3], axis=1)
    curve_band3 = np.sum(curves[:, 3:5], axis=1)
    curves = np.stack([curve_band1, curve_band2, curve_band3], axis=1)
    curves = np.log10(curves)
    curves = sp.stats.zscore(curves, axis=0)
    num_series = curves.shape[-1]
    print(curves.shape)

    # decide_best_order_by_aic(curves)

    # VAR(16) is the best VAR order determined by AIC.
    order = 16
    var = VAR(curves)
    fitres = var.fit(order)

    var_coefs = fitres.params[1:]
    var_coefs = np.reshape(var_coefs, (-1, num_series, num_series), "C")
    var_coefs = np.transpose(var_coefs, axes=(-3, -1, -2))
    np.save("var_coefficients", var_coefs)

    noise_cov = fitres.sigma_u
    np.save("var_noise_covariance", noise_cov)


if __name__ == "__main__":
    main()
