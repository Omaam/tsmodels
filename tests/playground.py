"""
"""
from tqdm import trange
from statsmodels.tsa import ar_model
import matplotlib.pyplot as plt
import numpy as np


def simulate_armodel(arcoefs, num_timesteps):
    num_burnin = 2000
    x = [0] * len(arcoefs)
    for i in range(num_burnin + num_timesteps):
        v = np.random.normal(0, 0.1)
        for j, c in enumerate(arcoefs):
            v += c*x[i+j]
        x.append(v)
    x = np.array(x[num_burnin:])
    return x


def main():
    arcoefs_actual = [0.5, 0.3]

    arcoefs_ = []
    for _ in trange(500):

        x = simulate_armodel(arcoefs_actual, 1000)

        sm_armodel = ar_model.AutoReg(x, 8)
        armodel_fitres = sm_armodel.fit()
        arcoefs_.append(armodel_fitres.params)

    arcoefs = np.asarray(arcoefs_)

    fig, ax = plt.subplots(1, len(arcoefs.T), sharey="row",
                           figsize=(12, 3))
    for i, v in enumerate(arcoefs.T):
        print(f"Par {i}: {np.mean(v):.3f} +- {np.std(v):.3f}")
        ax[i].hist(v)
    plt.show()


if __name__ == "__main__":
    main()
