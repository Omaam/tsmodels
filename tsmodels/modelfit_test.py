"""AR model fit module.
"""
from statsmodels.tsa import ar_model
import unittest
import numpy as np

import modelfit


class TestUtil(unittest.TestCase):

    def test_fit_armodel(self):
        self._create_testcase_00()
        arcoef_, error_variance_ = modelfit.fit_armodel(self.x, self.arorder)

    def _create_testcase_00(self):
        self.arorder = 2
        self.arcoef = [0.5, 0.3]

        num_timesteps = 1000
        np.random.seed(0)
        x = [0.1, -0.3]
        for i in range(num_timesteps):
            v = self.arcoef[0]*x[i] + self.arcoef[1]*x[i+1]
            v += np.random.normal(0, 0.1)
            x.append(v)
        self.x = np.array(x[self.arorder:])

        sm_armodel = ar_model.AutoReg(x, 2)
        armodel_fitres = sm_armodel.fit()
        print(armodel_fitres.params)


if __name__ == "__main__":
    unittest.main()
