"""General core modules.
"""

import numpy as np


def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array([x])
    return x
