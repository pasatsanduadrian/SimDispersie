import numpy as np
from app import calc_sigmas, calculate_plume_rise

def test_calc_sigmas_shape():
    sy, sz = calc_sigmas(1, np.array([100]))
    assert sy.shape == (1,)
    assert sz.shape == (1,)

def test_plume_rise_positive():
    assert calculate_plume_rise(150, 20, 2, 5) > 0
