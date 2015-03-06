import numpy as np
import sys
sys.path.append('../')
from stack_lya import rebin_flux

ex_flux = np.array([1., 2., 3., 4., 5., 6., 7.])
ex_rebin_2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 4.0])
ex_rebin_3 = np.array([2., 3., 4., 5., 6., 14./3., 10./3.])

def test_rebin_by_1():
    """
    Test case with rebin_number=1 (should return input array).
    """
    result = rebin_flux(ex_flux, 1)
    assert np.all(result == ex_flux)

def test_rebin_by_2():
    """
    Test averaging flux in pairs of bins.
    """
    result = rebin_flux(ex_flux, 2)
    assert np.all(result == ex_rebin_2)

def test_rebin_by_3():
    """
    Test averaging flux in groups of 3 bins.
    """
    result = rebin_flux(ex_flux, 3)
    assert np.all(result == ex_rebin_3)


