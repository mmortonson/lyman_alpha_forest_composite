import numpy as np
import sys
sys.path.append('../')
from stack_lya import stack_stats

def test_basic_stats():
    """
    Check that stack statistics are correct for 
    several short, simple columns.
    """

    ex_array = np.transpose(np.array([
                                [1., 1., 1., 1.],
                                [1., 2., 3., 4.],
                                [3., 1., 4., 2.],
                                [-2., -1., 0., 7.],
                                [np.nan, 2., 3., 4.],
                                [0., np.nan, 2., np.nan],
                                [np.nan, np.nan, np.nan, 4.],
                                [np.nan, np.nan, np.nan, np.nan]
                            ]))

    ex_number = np.array([4, 4, 4, 4, 3, 2, 1, 0])
    ex_mean = np.array([1., 2.5, 2.5, 1., 3., 1., 4., 0.])
    ex_median = np.array([1., 2.5, 2.5, -0.5, 3., 1., 4., 0.])

    number, mean, median = stack_stats(ex_array)
    assert np.all(number == ex_number), 'Number array is wrong.'
    assert np.allclose(mean, ex_mean), 'Mean array is wrong.'
    assert np.allclose(median, ex_median), 'Median array is wrong.'

def test_outlier_clipping():
    """
    Check that outlier_fraction option works correctly.
    """

    ex_array = np.transpose(np.array([[0., 1., 4., 9., 16.]]))
    outlier_fraction = 0.2
    ex_number = np.array([5])
    ex_mean = np.array([14./3.]) # result without clipping would be 6.
    ex_median = np.array([4.])

    number, mean, median = stack_stats(ex_array, \
                                           outlier_fraction=outlier_fraction)
    assert np.all(number == ex_number), 'Number array is wrong.'
    assert np.allclose(mean, ex_mean), 'Mean array is wrong.'
    assert np.allclose(median, ex_median), 'Median array is wrong.'
