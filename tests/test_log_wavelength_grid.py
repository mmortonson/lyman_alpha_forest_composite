import numpy as np
import sys
sys.path.append('../')
from stack_lya import log_wavelength_grid

# default parameters
d_wv_range = [100., 10000.]
d_dlog10_wv = 0.1
d_ref_wv = 1015.
d_ref_type = 'center'

def test_correct_range_exact():
    """
    Test whether the ends of the grid are the same as the input extent
    when the reference wavelength is an integer number of bins 
    from each end.
    """
    ref_wv = 1000.
    grid = log_wavelength_grid(d_wv_range, d_dlog10_wv, ref_wv, \
                               reference_type=d_ref_type)
    assert grid[0] == d_wv_range[0], 'Minimum wavelength is wrong.'
    assert grid[-1] == d_wv_range[1], 'Maximum wavelength is wrong.'

def test_correct_range_approx():
    """
    Test whether the ends of the grid are the same as the input extent
    when the reference wavelength is not an integer number of bins 
    from each end.
    """
    grid = log_wavelength_grid(d_wv_range, d_dlog10_wv, d_ref_wv, \
                               reference_type=d_ref_type)
    dlog10_min = np.abs(np.log10(grid[0]/d_wv_range[0]))
    dlog10_max = np.abs(np.log10(grid[-1]/d_wv_range[1]))
    assert dlog10_min < d_dlog10_wv, 'Minimum wavelength is wrong.'
    assert dlog10_max < d_dlog10_wv, 'Maximum wavelength is wrong.'

def test_reference_in_range_in_grid():
    """
    Test whether the reference wavelength is one of the grid 
    elements (for reference_type='center'), if it is within 
    the wavelength range of the grid.
    """
    grid = log_wavelength_grid(d_wv_range, d_dlog10_wv, d_ref_wv, \
                               reference_type=d_ref_type)
    assert np.any(np.isclose(grid, d_ref_wv)), \
               'Reference wavelength not found in grid.'

def test_reference_in_range_between_grid():
    """
    Test whether the reference wavelength is halfway (logarithmically)
    between two grid elements (for reference_type='edge'), if it is within 
    the wavelength range of the grid.
    """
    grid = log_wavelength_grid(d_wv_range, d_dlog10_wv, d_ref_wv, \
                               reference_type='edge')
    midpoints = np.sqrt( (grid*np.roll(grid, 1))[1:] )
    assert np.any(np.isclose(midpoints, d_ref_wv)), \
               'Reference wavelength not found at any grid edges.'

def test_reference_outside_range_correctly_aligned():
    """
    Test whether the reference wavelength is an integer number of
    bins away from the grid elements (for reference_type='center'), 
    if it is outside the wavelength range of the grid.
    """
    ref_wv = 42.
    grid = log_wavelength_grid(d_wv_range, d_dlog10_wv, ref_wv, \
                               reference_type=d_ref_type)
    n_bins_from_start = np.log10(ref_wv/grid[0]) / d_dlog10_wv
    assert np.abs(n_bins_from_start - np.round(n_bins_from_start)) < 1.e-5, \
               'Grid points are misaligned with reference wavelength.'

def test_invalid_reference_type():
    """
    Test that None is returned if reference_type is not either
    'center' or 'edge'.
    """
    grid = log_wavelength_grid(d_wv_range, d_dlog10_wv, d_ref_wv, \
                               reference_type='other')
    assert grid is None, 'Wrong return value for invalid reference_type.'
