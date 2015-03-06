import numpy as np
import sys
sys.path.append('../')
from stack_lya import select_pixels, wavelength_lya

#wavelength_lya = 1215.6701
ex_wavelength = wavelength_lya * np.linspace(3., 4., num=11)
ex_flux = np.array([0., 0.5, 0., 0.2, 0.3, 0.24, -0.2, 0.7, 1.2, 0.1, 0.5])
ex_sn = np.array([1., 1.5, 2., 3.5, 3.5, 4., 4., 3., 2., 2., 4.])

def test_flux_selection():
    """
    Test whether pixels in a flux bin are selected correctly
    (without rebinning).
    """
    selection = select_pixels(ex_wavelength, ex_flux, ex_sn, \
                              rebinning=1, flux_range=[0.2, 0.3])
    expected_selection = np.array([3, 5])
    assert np.all(selection == expected_selection), \
               'Wrong selection for flux bin.'

def test_flux_selection_with_rebinning_by_2():
    """
    Test whether pixels in a flux bin are selected correctly
    with rebinning in pairs of pixels.
    """
    selection = select_pixels(ex_wavelength, ex_flux, ex_sn, \
                              rebinning=2, flux_range=[0.2, 0.3])
    expected_selection = np.array([0, 4, 6])
    assert np.all(selection == expected_selection), \
               'Wrong selection for flux bin with rebinning=2.'

def test_flux_selection_with_rebinning_by_3():
    """
    Test whether pixels in a flux bin are selected correctly
    with rebinning in triplets of pixels.
    """
    selection = select_pixels(ex_wavelength, ex_flux, ex_sn, \
                              rebinning=3, flux_range=[0.2, 0.3])
    expected_selection = np.array([3])
    assert np.all(selection == expected_selection), \
               'Wrong selection for flux bin with rebinning=3.'

def test_sn_selection():
    """
    Test whether pixels above a S/N threshold are selected correctly.
    """
    selection = select_pixels(ex_wavelength, ex_flux, ex_sn, \
                              rebinning=1, flux_range=[-np.inf, np.inf], \
                              sn_range=[3., np.inf])
    expected_selection = np.array([3, 4, 5, 6, 7, 10])
    assert np.all(selection == expected_selection), \
               'Wrong selection for S/N threshold.'

def test_z_selection():
    """
    Test whether pixels in a redshift range are selected correctly.
    """
    selection = select_pixels(ex_wavelength, ex_flux, ex_sn, \
                              rebinning=1, flux_range=[-np.inf, np.inf], \
                              z_range=[2., 2.5])
    expected_selection = np.array([0, 1, 2, 3, 4])
    assert np.all(selection == expected_selection), \
               'Wrong selection for absorber redshift range.'

def test_all_selection():
    """
    Test whether correct pixels are selected with flux, S/N, and
    absorber redshift ranges, using rebinning=2.
    """
    selection = select_pixels(ex_wavelength, ex_flux, ex_sn, \
                              rebinning=2, flux_range=[0.2, 0.3], \
                              sn_range=[3., np.inf], z_range=[2., 2.5])
    expected_selection = np.array([4])
    assert np.all(selection == expected_selection), \
               'Wrong selection for combined criteria.'
