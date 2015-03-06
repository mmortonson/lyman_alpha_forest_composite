import numpy as np
import sys
sys.path.append('../')
from stack_lya import add_to_stack, wavelength_lya

log10_wv_lya = np.log10(wavelength_lya)

# 11 wavelengths from 1083 to 1364, Lya is index 5
wv_lya_center_11 = 10.**(np.linspace(-0.05, 0.05, num=11) + log10_wv_lya)
# 10 wavelengths from 1083 to 1364, Lya is between indices 4 and 5
wv_lya_edge_10 = 10.**(np.linspace(-0.05, 0.05, num=10) + log10_wv_lya)

flux_3 = np.linspace(-0.1, 0.1, num=3)
flux_10 = np.linspace(-0.4, 0.5, num=10)
flux_15 = np.linspace(-0.4, 1.0, num=15)

def test_wv_lya_center_11_flux_3():
    """
    Test adding a short flux array in the middle of a 11-pixel stack array
    where Lya is at the center of a pixel.
    """
    indices, flux = add_to_stack(wv_lya_center_11, flux_3, 1, 1)
    expected_indices = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    expected_flux = np.array([0, 0, 0, 0, -0.1, 0, 0.1, 0, 0, 0, 0])
    assert np.all(indices == expected_indices), 'Wrong indices.'
    assert np.allclose(flux, expected_flux), 'Wrong flux.'

def test_wv_lya_edge_10_flux_3_rebin2():
    """
    Test adding a short flux array in the middle of a 10-pixel stack array
    where Lya is at an edge between pixels.
    """
    indices, flux = add_to_stack(wv_lya_edge_10, flux_3, 1, 2)
    expected_indices = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    expected_flux = np.array([0, 0, 0, -0.1, 0, 0.1, 0, 0, 0, 0])
    assert np.all(indices == expected_indices), 'Wrong indices.'
    assert np.allclose(flux, expected_flux), 'Wrong flux.'

def test_wv_lya_edge_rebin3():
    """
    Test that results are None if the alignment of the stack wavelengths
    with Lya doesn't match the rebinning.
    """
    result = add_to_stack(wv_lya_edge_10, flux_3, 1, 3)
    assert result == (None, None), \
        'Should return None if there is a wavelength mismatch.'

def test_wv_lya_center_rebin2():
    """
    Test that results are None if the alignment of the stack wavelengths
    with Lya doesn't match the rebinning.
    """
    result = add_to_stack(wv_lya_center_11, flux_3, 1, 2)
    assert result == (None, None), \
        'Should return None if there is a wavelength mismatch.'

def test_wv_lya_center_11_flux_10_left():
    """
    Test adding a flux array that overlaps the left edge of the 
    wavelength array but not the right.
    """
    indices, flux = add_to_stack(wv_lya_center_11, flux_10, 9, 1)
    expected_indices = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    expected_flux = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0, 0, 0, 0, 0])
    assert np.all(indices == expected_indices), 'Wrong indices.'
    assert np.allclose(flux, expected_flux), 'Wrong flux.'

def test_wv_lya_center_11_flux_10_right():
    """
    Test adding a flux array that overlaps the right edge of the 
    wavelength array but not the left.
    """
    indices, flux = add_to_stack(wv_lya_center_11, flux_10, 0, 1)
    expected_indices = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    expected_flux = np.array([0, 0, 0, 0, 0, -0.4, -0.3, -0.2, -0.1, 0, 0.1])
    assert np.all(indices == expected_indices), 'Wrong indices.'
    assert np.allclose(flux, expected_flux), 'Wrong flux.'

def test_wv_lya_center_11_flux_15():
    """
    Test adding a flux array that is longer than the wavelength array
    and extends beyond both ends.
    """
    indices, flux = add_to_stack(wv_lya_center_11, flux_15, 7, 1)
    expected_indices = np.ones(11)
    expected_flux = np.array([-0.2, -0.1, 0, 0.1, 0.2, 0.3, \
                              0.4, 0.5, 0.6, 0.7, 0.8])
    assert np.all(indices == expected_indices), 'Wrong indices.'
    assert np.allclose(flux, expected_flux), 'Wrong flux.'
