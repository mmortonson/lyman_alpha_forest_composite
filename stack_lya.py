#!/usr/bin/env python

import numpy as np
import scipy.constants
from scipy.stats import scoreatpercentile
import sys
import os
import os.path
import re
import argparse
import datetime
from collections import OrderedDict

module_path = '/home/mmortonson/Code/modules/'
#module_path = '../../../Modules/'

sys.path.append(module_path + 'utils/')
import plotting, file_ops
from distinct_colours import get_distinct

wavelength_lya = 1215.6701


# settings
s = OrderedDict()
s['stack wavelength range'] = [1200., 1230.]
# rebinning: how many adjacent pixels to average flux in
#     1 to select using original pixels, 
#     2 to select using flux averaged in pixel pairs, etc.
s['pixel selection rebinning'] = 2
s['S/N threshold'] = 3.
s['flux bins'] = [\
                      (-0.05, 0.05), \
                      (0.05, 0.15), \
                      (0.15, 0.25), \
                      (0.25, 0.35), \
                      (0.35, 0.45) \
                     ]
s['absorber redshift range'] = [2.4, 3.1]
# pixels are spaced with Delta log lambda = 10^{-4} (about 69 km/s)
s['pixel delta log10 wavelength'] = 1.e-4


def log_wavelength_grid(extent, delta_log10_wavelength, \
                        reference_wavelength, reference_type='center'):
    """
    Create a grid of log-spaced wavelength pixels over a specified
    approximate range (extent) in which a given reference wavelength 
    is either at the center of a pixel or at an edge between pixels.
    Returns an array of wavelengths at the pixel centers.
    """
    # compute number of pixels on either side of reference wavelength
    n_pixels_blue = int(np.log10(reference_wavelength/extent[0]) / \
                        delta_log10_wavelength)
    n_pixels_red = int(np.log10(extent[1]/reference_wavelength) / \
                        delta_log10_wavelength)
    n_pixels = n_pixels_blue + n_pixels_red + 1
    # adjust pixel count and offset from reference if reference is 
    # at a pixel edge instead of the center
    if reference_type == 'edge':
        n_pixels -= 1
        pixel_offset = 0.5
    elif reference_type == 'center':
        pixel_offset = 0
    else:
        print 'log_wavelength_grid: valid reference_type values ' + \
                  'are "center" and "edge"'
        return None
    log10_wavelength_min = np.log10(reference_wavelength) - \
                               (n_pixels_blue-pixel_offset)* \
                               delta_log10_wavelength
    log10_wavelength_max = np.log10(reference_wavelength) + \
                               (n_pixels_red-pixel_offset)* \
                               delta_log10_wavelength
    return np.logspace(log10_wavelength_min, log10_wavelength_max, \
                       num=n_pixels)

def rebin_flux(flux, rebin_number):
    """
    Average the flux in N adjacent bins, where N=rebin_number.
    Each element of the input array is averaged with the next
    N-1 elements. Returns an array the same size as the input flux array.

    Notes
    -----
    If the length of flux is not evenly divisible by N, some 
    of the elements at the end of the flux array will be averaged 
    with elements at the start of the flux array - these will be
    in the last M<N elements of the returned array and should
    probably not be used.
    """
    rebinned_flux = np.zeros_like(flux)
    for i in range(rebin_number):
        rebinned_flux += np.roll(flux, -i)
    rebinned_flux /= float(rebin_number)
    return rebinned_flux

def select_pixels(wavelength, flux, signal_to_noise, rebinning=1, \
                  flux_range=[0., 1.], sn_range=[0., np.inf], \
                  z_range=[0., np.inf]):
    """
    Select pixels that satisfy several criteria:
        - If flux is rebinned by N, only select every Nth pixel.
        - Select pixels where the flux, S/N, and/or absorber
          redshift fall within specified ranges.
    Returns an array with the indices of the selected pixels.
    """
    rebinned_flux = rebin_flux(flux, rebinning)
    z_lya = wavelength/wavelength_lya - 1.
    mask_rebinned_pixels = (np.arange(len(flux)) % rebinning == 0) & \
                           (len(flux) - np.arange(len(flux)) >= rebinning)
    mask_flux = (flux_range[0] <= rebinned_flux) & \
                (rebinned_flux < flux_range[1])
    mask_sn = (sn_range[0] <= signal_to_noise) & \
              (signal_to_noise < sn_range[1])
    mask_z = (z_range[0] <= z_lya) & (z_lya < z_range[1])
    pixels = np.where(mask_rebinned_pixels & mask_flux & mask_sn & mask_z)[0]
    return pixels

def add_to_stack(stack_wavelength, flux, abs_pixel, rebinning):
    """
    For a Lya absorber selected at index abs_pixel in the 
    (possibly rebinned) flux array, find the indices of the pixels
    in the stack to add flux to and determine how much flux
    to add to each one.

    abs_pixel is the first index of the N pixels associated with
    the selected absorber, where rebinning=N. In the stack, this
    group of pixels is centered on the Lya rest wavelength.

    Note that even if absorbers are selected with rebinned flux,
    the flux added to the stack should always be the original, 
    non-rebinned array.
    """
    # find the pixel in the stack wavelength array to match with
    # the absorber pixel
    delta_log10_wavelength = np.diff(np.log10(stack_wavelength))[0]
    lya_pixel = np.log10(wavelength_lya/stack_wavelength[0]) / \
                    delta_log10_wavelength - float(rebinning-1)/2.
    # check that lya_pixel is consistent with rebinning
    # (should be close to integer, not half-integer)
    if int(np.round(10.*lya_pixel)) != 10*int(np.round(lya_pixel)):
        print 'Rebinning in add_to_stack inconsistent with ' + \
            'stack wavelength array.'
        return (None, None)
    # find the required range of indices in the flux array
    left_index = abs_pixel - int(np.round(lya_pixel))
    right_index = left_index + len(stack_wavelength) - 1
    # pad ends with zeros if necessary
    if left_index < 0 and right_index >= len(flux):
        new_pixels = np.hstack((np.zeros(-left_index), np.ones_like(flux), \
                                np.zeros(right_index-len(flux)+1)))
        new_flux = np.hstack((np.zeros(-left_index), flux, \
                              np.zeros(right_index-len(flux)+1)))
    elif left_index < 0:
        new_pixels = np.hstack((np.zeros(-left_index), \
                                np.ones(right_index+1)))
        new_flux = np.hstack((np.zeros(-left_index), flux[:right_index+1]))
    elif right_index >= len(flux):
        new_pixels = np.hstack((np.ones(len(flux)-left_index), \
                                np.zeros(right_index-len(flux)+1)))
        new_flux = np.hstack((flux[left_index:], \
                              np.zeros(right_index-len(flux)+1)))
    else:
        new_pixels = np.ones(right_index-left_index+1)
        new_flux = flux[left_index:right_index+1]
    return (new_pixels, new_flux)

def get_column_without_nans(array, n_col):
    """
    Return column number n_col from a 2D numpy array 
    with NaN values removed.
    """
    column = array[:,n_col]
    return column[~np.isnan(column)]

def stack_stats(stack_array, outlier_fraction=0.):
    """
    Given an MxN array, return N-D arrays with:
        - the number of valid (non NaN) values in each column
        - the mean of the values in each column (with outliers clipped)
        - the median of the values in each column
    """
    sh = stack_array.shape
    n_col = stack_array.shape[1]
    n_valid = np.zeros(n_col)
    mean = np.zeros(n_col)
    median = np.zeros(n_col)
    for i in range(n_col):
        flux_without_nans = get_column_without_nans(stack_array, i)
        cumulative_fraction = (np.arange(len(flux_without_nans))+0.5) / \
                                  len(flux_without_nans)
        outlier_mask = (cumulative_fraction > outlier_fraction) & \
                           (cumulative_fraction < 1.-outlier_fraction)
        flux_clipped = flux_without_nans[outlier_mask]
        n_valid[i] = len(flux_without_nans)
        if n_valid[i] > 0:
            mean[i] = np.mean(flux_clipped)
            median[i] = np.median(flux_without_nans)
        else:
            mean[i] = 0.
            median[i] = 0.
    return (n_valid, mean, median)

def stack_bootstrap_errors(stack_array, n_bootstrap, \
                               percentiles=(15.85, 84.15), \
                               outlier_fraction=0.):
    """
    Estimate errors by calling stack_stats on n_bootstrap
    samples of the input stack_array. Finds limits for the 
    mean and median at the specified percentiles (default encloses 
    68.3% probability, centered on the median). Returns a tuple
    with all limits for the mean followed by all limits for 
    the median, with each in the order given by percentiles.
    """
    (n_row, n_col) = stack_array.shape
    all_means = np.nan * np.ones(n_col)
    all_medians = np.nan * np.ones(n_col)
    for i in range(n_bootstrap):
        sample_indices = np.random.choice(n_row, n_row)
        sample_stack = stack_array[sample_indices,:]
        n_abs, mean, median = stack_stats(sample_stack, outlier_fraction)
        all_means = np.vstack((all_means, mean))
        all_medians = np.vstack((all_medians, median))
    mean_limits = []
    median_limits = []
    for p in percentiles:
        mean_limits.append(np.zeros(n_col))
        median_limits.append(np.zeros(n_col))
    for c in range(n_col):
        mean_col = get_column_without_nans(all_means, c)
        median_col = get_column_without_nans(all_medians, c)
        for i, p in enumerate(percentiles):
            mean_limits[i][c] = scoreatpercentile(mean_col, p)
            median_limits[i][c] = scoreatpercentile(median_col, p)
    return tuple(mean_limits) + tuple(median_limits)

def delta_log10_wavelength_to_delta_v_km_s(delta_log10_wavelength):
    """
    Convert log10(wavelength) differences to velocity differences
    in km/s.
    """
    speed_of_light_km_s = scipy.constants.c * 1.e-3
    return delta_log10_wavelength * np.log(10.) * speed_of_light_km_s

def plot_spectrum(wavelength, flux, flux_noiseless, signal_to_noise, selected_pixels=None):
    spectrum_plot = plotting.Figure(panel_grid=(2,1), fig_size=(10, 6))
    spectrum_plot.add_data(wavelength, flux, style='-', color='c')
    if selected_pixels != None:
        spectrum_plot.add_data(wavelength[selected_pixels], flux[selected_pixels], style='o', color='r')
    spectrum_plot.add_data(wavelength, flux_noiseless, style='-')
    spectrum_plot.add_data(wavelength, signal_to_noise, panel=(1, 0), style='-')
    spectrum_plot.add_labels('x', ['', '$\lambda\,[\AA]$'])
    spectrum_plot.add_labels('y', ['$F$', '$S/N$'])
    spectrum_plot.show_figure()


if __name__ == '__main__':

    # get data directories from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', nargs='+', \
                        help='one or more directories containing spectra')
    parser.add_argument('-l', '--label', help='label to identify run')
    parser.add_argument('--outlier_fraction', type=float, default=0., \
                        help='fraction of low and high flux outliers to ' + \
                             'exclude when computing mean')
    parser.add_argument('--n_bootstrap', type=int, default=0, \
                        help='number of bootstrap samples to use to ' + \
                             'estimate errors (0 to skip error estimation)')
    parser.add_argument('--noiseless', action='store_true', \
                        help='compute stacked spectrum without noise')
    args = parser.parse_args()
    s['spectra directories'] = args.dir
    s['outlier fraction'] = args.outlier_fraction
    s['bootstrap samples'] = args.n_bootstrap
    if args.noiseless:
        s['noiseless'] = True
    else:
        s['noiseless'] = False
    if not (0. <= s['outlier fraction'] <= 1.):
        print 'outlier_fraction must be between 0 and 1.'
        sys.exit()

    output_dir = os.path.join('..', 'Data', datetime.date.today().isoformat())
    output_file_format = 'stack_lya_stats{0:s}'
    if args.label:
        output_file_format += '_' + args.label
    output_file_format += '.{1:s}'
    output_format = os.path.join(output_dir, output_file_format)

    # output settings to log file
    writer = file_ops.open_with_path(output_format.format('', 'log'), 'w')
    writer.write('{0:s}\n'.format(os.path.basename(__file__)))
    for setting in s:
        writer.write(setting + ': ' + str(s[setting]) + '\n')
    writer.close()

    # if pixel selection rebinning is odd, Lya wavelength is one of the pixels
    # otherwise, Lya is between two pixels
    if s['pixel selection rebinning'] % 2 == 1:
        lya_pixel = 'center'
    else:
        lya_pixel = 'edge'

    # set up initial arrays for stack
    stack_wavelength = log_wavelength_grid(s['stack wavelength range'], \
                                           s['pixel delta log10 wavelength'], \
                                           wavelength_lya, \
                                           reference_type=lya_pixel)
    stack_delta_v_km_s = np.vectorize(delta_log10_wavelength_to_delta_v_km_s)\
                             (np.log10(stack_wavelength/wavelength_lya))

    stack_lya = {}
    # start with NaN's to set array shape (these will be ignored later)
    for flux_bin in s['flux bins']:
        stack_lya[flux_bin] = np.nan * np.ones_like(stack_wavelength)

    # loop over spectrum files
    # assume all .dat files in the given directories are spectra

    n_spectra = 0

    for d in s['spectra directories']:
        for entry in os.listdir(d):
            if os.path.isfile(os.path.join(d, entry)) and \
                   entry.split('.')[-1] == 'dat':
                n_spectra += 1
                log10_wavelength, flux_noiseless, noise, signal_to_noise = \
                    np.loadtxt(os.path.join(d, entry), \
                               skiprows=1, usecols=(0, 1, 2, 6), unpack=True)
                wavelength = 10.**log10_wavelength
                if s['noiseless']:
                    flux = flux_noiseless
                else:
                    flux = flux_noiseless + noise

                for flux_bin in s['flux bins']:
                    # select pixels to stack
                    selected_pixels = \
                        select_pixels(\
                            wavelength, flux, signal_to_noise, \
                            rebinning=s['pixel selection rebinning'], \
                            flux_range=flux_bin, \
                            sn_range=[s['S/N threshold'], np.inf], \
                            z_range=s['absorber redshift range'])

                    # loop over selected pixels
                    for p in selected_pixels:
                        new_stack_pixels, new_stack_flux = add_to_stack(\
                            stack_wavelength, flux, p, \
                            s['pixel selection rebinning'])
                        new_stack_row = np.where(new_stack_pixels==1, \
                                                 new_stack_flux, np.nan)
                        stack_lya[flux_bin] = np.vstack((stack_lya[flux_bin], \
                                                         new_stack_row))

    # compute, plot, and output mean and median spectra
    stack_plot = plotting.Figure(panel_grid=(3, 1), fig_size=(10, 10))
    max_n_abs = 0
    flux_bin_colors = get_distinct(len(s['flux bins']))
    #flux_bin_colors = plotting.plt.cm.Set1(np.linspace(0, 1, \
    #                                           len(s['flux bins'])))
    for flux_bin, color in zip(s['flux bins'], flux_bin_colors):
        bin_label = '_' + str(flux_bin[0]) + '_' + str(flux_bin[1])
        writer = file_ops.open_with_path(output_format.format(bin_label, \
                                                              'dat'), 'w')
        header = '# ' + str(flux_bin[0]) + ' < F < ' + str(flux_bin[1]) + '\n'
        header += '# See ' + output_file_format.format('', 'log') + \
                  ' for full list of settings.\n'
        header += '#\n#{0:11s} {1:12s}'.format(' wvl. (Ang)', 'N_abs')
        if s['bootstrap samples'] > 0:
            header += ' {0:13s} {1:13s} {2:13s} {3:13s} {4:13s} {5:13s}\n'\
                .format(' mean', ' 68.3% lower', ' 68.3% upper', \
                        ' median', ' 68.3% lower', ' 68.3% upper')
        else:
            header += ' {0:13s} {1:13s}\n'.format(' mean', ' median')
        writer.write(header)
        n_abs, mean, median = stack_stats(stack_lya[flux_bin], \
                                              s['outlier fraction'])
        max_n_abs = np.max([max_n_abs, np.max(n_abs)])
        if s['bootstrap samples'] > 0:
            mean_lower, mean_upper, median_lower, median_upper = \
                stack_bootstrap_errors(stack_lya[flux_bin], \
                                           s['bootstrap samples'], \
                                           outlier_fraction=\
                                               s['outlier fraction'])
            for limit in (mean_lower, mean_upper):
                stack_plot.add_data(stack_wavelength, limit, \
                                    panel=(0, 0), style=':', color=color)
            for limit in (median_lower, median_upper):
                stack_plot.add_data(stack_wavelength, limit, \
                                    panel=(1, 0), style=':', color=color)
        stack_plot.add_data(stack_wavelength, mean, \
                            panel=(0, 0), style='-', color=color, \
                            label=str(flux_bin))
        stack_plot.add_data(stack_wavelength, median, \
                            panel=(1, 0), style='-', color=color)
        stack_plot.add_data(stack_wavelength, n_abs, \
                            panel=(2, 0), style='-', color=color)
        for i, wv in enumerate(stack_wavelength):
            output_row = '{0:e} {1:e}'.format(wv, n_abs[i])
            if s['bootstrap samples'] > 0:
                output_row += ' {0: e} {1: e} {2: e}'.format(\
                    mean[i], mean_lower[i], mean_upper[i]) + \
                    ' {0: e} {1: e} {2: e}'.format(\
                    median[i], median_lower[i], median_upper[i])
            else:
                output_row += ' {0: e} {1: e}'.format(mean[i], median[i])
            writer.write(output_row + '\n')
        writer.close()
    stack_plot.set_limits('y', [0, 1], panel=(0, 0))
    stack_plot.set_limits('y', [0, 1], panel=(1, 0))
    stack_plot.set_limits('y', [0, 1.1*max_n_abs], panel=(2, 0))
    stack_plot.add_labels('x', ['', '', '$\lambda\,[\AA]$'])
    stack_plot.add_labels('y', ['${\\rm mean}\;F$', \
                                '${\\rm median}\;F$', '$N_{\\rm abs}$'])
    stack_plot.show_figure(legend_location=3)




