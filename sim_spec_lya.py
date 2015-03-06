#!/usr/bin/env python

import numpy as np
import scipy.constants
#from scipy.ndimage.filters import gaussian_filter1d
import fitsio
import sys
import os
import os.path
import argparse
import datetime
from collections import OrderedDict

module_path = '/home/mmortonson/Code/modules/'
#module_path = '../../../Modules/'

sys.path.append(module_path + 'utils/')
import plotting, file_ops
from distinct_colours import get_distinct

sys.path.append(module_path + 'science/cosmology/')
import cosm_models

"""
* should any of these be included in the settings dictionary?
"""
n_dr9 = 54468 # total number of spectra in DR9 LyaF catalog
n_wv_max = 4684 # maximum number of wavelength bins
log10_wv_min = 3.5496 # log10(lambda/A) for 1st bin
delta_log10_wv_pixel = 1.e-4
dz_specexbin = 1.5e-5

base_dir = '/home/mmortonson/Projects/LyA'
dr9_spec_dir = os.path.join(base_dir, 'BOSS', 'BOSSLyaDR9_spectra')
sightline_dir = os.path.join(base_dir, 'Code', 'data')


# settings
s = OrderedDict()

# first and last quasar spectra to use, starting at 0 (maximum = 54467)
s['quasar index range'] = [0, 0]

"""
* simpler to replace tile start, tile end, and LyaF range 
* with a range of wavelengths to tile?
"""
# tile start = 0: start at lowest observed wavelength 3545A
# tile start = 1: start at rest-frame 1041A, if > 3545A/(1+z_Q)
s['tile start'] = 1
# tile end = 0: end at rest-frame 1185A
# tile end = 1: end at rest-frame 1216A
s['tile end'] = 0
# wavelength range (A) of Lya forest to use
s['LyaF range'] = [1041., 1185.]

# tau_eff(z) model and parameters (Faucher-Giguere et al. 2008, dz=0.1 bins)
# 0: includes metals; 1: Schaye+ (2003) correction; 2: Kirkman+ (2005) corr.
s['FG08 tau_eff model'] = 1
if s['FG08 tau_eff model'] == 0:
    s['tau_eff a'] = -2.734
    s['tau_eff b'] = 3.924
elif s['FG08 tau_eff model'] == 1:
    s['tau_eff a'] = -2.876
    s['tau_eff b'] = 4.094
elif s['FG08 tau_eff model'] == 2:
    s['tau_eff a'] = -2.927
    s['tau_eff b'] = 4.192

# tau_eff measured in simulation
s['simulation tau_eff'] = 0.233

# spectral resolution (0: constant; 1: use wdisp from DR9 spectra)
s['resolution type'] = 1
s['constant R'] = 1800. # used only if resolution type is 0

# width for boxcar smoothing of S/N (in pixels)
s['S/N smoothing width'] = 100

# noise model
# 0: 1-parameter model sigma^2 = k*(f+s)
# 1: 2-param. sigma^2 = k1 + k2*(f+s)
# 2: 1-parameter model with weighted fit
s['noise model'] = 2

# flags for output files
# index 0: S/N list (*_sn_list.dat)
# index 1: local minima list (*_abs_flux.dat), 
# index 2: list of all pixels (*_all_flux.dat)
# index 3: simulated spectrum (*-plate-mjd-fiber.dat)
# index 4: high-resolution spectrum (*-plate-mjd-fiber_hires.dat)
s['output flags'] = [1, 1, 1, 1, 1]



def wv_obs_from_qso_rf(wv, z):
    return 10.**(delta_log10_wv_pixel * \
                 int(np.log10(wv*(1.+z))/ \
                     delta_log10_wv_pixel + 0.5))

def wv_pixel_diff(wv1, wv2):
    return np.floor(np.log10(wv2/wv1)/delta_log10_wv_pixel + 0.5)

def add_tiles(spectra, tiles):
    new_spectra = []
    for tile, spec in zip(tiles, spectra):
        new_spectra.append(np.hstack((spec, tile)))
    return tuple(new_spectra)

def reverse_arrays(arrays):
    new_arrays = []
    for arr in arrays:
        new_arrays.append(np.flipud(arr))
    return tuple(new_arrays)

def pixelize(x1, y1, x2, linlog=0):
    """
    Average array y1 with narrow bins x1 in wide bins x2
    to get pixelized array y2, where x2 bins are spaced evenly 
    in x2 (linlog=0) or log x2 (linlog=1).
    * simplify this using np.cumsum?
    """
    y2 = np.zeros_like(x2)
    a, b = (1., 0.)
    if linlog == 0:
        b = 0.5*(x2[1]-x2[0])
    elif linlog == 1:
        a = np.sqrt(x2[1]/x2[0])
    else:
        print('pixelize: linlog must be 0, 1, or 2')
        sys.exit()
    if x1[0] > a*x2[0]+b or x1[-1] < x2[-1]/a-b:
        print 'pixelize: not enough bins in x1 to cover x2'
        sys.exit()
    j = 0
    while j < len(x1) and x1[j] < x2[0]/a-b:
        j += 1
    for i in range(len(x2)):
        num = 0
        while j < len(x1) and x1[j] < a*x2[i]+b:
            y2[i] += y1[j]
            num += 1
            j += 1
        if num > 0:
            y2[i] = y2[i]/float(num)
    return y2

def boxsmooth(x, width):
    actual_width = 1 + 2*(int(width)/2)
    y = np.zeros_like(x)
    for ic in range(len(x)):
        y[ic] = 0.
        n_good = actual_width
        for i in range(ic-int(width)/2, ic+int(width)/2+1):
            if i <= 0:
                j = 0
            elif i >= len(x)-1:
                j = len(x)-1
            else:
                j = i
            if np.isnan(x[j]) or np.isinf(x[j]):
                n_good -= 1
            else:
                y[ic] += x[j]
        if n_good > 0:
            y[ic] = y[ic]/float(n_good)
        else:
            y[ic] = 0.
    return y



if __name__ == '__main__':

    # seed RNG and skip 1st number (for testing - comparison with simspec2.c)
    print '*** using hard-coded random seed 1356605568 for testing ***'
    np.random.seed(1356605568)
    r_skip = np.random.uniform()

    z_path_length = 0.

    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', help='label to identify run')
    args = parser.parse_args()

    output_dir = os.path.join('..', 'Data', datetime.date.today().isoformat())
    output_file_format = 'sim_spec_lya{0:s}'
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


    """
    * better to hard code these as settings?
    """
    # open simulation box parameter file and get parameter values
    sim_params = np.loadtxt('simspec/simboxparams.txt', unpack=True, \
                            usecols = (0,))
    z_box = sim_params[0]
    L_box = sim_params[1]
    omegam = sim_params[3]
    print '*** setting Omega_r=0 for testing ***'
    lcdm_cosm = cosm_models.lcdm(omegam=omegam, omegagamma=0.)
    H_box = lcdm_cosm.hubble(z_box) / (lcdm_cosm.h * cosm_models.C_HUB_MPC)

    # open ion file and read columns
    name_ion, lam_ion, osc_ion, id_ion = np.loadtxt('simspec/iontable.txt', \
                                             unpack=True, \
                                             dtype='S10,float,float,float')
    wv_lya = lam_ion[0]
    taufac_ion = np.zeros_like(lam_ion)
    taufacref = 1.
    for i, id_num in enumerate(id_ion):
        if id_num-int(id_num) < 0.001:
            taufacref = lam_ion[i]*osc_ion[i]
        taufac_ion[i] = lam_ion[i]*osc_ion[i]/taufacref

    # open DR9 LyaF catalog and read plate, mjd, fiber, z
    catalog = fitsio.FITS(os.path.join(dr9_spec_dir, 'BOSSLyaDR9_cat.fits'))
    catalog_rows = catalog[1].read(columns=['plate', 'mjd', 'fiberid', 'z_vi'])
    if len(catalog_rows) != n_dr9:
        print 'Number of catalog entries ' + len(catalog_rows) + \
              ' does not match expected number ' + n_dr9
        sys.exit()

    # read in flux calibration correction factors
    correction_log10_wv, correction_factor = \
        np.loadtxt(os.path.join(dr9_spec_dir, 'residcorr_v5_4_45.dat'), \
                   unpack=True)
    if np.abs(correction_log10_wv[0]-log10_wv_min) > 1.e-5:
        print 'Wavelength mismatch: ' + \
            'min. log(lambda) is {0:f} '.format(log10_wv_min) + \
            'in sim_spec_lya.py'
        print 'and {0:f} '.format(correction_log10_wv[0]) + \
            'in residcorr_v5_4_45.dat'
        sys.exit()

    # read in noise fit coefficients
    if s['noise model'] == 0:
        noise_model_wv, noise_model_k = \
            np.loadtxt('simspec/noisefit_lyaf_fs_1p.dat', unpack=True, \
                       usecols=(0,1))
    elif s['noise model'] == 1:
        noise_model_wv, noise_model_k1, noise_model_k2 = \
            np.loadtxt('simspec/noisefit_lyaf_fs_2p.dat', unpack=True, \
                       usecols=(0,1,2))
    elif s['noise model'] == 2:
        noise_model_wv, noise_model_k = \
            np.loadtxt('simspec/noisefit_lyaf_fs_1pw.dat', unpack=True, \
                       usecols=(0,1))

    # set up output files that use all spectra
    if s['output flags'][0]:
        writer_sn = file_ops.open_with_path(\
            output_format.format('_sn_list', 'dat'), 'w')
    if s['output flags'][1]:
        writer_local_min_flux = file_ops.open_with_path(\
            output_format.format('_abs_flux', 'dat'), 'w')
    if s['output flags'][2]:
        writer_all_flux = file_ops.open_with_path(\
            output_format.format('_all_flux', 'dat'), 'w')


    # ************** Start loop over quasar spectra ***************

    for iq in range(s['quasar index range'][0], \
                    s['quasar index range'][1]+1):

        zq = catalog_rows[iq][3]
        print '{0:7d}: z_Q = {1:5.3f}'.format(iq+1, zq)

        plate_mjd_fiber = '-{0:04d}-{1:05d}-{2:04d}'.format( \
            catalog_rows[iq][0], catalog_rows[iq][1], catalog_rows[iq][2])
        if s['output flags'][3]:
            writer_spec = file_ops.open_with_path(\
                output_format.format(plate_mjd_fiber, 'dat'), 'w')
        if s['output flags'][4]:
            writer_spec_hires = file_ops.open_with_path(\
                output_format.format(plate_mjd_fiber + '_hires', 'dat'), 'w')

        qso_spectrum = fitsio.FITS(os.path.join(\
            dr9_spec_dir, '{0:04d}'.format(catalog_rows[iq][0]), \
            'speclya' + plate_mjd_fiber + '.fits'))

        # read columns individually (otherwise the order is wrong)
        qso_log10_wv = np.array(zip(*qso_spectrum[1].read(\
                    columns=['loglam']))[0])
        qso_continuum = np.array(zip(*qso_spectrum[1].read(\
                    columns=['cont']))[0])
        qso_sky = np.array(zip(*qso_spectrum[1].read(\
                    columns=['sky']))[0])
        qso_dispersion = np.array(zip(*qso_spectrum[1].read(\
                    columns=['wdisp']))[0])

        qso_wv = 10.**qso_log10_wv
        # wavelength-dependent resolution
        qso_resolution = 1. / (qso_dispersion * np.sqrt(8.*np.log(2.)) * \
                               np.log(10.) * delta_log10_wv_pixel)
        qso_spectrum.close()

        if len(qso_log10_wv) > n_wv_max:
            print 'Number of wavelengths in spectrum ' + \
                '({0:d}) exceeds n_wv_max ({1:d}).'.format(\
                len(qso_log10_wv), n_wv_max)
            sys.exit()

        # set up mask (0: not masked, 1: masked)
        qso_mask = np.zeros_like(qso_wv)

        # divide sky flux by correction factor
        """
        * possible to do this without the loop?
        """
        for i in range(len(qso_sky)):
            if np.isnan(qso_sky[i]):
                qso_sky[i] = 0.
                qso_mask[i] = 1.
            index_offset = np.floor((qso_log10_wv[0]-log10_wv_min) / \
                                    delta_log10_wv_pixel + 0.5)
            if i+index_offset < len(correction_log10_wv):
                qso_sky[i] /= correction_factor[i+index_offset]

        # shift from quasar rest-frame to observed wavelengths
        wv_obs_lyaf_range = []
        for wv in s['LyaF range']:
            wv_obs = wv_obs_from_qso_rf(wv, zq)
            wv_obs_lyaf_range.append(wv_obs)
        wv_obs_lya = wv_obs_from_qso_rf(wv_lya, zq)

        # determine number of simulation sightlines needed to tile spectrum
        tile_wv_min = qso_wv[0] / (1.+zq)
        if (s['tile start'] == 1) and (tile_wv_min < s['LyaF range'][0]):
            tile_wv_min = s['LyaF range'][0]
        if s['tile end'] == 0:
            tile_wv_max = s['LyaF range'][1]
        else:
            tile_wv_max = wv_lya
        tile_z_min = tile_wv_min*(1.+zq)/wv_lya - 1.
        tile_z_max = tile_wv_max*(1.+zq)/wv_lya - 1.
        z_path_length += tile_z_max - tile_z_min
        n_tile = 1 + int( \
            (lcdm_cosm.dist_com(tile_z_max)-lcdm_cosm.dist_com(tile_z_min))* \
            lcdm_cosm.h * cosm_models.C_HUB_MPC / L_box
            )
        if n_tile <= 0:
            print 'Number of tiles <= 0.'
            print 'Check wavelength range and quasar redshift range.'
            sys.exit()

        # *** Start loop over sightlines used to tile spectrum ***

        prev_tile_z_min = 0.

        z_sim = np.array([])
        wv_sim = np.array([])
        tau_sim = np.array([])
        flux_sim = np.array([])

        for it in range(n_tile):

            # choose a random sightline and read tau_Lya(z)
            sightline_id = int(400.*np.random.uniform()) + 1
            print '*** drawing sightline from 400 with self-shielding ***'
            sightline_filename = os.path.join(sightline_dir, \
                'specaim.c100n576vzw15.z01ss.{0:04d}'.format(sightline_id))
            tau_tile = np.loadtxt(sightline_filename, unpack=True, \
                                      usecols=(7,))

            # shift from simulation redshifts to tile redshifts
            z_tile = np.zeros_like(tau_tile)
            if it == 0:
                z_tile[0] = tile_z_max
            else:
                z_tile[0] = prev_tile_z_min - dz_specexbin / H_box * \
                    lcdm_cosm.hubble(prev_tile_z_min) / \
                    (lcdm_cosm.h * cosm_models.C_HUB_MPC)
            for i in range(1, len(z_tile)):
                z_tile[i] = z_tile[i-1] - dz_specexbin / H_box * \
                    lcdm_cosm.hubble(z_tile[i-1]) / \
                    (lcdm_cosm.h * cosm_models.C_HUB_MPC)
            prev_tile_z_min = z_tile[-1]

            # compute wavelengths (observed)
            wv_tile = wv_lya*(1.+z_tile)

            # shift tau(z) by a random displacement along the sightline
	    # (using periodic boundary conditions)
            # and rescale from simulation redshift using tau_eff(z)
            tau_tile =  10.**s['tau_eff a']*(1.+z_tile)**s['tau_eff b'] / \
                s['simulation tau_eff'] * \
                np.roll(tau_tile, \
                            -int(np.random.uniform()*len(tau_tile)))

	    # compute transmitted flux fraction F = exp(-tau)
            flux_tile = np.exp(-tau_tile)

            # add the tile to the spectrum
            z_sim, wv_sim, tau_sim, flux_sim = \
                add_tiles((z_sim, wv_sim, tau_sim, flux_sim), \
                          (z_tile, wv_tile, tau_tile, flux_tile))

        # reverse spectrum arrays (in order of increasing z and wavelength)
        z_sim, wv_sim, tau_sim, flux_sim = \
            reverse_arrays((z_sim, wv_sim, tau_sim, flux_sim))

        # output full (simulation) resolution spectrum
        if s['output flags'][4]:
            writer_spec_hires.write('{0:.3f}\n'.format(zq))
            for i in range(len(wv_sim)):
                writer_spec_hires.write(\
                    '{0:8.4f} {1:12.5e}\n'.format(np.log10(wv_sim[i]), \
                                                      flux_sim[i]))
            writer_spec_hires.close()


        # smooth spectrum with a Gaussian (with varying width)
        """
        * speed up later by smoothing whole spectrum by 
        * several different Gaussians, then interpolating?
        """
        flux_smoothed = np.zeros_like(flux_sim)
        dwv_times_flux = np.gradient(wv_sim)*flux_sim
        for i, x in enumerate(wv_sim):
            if s['resolution type'] == 0:
                R = s['constant R']
            elif s['resolution type'] == 1:
                R = np.interp(x, qso_wv, qso_resolution)
            #sigma = (1.+z_sim[i]) / dz_specexbin / (R * np.sqrt(8.*np.log(2.)))
            #flux_smoothed[i] = gaussian_filter1d(flux_sim, sigma, mode='constant')[i]
            var_wv = (x/R)**2 / (8.*np.log(2.))
            gaussian = np.exp(-0.5*(wv_sim-x)**2/var_wv)/ \
                np.sqrt(2.*np.pi*var_wv)
            flux_smoothed[i] = np.sum(gaussian*dwv_times_flux)


        # average flux in constant Delta log10(wavelength) pixels
        wv_min = wv_obs_from_qso_rf(tile_wv_min, zq)
        wv_max = wv_obs_from_qso_rf(tile_wv_max, zq)
        wv = np.logspace(np.log10(wv_min), np.log10(wv_max), \
                         num=1+np.log10(wv_max/wv_min)/delta_log10_wv_pixel)
        flux = pixelize(wv_sim, flux_smoothed, wv, linlog=1)


        # add noise in Lya forest
        #
        # determine pixel indices corresponding to 
        # lamlyafmin (ip0) and lamlyafmax (ipn)
        # (if tilestart==1 and tileend==0, ip0=0 and ipn=npix)
        """
        * simplify this and put into function(s)
        """
        noise = np.zeros_like(wv)
        sigma_f = np.zeros_like(wv)
        sn_f = np.zeros_like(wv)
        sigma_flux = np.zeros_like(wv)
        mask = np.zeros_like(wv)
        ip0 = 0
        ipn = len(wv)
        if qso_wv[0] < wv_obs_lyaf_range[0]:
            if s['tile end'] == 1:
                ipn = wv_pixel_diff(*wv_obs_lyaf_range)
            if s['tile start'] == 0:
                ip0 = wv_pixel_diff(wv[0], wv_obs_lyaf_range[0])
        else:
            if s['tile end'] == 1:
                ipn = wv_pixel_diff(qso_wv[0], wv_obs_lyaf_range[1])
            if s['tile start'] == 0:
                ip0 = wv_pixel_diff(wv[0], qso_wv[0])
        ipn += ip0
        # compute pixel offsets for observed spectrum (ipd)
        # and for noise fitting function (ipd2)
        # - ipd2 should only be used to set i_noise
        ipd = wv_pixel_diff(qso_wv[0], wv_obs_lyaf_range[0])
        if ipd < 0:
            ipd = 0
        if ipd == 0:
            ipd2 = wv_pixel_diff(noise_model_wv[0], qso_wv[0])
        else:
            ipd2 = wv_pixel_diff(noise_model_wv[0], wv_obs_lyaf_range[0])

        for ip in range(ip0, ipn):
            noise_out_of_range = 0
            i_noise = ip-ip0+ipd2
            if i_noise < 0:
                noise_out_of_range = 1
                i_noise = 0
            elif i_noise >= len(noise_model_wv):
                noise_out_of_range = 1
                i_noise = len(noise_model_wv)-1
            if qso_continuum[ip-ip0+ipd] <= 0.:
                noise[ip] = 1.e10
                mask[ip] = 1
            else:
                if s['noise model'] == 0 or s['noise model'] == 2:
                    sigma_f[ip] = np.sqrt(\
                        (flux[ip]*qso_continuum[ip-ip0+ipd] \
                             +qso_sky[ip-ip0+ipd])/ \
                            noise_model_k[i_noise])
                elif s['noise model'] == 1:
                    sigma_f[ip] = np.sqrt(\
                        (flux[ip]*qso_continuum[ip-ip0+ipd] \
                             +qso_sky[ip-ip0+ipd]-noise_model_k1[i_noise]) / \
                            noise_model_k2[i_noise])
                noise[ip] = np.random.normal(scale=sigma_f[ip]) / \
                    qso_continuum[ip-ip0+ipd]
                if qso_mask[ip-ip0+ipd] == 1 or np.isnan(flux[ip]):
                    mask[ip] = 1
            sn_f[ip] = flux[ip]*qso_continuum[ip-ip0+ipd]/sigma_f[ip]
            sigma_flux[ip] = sigma_f[ip]/qso_continuum[ip-ip0+ipd]
            # check that wavelengths are aligned correctly
            if np.abs(np.log10(wv[ip]/qso_wv[ip-ip0+ipd])) > \
                    delta_log10_wv_pixel:
                print 'Wavelength vectors misaligned: wv and qso_wv'
                print 'wv[{0:d}]={1:f}, qso_wv[{2:d}]={3:f}\n'.format(\
                    ip, wv[ip], ip-ip0+ipd, qso_wv[ip-ip0+ipd])
                sys.exit()
            if np.abs(np.log10(wv[ip]/noise_model_wv[i_noise])) > \
                    delta_log10_wv_pixel:
                print 'Wavelength vectors misaligned: wv and noise_model_wv'
                print 'wv[{0:d}]={1:f}, noise_model_wv[{2:d}]={3:f}\n'.format(\
                    ip, wv[ip], i_noise, noise_model_wv[i_noise])
                sys.exit()

        # median S/N per pixel of measured flux (F*continuum)
        sn_f_median = np.median(sn_f)

        # boxcar-smoothed S/N of F
        flux_boxsmooth = boxsmooth(flux, s['S/N smoothing width'])
        sigma_flux_boxsmooth = boxsmooth(sigma_flux, s['S/N smoothing width'])
        sn_boxsmooth = flux_boxsmooth/sigma_flux_boxsmooth


        # output final spectrum
        if s['output flags'][3]:
            writer_spec.write('{0:.3f} {1:.3f}\n'.format(zq, sn_f_median))
            for i in range(len(wv)):
                fmt_str = '{0:8.4f} {1:12.5e} {2:12.5e} ' + \
                    '{3:12.5e} {4:12.5e} {5:12.5e} {6:12.5e}\n'
                writer_spec.write(\
                    fmt_str.format(np.log10(wv[i]), flux[i], noise[i], \
                                   sigma_flux[i], qso_continuum[i-ip0+ipd], \
                                   qso_sky[i-ip0+ipd], sn_boxsmooth[i]))
            writer_spec.close()

        # output median S/N
        if s['output flags'][0]:
            writer_sn.write('{0:7d} {1:8.4f}\n'.format(iq+1, sn_f_median))

        # output flux and noise for local minima and for all pixels
        if s['output flags'][1] or s['output flags'][2]:
            for i in range(1, len(wv)-1):
                if s['output flags'][2]:
                    writer_all_flux.write(\
                        '{0:10.3e} {1:10.3e} {2:8.4f} {3:8.4f} {4:f}\n'.format(\
                            flux[i], noise[i], sn_f_median, sn_boxsmooth[i], \
                                mask[i]))
                    if not mask[i]:
                        if s['output flags'][1]:
                            if flux[i]+noise[i] < flux[i-1]+noise[i-1] and \
                               flux[i]+noise[i] < flux[i+1]+noise[i+1]:
                                writer_local_min_flux.write(\
                                    '{0:10.3e} {1:10.3e} {2:8.4f} {3:8.4f}\n'.format(\
                                        flux[i], noise[i], sn_f_median, sn_boxsmooth[i]))



    catalog.close()

    if s['output flags'][0]:
        writer_sn.close()
    if s['output flags'][1]:
        writer_local_min_flux.close()
    if s['output flags'][2]:
        writer_all_flux.close()
