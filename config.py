NB_PROF_SLICE = 3000 # number of profiles in a slice to process at a time
NB_PROF_OVERLAP = 510 # number of profiles shifted in next slice to avoid edge effects
                      # This number needs to be large enough in order to detect widespread very thin pattern
                      # Need to be a multiple of 15 for 2D-McDA-PSC
NB_PROF_EDGE = int(NB_PROF_OVERLAP/2) # number of profiles at the edge (in the overlap region) not overwritten

FLAG_NOTHING         = 0
FLAG_DETECT          = 1
FLAG_MAYBE           = 255
FLAG_SURFACE         = 254
FLAG_LIKELY_ARTIFACT = 253
FLAG_FA              = 252
FLAG_AFA             = 251
FLAG_SMALL_STRIPS    = 250
FLAG_SPIKES          = 249
FLAG_LOW_CONFIDENCE  = 248


# def get_nb_pixels(wl):
#     """Input: wl = wavelength of the lidar channel
#        Output: nb_pixels = number of original resolution lidar "pixels"
#                            (bins) averaged together at each altitude
#                            level at the wavelength channel wl"""
#
#     nb_pixels = np.ones(583)*-9999.
#
#     if wl==532:
#         nb_pixels[0:33] = 15*20 # horizontal × vertical (see ATBD)
#         nb_pixels[33:88] = 5*12
#         nb_pixels[88:288] = 3*4
#         nb_pixels[288:578] = 1*2
#         nb_pixels[578:583] = 1*20
#
#
#     elif wl==1064:
#         nb_pixels[0:33] = 15*20 # no values here at 1064 nm
#         nb_pixels[33:88] = 5*12
#         nb_pixels[88:288] = 3*4
#         nb_pixels[288:578] = 1*4
#         nb_pixels[578:583] = 1*20
#
#     else:
#         sys.exit('Unrecognized wavelength: %d; use 532 or 1064 instead\n\n' %
#                  wl)
#
#     return nb_pixels
#
#
# def get_fcorr(wl):
#     """Input: wl = wavelength of the lidar channel
#        Output: fcorr = correction function at each altitude level from
#                        Table 2 of Liu (2011), updated values sent by M.
#                        Vaughan"""
#
#     fcorr = np.ones((583, 11)) # fcorr[bin index range, Nshift]
#     fcorr[  0: 33] = [1.596, 1.448, 1.322, 1.224, 1.161, 1.140, 1.161,
#                       1.224, 1.322, 1.448, 1.596]
#     fcorr[ 33: 88] = [1.573, 1.345, 1.188, 1.131, 1.188, 1.345, 1.573,
#                       1.345, 1.188, 1.130, 1.188]
#     fcorr[ 88:288] = [1.451, 1.080, 1.451, 1.080, 1.451, 1.080, 1.451,
#                       1.080, 1.451, 1.080, 1.451]
#     if wl==532:
#         fcorr[288:578] = [1.269, 1.269, 1.269, 1.269, 1.269, 1.269, 1.269,
#                           1.269, 1.269, 1.269, 1.269]
#     elif wl==1064:
#         fcorr[288:578] = [1.451, 1.451, 1.451, 1.451, 1.451, 1.451, 1.451,
#                           1.451, 1.451, 1.451, 1.451]
#     else:
#         sys.exit('Unrecognized wavelength: %d; use 532 or 1064 instead\n\n' %
#                  wl)
#     fcorr[578:583] = [1.596, 1.448, 1.322, 1.224, 1.161, 1.140, 1.161,
#                       1.224, 1.322, 1.448, 1.596]
#
#     return fcorr

class SurfaceDetectionParameters():
    def __init__(self, channel):
        self.offset_dem_water = 3 # +-90 m
        self.offset_dem_perm_snow = 17 # +-510 m
        self.offset_dem_other = 5 # +-150 m
        self.offset_dem_false_positive = 1 # 1: +-30m, None: not used
        self.coef_nb_std = 5 # nb of background noise std for detection
    
        # Determine minimum bins difference between i_min and i_max
        if (channel == '532_par') | (channel == '532_per'):
            self.N = 2
        elif channel == '1064':
            self.N = 4
        else:
            raise Exception(f"Unrecognized channel: {channel}")


class FeatureDetectionParameters():
    def __init__(self, channel):
        self.S_liquid = 10 # eff. lidar ratio for attenuation (at least one part above -38 °C)
        self.S_ice = 18 # effective lidar ratio for attenuation (all below -38 °C)
        self.temp_ice_liquid = -38 # transition temperature to determine the lidar ratio
        # twoway_transmittance_limit = 0.2 # don't correct more that this value of transmittance
        self.twoway_transmittance_limit = 1.0 # no transmittance correction if 1.0
        self.mult_scatt = 0.7 # multiple scattering factor
    
        # Determine extent below very high echo to flag as low confidence
        self.nb_bins_PMT_artifact = 20 # 20 × 30 m = 600 m
    
        # Determine params for AFA flag
        if channel == '532_par':
            self.weak_signal_ratio = 0.3
            self.weak_signal_ratio_threshold = 0.1
        elif channel == '532_per':
            self.weak_signal_ratio = 0.9
            self.weak_signal_ratio_threshold = 1
        elif channel == '1064':
            self.weak_signal_ratio = 0.85
            self.weak_signal_ratio_threshold = 1
        else:
            raise Exception(f"Unrecognized channel: {channel}")
        
        # Define minimum number of pixels (horizontally) for a strip of signal
        # between (A)FA or 'Likely Artiafct' to be kept; else the strip is flag
        # as 'Low confidence small strips'
        self.nb_prof_min_small_strips = 15 # profiles


def get_feature_detection_coef(channel, level):
    """Coefficients for each detection level
    k: nb of STD(noise) in detetion threshold (ATSR > 1 + k*STD(noise))
    n: min nb of neighbors
    s: (vert, horiz) nb of pixels in smoothing window
    a: (horiz, gauss_sigma) nb of horizontal pixels in averaging window and
       sigma coefficient for Gaussian (Note: window with heigh > 1 pixel not
       coded yet)"""

    if channel == '532_par':
        k = [  50,   10,        5,       2,       5]
        n = [   5,   10,       20,      50,    1000]
        s = [None, None,   (5, 5),  (3, 9), (9, 21)]
        a = [None, None,     None,    None,  (11, 5)]

    elif channel == '532_per':
        k = [ 500,   50,       20,       2,       4]
        n = [   5,   50,      100,       5,    1000]
        s = [None, None,   (5, 5),  (3, 9), (9, 21)]
        a = [None, None,     None,    None,  (11, 5)]

    elif channel == '1064':
        k = [  50,   10,        5,       1,       5]
        n = [   5,   10,       20,      50,    1000]
        s = [None, None,   (5, 5),  (3, 9), (9, 21)]
        a = [None, None,     None,    None,  (11, 5)]

    # if channel == '532_par':
    #     k = [  50,   10,        5,       2,        2]
    #     n = [   5,   10,       20,      50,     1000]
    #     s = [None, None,   (5, 5),  (3, 9),  (9, 21)]
    #     a = [None, None,     None,    None,  (29, 11)]

    # elif channel == '532_per':
    #     k = [ 500,   50,       20,       2,        2]
    #     n = [   5,   50,      100,      50,     1000]
    #     s = [None, None,   (5, 5),  (3, 9),  (9, 21)]
    #     a = [None, None,     None,    None, (29, 11)]

    # elif channel == '1064':
    #     k = [  50,   10,        5,       1,        1]
    #     n = [   5,   10,       20,      50,     1000]
    #     s = [None, None,   (5, 5),  (3, 9),  (9, 21)]
    #     a = [None, None,     None,    None, (29, 11)]

    else:
        sys.exit(f"Unrecognized channel: {channel}")

    return k[level], n[level], s[level], a[level]


# def get_params_bkg_conv():
#     """Outputs: Parameters to convert background signal (from the background
#                 energy monitor), in GA-normalized digitizer counts"""
#
#     C0_par = -0.1782756e-6
#     C0_per = 0.1844652e-6
#     BGMonSens_par = 0.000760019e-6
#     BGMonSens_per = 0.000759954e-6
#     TIAGain = 2.49e3
#     PostAmpGain = 1.25
#     SciDigSens = 8192
#
#     return C0_par, C0_per, BGMonSens_par, BGMonSens_per, TIAGain, PostAmpGain,\
#            SciDigSens
#
#
# def get_R_MID():
#     """Output: R_MID = mid range of baseline region used in current
#                        algorithm"""
#
#     R_MID = 617.5 # (km)
#
#     return R_MID