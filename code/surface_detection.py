#!/usr/bin/env python
# coding: utf8

from datetime import datetime
import numpy as np
import sys

from standard_outputs import print_elapsed_time
from config import SurfaceDetectionParameters
from readers.calipso_reader import range_from_altitude, rms_from_P_domain_to_betap_domain
from calipso_constants import *

FILL_VALUE_INT = 999

def surf_search_region(surf_type, est_surf_alt, est_surf_alt_index, alt, params):
    """Define surface search region"""

    # Initialization
    nb_prof = surf_type.size
    min_index_search_region = np.ones(nb_prof, dtype=int)*FILL_VALUE_INT
    max_index_search_region = np.ones(nb_prof, dtype=int)*FILL_VALUE_INT

    # Define (min, max) bin index of surface search region
    for i_prof in np.arange(nb_prof):

        # Select offset according to surface type
        if (surf_type[i_prof] == 17) & (est_surf_alt[i_prof] == 0): # Water & z_DEM = 0
            offset_DEM = np.copy(params.offset_dem_water)
        elif surf_type[i_prof] == 15: # Permanent-Snow
            offset_DEM = np.copy(params.offset_dem_perm_snow)
        else:
            offset_DEM = np.copy(params.offset_dem_other)

        # Determine (min, max) 
        min_index_search_region[i_prof] = est_surf_alt_index[i_prof]-offset_DEM
        max_index_search_region[i_prof] = est_surf_alt_index[i_prof]+offset_DEM

        # Take into account the limit of the size of alt
        min_index_search_region[i_prof] = max(min_index_search_region[i_prof], 0)
        if max_index_search_region[i_prof] >= alt.size:
            max_index_search_region[i_prof] = alt.size - 1

    return min_index_search_region, max_index_search_region


def compute_deriv(ab, alt):
    """Compute the derivatives of the lidar signal"""
    return (ab[:, :-1] - ab[:, 1:]) / (alt[np.newaxis, :-1] - alt[np.newaxis, 1:])


def get_min_max_deriv(deriv, alt, min_index_search_region, max_index_search_region):
    """Get the altitudes of the minimum and maximum values of the derivatives in
    the surface search region, and their respective bin index"""

    # Initialization
    nb_prof = deriv.shape[0]
    i_min = np.ones(nb_prof, dtype=int)*FILL_VALUE_INT
    i_max = np.ones(nb_prof, dtype=int)*FILL_VALUE_INT
    alt_min = np.ones(nb_prof)*FILL_VALUE_FLOAT
    alt_max = np.ones(nb_prof)*FILL_VALUE_FLOAT

    # Loop on each profile
    for i_prof in np.arange(nb_prof):

        # Get min and max indexes of search region
        min_index = min_index_search_region[i_prof]
        max_index = max_index_search_region[i_prof]+1

        # Get indexes of min and max derivative in search region
        i_min[i_prof] = np.argmin(deriv[i_prof, min_index:max_index]) + min_index
        i_max[i_prof] = np.argmax(deriv[i_prof, min_index:max_index]) + min_index

        # Get corresponding altitude
        alt_min[i_prof] = alt[i_min[i_prof]]
        alt_max[i_prof] = alt[i_max[i_prof]]

    return i_min, i_max, alt_min, alt_max


def get_max_ab_signal(ab, i_min, i_max):
    """Get the maximum signal magnitude lying between alt_min and alt_max"""

    # Initialization
    nb_prof = ab.shape[0]
    ab_max = np.ones(nb_prof)*FILL_VALUE_FLOAT
    ab_argmax = np.ones(nb_prof)*FILL_VALUE_INT

    # Loop on each profile
    for i in np.arange(nb_prof):

        # Get min and max index in i_min and i_max (i_min can be above i_max)
        min_index = np.min((i_min[i], i_max[i]))
        max_index = np.max((i_min[i], i_max[i]))

        # Get the maximum signal magnitude
        ab_max[i] = np.max(ab[i, min_index:max_index+1])

        # Get corresponding index
        ab_argmax[i] = np.argmax(ab[i, min_index:max_index+1]) + min_index

    return ab_max, ab_argmax


def apply_surf_detection_rules(i_min, i_max, alt_min, alt_max, ab_max, ab_argmax, params,
                               rms_betap_surf, ab, deriv, alt, channel):
    """Test the three following rules to determine if we have identified 
    a legitimate surface return"""

    # Initialization
    nb_prof = ab.shape[0]
    i_surf = np.ones(nb_prof, dtype=int)*FILL_VALUE_INT
    alt_surf = np.ones(nb_prof)*FILL_VALUE_FLOAT

    # Conditions
    condition_1 = alt_min > alt_max
    condition_2 = np.abs(i_min - i_max) <= params.N
    condition_3 = ab_max > (params.coef_nb_std*rms_betap_surf)
    # #TEST#################
    # condition_4 = np.ones(ab_argmax.size, dtype=bool)
    # for i in np.arange(ab_argmax.size):
    #     if ab_argmax[i] >= 0:
    #         # ATB 532 per around max of current ATB not to high
    #         # I should take the peak of ATB 532 per (need some recoding)
    #         condition_4[i] = np.max(ab_532_per[i, :np.int(ab_argmax[i])+3]) < 0.04
    # ######################
    rules_passed = (condition_1 & condition_2 & condition_3)# & condition_4)

    for i in np.arange(nb_prof):
        if rules_passed[i]:
            i_ab = i_min[i]+1 # index just above (in altitude)
            if (deriv[i, i_ab] > 0) | (ab[i, i_ab] <= 0):
                i_surf[i] = i_min[i]
                alt_surf[i] = alt[i_min[i]]
            else:
                if channel == '1064':
                    i_surf[i] = i_min[i]+2
                    alt_surf[i] = alt[i_min[i]+2]
                else:
                    i_surf[i] = i_min[i]+1
                    alt_surf[i] = alt[i_min[i]+1]

    return i_surf, alt_surf


def remove_false_pos(i_surf, alt_surf, params, est_surf_alt_index):
    """Remove surface detection if detection is isolated (no neighbor profiles)
    with surface detection"""
    
    nb_prof = est_surf_alt_index.size
    if params.offset_dem_false_positive:
        for i in np.arange(nb_prof-2)+1: # no test on first and last profile
            prof_detect_alone = (i_surf[i-1] == FILL_VALUE_INT) & (i_surf[i] != FILL_VALUE_INT) &\
                                (i_surf[i+1] == FILL_VALUE_INT)# prof with surface detect isolated
            detect_far_from_DEM = np.abs(i_surf[i] - est_surf_alt_index[i]) >\
                                  params.offset_dem_false_positive
            if prof_detect_alone & detect_far_from_DEM:
                # cancel detection
                i_surf[i] = FILL_VALUE_INT
                alt_surf[i] = FILL_VALUE_FLOAT

    return i_surf, alt_surf


# **********************************************************************
# MAIN FUNCTION
# **********************************************************************
def detect_surface(ab, surf_type, est_surf_alt, alt_sat, alt, rms, energy, calib, pgr, gain,
                   caliop_lidar_tilt, channel):
    """Detect surface using the attenuated backscatter signal in lidar channel"""
    
    tic_function = datetime.now()

    # Print channel
    print(f"\n\t***{channel}***")
    
    # Get surface detection parameters
    params = SurfaceDetectionParameters(channel)


    ###########################################################
    #### Test if DEM estimated surface altitude is correct ####
    print("\t=> Test if DEM estimated surface altitude is correct...", end='')
    for i in np.arange(est_surf_alt.size):
        if (est_surf_alt[i] < np.min(alt)) | (est_surf_alt[i] > np.max(alt)):
            raise Exception(f"DEM surface_elevation[{i}] = {est_surf_alt[i]:.3f} looks uncorrect")

    # Show elapsed time
    tic = print_elapsed_time(tic_function)


    ################################################
    #### Compute RMS in beta' domain at surface ####
    print("\t=> Compute RMS in beta' domain at surface...", end='')
    r_surf = range_from_altitude(alt_sat, est_surf_alt, caliop_lidar_tilt)
    rms_betap_surf = rms_from_P_domain_to_betap_domain(rms, r_surf, energy, gain, calib, pgr)

    # Show elapsed time
    tic = print_elapsed_time(tic)
    

    #########################################################
    #### Compute bin index of estimated surface altitude ####
    print("\t=> Compute bin index of estimated surface altitude...", end='')
    est_surf_alt_index = np.argmin(np.abs(alt - est_surf_alt[:, np.newaxis]), axis=1)

    # Show elapsed time
    tic = print_elapsed_time(tic)


    ######################################
    #### Define surface search region ####
    print("\t=> Define surface search region...", end='')
    min_index_search_region, max_index_search_region =\
        surf_search_region(surf_type, est_surf_alt, est_surf_alt_index, alt, params)
    
    # Show elapsed time
    tic = print_elapsed_time(tic)


    #############################
    #### Compute derivatives ####
    print("\t=> Compute derivatives...", end='')
    deriv = compute_deriv(ab, alt)

    # Show elapsed time
    tic = print_elapsed_time(tic)


    ##########################################
    #### Get min and max in search region ####
    print("\t=> Get min and max in search region...", end='')
    i_min, i_max, alt_min, alt_max = get_min_max_deriv(deriv, alt, min_index_search_region,
                                                       max_index_search_region)

    # Show elapsed time
    tic = print_elapsed_time(tic)


    ######################################
    #### Get maximum signal magnitude ####
    print("\t=> Get maximum signal magnitude...", end='')
    ab_max, ab_argmax = get_max_ab_signal(ab, i_min, i_max)

    # Show elapsed time
    tic = print_elapsed_time(tic)


    ###########################
    #### Surface detection ####
    print("\t=> Surface detection...", end='')
    i_surf, alt_surf = apply_surf_detection_rules(i_min, i_max, alt_min, alt_max, ab_max,
                                                  ab_argmax, params, rms_betap_surf, ab, deriv,
                                                  alt, channel)

    # Show elapsed time
    tic = print_elapsed_time(tic)

    
    ################################
    #### Remove false positives ####
    print("\t=> Remove false positives...", end='')
    i_surf, alt_surf = remove_false_pos(i_surf, alt_surf, params, est_surf_alt_index)

    # Show elapsed time
    print_elapsed_time(tic)

    print(f'\t(Elapsed time: {datetime.now() - tic_function})')

    return i_surf
