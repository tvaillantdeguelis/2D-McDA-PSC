#!/usr/bin/env python
# coding: utf8

from datetime import datetime
import numpy as np
import os
import sys

from config import *


# # Global variables
MAX_DETECT_LEVEL = 5


def change_detection_values(mask):
    """Put all 'flag_detection_level' = 1, 2,... to 'FLAG_DETECT'
    Put 'FLAG_LIKELY_ARTIFACT', 'FLAG_AFA', and 'FLAG_SMALL_STRIPS' to
    'FLAG_LOW_CONFIDENCE'"""

    new_mask = np.ma.copy(mask)

    # Put 'FLAG_DETECT'
    new_mask[(new_mask >= 1) & (new_mask <= MAX_DETECT_LEVEL)] = FLAG_DETECT

    # Put 'FLAG_LOW_CONFIDENCE'
    new_mask[(new_mask == FLAG_LIKELY_ARTIFACT) |\
             (new_mask == FLAG_AFA) |\
             (new_mask == FLAG_SMALL_STRIPS) |\
             (new_mask == FLAG_SPIKES)] = FLAG_LOW_CONFIDENCE

    return new_mask


# **********************************************************************
# MAIN FUNCTION
# **********************************************************************
def merged_feature_masks(mask_532_par, mask_532_per, mask_1064):
    """Merged the feature masks from the 3 channels"""

    tic_function = datetime.now()

    # Check if flag values not declared in global variables
    if not np.all((mask_532_par <= MAX_DETECT_LEVEL) |\
                  (mask_532_par >= FLAG_SPIKES)):
        sys.exit("mask_532_par has values not declared in global variables")
    if not np.all((mask_532_per <= MAX_DETECT_LEVEL) |\
                  (mask_532_per >= FLAG_SPIKES)):
        sys.exit("mask_532_per has values not declared in global variables")
    if not np.all((mask_1064 <= MAX_DETECT_LEVEL) |\
                  (mask_1064 >= FLAG_SPIKES)):
        sys.exit("mask_1064 has values not declared in global variables")


    #################################
    #### Create a composite mask ####
    print("\t=> Create a composite mask from the 3 channel "\
          "feature masks...")

    # Put 'FLAG_DETECT' and 'FLAG_LOW_CONFIDENCE'
    flag_532_par = change_detection_values(mask_532_par)
    flag_532_per = change_detection_values(mask_532_per)
    flag_1064    = change_detection_values(mask_1064)

    #--------------------------------------------------------------------------
    # Bits 1–3: Classification
    # Note: the order in which to write the classification is important
    # e.g.: 'surface' will overwrite 'surface' above 'detect'

    # 0: invalid (bad or missing data) (initialization)
    merged_mask = np.zeros(mask_532_par.shape, dtype=np.uint8)

    # 3: low confidence
    merged_mask[(flag_532_par == FLAG_LOW_CONFIDENCE) |\
                (flag_532_per == FLAG_LOW_CONFIDENCE) |\
                (flag_1064    == FLAG_LOW_CONFIDENCE)] = 3

    # 1: "clear air"
    merged_mask[(flag_532_par == FLAG_NOTHING) |\
                (flag_532_per == FLAG_NOTHING) |\
                (flag_1064    == FLAG_NOTHING)] = 1

    # 3: low confidence if "clear air" only from 532_per
    merged_mask[( (flag_532_par == FLAG_LOW_CONFIDENCE) &\
                  (flag_532_per == FLAG_NOTHING) &\
                  (flag_1064    == FLAG_LOW_CONFIDENCE) ) |
                ( (flag_532_par == FLAG_LOW_CONFIDENCE) &\
                  (flag_532_per == FLAG_NOTHING) &\
                  (flag_1064    == FLAG_FA) ) |
                ( (flag_532_par == FLAG_FA) &\
                  (flag_532_per == FLAG_NOTHING) &\
                  (flag_1064    == FLAG_LOW_CONFIDENCE) )] = 3

    # 2: atmospheric feature
    merged_mask[(flag_532_par == FLAG_DETECT) |\
                (flag_532_per == FLAG_DETECT) |\
                (flag_1064    == FLAG_DETECT)] = 2

    # 4: not used

    # 5: surface or subsurface
    merged_mask[(flag_532_par == FLAG_SURFACE) |\
                (flag_532_per == FLAG_SURFACE) |\
                (flag_1064    == FLAG_SURFACE)] = 5

    # 6: not used

    # 7: totally attenuated
    # if 532_par and 1064 FA and 532_per FA, nothing, or low_confidence
    merged_mask[ (flag_532_par == FLAG_FA) &\
                 (flag_1064    == FLAG_FA) &\
                ((flag_532_per == FLAG_FA) |\
                 (flag_532_per == FLAG_NOTHING) |\
                 (flag_532_per == FLAG_LOW_CONFIDENCE))] = 7

    #--------------------------------------------------------------------------
    # Bits 4: 532 nm parallel channel detection status %1000 = 8
    merged_mask[(flag_532_par == FLAG_DETECT) |\
                (flag_532_par == FLAG_SURFACE)] += 8

    #--------------------------------------------------------------------------
    # Bits 5: 532 nm perpendicular channel detection status %10000 = 16
    merged_mask[(flag_532_per == FLAG_DETECT) |\
                (flag_532_per == FLAG_SURFACE)] += 16

    #--------------------------------------------------------------------------
    # Bits 6: 532 nm perpendicular channel detection status %100000 = 32
    merged_mask[(flag_1064 == FLAG_DETECT) |\
                (flag_1064 == FLAG_SURFACE)] += 32


    print(f'\t(Elapsed time: {datetime.now() - tic_function})')

    return merged_mask
