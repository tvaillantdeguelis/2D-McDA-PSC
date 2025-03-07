#!/usr/bin/env python
# coding: utf8

import os
import sys
from datetime import datetime

import numpy as np
from numba import jit
import matplotlib as mpl
from matplotlib import cm, gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
import seaborn as sns

from my_modules.standard_outputs import print_elapsed_time
from my_modules.calipso_constants import FILL_VALUE_FLOAT
from my_modules.figuretools import setstyle, cm2in

from config import *


def apply_threshold(k, ab_mol, feature, ab_signal, ab_sigma, where_FA=False):
    """Put FLAG_MAYBE where signal above threshold"""

    # Initialization
    new_feature = np.ma.copy(feature)

    # Define threshold
    ab_maybe = ab_mol + k*ab_sigma

    if where_FA:
        # Put flag where AB > threshold and where AB is not masked
        new_feature[np.ma.where(ab_signal>ab_maybe)] = FLAG_MAYBE
    else:
        # Put flag where AB > threshold and where feature is still "nothing"
        new_feature[np.ma.where((ab_signal>ab_maybe) &\
                            (new_feature==FLAG_NOTHING))] = FLAG_MAYBE

    return new_feature


@jit(nopython=True)
def apply_window_jit(w_side, h_side, feature, nb_pixels_window, min_percent, detected_pixels,
                     flag_detection_level):
    """Part extracted from apply_window function for faster processing with
    @jit"""

    for i in np.arange(w_side, feature.shape[0]-w_side):
        for j in np.arange(h_side, feature.shape[1]-h_side):
            if (feature[i, j]==FLAG_NOTHING) | (feature[i, j]==FLAG_MAYBE):
                # Tuple with indexes of the window
                window = (slice(i-w_side, i+w_side+1), 
                          slice(j-h_side, j+h_side+1))

                # Count nb of "maybe"
                nb_maybe = list(feature[window].flatten()).count(FLAG_MAYBE)

                # Count nb of "previous detection level" (d-1)
                nb_detected_1 = 0
                if flag_detection_level > 1: # if previous detection exists
                    prev_flag_detection_level = flag_detection_level-1
                    nb_detected_1 = list(feature[window].flatten()).count(prev_flag_detection_level)
                
                # Total detected at n and n-1
                nb_tot = nb_maybe + nb_detected_1

                # Count nb of special (and detection <= d-2) and remove 
                # from nb_pixels_window
                nb_surface = list(feature[window].flatten()).count(FLAG_SURFACE)
                nb_likely_artifact = list(feature[window].flatten()).count(FLAG_LIKELY_ARTIFACT)
                nb_FA = list(feature[window].flatten()).count(FLAG_FA)
                nb_AFA = list(feature[window].flatten()).count(FLAG_AFA)
                nb_small_strips = list(feature[window].flatten()).count(FLAG_SMALL_STRIPS)
                nb_detected_2_and_before = 0
                if flag_detection_level > 1: # if previous detection exists
                    for prev_flag_detection_level in np.arange(1, flag_detection_level-1):
                        nb_detected_2_and_before += list(feature[window].flatten()).\
                                      count(prev_flag_detection_level)
                nb_pixels_window_2 = nb_pixels_window - nb_FA - nb_AFA - nb_surface - \
                                     nb_likely_artifact - nb_small_strips - nb_detected_2_and_before

                # Flag detected if amount above limit
                nb_min_tot = nb_pixels_window_2*min_percent
                if (nb_tot >= nb_min_tot):
                    detected_pixels[i, j] = 1
    
    return feature


def apply_window(height_window, width_window, feature, flag_detection_level, min_percent=0.5):
    # min_percent: min pourcentage of total counted pixels in the  window to flag the center as "detected"

    # Initialization
    new_feature = np.ma.copy(feature)
    new_feature = new_feature.filled(FILL_VALUE_FLOAT)

    # height_window and width_window should be odd numbers
    if (height_window%2 != 1) | (width_window%2 != 1):
        sys.exit(f"height_window (= {height_window}) and width_window "\
                 f"(= {width_window}) should be odd numbers")

    # Initialization
    detected_pixels = np.zeros(new_feature.shape, dtype=bool)
    nb_pixels_window = width_window * height_window
    h_side = int(height_window/2) # nb of pixel each side of the center
    w_side = int(width_window/2) # nb of pixel each side of the center

    # Apply moving window
    new_feature = apply_window_jit(w_side, h_side, new_feature, nb_pixels_window, min_percent,
                                   detected_pixels, flag_detection_level)
    new_feature = np.ma.masked_where(new_feature==FILL_VALUE_FLOAT, new_feature)

    # Remove previous "maybe" pixels (or not if keep_all==True)
    new_feature[new_feature==FLAG_MAYBE] = FLAG_NOTHING

    # Replace by those which result from the windowing
    new_feature[detected_pixels==1] = FLAG_MAYBE

    return new_feature


@jit(nopython=True)
def neighbors(shape, p):
    """Get neighbors of a pixel"""

    v = []

    if p[0] != shape[0]-1: # if not extreme right
        v.append( (p[0]+1, p[1]) ) # add right neighbor

    if p[0] != 0: # if not extreme left
        v.append( (p[0]-1, p[1]) ) # add left neighbor

    if p[1] != shape[1]-1: # if not extreme top
        v.append( (p[0], p[1]+1) ) # add top neighbor

    if p[1] != 0: # if not extreme bottom
        v.append( (p[0], p[1]-1) ) # add bottom neighbor

    return v


@jit(nopython=True)
def replace_maybe_jit(nb_lim, feature, seen_pixels, flag_detection_level,
                      prev_detect, prevprev_detect):
    """Part extracted from replace_maybe function for faster processing with @jit"""

    for i in np.arange(feature.shape[0]):
        for j in np.arange(feature.shape[1]):
            if not seen_pixels[i, j]:
                if feature[i, j] == FLAG_MAYBE:
                    connected_to_detected_pattern = False                      
                    # Count neighbors
                    accessible_pixels = [(i, j)]
                    pattern_pixels = np.zeros(feature.shape)
                    pattern_pixels[i, j] = True
                    while (len(accessible_pixels) != 0):
                        p = accessible_pixels[0] # 1st pixel of the list
                        accessible_pixels = accessible_pixels[1:] # Remove 1st
                        #-----------------------------------------------
                        # if FA/low confidence on the left, check on the 
                        # left of FA/low confidence if pattern connect 
                        # to an already detected pattern
                        i_FA_left = 1
                        while (i - i_FA_left >= 1) &\
                              (feature[p[0]-i_FA_left, p[1]] == FLAG_FA) |\
                              (feature[p[0]-i_FA_left, p[1]] == FLAG_AFA) |\
                              (feature[p[0]-i_FA_left, p[1]] == FLAG_SMALL_STRIPS):
                            i_FA_left +=1
                        if (i - i_FA_left >= 0) &\
                           (feature[p[0]-i_FA_left, p[1]] == flag_detection_level):
                            connected_to_detected_pattern = True
                        #-----------------------------------------------
                        if not seen_pixels[p]:
                            seen_pixels[p] = True # We note that we see this 
                                                  # pixel
                            v = neighbors(feature.shape, p) # Get pixel 
                                                            # neighbors 
                            # Look for neighbors
                            for voisin in v:
                                c1 = seen_pixels[voisin]
                                c2 = feature[voisin] == FLAG_MAYBE # level n
                                c3 = False
                                if flag_detection_level > 1: # if previous
                                                           # detection exists
                                    if prev_detect:
                                        c3 = feature[voisin]==flag_detection_level-1
                                                                  # level n - 1
                                    else:
                                        c3 == False
                                    if prevprev_detect:
                                        c4 = feature[voisin]==flag_detection_level-2
                                                                  # level n - 2
                                    else:
                                        c4== False
                                if (not c1) & (c2 | c3 | c4):
                                    accessible_pixels.append(voisin)
                                    pattern_pixels[voisin] = 1
                    
                    count = np.int64(np.round(np.sum(pattern_pixels)))
                    # remove the too small "maybe pattern"
                    px, py = np.where(pattern_pixels==1)
                    for pix_i in np.arange(px.size):
                        # unless already classify
                        if feature[px[pix_i], py[pix_i]] == FLAG_MAYBE:
                            if (count < nb_lim) &\
                               (not connected_to_detected_pattern):
                                # replace "maybe" by "nothing"
                                feature[px[pix_i], py[pix_i]] = FLAG_NOTHING
                            else:
                                # replace "maybe" by the next level of detection
                                feature[px[pix_i], py[pix_i]] = flag_detection_level

    return feature


def replace_maybe(n, feature, flag_detection_level, prev_detect=True,
                  prevprev_detect=False):
    """Put flag 'flag_detection_level' where patterns of connected 'FLAG_MAYBE'
    pixels consist of more than n pixels
    if prev_detect=True means that we also count detection pixels n-1
    if prevprev_detect=True means that we also count detection pixels n-2"""

    # Initialization
    new_feature = np.ma.copy(feature)
    new_feature = new_feature.filled(FILL_VALUE_FLOAT)
    seen_pixels = np.zeros(new_feature.shape, dtype=bool)
    
    # Look for a "maybe" pixel and decide if it's really part of a pattern
    # based on nb of neighbors (neighbors in "level n" + "level n-1")
    if n == 1: # keep all
        new_feature[new_feature==FLAG_MAYBE] = flag_detection_level
    else:
        new_feature = replace_maybe_jit(n, new_feature, seen_pixels, flag_detection_level,
                                        prev_detect, prevprev_detect)
    
    new_feature = np.ma.masked_where(new_feature==FILL_VALUE_FLOAT, new_feature)

    return new_feature


@jit(nopython=True)
def flag_isolated_spikes_jit(nb_lim, feature, seen_pixels):
    """Part extracted from flag_isolated_spikes function for faster processing with @jit"""

    for i in np.arange(feature.shape[0]):
        for j in np.arange(feature.shape[1]):
            if not seen_pixels[i, j]:
                if feature[i, j] == FLAG_MAYBE:                   
                    # Count neighbors
                    accessible_pixels = [(i, j)]
                    pattern_pixels = np.zeros(feature.shape)
                    pattern_pixels[i, j] = True
                    while (len(accessible_pixels) != 0):
                        p = accessible_pixels[0] # 1st pixel of the list
                        accessible_pixels = accessible_pixels[1:] # Remove 1st
                        if not seen_pixels[p]:
                            seen_pixels[p] = True # We note that we see this pixel
                            v = neighbors(feature.shape, p) # Get pixel neighbors 
                            # Look for neighbors
                            for voisin in v:
                                c1 = seen_pixels[voisin]
                                c2 = feature[voisin] == FLAG_MAYBE
                                if (not c1) & c2:
                                    accessible_pixels.append(voisin)
                                    pattern_pixels[voisin] = 1
                    
                    count = np.int64(np.round(np.sum(pattern_pixels)))
                    # keep only the small "maybe pattern" (= spikes)
                    px, py = np.where(pattern_pixels == 1)
                    for pix_i in np.arange(px.size):
                        # unless already classify
                        if feature[px[pix_i], py[pix_i]] == FLAG_MAYBE:
                            if count <= nb_lim:
                                # replace "maybe" by "spikes"
                                feature[px[pix_i], py[pix_i]] = FLAG_SPIKES
                            else:
                                # replace "maybe" by "nothing"
                                feature[px[pix_i], py[pix_i]] = FLAG_NOTHING

    return feature


def flag_isolated_spikes(n, feature):
    """Put flag 'FLAG_SPIKES' where patterns of connected 'FLAG_MAYBE'
    pixels consist of less than n pixels"""

    # Initialization
    new_feature = np.ma.copy(feature)
    new_feature = new_feature.filled(FILL_VALUE_FLOAT)
    seen_pixels = np.zeros(new_feature.shape, dtype=bool)
    
    new_feature = flag_isolated_spikes_jit(n, new_feature, seen_pixels)
    
    new_feature = np.ma.masked_where(new_feature==FILL_VALUE_FLOAT, new_feature)

    spikes = np.ma.copy(new_feature)
    spikes[spikes != FLAG_SPIKES] = 0
    spikes[spikes == FLAG_SPIKES] = 1

    return new_feature, spikes


def mask_detected_from_ab(ab_signal, feature):
    """Remove detected pixel from the AB signal"""

    # Mask where not "nothing"
    new_ab = np.ma.masked_where(feature != FLAG_NOTHING, ab_signal)

    return new_ab


def release_flag_spikes(feature):
    """Put back FLAG_NOTHING where FLAG_SPIKES"""

    # Initialization
    new_feature = np.ma.copy(feature)

    new_feature[new_feature == FLAG_SPIKES] = FLAG_NOTHING

    return new_feature


def mask_spikes_in_ab(ab_signal, feature):
    """Mask pixels flagged as spikes in the AB signal"""

    # Mask where not "nothing"
    new_ab = np.ma.masked_where(feature == FLAG_SPIKES, ab_signal)

    return new_ab


@jit(nopython=True)
def gaussian_line_window_jit(nb_prof, nb_alt, width_window, gauss_sigma, ab_signal, 
                             new_ab, fill_value, feature):
    """Part extracted from gaussian_line_window function for faster processing 
    with @jit"""

    # Loop on profiles
    for i in np.arange(nb_prof):
        # From bottom go up
        for j in np.arange(nb_alt):
            # Apply n-elements line gaussian sliding window
            nside = np.int64((width_window-1)/2)
            x = np.arange(width_window) - nside
            gaussian = np.exp(-x**2/(2*gauss_sigma**2))
            nb_prof_averaged = np.sum(gaussian)
            if ab_signal[i, j] != fill_value:
                # Horizontal averaging (if not left/right edge)
                if not (i < nside) | (i > nb_prof - (nside+1)):
                    line = ab_signal[i-nside:i+(nside+1), j]
                    gaussian = gaussian[line!=fill_value] # first
                    line = line[line!=fill_value] # then (in this order!)
                    new_ab[i, j] = np.sum(line*gaussian)/np.sum(gaussian)
            # Also apply window where special flags (FA, AFA, likely artifact, 
            # and no confidence)
            elif (feature[i,j] == FLAG_FA) | (feature[i,j] == FLAG_AFA) |\
                 (feature[i,j] == FLAG_LIKELY_ARTIFACT) |\
                 (feature[i,j] == FLAG_SMALL_STRIPS):
                if not (i < nside) | (i > nb_prof - (nside+1)):
                    line = ab_signal[i-nside:i+(nside+1), j]
                    no_special_flag = line!=fill_value
                    nb_no_special_flag = np.sum(no_special_flag)
                    if nb_no_special_flag != 0: # if not all FA or AFA
                        gaussian = gaussian[line!=fill_value] # first
                        line = line[line!=fill_value] # then (in this order!)
                        new_ab[i, j] = np.sum(line*gaussian)/np.sum(gaussian)

    return new_ab, nb_prof_averaged


@jit(nopython=True)
def gaussian_2d_window_jit(nb_prof, nb_alt, ab_signal, new_ab, gaussian_2d, h_nside, v_nside, fill_value, feature):
    """Part extracted from gaussian_line_window function for faster processing 
    with @jit"""

    # Loop on profiles
    for i in np.arange(nb_prof):
        # From bottom go up
        for j in np.arange(nb_alt):
            # Apply 2-D gaussian sliding window
            if ab_signal[i, j] != fill_value:
                # Averaging (if not at an edge of the image)
                if not (i < h_nside) | (i > nb_prof - (h_nside+1)) | (j < v_nside) | (j > nb_alt - (v_nside+1)):
                    window = ab_signal[i-h_nside:i+(h_nside+1), j-v_nside:j+(v_nside+1)]
                    gaussian_2d_vector_without_fillvalues = np.array(())
                    window_vector_without_fillvalues = np.array(())
                    for i_window in np.arange(window.shape[0]):
                        for j_window in np.arange(window.shape[1]):
                            if window[i_window, j_window] != fill_value:
                                gaussian_2d_vector_without_fillvalues = np.append(gaussian_2d_vector_without_fillvalues, gaussian_2d[i_window, j_window])
                                window_vector_without_fillvalues = np.append(window_vector_without_fillvalues, window[i_window, j_window])
                    new_ab[i, j] = np.sum(window_vector_without_fillvalues*gaussian_2d_vector_without_fillvalues)/np.sum(gaussian_2d_vector_without_fillvalues)
            # Also apply window where special flags (FA, AFA, likely artifact, no confidence, and spikes)
            elif (feature[i,j] == FLAG_FA) |\
                 (feature[i,j] == FLAG_AFA) |\
                 (feature[i,j] == FLAG_LIKELY_ARTIFACT) |\
                 (feature[i,j] == FLAG_SMALL_STRIPS) |\
                 (feature[i,j] == FLAG_SPIKES):
                if not (i < h_nside) | (i > nb_prof - (h_nside+1)) | (j < v_nside) | (j > nb_alt - (v_nside+1)):
                    window = ab_signal[i-h_nside:i+(h_nside+1), j-v_nside:j+(v_nside+1)]
                    no_special_flag = window != fill_value
                    nb_no_special_flag = np.sum(no_special_flag)
                    if nb_no_special_flag != 0: # if not all FA or AFA
                        gaussian_2d_vector_without_fillvalues = np.array(())
                        window_vector_without_fillvalues = np.array(())
                        for i_window in np.arange(window.shape[0]):
                            for j_window in np.arange(window.shape[1]):
                                if window[i_window, j_window] != fill_value:
                                    gaussian_2d_vector_without_fillvalues = np.append(gaussian_2d_vector_without_fillvalues, gaussian_2d[i_window, j_window])
                                    window_vector_without_fillvalues = np.append(window_vector_without_fillvalues, window[i_window, j_window])
                        new_ab[i, j] = np.sum(window_vector_without_fillvalues*gaussian_2d_vector_without_fillvalues)/np.sum(gaussian_2d_vector_without_fillvalues)

    return new_ab


def gaussian_line_window(width_window, gauss_sigma, ab_signal, feature, ab_sigma):
    """Apply a horizontal gaussian averaging window to the AB signal"""

    # Initialization
    nb_prof = ab_signal.shape[0]
    nb_alt = ab_signal.shape[1]
    ab2 = np.ma.copy(ab_signal)
    # ab2 = np.ma.masked_where(feature != FLAG_NOTHING, ab2) # mask where not "nothing"
    ab2 = ab2.filled(FILL_VALUE_FLOAT) # fill mask value to use jit
    new_ab = np.ma.ones(ab_signal.shape)*FILL_VALUE_FLOAT
    new_ab = new_ab.filled(FILL_VALUE_FLOAT)
    copy_feature = np.ma.copy(feature)  
    copy_feature = copy_feature.filled(FILL_VALUE_FLOAT)

    # width_window should be odd numbers
    if width_window%2 != 1:
        sys.exit(f"width_window (= {width_window}) should be odd numbers")

    # Apply gaussian line averaging
    new_ab, nb_prof_averaged = gaussian_line_window_jit(nb_prof, nb_alt, width_window, gauss_sigma,
                                                        ab2, new_ab, FILL_VALUE_FLOAT, copy_feature)
    
    # Mask where FILL_VALUE_FLOAT
    new_ab = np.ma.masked_where(new_ab==FILL_VALUE_FLOAT, new_ab)
    ab2 = np.ma.masked_where(ab2==FILL_VALUE_FLOAT, ab2)
    copy_feature = np.ma.masked_where(copy_feature==FILL_VALUE_FLOAT, copy_feature)

    # Adapt SR threshold
    ab_sigma = ab_sigma/np.sqrt(nb_prof_averaged)

    return new_ab, ab_sigma


def gaussian_2d_window(width_window, horizontal_gauss_sigma, ab_signal, feature, ab_sigma, height_window=5, vertical_gauss_sigma=3):
    """Apply a 2-D gaussian averaging window to the AB signal"""

    # Initialization
    nb_prof = ab_signal.shape[0]
    nb_alt = ab_signal.shape[1]
    ab2 = np.ma.copy(ab_signal)
    # ab2 = np.ma.masked_where(feature != FLAG_NOTHING, ab_signal) # mask where not "nothing"
    ab2 = ab2.filled(FILL_VALUE_FLOAT) # fill mask value to use jit
    new_ab = np.ma.ones(ab_signal.shape)*FILL_VALUE_FLOAT
    new_ab = new_ab.filled(FILL_VALUE_FLOAT)
    copy_feature = np.ma.copy(feature)  
    copy_feature = copy_feature.filled(FILL_VALUE_FLOAT)

    # width_window should be odd numbers
    if width_window%2 != 1:
        sys.exit(f"width_window (= {width_window}) should be odd numbers")

    # Apply gaussian 2-D averaging
    h_nside = np.int64((width_window-1)/2)
    x = np.arange(width_window) - h_nside
    v_nside = np.int64((height_window-1)/2)
    y = np.arange(height_window) - v_nside
    gaussian_2d = np.outer(np.exp(-x**2/(2*horizontal_gauss_sigma**2)), np.exp(-y**2/(2*vertical_gauss_sigma**2))) # np.outer: matrix from product of two vectors
    nb_prof_averaged = np.sum(gaussian_2d)
    new_ab = gaussian_2d_window_jit(nb_prof, nb_alt, ab2, new_ab, gaussian_2d, h_nside, v_nside, FILL_VALUE_FLOAT, copy_feature)
    
    # Mask where FILL_VALUE_FLOAT
    new_ab = np.ma.masked_where(new_ab==FILL_VALUE_FLOAT, new_ab)
    ab2 = np.ma.masked_where(ab2==FILL_VALUE_FLOAT, ab2)
    copy_feature = np.ma.masked_where(copy_feature==FILL_VALUE_FLOAT, copy_feature)

    # Adapt SR threshold
    ab_sigma = ab_sigma/np.sqrt(nb_prof_averaged)

    return new_ab, ab_sigma




# For algorithm development purpose
def plot_mask(mask):

    # Labels
    clabels = ["No detection",] +\
                ["Detection level %d" % i for i in np.arange(5)+1] +\
                ["Potential detection",]

    # Colormap
    palette = sns.cubehelix_palette(n_colors=5, start=2, rot=1, hue=1., gamma=1., light=0.8,
                                    dark=0.2, reverse=True)
    palette.insert(0, [1.0, 1.0, 1.0]) # 0 = Nothing
    palette.append([1.0, 0.5, 0.0])  # 255 = Maybe
    my_cmap = mpl.colors.ListedColormap(palette)
    bounds = np.array((-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 255.5))
    colors = my_cmap(np.arange(len(palette)))
    my_cmap, my_norm = from_levels_and_colors(bounds, colors)

    # Figure style
    setstyle("ticks_nogrid")

    # Create figure
    fig_w = cm2in(17.7) # cm
    fig_h = cm2in(6) # cm
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)

    # Plot figure
    ax0 = plt.subplot(gs0[0])
    pc = plt.pcolormesh(mask.T, cmap=my_cmap, norm=my_norm, rasterized=True)

    # Plot colorbar
    ax1 = plt.subplot(gs0[1])
    cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
    fontsize_clabel = 5
    cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
    for j, lab in enumerate(clabels):
        cbar.ax.text(1.5, 1/(float(bounds.size-1)*2) + j/float(bounds.size-1), lab,
                        va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
    cbar.set_label("Level of detection", labelpad=80)
    
    # Close figure
    plt.show()


def get_last_key(dict):

    last_key = next(reversed(dict))

    return last_key


def process_detection_level(channel, level, ab_mol, ab_sigma, feature_dict, ab_dict, step):
    """Process a single detection level."""

    print(f"\t=> Detection level {level}:")

    # Initialization
    tic = datetime.now()
    k, n, s, a = get_feature_detection_coef(channel, level - 1)
    spikes = None

    # Apply Averaging Window: Smooth the 2D data signal by applying an averaging window.
    if a:
        # Filter spikes before averaging the signal
        k = 20 # threshold
        n = 5 # max number of connected pixels
        feature_dict, ab_dict, spikes, step = filter_spikes(feature_dict, ab_dict, ab_mol, ab_sigma, step, k, n)

        # Mask Previously Detected Features: Mask features detected during previous detection levels from the AB signal.
        # print("\t\t- Apply a gaussian horizontal line window averaging...", end='')
        # ab_dict[step := step+1], ab_sigma = gaussian_line_window(a[0], a[1], ab_dict[get_last_key(ab_dict)],
        #                                             feature_dict[get_last_key(feature_dict)], ab_sigma)
        print("\t\t- Apply a 2D gaussian window averaging...", end='')
        ab_dict[step := step+1], ab_sigma = gaussian_2d_window(a[0], a[1], ab_dict[get_last_key(ab_dict)],
                                                    feature_dict[get_last_key(feature_dict)], ab_sigma)
        tic = print_elapsed_time(tic)
    
    # Threshold Application: Apply a threshold to the smoothed signal to generate a binary mask, distinguishing regions above and below the threshold.
    print("\t\t- Apply threshold...", end='')
    feature_dict[step := step+1] = apply_threshold(k, ab_mol, feature_dict[get_last_key(feature_dict)], ab_dict[get_last_key(ab_dict)], ab_sigma)
    tic = print_elapsed_time(tic)

    # Smooth Binary Mask: Apply a smoothing window to the binary mask to improve feature continuity.
    if s:
        print("\t\t- Windowing on the 'maybe' pixels...", end='')
        feature_dict[step := step+1] = apply_window(s[0], s[1], feature_dict[get_last_key(feature_dict)], level)
        tic = print_elapsed_time(tic)

    # Filter Small Features: Remove regions in the smoothed binary mask that contain fewer connected pixels than the specified minimum threshold.
    print("\t\t- Flag 'Detected' where patterns of 'FLAG_MAYBE' pixels meet neighbors number limit condition...", end='')
    feature_dict[step := step+1] = replace_maybe(n, feature_dict[get_last_key(feature_dict)], level)
    tic = print_elapsed_time(tic)

    # Remove detected pixel from AB
    print("\t=> Remove detected pixel from AB...", end='')
    ab_dict[step := step+1] = mask_detected_from_ab(ab_dict[get_last_key(ab_dict)], feature_dict[get_last_key(feature_dict)])
    tic = print_elapsed_time(tic)

    return feature_dict, ab_dict, spikes, step


def filter_spikes(feature_dict, ab_dict, ab_mol, ab_sigma, step, k, n):
    """Remove signal spikes consisting of small clusters of less than n connected pixels
    where the signal is above the threshold defined with k."""

    print("\t=> Remove spikes:")

    # Initialization
    tic = datetime.now()

    # Apply threshold to get the 'maybe' pixels
    print("\t\t- Apply threshold...", end='')
    feature_dict[step := step+1] = apply_threshold(k, ab_mol, feature_dict[get_last_key(feature_dict)], ab_dict[get_last_key(ab_dict)], ab_sigma)
    tic = print_elapsed_time(tic)

    # Get isolated spikes
    print("\t\t- Get isolated spikes...", end='')
    feature_dict[step := step+1], spikes = flag_isolated_spikes(n, feature_dict[get_last_key(feature_dict)])
    tic = print_elapsed_time(tic)

    # Mask isolated spike signal in AB
    print("\t\t- Mask isolated spike signal in AB...", end='')
    ab_dict[step := step+1] = mask_spikes_in_ab(ab_dict[get_last_key(ab_dict)], feature_dict[get_last_key(feature_dict)])
    tic = print_elapsed_time(tic)

    # Remove spike flags from mask
    print("\t\t- Remove spike flags from mask...", end='')
    feature_dict[step := step+1] = release_flag_spikes(feature_dict[get_last_key(feature_dict)])
    tic = print_elapsed_time(tic)

    return feature_dict, ab_dict, spikes, step


def transform_feature_and_ab_dicts_to_arrays(feature_dict, ab_dict, step):

    print("\t=> Transform feature and ab_signal dictionaries to 3D arrays...", end='')
    
    # Initialization
    tic = datetime.now()
    array_shape_0 = ab_dict[get_last_key(ab_dict)].shape[0]
    array_shape_1 = ab_dict[get_last_key(ab_dict)].shape[1]
    feature_array_steps = np.ma.zeros((step+1, array_shape_0, array_shape_1), dtype=np.uint8)
    ab_array_steps = np.ma.ones((step+1, array_shape_0, array_shape_1))*FILL_VALUE_FLOAT

    # Transform dictionaries to arrays
    for i_step in np.arange(step+1):
        try: # Test if feature_dict has a i_step
            feature_dict[i_step]
        except: # If not go directly to next step
            continue
        feature_array_steps[i_step, :, :] = feature_dict[i_step]

    for i_step in np.arange(step+1):
        try: # Test if ab_dict has a i_step
            ab_dict[i_step]
        except: # If not go directly to next step
            continue   
        ab_array_steps[i_step, :, :] = ab_dict[i_step]

    # Show elapsed time
    tic = print_elapsed_time(tic)

    return feature_array_steps, ab_array_steps


# **********************************************************************
# MAIN FUNCTION
# **********************************************************************
def detect_features(ab_signal, ab_mol, ab_sigma, channel):
    """Detect features in AB signal of lidar channel"""

    # Initialization
    tic_function = datetime.now()
    NB_DETECTION_LEVELS = 5
    ab_dict = {0: np.ma.copy(ab_signal)}
    feature_dict = {0: np.ma.zeros(ab_signal.shape, dtype=np.uint8)}
    step = 0

    # Print channel
    print(f"\n\t***{channel}***")

    # Get feature detection parameters
    params = FeatureDetectionParameters(channel)

    # Process detection levels
    for level in range(1, NB_DETECTION_LEVELS+1):
        feature_dict, ab_dict, spikes, step = process_detection_level(channel, level, ab_mol, ab_sigma, feature_dict, ab_dict, step)

    # Transform feature and ab_signal dictionaries to 3D arrays
    feature_array_steps, ab_array_steps = transform_feature_and_ab_dicts_to_arrays(feature_dict, ab_dict, step)

    print(f'\t(Elapsed time: {datetime.now() - tic_function})')

    return feature_dict[get_last_key(feature_dict)], feature_array_steps, ab_array_steps, spikes
