#!/usr/bin/env python
# coding: utf8

import os
import sys

from datetime import datetime
import numpy as np
from pyhdf.SD import SD
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from matplotlib.colors import LogNorm, from_levels_and_colors
from matplotlib.ticker import MultipleLocator, FixedLocator, LogLocator
import seaborn as sns
import cartopy
import cartopy.crs as ccrs
import cmocean
import cmlidar

from my_modules.standard_outputs import print_time
from my_modules.readers.calipso_reader import CALIPSOReader, get_prof_min_max_indexes_from_lon
from my_modules.figuretools import setstyle, takecmap, cm2in, compute_bounds, lat_lon_dist_xaxis, \
    CALIOPFigureMaker, remove_edges
from my_modules.geotools import neighbors, geo_distance, get_monotical_lon, granule_date_decomposition

FILL_VALUE_FLOAT = -9999.0


class FigureMaker(CALIOPFigureMaker):
    def __init__(self):
        super().__init__()
        self.fig_w = cm2in(17.7) # cm
        self.fig_h = cm2in(6) # cm
        self.axes_titlesize = 8
        self.axes_title_pad = 1.12
        self.clabelpad = 50

    def set_max_detect_level(self, max_detect_level):
        self.max_detect_level = max_detect_level
        
    def plot_map(self, lat_granule, lon_granule, granule_date):
        """Plot CALIPSO track on a map"""

        # Figure style
        setstyle("ticks_nogrid")

        # Get mid lat/lon
        lat_0 = np.squeeze(self.lat[int(self.lat.size / 2)])
        lon_0 = np.squeeze(self.lon[int(self.lon.size / 2)])

        # # Get monotical longitudes
        lon_granule_plot = get_monotical_lon(lon_granule)
        lon_plot = get_monotical_lon(self.lon)

        # Figure
        fig = plt.figure(figsize=(cm2in(15), cm2in(15)))
        ax_proj = ccrs.Orthographic(lon_0, lat_0)
        # ax_proj = ccrs.NearsidePerspective(lon_0, lat_0)
        ax = plt.axes(projection=ax_proj)
        ax.coastlines()
        ax.add_feature(cartopy.feature.LAKES, edgecolor='k')
        ax.stock_img()
        # Add lat-lon grid
        gl = ax.gridlines(color='k', linewidth=0.5, linestyle='--', alpha=0.2)
        gl.xlocator = MultipleLocator(10)
        gl.ylocator = MultipleLocator(10)
        gl = ax.gridlines(color='k', linewidth=1., linestyle='-', alpha=0.2)
        gl.xlocator = MultipleLocator(30)
        gl.ylocator = MultipleLocator(30)
        ax.set_global()
        # Plot
        plt.plot(lon_granule_plot, lat_granule, c='b', lw=4, alpha=0.1, transform=ccrs.PlateCarree())
        plt.plot(lon_plot, self.lat, c='#a81c07', lw=4, alpha=1, transform=ccrs.PlateCarree())
        title = f"{granule_date}"
        ypos = 0.95
        ax.text(0.5, ypos, title, weight='bold', ha='center', va='center',fontsize=16, transform=fig.transFigure)

        # Save figure
        filename = f"map"
        self.save_fig(filename, transparent=True, adjust=(0.02, 0.02, 0.98, 0.9))

        # Close figure
        plt.close(fig)


    def plot_mask(self, step, mask, channel):
        """Plot mask for each step. If only final mask, put step = None"""

        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            mask = remove_edges(mask, self.edges_removal)

        # Labels
        clabels = ["No detection",] +\
                  ["Detection level %d" % i for i in np.arange(self.max_detect_level)+1] 
        if step:
            clabels = clabels + ["Spikes",]
            clabels = clabels + ["Potential detection",]

        # Colormap
        nb_colors = self.max_detect_level
        palette = sns.cubehelix_palette(nb_colors, start=2, rot=1, hue=1., gamma=1., light=0.8,
                                        dark=0.2, reverse=True)
        palette.insert(0, [1.0, 1.0, 1.0]) # 0 = Nothing
        if step:
            palette.append([1.0, 0.0, 0.0])  # 249 = Spikes
            palette.append([1.0, 0.5, 0.0])  # 255 = Maybe
            colorbins = np.array((-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 249.5, 255.5))
        else:
            colorbins = np.array((-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5))
        my_cmap = mpl.colors.ListedColormap(palette)
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
    
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, mask.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=False)
        if step:
            plt.title(f'{channel} (step {step})', weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        else:
            if channel == '532_par':
                title = "2D-McDA-PSC 532 nm parallel detection feature mask"
            elif channel == '532_per':
                title = "2D-McDA-PSC 532 nm perpendicular detection feature mask"
            elif channel == '1064':
                title = "2D-McDA-PSC 1064 nm detection feature mask"
            else:
                raise ValueError(f"Unknown channel = {channel}")
            plt.title(title, fontweight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)

        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        fontsize_clabel = 5
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
        # cbar.set_label("Level of detection", labelpad=55)
       
        # Save figure
        if step:
            filename = f"{channel}_step{step:d}_mask"
        else:
            filename = f"mask_{channel}"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_best_detection_mask(self, mask):

        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            mask = remove_edges(mask, self.edges_removal)

        # Labels
        clabels = ["No detection",] +\
                  ["Detection level %d" % i for i in np.arange(self.max_detect_level)+1] 

        # Colormap
        nb_colors = self.max_detect_level
        palette = sns.cubehelix_palette(nb_colors, start=2, rot=1, hue=1., gamma=1., light=0.8,
                                        dark=0.2, reverse=True)
        palette.insert(0, [1.0, 1.0, 1.0]) # 0 = Nothing
        colorbins = np.array((-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5))
        my_cmap = mpl.colors.ListedColormap(palette)
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
    
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, mask.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=False)
        title = "Composite\ best\ detection\ level"
        plt.title(r'$\mathbf{%s}$' % title, y=1.35)

        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        fontsize_clabel = 5
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
        cbar.set_label("Level of detection", labelpad=55)
       
        # Save figure
        filename = f"mask_best_detection"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def backscatter_cbar_labels(self, cbar):
        cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
        cbar.ax.tick_params(which='both', labelright=False)
        minor_bounds = cmlidar.cm.BACKSCATTER_DISCRETE_BOUNDS
        cbar.ax.yaxis.set_minor_locator(FixedLocator(minor_bounds))
        cbar_minor_label = ['1.0',
                            '1.0', '3.0', '6.0',
                            '1.0', '1.5', '2.0', '3.0', '4.0', '5.0', '6.0', '8.0',
                            '1.0', '1.5', '2.0', '3.0', '5.0']
        for j, bound in enumerate(minor_bounds):
            cbar.ax.text(2, bound, cbar_minor_label[j], va='center', fontsize=6)
        cbar_major_label = ['$\mathbf{×10^{-5}}$', '$\mathbf{×10^{-4}}$', '$\mathbf{×10^{-3}}$', '$\mathbf{×10^{-2}}$']
        c_bar_major_values = np.array((1e-5, 1e-4, 1e-3, 1e-2))
        for j, bound in enumerate(c_bar_major_values):
            cbar.ax.text(3.5, bound, cbar_major_label[j], va='center', fontsize=8)


    def plot_ab_signal(self, ab_signal, title, filename):
        # sourcery skip: merge-comparisons, merge-duplicate-blocks, remove-redundant-if
        
        # Mask where fill_value
        # ab_signal = np.ma.masked_where(ab_signal == 0, ab_signal)
    
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            ab_signal = remove_edges(ab_signal, self.edges_removal)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)

        # Plot figure
        ax0 = plt.subplot(gs0[0])
        # my_cmap = cmocean.cm.thermal
        # my_cmap = takecmap("extthermal")
        # my_cmap.colorbar_extend = 'both'
        # my_norm = mpl.colors.LogNorm()
        my_cmap = "backscatter_continuous"
        my_norm = cmlidar.cm.backscatter_continuous_norm

        # Put negative values to 1e-9 so they don't appear transparent
        ab_signal[(ab_signal<0) & ~ab_signal.mask] = 1e-9
        pc = plt.pcolormesh(self.pindexbins, self.altbins, ab_signal.T, cmap=my_cmap, norm=my_norm, rasterized=True)
        # plt.clim(1e-5, 2e-3)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=False)
        plt.title(title, weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        # plt.text(0.02, 0.85, "(5km×180m)", ha='left', va='center', transform=fig.transFigure)
        
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', extend='both')
        self.backscatter_cbar_labels(cbar)
        cbar.set_label(label=r"$\beta'$ (km$^{-1}$ sr$^{-1}$)", fontsize=8, labelpad=self.clabelpad)

        # Save figure
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_h2o(self, h2o, title, filename):
        
        # Mask where fill_value
        h2o = np.ma.masked_where(h2o == 0, h2o)
    
        # # Remove edges
        # if self.edges_removal != 0: # to avoid error with "-0"
        #     h2o = remove_edges(h2o, self.edges_removal)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)

        # Plot figure
        ax0 = plt.subplot(gs0[0])
        # ax0.set_facecolor('0.5')
        my_cmap = cm.viridis_r
        # my_cmap = cmocean.cm.thermal
        # my_cmap = takecmap("extthermal")
        # my_cmap.colorbar_extend = 'both'

        # Put negative values to 1e-9 so they don't appear transparent
        # h2o[(h2o<0) & ~h2o.mask] = 1e-9
        pc = plt.pcolormesh(self.pindexbins, self.altbins, h2o.T, cmap=my_cmap, vmin=0, vmax=8, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=False)
        plt.title(title, fontweight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        # plt.text(0.02, 0.85, "(5km×180m)", ha='left', va='center', transform=fig.transFigure)
        
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', extend='max')
        cbar.set_label(label="ppmv")

        # Save figure
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_hno3(self, hno3, title, filename):
        
        # Mask where fill_value
        hno3 = np.ma.masked_where(hno3 == 0, hno3)
    
        # # Remove edges
        # if self.edges_removal != 0: # to avoid error with "-0"
        #     hno3 = remove_edges(hno3, self.edges_removal)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)

        # Plot figure
        ax0 = plt.subplot(gs0[0])
        # ax0.set_facecolor('0.5')
        my_cmap = cmocean.cm.thermal_r
        # my_cmap = takecmap("extthermal")
        # my_cmap.colorbar_extend = 'both'

        # Put negative values to 1e-9 so they don't appear transparent
        # hno3[(hno3<0) & ~hno3.mask] = 1e-9
        pc = plt.pcolormesh(self.pindexbins, self.altbins, hno3.T, cmap=my_cmap, vmin=0, vmax=15, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=False)
        plt.title(title, fontweight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        # plt.text(0.02, 0.85, "(5km×180m)", ha='left', va='center', transform=fig.transFigure)
        
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', extend='max')
        cbar.set_label(label="ppbv")

        # Save figure
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_hom_4_colors(self, mask, title, filename):
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            mask = remove_edges(mask, self.edges_removal)

        # Set color to separate homogeneous feature
        nb_colors = 7
        color_number_array = np.arange(nb_colors) + 1
        mask = set_homogeneous_feature_color(mask, color_number_array)
        mask[mask==0] = np.ma.masked # mask where no feature, if not appear like first color
        
        # Figure style
        setstyle("ticks_nogrid")

        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        
        # Colormap
        # mpl.rcParams["axes.facecolor"]='0.5' # dark grey background
        cmaplist = ["#FF0000",
                    "#00FFFF",
                    "#0000FF",
                    "#FFFF00",
                    "#FF00FF",
                    "#00FF00",
                    "#006600"]
        cmaplist = cmaplist[:nb_colors]
        my_cmap = mpl.colors.ListedColormap(cmaplist)
        colorbins = np.arange(nb_colors+1)+0.5
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)
        
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, mask.T, cmap=my_cmap, norm=my_norm,
                            rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=False)
        plt.title(title, weight='bold', y=1.35)
        
        # Save figure
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)
    

    def plot_hom_ab_signal(self, ab_signal, title, filename):        
        # Mask where fill_value
        ab_signal = np.ma.masked_where(ab_signal == FILL_VALUE_FLOAT, ab_signal)
    
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            ab_signal = remove_edges(ab_signal, self.edges_removal)

        # Put negative values to 1e-9 so they don't appear transparent
        ab_signal[(ab_signal<0) & ~ab_signal.mask] = 1e-9

        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
    
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        my_cmap = cm.plasma
        nb_colors = 256
        palette = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
        my_cmap = mpl.colors.ListedColormap(palette[1:-1])
        my_cmap.set_under(palette[0])
        my_cmap.set_over(palette[-1])
        # my_cmap.colorbar_extend = 'both'
        pc = plt.pcolormesh(self.pindexbins, self.altbins, ab_signal.T, cmap=my_cmap, norm=LogNorm())
        plt.clim(1e-6, 1e-3)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=False)
        plt.title(title, weight='bold', y=1.35)
        plt.text(0.02, 0.85, "(5km×180m)", ha='left', va='center', transform=fig.transFigure)
        
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', extend='both', drawedges=False)
        cbar.set_label(label=r"$\langle\beta'\rangle$ (km$^{-1}$ sr$^{-1}$)")
        cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
        cbar.ax.yaxis.set_minor_locator(LogLocator(numticks=15, subs=np.arange(0.2, 1, 0.1)))
        
        # Save figure
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_hom_asr_signal(self, sr_signal, title, filename):

        # Mask where fill_value
        sr_signal = np.ma.masked_where(sr_signal == FILL_VALUE_FLOAT, sr_signal)
    
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            sr_signal = remove_edges(sr_signal, self.edges_removal)

        # Put negative values to 1e-9 so they don't appear transparent
        sr_signal[(sr_signal<0) & ~sr_signal.mask] = 1e-9

        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
    
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, sr_signal.T, cmap=cm.viridis, norm=LogNorm())
        plt.clim(1, 50)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=False)
        plt.title(title, weight='bold', y=1.35)
        plt.text(0.02, 0.85, "(5km×180m)", ha='left', va='center', transform=fig.transFigure)
        
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', extend='both', drawedges=False)
        cbar.set_label(label=r"$\langle R'\rangle$")
        cbar.ax.yaxis.set_major_locator(FixedLocator(np.array((1, 10, 50))))
        cbar.ax.set_yticklabels(['1', '10', '50'])

        # Save figure
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)

    
    def plot_steps(self, mask, ab_signal, channel):
    
        # Get number of steps
        nb_steps = mask.shape[0]
    
        for step in np.arange(nb_steps):
            if not np.all(mask[step, :, :] == 0):
                self.plot_mask(step, mask[step, :, :], channel)
            if not np.all(ab_signal[step, :, :].mask):
                filename = f"{channel}_step{step:d}_signal"
                title = f'{channel} (step {step})'
                self.plot_ab_signal(ab_signal[step, :, :], title, filename)


    def plot_composite_mask(self, mask):
        """Plot composite mask"""
    
        # Take only Bits 1–3
        mask = np.bitwise_and(mask, 7)
    
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            mask = remove_edges(mask, self.edges_removal)
        
        # Labels
        clabels = ['Clear air',
                   'Atmospheric feature']
    
        # Colormap
        cmaplist = ['1',
                    "#00539c"]
        my_cmap = mpl.colors.ListedColormap(cmaplist)
        colorbins = np.array((0.5, 1.5, 2.5))
                    # 1: Clear air
                    # 2: Atmospheric feature
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, mask.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=False)
        plt.title('2D-McDA-PSC composite detection feature mask', fontweight='bold', y=1.35)
    
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        fontsize_clabel = 7
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
       
        # Save figure
        filename = "mask_composite"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_composite_mask_strong_weak(self, mask):
        """Plot composite mask"""
    
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            mask = remove_edges(mask, self.edges_removal)
    
        # Labels
        clabels = ['Clear',
                   'Weak feature',
                   'Strong feature']
    
        # Colormap
        cmaplist = ['1',
                    "#fdac53",
                    "#34568b"]
        my_cmap = mpl.colors.ListedColormap(cmaplist)
        colorbins = np.array((0.5, 1.5, 2.25, 2.75))
                    # 1: Clear air
                    # 2: Weak feature
                    # 2.5: Strong feature
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
    
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, mask.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=False)
        plt.title('2D-McDA-PSC composite detection feature mask', fontweight='bold', y=1.35)
    
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        fontsize_clabel = 7
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center',fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
       
        # Save figure
        filename = "mask_composite_strong_weak"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_composite_mask_channel(self, mask):
        """Plot composite mask"""
    
        # Change value of mask
        new_mask = np.ma.zeros(mask.shape)
        new_mask[ np.bitwise_and(mask, int('000111', 2)) == 1] = 1
        new_mask[(np.bitwise_and(mask, int('000111', 2)) == 2) &\
                 (np.bitwise_and(mask, int('111000', 2)) == 8)] = 2
        new_mask[(np.bitwise_and(mask, int('000111', 2)) == 2) &\
                 (np.bitwise_and(mask, int('111000', 2)) == 16)] = 3
        new_mask[(np.bitwise_and(mask, int('000111', 2)) == 2) &\
                 (np.bitwise_and(mask, int('111000', 2)) == 32)] = 4
        new_mask[(np.bitwise_and(mask, int('000111', 2)) == 2) &\
                 (np.bitwise_and(mask, int('111000', 2)) == (8+16))] = 5
        new_mask[(np.bitwise_and(mask, int('000111', 2)) == 2) &\
                 (np.bitwise_and(mask, int('111000', 2)) == (8+32))] = 6
        new_mask[(np.bitwise_and(mask, int('000111', 2)) == 2) &\
                 (np.bitwise_and(mask, int('111000', 2)) == (16+32))] = 7
        new_mask[(np.bitwise_and(mask, int('000111', 2)) == 2) &\
                 (np.bitwise_and(mask, int('111000', 2)) == (8+16+32))] = 8
        new_mask[ np.bitwise_and(mask, int('000111', 2)) == 3] = 9
        new_mask[ np.bitwise_and(mask, int('000111', 2)) == 5] = 10
        new_mask[ np.bitwise_and(mask, int('000111', 2)) == 7] = 11
    
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            new_mask = remove_edges(new_mask, self.edges_removal)
        
        # Labels
        clabels = ['Clear air',
                   '532 par only',
                   '532 per only',
                   '1064 only',
                   '532 par\n+ 532 per',
                   '532 par\n+ 1064',
                   '532 per\n+ 1064',
                   '532 par\n+ 532 per\n+ 1064']
    
        # Colormap
        palette = ([1.0,           1.0,      1.0], # 1: Clear air
                   [0.9,           0.0,      0.0], # 2: 532 par only (red)
                   [0.9,           0.9,      0.0], # 3: 532 per only (yellow)
                   [135./255, 206./255, 235./255], # 4: 1064 only (blue)
                   [255./255, 140./255,   0./255], # 5: 532 par + 532 per (orange)
                   [102./255,   0./255, 153./255], # 6: 532 par + 1064 (violet)
                   [152./255, 251./255, 152./255], # 7: 532 per + 1064 (green)
                   [0.0,           0.0,      0.0]) # 8: 532 par + 532 per + 1064 (black)
        my_cmap = mpl.colors.ListedColormap(palette)
        colorbins = np.arange(len(palette) + 1) + 0.5
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
        
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, new_mask.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=False)
        plt.title('2D-McDA-PSC composite detection feature mask', fontweight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
    
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        fontsize_clabel = 5
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
       
        # Save figure
        filename = "mask_composite_channel"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


    def plot_attenuated_color_ratio(self, acr_signal):
        
        # Mask where fill_value
        acr_signal = np.ma.masked_where(acr_signal == FILL_VALUE_FLOAT, acr_signal)
    
        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            acr_signal = remove_edges(acr_signal, self.edges_removal)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
    
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        ax0.set_facecolor((0.75, 0.75, 0.75))
        bounds = np.array((0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6))
        colors = ["#C7C7C7",
                    "#000000",
                    "#160D84",
                    "#700AA4",
                    "#B83A87",
                    "#E8735C",
                    "#FCC140",
                    "#EFF941",
                    "#FFFFFF"]
        my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')
        # Put negative values to 1e-9 so they don't appear transparent
        acr_signal[(acr_signal<0) & ~acr_signal.mask] = 1e-9
        pc = plt.pcolormesh(self.pindexbins, self.altbins, acr_signal.T, cmap=my_cmap, norm=my_norm)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=False)
        plt.title("$\mathbf{Attenuated\ Color\ Ratio}\ \\frac{\\beta'_{1064}}{\\beta'_{532}}$", y=1.35)
        plt.text(0.02, 0.85, "(5km×180m)", ha='left', va='center', transform=fig.transFigure)
        
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', extend='both', drawedges=True)
        cbar.set_label(label=r"Attenuated Color Ratio", labelpad=5)
        cbar.ax.yaxis.set_minor_locator(MultipleLocator(1000.))
        
        # Save figure
        self.save_fig("acr")

        # Close figure
        plt.close(fig)


    def plot_nsf(self, nsf, title, filename):
        
        # Mask where fill_value
        nsf = np.ma.masked_where(nsf == FILL_VALUE_FLOAT, nsf)

        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            nsf = remove_edges(nsf, self.edges_removal)

        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
    
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        ax0.set_facecolor((0.75, 0.75, 0.75))
        my_cmap = takecmap('extviridis')
        pc = plt.pcolormesh(self.pindexbins, self.altbins, nsf.T, cmap=my_cmap)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=False)
        plt.title(title, y=1.35)
        plt.text(0.02, 0.85, "(5km×180m)", ha='left', va='center', transform=fig.transFigure)
        
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical')
        plt.clim(0.045, 0.060)
        cbar.set_label(label=r"NSF", labelpad=5)

        # Save figure
        self.save_fig(filename)

        # Close figure
        plt.close(fig)


    def plot_psc_composition(self, mask):

        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            mask = remove_edges(mask, self.edges_removal)

        # Labels
        clabels = ["Likely tropo. ice", # -4
                  "Not determinable", # -1
                  "No detection", # 0
                  "STS", # 1
                  "NAT", # 2
                  "Ice", # 4
                  "Enhanced NAT", # 5
                  "Wave ice"] # 6

        # Colormap
        nb_colors = len(clabels)
        palette = ["#000000",
                   "#FFFFFF",
                   "#888888",
                   "#00FF26",
                   "#FAFF00",
                   "#00BBFF",
                   "#FF0000",
                   "#4700C3"]
        colorbins = np.array((-5, -3, -0.5, 0.5, 1.5, 3.5, 4.5, 5.5, 6.5))
        my_cmap = mpl.colors.ListedColormap(palette)
        my_norm = mpl.colors.BoundaryNorm(colorbins, my_cmap.N)
    
        # Figure style
        setstyle("ticks_nogrid")
    
        # Create figure
        fig = plt.figure(figsize=(self.fig_w, self.fig_h))
        gs0 = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.1)
    
        # Plot figure
        ax0 = plt.subplot(gs0[0])
        pc = plt.pcolormesh(self.pindexbins, self.altbins, mask.T, cmap=my_cmap,
                            norm=my_norm, rasterized=True)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=False)
        plt.title(f'PSC v2 Composition (draft code)', weight='bold', y=1.35)

        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        fontsize_clabel = 7
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
       
        # Save figure
        filename = f"PSC_v2_Composition"
        self.save_fig(filename)
        
        # Close figure
        plt.close(fig)


def set_homogeneous_feature_color(mask, color_number_array):
    """Give a color number to each homogeneous feature in order that two connected feature
    do not get the same color number.
    Need at least 4 colors in the array according to "four color theorem"."""
    
    # Initialization
    nb_colors = color_number_array.size
    seen_pixels = np.zeros(mask.shape, dtype=bool)
    color_mask = np.ma.zeros(mask.shape)
    i_color = 0
    
    # Set a color to each homogeneous pattern
    for i in np.arange(mask.shape[0]):
        for j in np.arange(mask.shape[1]):
            if not seen_pixels[i, j] and mask[i, j] != 0: # pixel not processed and with atmospheric feature
                accessible_pixels = [(i, j)] # start list of accessible pixels
                connected_pattern_colors = [] # start list of connected pattern colors
                pattern_pixels = np.zeros(mask.shape, dtype=bool)
                pattern_pixels[i, j] = True
                while (len(accessible_pixels) != 0):
                    p = accessible_pixels[0] # 1st pixel of the list
                    accessible_pixels = accessible_pixels[1:] # Remove 1st
                    if not seen_pixels[p]:
                        seen_pixels[p] = True # We note that we see this pixel
                        v = neighbors(mask.shape, p) # Get pixel neighbors
                        # Look for neighbors
                        for voisin in v:
                            if seen_pixels[voisin]: # pixel already processed
                                if mask[voisin] != mask[i, j] and mask[voisin] != 0: # and not part of the feature and part of another feature (received color number)
                                    if color_mask[voisin] not in connected_pattern_colors:
                                        connected_pattern_colors.append(color_mask[voisin])
                            else: # pixel not already processed
                                if mask[voisin] == mask[i, j]: # and part of the feature
                                    accessible_pixels.append(voisin)
                                    pattern_pixels[voisin] = True
                # Set color to the pattern
                if i_color >= 4:
                    i_color = 0 # use the 4 first colors primarily
                first_i_color = i_color
                while color_number_array[i_color] in connected_pattern_colors:
                    i_color = change_color(i_color, nb_colors) # loop on color array
                    if i_color == first_i_color: # new color not possible
                        print("Warning: Not enough colors, two connected patterns with same color.")
                        break
                color_mask[pattern_pixels] = color_number_array[i_color]
                i_color = change_color(i_color, nb_colors) # new color for next one
        
    return color_mask

def change_color(i_color, nb_colors):
    """Loop on color array"""
    if i_color < nb_colors - 1:
        i_color += 1
    else:
        i_color = 0
    return i_color
    

if __name__ == '__main__':
    tic_main_program = print_time()

    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # PARAMETERS
    INDATA_FOLDER = "/home/vaillant/codes/projects/2D_McDA_for_PSCs/out/data/CLaMS_ZOMM/"
    FILENAME_2D_McDA_PSCs_ZOMM_CLAMS = "2D_McDA_PSCs-PSC_ZOMM_CLAMS_BKS_2010d018_0000.nc" # 
    VERSION_2D_McDA = "V1.0"
    TYPE_2D_McDA = "Prototype"
    EDGES_REMOVAL = 0 # number of prof to remove on both edges of plot
    MAX_DETECT_LEVEL = 5
    PLOT_ALL_STEPS = False
    INVERT_XAXIS = False
    YMIN = 8
    YMAX = 30
    FIGURES_PATH = "/home/vaillant/codes/projects/2D_McDA_for_PSCs/out/figures/CLaMS_ZOMM/"
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    
    
    # **********************************
    # *** Load 2D-McDA HDF data file ***
    print("\n*****Load 2D-McDA HDF data file...*****")
    
    # Get filename and filepath
    hdffile = os.path.join(INDATA_FOLDER, FILENAME_2D_McDA_PSCs_ZOMM_CLAMS)

    # Open HDF file
    print(f"\tGranule path: {hdffile}")
    cal_2d_mcda = CALIPSOReader(hdffile)

    # Load 2D-McDA parameters
    data_dict_cal_2d_mcda = {}
    cal_2d_mcda_keys = [
        "Latitude",
        "Longitude",
        "Profile_ID",
        "Altitude",
        "Parallel_Detection_Flags_532",
        "Perpendicular_Detection_Flags_532",
        "Detection_Flags_1064",
        "Parallel_Attenuated_Backscatter_532",
        "Perpendicular_Attenuated_Backscatter_532",
        "Attenuated_Backscatter_1064",
        "Composite_Detection_Flags",
        "H2O",
        "HNO3",
        "H2OC_ICE",
        "HNO3C_NAT"
    ]
    for key in cal_2d_mcda_keys:
        data_dict_cal_2d_mcda[key] = cal_2d_mcda.get_data(key)

    data_dict_cal_2d_mcda_steps = {}
    if PLOT_ALL_STEPS:
        cal_2d_mcda_steps_keys = [
            "Parallel_Detection_Flags_532_steps",
            "Perpendicular_Detection_Flags_532_steps",
            "Detection_Flags_1064_steps",
            "Parallel_Attenuated_Backscatter_532_steps",
            "Perpendicular_Attenuated_Backscatter_532_steps",
            "Attenuated_Backscatter_1064_steps"
        ]
        for key in cal_2d_mcda_steps_keys:
            data_dict_cal_2d_mcda_steps[key] = cal_2d_mcda.get_data(key)


    # ###########################
    # # Weak/Strong feature mask
    # mask_weak_strong = np.bitwise_and(data_dict_cal_2d_mcda["Composite_Detection_Flags"], 7).astype(float)
    # # 2.5 = 'Strong' were detection without averaging at least in one channel,
    # # keep 2 = 'Weak' elsewhere
    # where_strong = ((data_dict_cal_2d_mcda["Parallel_Detection_Flags_532"] >= 1) &
    #                 (data_dict_cal_2d_mcda["Parallel_Detection_Flags_532"] <= 4)) |\
    #                ((data_dict_cal_2d_mcda["Perpendicular_Detection_Flags_532"] >= 1) &
    #                 (data_dict_cal_2d_mcda["Perpendicular_Detection_Flags_532"] <= 4)) |\
    #                ((data_dict_cal_2d_mcda["Detection_Flags_1064"] >= 1) &
    #                 (data_dict_cal_2d_mcda["Detection_Flags_1064"] <= 4))
    # mask_weak_strong[where_strong] = 2.5
    

    # ###########################
    # # Best detection level mask
    # detection_level_masks = []
    # for i in np.arange(5) + 1:
    #     detection_level_masks.append((data_dict_cal_2d_mcda["Parallel_Detection_Flags_532"] == i) + (data_dict_cal_2d_mcda["Perpendicular_Detection_Flags_532"] == i) + (data_dict_cal_2d_mcda["Detection_Flags_1064"] == i))

    # # Initialization
    # best_detection_level_mask = np.ma.zeros(data_dict_cal_2d_mcda["Parallel_Detection_Flags_532"].shape)

    # for i in np.arange(5, 0, -1):
    #     best_detection_level_mask[detection_level_masks[i-1]] = i

    
    # ************
    # *** Plot ***
    print("\n\n*****Plot...*****")
    
    # Initialize instance of FigureMaker
    plot_fig = FigureMaker()
    plot_fig.fig_w = cm2in(16) # cm
    plot_fig.fig_h = cm2in(8) # cm
    plot_fig.adj_left = 0.08
    plot_fig.adj_bottom = 0.11
    plot_fig.adj_right = 0.87
    plot_fig.adj_top = 0.81
    plot_fig.axes_title_pad = 1.15
    plot_fig.clabelpad = 40
    plot_fig.axes_titlesize = 8
    plot_fig.fig_folder = FIGURES_PATH
    plot_fig.head_filename = FILENAME_2D_McDA_PSCs_ZOMM_CLAMS[:-3]
    plot_fig.set_edges_removal(EDGES_REMOVAL)
    plot_fig.set_max_detect_level(MAX_DETECT_LEVEL)
    plot_fig.set_coordinates(data_dict_cal_2d_mcda["Latitude"], data_dict_cal_2d_mcda["Longitude"],
                             data_dict_cal_2d_mcda["Altitude"])
    
    # Plot signals
    if True:
        filename = "ab_532_par"
        title = "CLaMS-ZOMM 532 nm Parallel Backscatter $\mathit{\\beta_{532,\\parallel}}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Parallel_Attenuated_Backscatter_532"], title, filename)

        filename = "ab_532_per"
        title = "CLaMS-ZOMM 532 nm Perpendicular Backscatter $\mathit{\\beta_{532,\\bot}}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Perpendicular_Attenuated_Backscatter_532"], title, filename)

        filename = "ab_1064"
        title = "1064 nm Backscatter $\mathit{\\beta_{1064}}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Attenuated_Backscatter_1064"], title, filename)

        filename = "h2o"
        title = "$\mathbf{H_{2}O}$ vapor"
        plot_fig.plot_h2o(data_dict_cal_2d_mcda["H2O"]*1e6, title, filename) # in ppmv

        filename = "h2oc_ice"
        title = "$\mathbf{H_{2}O}$ condensed ice"
        plot_fig.plot_h2o(data_dict_cal_2d_mcda["H2OC_ICE"]*1e6, title, filename) # in ppmv

        filename = "hno3"
        title = "$\mathbf{HNO_{3}}$ vapor"
        plot_fig.plot_hno3(data_dict_cal_2d_mcda["HNO3"]*1e9, title, filename) # in ppmv

        filename = "hno3c_nat"
        title = "$\mathbf{HNO_{3}}$ condensed NAT"
        plot_fig.plot_hno3(data_dict_cal_2d_mcda["HNO3C_NAT"]*1e9, title, filename) # in ppmv

    if False:
        filename = "mol_ab_532_par"
        title = "$\mathbf{532\ nm\ Molecular\ Parallel\ Attenuated\ Backscatter}\ \\beta'_{m,532,\\parallel}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Molecular_Parallel_Attenuated_Backscatter_532"], title, filename)

        filename = "mol_ab_532_per"
        title = "$\mathbf{532\ nm\ Molecular\ Perpendicular\ Attenuated\ Backscatter}\ \\beta'_{m,532,\\perp}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Molecular_Perpendicular_Attenuated_Backscatter_532"], title, filename)

        filename = "mol_ab_1064"
        title = "$\mathbf{1064\ nm\ Molecular\ Attenuated\ Backscatter}\ \\beta'_{m,1064}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Molecular_Attenuated_Backscatter_1064"], title, filename)
        
        filename = "background_noise_532_par"
        title = "$\mathbf{532\ nm\ Parallel\ Background\ Noise}\ \\Delta\\beta'_{b,532,\\parallel}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Background_Noise_532_Parallel"], title, filename)

        filename = "background_noise_532_per"
        title = "$\mathbf{532\ nm\ Perpendicular\ Background\ Noise}\ \\Delta\\beta'_{b,532,\\perp}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Background_Noise_532_Perpendicular"], title, filename)

        filename = "background_noise_1064"
        title = "$\mathbf{1064\ nm\ Background\ Noise}\ \\Delta\\beta'_{b,1064}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Background_Noise_1064"], title, filename)

        filename = "nsf_532_par"
        title = "$\mathbf{532\ nm\ Parallel\ Noise\ Scale\ Factor}\ \mathrm{NSF}_{\\beta',532,\\parallel}$"
        plot_fig.plot_nsf(data_dict_cal_2d_mcda["Noise_Scale_Factor_532_Parallel_AB_domain"], title, filename)

        filename = "nsf_532_per"
        title = "$\mathbf{532\ nm\ Perpendicular\ Noise\ Scale\ Factor}\ \mathrm{NSF}_{\\beta',532,\\perp}$"
        plot_fig.plot_nsf(data_dict_cal_2d_mcda["Noise_Scale_Factor_532_Perpendicular_AB_domain"], title, filename)

        filename = "nsf_1064"
        title = "$\mathbf{1064\ nm\ Noise\ Scale\ Factor}\ \mathrm{NSF}_{\\beta',1064}$"
        plot_fig.plot_nsf(data_dict_cal_2d_mcda["Noise_Scale_Factor_1064_AB_domain"], title, filename)

        filename = "shot_noise_532_par"
        title = "$\mathbf{532\ nm\ Parallel\ Shot\ Noise}\ \mathrm{NSF}_{\\beta',532,\\parallel} \sqrt{\\beta'_{m,532,\\parallel}}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Shot_Noise_532_Parallel"], title, filename)

        filename = "shot_noise_532_per"
        title = "$\mathbf{532\ nm\ Perpendicular\ Shot\ Noise}\ \mathrm{NSF}_{\\beta',532,\\perp} \sqrt{\\beta'_{m,532,\\perp}}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Shot_Noise_532_Perpendicular"], title, filename)

        filename = "shot_noise_1064"
        title = "$\mathbf{1064\ nm\ Shot\ Noise}\ \mathrm{NSF}_{\\beta',1064} \sqrt{\\beta'_{m,1064}}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Shot_Noise_1064"], title, filename)

        filename = "ab_std_532_par"
        title = "$\mathbf{532\ Parallel\ nm\ Attenuated\ Backscatter\ Uncertainty}\ \\sigma_{\\beta',532,\\parallel}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Parallel"], title, filename)

        filename = "ab_std_532_per"
        title = "$\mathbf{532\ Perpendicular\ nm\ Attenuated\ Backscatter\ Uncertainty}\ \\sigma_{\\beta',532,\\perp}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Perpendicular"], title, filename)

        filename = "ab_std_1064"
        title = "$\mathbf{1064\ nm\ Attenuated\ Backscatter\ Uncertainty}\ \\sigma_{\\beta',1064}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Attenuated_Backscatter_Uncertainty_Standard_Deviation_1064"], title, filename)

        filename = "threshold_k1_532_par"
        title = "$\mathbf{532\ Parallel\ nm\ Detection\ Threshold\ (k=1)}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Molecular_Parallel_Attenuated_Backscatter_532"]+data_dict_cal_2d_mcda["Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Parallel"], title, filename)

        filename = "threshold_k1_532_per"
        title = "$\mathbf{532\ Perpendicular\ nm\ Detection\ Threshold\ (k=1)}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Molecular_Perpendicular_Attenuated_Backscatter_532"]+data_dict_cal_2d_mcda["Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Perpendicular"], title, filename)

        filename = "threshold_k1_1064"
        title = "$\mathbf{1064\ nm\ Detection\ Threshold\ (k=1)}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Molecular_Attenuated_Backscatter_1064"]+data_dict_cal_2d_mcda["Attenuated_Backscatter_Uncertainty_Standard_Deviation_1064"], title, filename)

        # plot_fig.plot_attenuated_color_ratio(data_dict_cal_2d_mcda['Attenuated_Backscatter_1064']/\
        #                                       (data_dict_cal_2d_mcda['Parallel_Attenuated_Backscatter_532']+data_dict_cal_2d_mcda["Perpendicular_Attenuated_Backscatter_532"]))


    # # Plot map (function to finish)
    # plot_fig.plot_map(data_dict_cal_2d_mcda["Latitude"], data_dict_cal_2d_mcda["Longitude"], FILENAME_2D_McDA_PSCs_ZOMM_CLAMS[-19:-6])

    # Plot the 3 channel masks
    plot_fig.plot_mask(None, data_dict_cal_2d_mcda["Parallel_Detection_Flags_532"], '532_par')
    plot_fig.plot_mask(None, data_dict_cal_2d_mcda["Perpendicular_Detection_Flags_532"], '532_per')
    plot_fig.plot_mask(None, data_dict_cal_2d_mcda["Detection_Flags_1064"], '1064')

    # Plot the composite masks
    # plot_fig.plot_composite_mask(data_dict_cal_2d_mcda["Composite_Detection_Flags"])
    # plot_fig.plot_composite_mask_strong_weak(mask_weak_strong)
    plot_fig.plot_composite_mask_channel(data_dict_cal_2d_mcda["Composite_Detection_Flags"])
    # plot_fig.plot_best_detection_mask(best_detection_level_mask)
    if False:
        plot_fig.plot_psc_composition(data_dict_cal_2d_mcda["Homogeneous_Feature_Classification"])

        filename = "homogeneous_feature_separation_in_4_colors"
        title = "$\mathbf{Homogeneous\ Features\ (4\ colors)}$"
        plot_fig.plot_hom_4_colors(data_dict_cal_2d_mcda["Homogeneous_Feature_Mask"], title, filename)

        filename = "homogeneous_feature_mean_ab_532_per"
        title = "$\mathbf{Mean\ 532\ nm\ Perpendicular\ Attenuated\ Backscatter}\ \\langle\\beta'_{532,\\perp}\\rangle$"
        plot_fig.plot_hom_ab_signal(data_dict_cal_2d_mcda["Homogeneous_Feature_Mean_Perpendicular_Attenuated_Backscatter_532"], title, filename)

        filename = "homogeneous_feature_mean_asr_532"
        title = "$\mathbf{Mean\ Attenuated\ 532\ nm\ Scattering\ Ratio}\ \\langle R'_{532}\\rangle$"
        plot_fig.plot_hom_asr_signal(data_dict_cal_2d_mcda["Homogeneous_Feature_Mean_Attenuated_Scattering_Ratio_532"], title, filename)

    # Plot every steps
    if PLOT_ALL_STEPS:
        plot_fig.plot_steps(data_dict_cal_2d_mcda_steps["Parallel_Detection_Flags_532_steps"],
                   data_dict_cal_2d_mcda_steps["Parallel_Attenuated_Backscatter_532_steps"],
                   '532_par')
        plot_fig.plot_steps(data_dict_cal_2d_mcda_steps["Perpendicular_Detection_Flags_532_steps"],
                   data_dict_cal_2d_mcda_steps["Perpendicular_Attenuated_Backscatter_532_steps"],
                   '532_per')
        # plot_fig.plot_steps(data_dict_cal_2d_mcda_steps["Detection_Flags_1064_steps"],
        #            data_dict_cal_2d_mcda_steps["Attenuated_Backscatter_1064_steps"],
        #            '1064')
    
    
    print_time(tic_main_program)
