#!/usr/bin/env python
# coding: utf8

from datetime import datetime
import numpy as np
from pyhdf.SD import SD
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from matplotlib.colors import LogNorm, from_levels_and_colors
from matplotlib.ticker import MultipleLocator, FixedLocator, LogLocator
import seaborn as sns
import os
import cmlidar
import seaborn as sns

from my_modules.standard_outputs import print_time
from my_modules.readers.calipso_reader import CALIPSOReader, get_prof_min_max_indexes_from_lon
from my_modules.paths import split_granule_date
from my_modules.figuretools import setstyle, takecmap, cm2in, compute_bounds, lat_lon_dist_xaxis, \
    CALIOPFigureMaker, remove_edges
from my_modules.geotools import neighbors

FILL_VALUE_FLOAT = -9999.0


class FigureMaker(CALIOPFigureMaker):
    def __init__(self):
        super().__init__()
        self.fig_w = cm2in(17.7) # cm
        self.fig_h = cm2in(6) # cm
        self.axes_titlesize = 8
        self.axes_title_pad = 1.14
        self.clabelpad = 40

    def set_max_detect_level(self, max_detect_level):
        self.max_detect_level = max_detect_level
        
        
    def plot_mask(self, step, mask, channel):
        """Plot mask for each step. If only final mask, put step = None"""

        # Remove edges
        if self.edges_removal != 0: # to avoid error with "-0"
            mask = remove_edges(mask, self.edges_removal)

        # Labels
        clabels = ["No\ndetect.",] +\
                  ["%d" % i for i in np.arange(self.max_detect_level)+1] 
                #   ["No detection",] +\
                #   ["Detection level %d" % i for i in np.arange(self.max_detect_level)+1] 
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
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
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
        fontsize_clabel = 8
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
            if j == 3:
                cbar.ax.text(4, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), 'Detection level',
                             va='center', rotation=90, fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
        # cbar.set_label("Level of detection", labelpad=self.clabelpad)
       
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
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        title = "Composite\ best\ detection\ level"
        plt.title(r'$\mathbf{%s}$' % title, fontsize=self.axes_titlesize, y=self.axes_title_pad)

        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', drawedges=True)
        fontsize_clabel = 5
        cbar.ax.tick_params(axis='y', which='both', right=False, labelright=False)
        for j, lab in enumerate(clabels):
            cbar.ax.text(1.5, 1/(float(colorbins.size-1)*2) + j/float(colorbins.size-1), lab,
                         va='center', fontsize=fontsize_clabel, transform=cbar.ax.transAxes)
        # cbar.set_label("Level of detection", labelpad=self.clabelpad)
       
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
        cbar.set_label(label=r"$\langle\beta'\rangle$ (km$^{-1}$ sr$^{-1}$)")

    def plot_ab_signal(self, ab_signal, title, filename):
        # sourcery skip: merge-comparisons, merge-duplicate-blocks, remove-redundant-if
        
        # Mask where fill_value
        ab_signal = np.ma.masked_where(ab_signal == FILL_VALUE_FLOAT, ab_signal)
    
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
        # palette = ["#000000",
        #             '#000444', '#010846', '#030b48', '#050e4a', '#08104c', '#0b134f', '#0e1551', '#101753', '#131a55', '#151c57', '#171e59', '#1a215c', '#1c235e', '#1e2560', '#202862', '#222a65', '#242d67', '#262f69', '#28316b', '#2a346e', '#2c3670', '#2e3972', '#303b74', '#323e77', '#344079', '#36437b', '#37467d', '#394880', '#3b4b82', '#3d4d84', '#3f5087', '#415389', '#43558b', '#44588e', '#465b90', '#485d92', '#4a6095', '#4c6397', '#4e6599', '#4f689c', '#516b9e', '#536da1', '#5570a3', '#5773a5', '#5876a8', '#5a79aa', '#5c7bad', '#5e7eaf', '#6081b1', '#6284b4', '#6387b6', '#6589b9', '#678cbb', '#698fbe', '#6b92c0', '#6d95c3', '#6e98c5', '#709bc7', '#729eca', '#74a1cc', '#76a4cf', '#78a7d1', '#79a9d4', '#7bacd6', '#7dafd9', '#7fb2db', '#81b5de', '#83b8e0', '#85bbe3', '#86bee5', '#88c1e8', '#8ac4eb', '#8cc7ed', '#8ecbf0', '#90cef2', '#92d1f5', '#93d4f7', '#95d7fa', '#97dafc',
        #             '#99ddff', '#a8e1f7', '#b6e4ef', '#c3e8e6', '#ceecde', '#d9efd5', '#e3f3cd', '#edf7c4', '#f6fbbb', '#ffffb2',
        #             '#fffcb0', '#fff9ae', '#fef6ac', '#fef3aa', '#fdf1a8', '#fdeea6', '#fceba4', '#fce8a1', '#fbe59f', '#fbe29d', '#fae09b', '#fadd99', '#f9da97', '#f9d795', '#f8d493', '#f7d191', '#f7cf8f', '#f6cc8d', '#f6c98b', '#f5c689', '#f4c387', '#f4c085', '#f3be83', '#f2bb81', '#f1b87f', '#f1b57d', '#f0b27b', '#efaf79', '#eead77', '#eeaa76', '#eda774', '#eca472', '#eba170', '#ea9e6e', '#e99c6c', '#e9996a', '#e89668', '#e79366', '#e69064', '#e58d62', '#e48a61', '#e3875f', '#e2845d', '#e1825b', '#e07f59', '#df7c57', '#de7956', '#dd7654', '#dc7352', '#db7050', '#da6d4e', '#d96a4c', '#d8674b', '#d76449', '#d66047', '#d45d45', '#d35a44', '#d25742', '#d15440', '#d0503e', '#cf4d3d', '#ce493b', '#cc4639', '#cb4238', '#ca3e36', '#c93b34', '#c73633', '#c63231', '#c52e2f', '#c4292e', '#c2242c', '#c11e2b', '#c01629', '#be0d28',
        #             '#bd0026', '#b31e2b', '#a82c30', '#9e3635', '#933d3a', '#88433f', '#7c4844', '#704c49', '#62504e', '#535353',
        #             '#555555', '#575757', '#595959', '#5b5b5b', '#5d5d5d', '#5f5f5f', '#616161', '#636363', '#656565', '#676767', '#696969', '#6b6b6b', '#6d6d6d', '#6f6f6f', '#717171', '#737373', '#757575', '#777777', '#797979', '#7b7b7b', '#7d7d7d', '#7f7f7f', '#818181', '#848484', '#868686', '#888888', '#8a8a8a', '#8c8c8c', '#8e8e8e', '#909090', '#929292', '#949494', '#979797', '#999999', '#9b9b9b', '#9d9d9d', '#9f9f9f', '#a1a1a1', '#a4a4a4', '#a6a6a6', '#a8a8a8', '#aaaaaa', '#acacac', '#afafaf', '#b1b1b1', '#b3b3b3', '#b5b5b5', '#b8b8b8', '#bababa', '#bcbcbc', '#bebebe', '#c1c1c1', '#c3c3c3', '#c5c5c5', '#c7c7c7', '#cacaca', '#cccccc', '#cecece', '#d0d0d0', '#d3d3d3', '#d5d5d5', '#d7d7d7', '#dadada', '#dcdcdc', '#dedede', '#e0e0e0', '#e3e3e3', '#e5e5e5', '#e7e7e7', '#eaeaea', '#ececec', '#eeeeee', '#f1f1f1', '#f3f3f3', '#f6f6f6', '#f8f8f8', '#fafafa', '#fdfdfd', '#ffffff'
        #             ]
        # my_cmap = mpl.colors.ListedColormap(palette)
        # b1 = np.logspace(-5, -3+np.log10(1.5), 85)
        # b2 = np.logspace(-3+np.log10(1.5), -3+np.log10(6.5), 85)[1:]
        # b3 = np.logspace(-3+np.log10(6.5), -1, 84)[1:]
        # bounds = np.concatenate((b1, b2, b3))
        # colors = my_cmap(np.arange(len(palette)))
        # my_cmap, my_norm = from_levels_and_colors(bounds, colors, extend='both')

        ax0.set_facecolor('0.5')

        # Put negative values to 1e-9 so they don't appear transparent
        ab_signal[(ab_signal<0) & ~ab_signal.mask] = 1e-9
        # pc = plt.pcolormesh(self.pindexbins, self.altbins, ab_signal.T, cmap=my_cmap, norm=my_norm)
        # plt.clim(1e-5, 1e-1)
        # pc = plt.pcolormesh(self.pindexbins, self.altbins, ab_signal.T, cmap="backscatter_continuous", norm=cmlidar.cm.backscatter_continuous_norm)
        # self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        plt.title(title, weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        # plt.text(0.02, 0.85, "(5km×180m)", ha='left', va='center', transform=fig.transFigure)
        my_cmap = takecmap('extthermal')
        my_cmap.colorbar_extend = 'both'
        pc = plt.pcolormesh(self.pindexbins, self.altbins, ab_signal.T, cmap=my_cmap, norm=LogNorm(), rasterized=True)
        plt.clim(1e-6, 2e-3)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', extend='both', drawedges=False)
        cbar.set_label(label=r"$\beta'$ (km$^{-1}$ sr$^{-1}$)", labelpad=45)
        # cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
        # cbar.ax.tick_params(which='both', labelright=False)
        # cbar_major_label = ['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$']
        # for j, bound in enumerate(np.array((1e-5, 1e-4, 1e-3, 1e-2))):
        #     cbar.ax.text(2, bound, cbar_major_label[j], va='center', fontsize=9)
        # # cbar.ax.ticklabel_format(style="scientific", scilimits=(0, 0))
        # minor_locators = np.concatenate((np.arange(2,10)*1e-6,
        #                                  np.arange(2,10)*1e-5,
        #                                  np.arange(2,10)*1e-4,
        #                                  np.arange(2,10)*1e-3,
        #                                  np.arange(2,10)*1e-2))
        # cbar.ax.yaxis.set_minor_locator(FixedLocator(minor_locators))
        # cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', extend='both')
        # cbar.set_label(label=r"$\beta'$ (km$^{-1}$ sr$^{-1}$)", fontsize=8, labelpad=45)
        # self.backscatter_cbar_labels(cbar)

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
        mask = set_homogeneous_chunks_color(mask, color_number_array)
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
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        plt.title(title, weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        
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
        # my_cmap = cm.plasma
        # nb_colors = 256
        # palette = my_cmap(np.linspace(0, 255, nb_colors).astype(int))
        # my_cmap = mpl.colors.ListedColormap(palette[1:-1])
        # my_cmap.set_under(palette[0])
        # my_cmap.set_over(palette[-1])
        # my_cmap.colorbar_extend = 'both'
        # pc = plt.pcolormesh(self.pindexbins, self.altbins, ab_signal.T, cmap=my_cmap, norm=LogNorm())
        # plt.clim(1e-6, 1e-3)
        pc = plt.pcolormesh(self.pindexbins, self.altbins, ab_signal.T, cmap="backscatter_continuous", norm=cmlidar.cm.backscatter_continuous_norm)
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        plt.title(title, weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
        # plt.text(0.02, 0.85, "(5km×180m)", ha='left', va='center', transform=fig.transFigure)
        
        # Plot colorbar
        ax1 = plt.subplot(gs0[1])
        # cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', extend='both', drawedges=False)
        # cbar.set_label(label=r"$\langle\beta'\rangle$ (km$^{-1}$ sr$^{-1}$)")
        # cbar.ax.yaxis.set_major_locator(LogLocator(numticks=15))
        # cbar.ax.yaxis.set_minor_locator(LogLocator(numticks=15, subs=np.arange(0.2, 1, 0.1)))
        cbar = plt.colorbar(pc, cax=ax1, orientation='vertical', extend='both')
        cbar.set_label(label=r"$\langle\beta'\rangle$ (km$^{-1}$ sr$^{-1}$)", fontsize=8, labelpad=40)
        self.backscatter_cbar_labels(cbar)

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
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        plt.title(title, weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
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
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        plt.title('2D-McDA-PSC composite detection feature mask', fontweight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
    
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
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        plt.title('2D-McDA-PSC composite detection feature mask', fontweight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)
    
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
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
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
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        plt.title("$\mathbf{Attenuated\ Color\ Ratio}\ \\frac{\\beta'_{1064}}{\\beta'_{532}}$", fontsize=self.axes_titlesize, y=self.axes_title_pad5)
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
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        plt.title(title, fontsize=self.axes_titlesize, y=self.axes_title_pad)
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
        self.plot_params(ax0, YMIN, YMAX, INVERT_XAXIS, flag_granule=FLAG_GRANULE)
        plt.title(f'PSC v2 Composition (draft code)', weight='bold', fontsize=self.axes_titlesize, y=self.axes_title_pad)

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
    

    def plot_ab_signal_distribution(self, data, title, filename):

        plot_fig.adj_left = 0.08
        plot_fig.adj_bottom = 0.11
        plot_fig.adj_right = 0.95
        plot_fig.adj_top = 0.9

        bins_par = np.logspace(-6, np.log10(2e-3), 100)
        plt.figure(figsize=(10, 5))
        plt.hist(data.flatten(), bins=bins_par)
        plt.xscale("log")
        plt.xlabel(r"$\beta'_{\parallel}$ (km$^{-1}$ sr$^{-1}$)")
        plt.ylabel("Occurrence")
        plt.title(title, weight='bold')

        # Save figure
        self.save_fig(filename)


    def plot_532per_vs_sr532(self, sr532, per532, title, filename):
        
        # Mask where negatice
        sr532 = np.ma.masked_where(sr532 <=0, sr532)
        per532 = np.ma.masked_where(per532 <=0, per532)

        # Make sure the masked values are the same
        sr532 = np.ma.masked_where(per532.mask, sr532)
        per532 = np.ma.masked_where(sr532.mask, per532)

        sr532 = sr532.compressed()        
        per532 = per532.compressed()
        
        plot_fig.adj_left = 0.08
        plot_fig.adj_bottom = 0.11
        plot_fig.adj_right = 0.95
        plot_fig.adj_top = 0.9

        plt.figure(figsize=(7, 5))
        ax = plt.subplot(111)
        hb = ax.hexbin(sr532, per532, gridsize=50, xscale='log', yscale='log', 
                       extent=(np.log10(1.1), np.log10(60), np.log10(1e-6), np.log10(2.5e-3)), 
                       cmap=takecmap('thermal'), mincnt=1)
        ax.set_xlim(1.1, 60)
        ax.set_ylim(1e-6, 2.5e-3)
        custom_ticks = [1.1, 2, 5, 10, 20, 50]
        ax.set_xticks(custom_ticks)
        ax.set_xticklabels([f'{tick}' for tick in custom_ticks])
        plt.colorbar(hb, label="Occurrence")
        plt.xlabel("$R'_{532}$")
        plt.ylabel("$\\beta'_{532,\\perp}$")
        plt.title(title, weight='bold')

        # Save figure
        self.save_fig(filename)


def set_homogeneous_chunks_color(mask, color_number_array):
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
    INDATA_FOLDER = "/home/vaillant/codes/projects/2D_McDA_for_PSCs/out/data/"
    GRANULE_DATE = "2010-01-18T00-19-57ZN_lon_170.68_27.93" # "2011-06-25T00-11-52ZN_lon_5.95_-150.07"
    VERSION_2D_McDA = "V1.2.0"
    TYPE_2D_McDA = "Prototype"
    SLICE_START_END_TYPE = 'longitude' # 'profindex' (of the 2D-McDA file) or 'longitude'
    SLICE_START = 170.59 # profindex or longitude
    SLICE_END = 27.95 # profindex or longitude
    EDGES_REMOVAL = 0 # number of prof to remove on both edges of plot
    MAX_DETECT_LEVEL = 5
    PLOT_ALL_STEPS = False
    INVERT_XAXIS = False
    YMIN = 15
    YMAX = 30
    PLOT_ASPECT_RATIO = "browse" # "browse", "spec" or None
    FIGURES_PATH = "/home/vaillant/codes/projects/2D_McDA_for_PSCs/out/figures/"
    FIGURES_FILETYPE = 'png' #'png' 'svg'
    FLAG_GRANULE = False # Write granule name in the plots
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    
    
    # **********************************
    # *** Load 2D-McDA HDF data file ***
    print("\n*****Load 2D-McDA HDF data file...*****")
    
    # Get filename and filepath
    filename_2d_mcda = f"CAL_LID_L2_2D_McDA_PSCs-{TYPE_2D_McDA}-{VERSION_2D_McDA.replace('.', '-')}." \
                       f"{GRANULE_DATE}.hdf"
    granule_date_dict = split_granule_date(GRANULE_DATE)
    hdffile = os.path.join(INDATA_FOLDER, f"2D_McDA_PSCs.{VERSION_2D_McDA.replace('V', 'v')}",
                           str(granule_date_dict['year']),
                           f"{granule_date_dict['year']}_{granule_date_dict['month']:02d}_"
                           f"{granule_date_dict['day']:02d}",
                           filename_2d_mcda)

    # Open HDF file
    print(f"\tGranule path: {hdffile}")
    cal_2d_mcda = CALIPSOReader(hdffile)

    # Get prof_min and prof_max from longitudes
    lat = cal_2d_mcda.get_data("Latitude")
    print(lat.size)
    lon = cal_2d_mcda.get_data("Longitude")
    if SLICE_START_END_TYPE == 'longitude':
        prof_min, prof_max = get_prof_min_max_indexes_from_lon(lon, SLICE_START, SLICE_END)
    else:
        prof_min = SLICE_START
        prof_max = SLICE_END
        
    # Print lat/lon of min and max prof indices
    print(f"\tFrom min profile index {prof_min:d} "
          f"(lat = {lat[prof_min]:.2f} / lon = {lon[prof_min]:.2f}) "
          f"to max profile index {prof_max:d} "
          f"(lat = {lat[prof_max]:.2f} / lon = {lon[prof_max]:.2f})")
    
    # Load 2D-McDA parameters
    data_dict_cal_2d_mcda = {}
    cal_2d_mcda_keys = [
        "Latitude",
        "Longitude",
        "Profile_ID",
        "Profile_Time",
        "Profile_UTC_Time",
        "Altitude",
        "Parallel_Detection_Flags_532",
        "Perpendicular_Detection_Flags_532",
        "Detection_Flags_1064",
        "Composite_Detection_Flags",
        "Homogeneous_Chunks_Mask",
        "Homogeneous_Chunks_Classification",
        "Homogeneous_Chunks_Mean_Parallel_Attenuated_Backscatter_532",
        "Homogeneous_Chunks_Mean_Perpendicular_Attenuated_Backscatter_532",
        "Homogeneous_Chunks_Mean_Attenuated_Backscatter_1064",
        "Homogeneous_Chunks_Mean_Attenuated_Scattering_Ratio_532"
    ] + ["Parallel_Attenuated_Backscatter_532",
         "Perpendicular_Attenuated_Backscatter_532",
         "Attenuated_Backscatter_1064",
         "Molecular_Parallel_Attenuated_Backscatter_532",
         "Molecular_Perpendicular_Attenuated_Backscatter_532"]
    for key in cal_2d_mcda_keys:
        data_dict_cal_2d_mcda[key] = cal_2d_mcda.get_data(key, SLICE_START, SLICE_END, SLICE_START_END_TYPE)

    data_dict_cal_2d_mcda_steps = {}
    if PLOT_ALL_STEPS:
        cal_2d_mcda_steps_keys = [
            "Parallel_Detection_Flags_532_steps",
            "Perpendicular_Detection_Flags_532_steps",
            "Detection_Flags_1064_steps",
            "Parallel_Attenuated_Backscatter_532_steps",
            "Perpendicular_Attenuated_Backscatter_532_steps",
            "Attenuated_Backscatter_1064_steps",
            "Parallel_Attenuated_Backscatter_532",
            "Perpendicular_Attenuated_Backscatter_532",
            "Attenuated_Backscatter_1064",
            "Molecular_Parallel_Attenuated_Backscatter_532",
            "Molecular_Perpendicular_Attenuated_Backscatter_532",
            "Molecular_Attenuated_Backscatter_1064",
            "Background_Noise_532_Parallel",
            "Background_Noise_532_Perpendicular",
            "Background_Noise_1064",
            "Noise_Scale_Factor_532_Parallel_AB_domain",
            "Noise_Scale_Factor_532_Perpendicular_AB_domain",
            "Noise_Scale_Factor_1064_AB_domain",
            "Shot_Noise_532_Parallel",
            "Shot_Noise_532_Perpendicular",
            "Shot_Noise_1064",
            "Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Parallel",
            "Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Perpendicular",
            "Attenuated_Backscatter_Uncertainty_Standard_Deviation_1064"
        ]
        for key in cal_2d_mcda_steps_keys:
            data_dict_cal_2d_mcda_steps[key] = cal_2d_mcda.get_data(key, SLICE_START, SLICE_END, SLICE_START_END_TYPE)


    ###########################
    # Weak/Strong feature mask
    mask_weak_strong = np.bitwise_and(data_dict_cal_2d_mcda["Composite_Detection_Flags"], 7).astype(float)
    # 2.5 = 'Strong' were detection without averaging at least in one channel,
    # keep 2 = 'Weak' elsewhere
    where_strong = ((data_dict_cal_2d_mcda["Parallel_Detection_Flags_532"] >= 1) &
                    (data_dict_cal_2d_mcda["Parallel_Detection_Flags_532"] <= 4)) |\
                   ((data_dict_cal_2d_mcda["Perpendicular_Detection_Flags_532"] >= 1) &
                    (data_dict_cal_2d_mcda["Perpendicular_Detection_Flags_532"] <= 4)) |\
                   ((data_dict_cal_2d_mcda["Detection_Flags_1064"] >= 1) &
                    (data_dict_cal_2d_mcda["Detection_Flags_1064"] <= 4))
    mask_weak_strong[where_strong] = 2.5
    

    ###########################
    # Best detection level mask
    detection_level_masks = []
    for i in np.arange(5) + 1:
        detection_level_masks.append((data_dict_cal_2d_mcda["Parallel_Detection_Flags_532"] == i) + (data_dict_cal_2d_mcda["Perpendicular_Detection_Flags_532"] == i) + (data_dict_cal_2d_mcda["Detection_Flags_1064"] == i))

    # Initialization
    best_detection_level_mask = np.ma.zeros(data_dict_cal_2d_mcda["Parallel_Detection_Flags_532"].shape)

    for i in np.arange(5, 0, -1):
        best_detection_level_mask[detection_level_masks[i-1]] = i

    
    # ************
    # *** Plot ***
    print("\n\n*****Plot...*****")
    
    # Initialize instance of FigureMaker
    plot_fig = FigureMaker()
    if PLOT_ASPECT_RATIO == 'browse':
        plot_fig.fig_w = cm2in(16) # cm
        plot_fig.fig_h = cm2in(8) # cm
        plot_fig.adj_left = 0.08
        plot_fig.adj_bottom = 0.11
        plot_fig.adj_right = 0.87
        plot_fig.adj_top = 0.81
    elif PLOT_ASPECT_RATIO == 'spec':
        plot_fig.fig_w = cm2in(12) # cm
        plot_fig.fig_h = cm2in(8) # cm
        plot_fig.adj_left = 0.1
        plot_fig.adj_bottom = 0.11
        plot_fig.adj_right = 0.83
        plot_fig.adj_top = 0.81
        plot_fig.axes_title_pad = 1.15
        plot_fig.clabelpad = 40
    plot_fig.filetype = FIGURES_FILETYPE
    plot_fig.set_and_create_fig_folder(FIGURES_PATH, GRANULE_DATE, lon[prof_min], lon[prof_max])
    plot_fig.set_head_filename(GRANULE_DATE, lon[prof_min], lon[prof_max])
    plot_fig.set_edges_removal(EDGES_REMOVAL)
    plot_fig.set_max_detect_level(MAX_DETECT_LEVEL)
    plot_fig.set_coordinates(data_dict_cal_2d_mcda["Latitude"], data_dict_cal_2d_mcda["Longitude"],
                             data_dict_cal_2d_mcda["Altitude"])
    
    # Plot signals
    if False:
        filename = "ab_532_par"
        title = "$\mathbf{532\ nm\ Parallel\ Attenuated\ Backscatter}\ \\beta'_{532,\\parallel}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Parallel_Attenuated_Backscatter_532"], title, filename)

        filename = "ab_532_per"
        title = "$\mathbf{532\ nm\ Perpendicular\ Attenuated\ Backscatter}\ \\beta'_{532,\\perp}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Perpendicular_Attenuated_Backscatter_532"], title, filename)

        filename = "ab_1064"
        title = "$\mathbf{1064\ nm\ Attenuated\ Backscatter}\ \\beta'_{1064}$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Attenuated_Backscatter_1064"], title, filename)

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


    # Plot the 3 channel masks
    if False:
        plot_fig.plot_mask(None, data_dict_cal_2d_mcda["Parallel_Detection_Flags_532"], '532_par')
        plot_fig.plot_mask(None, data_dict_cal_2d_mcda["Perpendicular_Detection_Flags_532"], '532_per')
        plot_fig.plot_mask(None, data_dict_cal_2d_mcda["Detection_Flags_1064"], '1064')

    # Plot the composite masks
    if False:
        plot_fig.plot_composite_mask(data_dict_cal_2d_mcda["Composite_Detection_Flags"])
        plot_fig.plot_composite_mask_strong_weak(mask_weak_strong)
        plot_fig.plot_composite_mask_channel(data_dict_cal_2d_mcda["Composite_Detection_Flags"])
        plot_fig.plot_best_detection_mask(best_detection_level_mask)
    if False:
        plot_fig.plot_psc_composition(data_dict_cal_2d_mcda["Homogeneous_Chunks_Classification"])

        filename = "homogeneous_chunks_separation_in_4_colors"
        title = "$\mathbf{Homogeneous\ Chunks\ Boundaries}$"
        plot_fig.plot_hom_4_colors(data_dict_cal_2d_mcda["Homogeneous_Chunks_Mask"], title, filename)

        filename = "homogeneous_chunks_mean_ab_532_par"
        title = "$\mathbf{Homogeneous\ Chunks\ Mean\ 532\ nm\ Parallel\ Attenuated\ Backscatter}\ \\langle\\beta'_{532,\\parallel}\\rangle$"
        plot_fig.plot_hom_ab_signal(data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Parallel_Attenuated_Backscatter_532"], title, filename)

        filename = "homogeneous_chunks_mean_ab_532_per"
        title = "$\mathbf{Homogeneous\ Chunks\ Mean\ 532\ nm\ Perpendicular\ Attenuated\ Backscatter}\ \\langle\\beta'_{532,\\perp}\\rangle$"
        plot_fig.plot_hom_ab_signal(data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Perpendicular_Attenuated_Backscatter_532"], title, filename)

        filename = "homogeneous_chunks_mean_ab_1064"
        title = "$\mathbf{Homogeneous\ Chunks\ Mean\ 1064\ nm\ Attenuated\ Backscatter}\ \\langle\\beta'_{1064}\\rangle$"
        plot_fig.plot_hom_ab_signal(data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Attenuated_Backscatter_1064"], title, filename)

        filename = "homogeneous_chunks_mean_asr_532"
        title = "$\mathbf{Mean\ Attenuated\ 532\ nm\ Scattering\ Ratio}\ \\langle R'_{532}\\rangle$"
        plot_fig.plot_hom_asr_signal(data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Attenuated_Scattering_Ratio_532"], title, filename)

    # Plot signal distribution
    if True:
        alt_idx_above_15km = np.where(data_dict_cal_2d_mcda["Altitude"] >= 15)[0]

        # 532par
        hom_chunks_mean_532par_signal = data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Parallel_Attenuated_Backscatter_532"][:, alt_idx_above_15km]
        filename = "homogeneous_chunks_mean_ab_532_par_distribution"
        title = "$\mathbf{Homogeneous\ Chunks\ Mean\ 532\ nm\ Parallel\ Attenuated\ Backscatter}\ \\langle\\beta'_{532,\\parallel}\\rangle$"
        plot_fig.plot_ab_signal_distribution(hom_chunks_mean_532par_signal, title, filename)

        feature_5km_180m_532par_signal = np.ma.masked_where(data_dict_cal_2d_mcda["Composite_Detection_Flags"] == 1, data_dict_cal_2d_mcda["Parallel_Attenuated_Backscatter_532"])
        feature_5km_180m_532par_signal = feature_5km_180m_532par_signal[:, alt_idx_above_15km]
        filename = "feature_5km_180m_ab_532_par_distribution"
        title = "$\mathbf{Feature\ 5km×180m\ 532\ nm\ Parallel\ Attenuated\ Backscatter}\ \\beta'_{532,\\parallel}$"
        plot_fig.plot_ab_signal_distribution(feature_5km_180m_532par_signal.compressed(), title, filename)

        # 532per
        hom_chunks_mean_532per_signal = data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Perpendicular_Attenuated_Backscatter_532"][:, alt_idx_above_15km]
        filename = "homogeneous_chunks_mean_ab_532_per_distribution"
        title = "$\mathbf{Homogeneous\ Chunks\ Mean\ 532\ nm\ Perpendicular\ Attenuated\ Backscatter}\ \\langle\\beta'_{532,\\perp}\\rangle$"
        plot_fig.plot_ab_signal_distribution(hom_chunks_mean_532per_signal, title, filename)

        feature_5km_180m_532per_signal = np.ma.masked_where(data_dict_cal_2d_mcda["Composite_Detection_Flags"] == 1, data_dict_cal_2d_mcda["Perpendicular_Attenuated_Backscatter_532"])
        feature_5km_180m_532per_signal = feature_5km_180m_532per_signal[:, alt_idx_above_15km]
        filename = "feature_5km_180m_ab_532_per_distribution"
        title = "$\mathbf{Feature\ 5km×180m\ 532\ nm\ Perpendicular\ Attenuated\ Backscatter}\ \\beta'_{532,\\perp}$"
        plot_fig.plot_ab_signal_distribution(feature_5km_180m_532per_signal.compressed(), title, filename)

        # 1064
        hom_chunks_mean_1064_signal = data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Attenuated_Backscatter_1064"][:, alt_idx_above_15km]
        filename = "homogeneous_chunks_mean_ab_1064_distribution"
        title = "$\mathbf{Homogeneous\ Chunks\ Mean\ 1064\ nm\ Attenuated\ Backscatter}\ \\langle\\beta'_{1064}\\rangle$"
        plot_fig.plot_ab_signal_distribution(hom_chunks_mean_1064_signal, title, filename)

        feature_5km_180m_1064_signal = np.ma.masked_where(data_dict_cal_2d_mcda["Composite_Detection_Flags"] == 1, data_dict_cal_2d_mcda["Attenuated_Backscatter_1064"])
        feature_5km_180m_1064_signal = feature_5km_180m_1064_signal[:, alt_idx_above_15km]
        filename = "feature_5km_180m_ab_1064_distribution"
        title = "$\mathbf{Feature\ 5km×180m\ 1064\ nm\ Attenuated\ Backscatter}\ \\beta'_{1064}$"
        plot_fig.plot_ab_signal_distribution(feature_5km_180m_1064_signal.compressed(), title, filename)

        # SR532 vs 532per
        sr532 = (data_dict_cal_2d_mcda["Parallel_Attenuated_Backscatter_532"] + data_dict_cal_2d_mcda["Perpendicular_Attenuated_Backscatter_532"])/(data_dict_cal_2d_mcda["Molecular_Parallel_Attenuated_Backscatter_532"] + data_dict_cal_2d_mcda["Molecular_Perpendicular_Attenuated_Backscatter_532"])
        feature_5km_180m_sr532_signal = np.ma.masked_where(data_dict_cal_2d_mcda["Composite_Detection_Flags"] == 1, sr532)
        feature_5km_180m_sr532_signal = feature_5km_180m_sr532_signal[:, alt_idx_above_15km]
        filename = "feature_5km_180m_sr532_vs_532per_distribution"
        title = "$\mathbf{Feature\ 5km×180m}\ \\beta'_{532,\\perp}\ \mathbf{vs}\ R'_{532}$"
        plot_fig.plot_532per_vs_sr532(feature_5km_180m_sr532_signal, feature_5km_180m_532per_signal, title, filename)

    # Plot signal in detection mask
    if False:
        # 532par
        filename = "homogeneous_chunks_mean_ab_532_par_bis"
        title = "$\mathbf{Homogeneous\ Chunks\ Mean\ 532\ nm\ Parallel\ Attenuated\ Backscatter}\ \\langle\\beta'_{532,\\parallel}\\rangle$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Parallel_Attenuated_Backscatter_532"], title, filename)

        feature_5km_180m_par_signal = np.ma.masked_where(data_dict_cal_2d_mcda["Composite_Detection_Flags"] == 1, data_dict_cal_2d_mcda["Parallel_Attenuated_Backscatter_532"])
        filename = "feature_5km_180m_ab_532_par"
        title = "$\mathbf{Feature\ 5km×180m\ 532\ nm\ Parallel\ Attenuated\ Backscatter}\ \\beta'_{532,\\parallel}$"
        plot_fig.plot_ab_signal(feature_5km_180m_par_signal, title, filename)

        # 532per
        filename = "homogeneous_chunks_mean_ab_532_per_bis"
        title = "$\mathbf{Homogeneous\ Chunks\ Mean\ 532\ nm\ Perpendicular\ Attenuated\ Backscatter}\ \\langle\\beta'_{532,\\perp}\\rangle$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Perpendicular_Attenuated_Backscatter_532"], title, filename)

        feature_5km_180m_per_signal = np.ma.masked_where(data_dict_cal_2d_mcda["Composite_Detection_Flags"] == 1, data_dict_cal_2d_mcda["Perpendicular_Attenuated_Backscatter_532"])
        filename = "feature_5km_180m_ab_532_per"
        title = "$\mathbf{Feature\ 5km×180m\ 532\ nm\ Perpendicular\ Attenuated\ Backscatter}\ \\beta'_{532,\\perp}$"
        plot_fig.plot_ab_signal(feature_5km_180m_per_signal, title, filename)

        # 1064
        filename = "homogeneous_chunks_mean_ab_1064_bis"
        title = "$\mathbf{Homogeneous\ Chunks\ Mean\ 1064\ nm\ Attenuated\ Backscatter}\ \\langle\\beta'_{1064}\\rangle$"
        plot_fig.plot_ab_signal(data_dict_cal_2d_mcda["Homogeneous_Chunks_Mean_Attenuated_Backscatter_1064"], title, filename)

        feature_5km_180m_per_signal = np.ma.masked_where(data_dict_cal_2d_mcda["Composite_Detection_Flags"] == 1, data_dict_cal_2d_mcda["Attenuated_Backscatter_1064"])
        filename = "feature_5km_180m_ab_1064"
        title = "$\mathbf{Feature\ 5km×180m\ 1064\ nm\ Attenuated\ Backscatter}\ \\beta'_{1064}$"
        plot_fig.plot_ab_signal(feature_5km_180m_per_signal, title, filename)

    # Plot every steps
    if PLOT_ALL_STEPS:
        plot_fig.plot_steps(data_dict_cal_2d_mcda_steps["Parallel_Detection_Flags_532_steps"],
                   data_dict_cal_2d_mcda_steps["Parallel_Attenuated_Backscatter_532_steps"],
                   '532_par')
        plot_fig.plot_steps(data_dict_cal_2d_mcda_steps["Perpendicular_Detection_Flags_532_steps"],
                   data_dict_cal_2d_mcda_steps["Perpendicular_Attenuated_Backscatter_532_steps"],
                   '532_per')
        plot_fig.plot_steps(data_dict_cal_2d_mcda_steps["Detection_Flags_1064_steps"],
                   data_dict_cal_2d_mcda_steps["Attenuated_Backscatter_1064_steps"],
                   '1064')
    
    
    print_time(tic_main_program)
