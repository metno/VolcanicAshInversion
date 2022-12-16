#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import datetime
import logging
import pprint
import time
from netCDF4 import Dataset, num2date
import os
import re
import json


from AshInv import Misc
from AshInv import Plot
from cartopy import crs as ccrs

from IPython.display import clear_output
from IPython.display import Video
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import rc
from matplotlib import colors
from matplotlib.gridspec import GridSpec

from copy import copy

plt.rcParams["animation.html"] = "html5" #mp4
plt.rcParams["figure.dpi"] = 100.0 #highres movies/plots

plt.rcParams["animation.writer"] = 'ffmpeg' 
plt.rcParams["animation.codec"] = 'h264' 

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],  'size' : 18})
# for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
#rc('text', fontsize=14)


if __name__ == "__main__":
    import configargparse
    parser = configargparse.ArgParser(description='Generate video from synthetic satellite images (generate_satellite).')
    parser.add("--verbose", action="store_true", help="Enable verbose mode")
    
    parser.add('--input_npz', type=str,
                        help='NPZ-file to generate movie from', required=True)
    parser.add('--output_mp4', type=str,
                        help='Output mp4 file to write to', required=True)

    args = parser.parse_args()

    print("Arguments: ")
    for var, val in vars(args).items():
        print("{:s} = {:s}".format(var, str(val)))


    class Timer(object):
        def __init__(self, tag):
            self.tag = tag

        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *args):
            self.end = time.time()
            self.secs = self.end - self.start
            self.msecs = self.secs * 1000 # millisecs
            print("{:s}: {:f} ms".format(self.tag, self.msecs))


    #Read data from npz and json
    print("Reading data")
    with Timer('Reading data') as t:
        existing_data = np.load(args.input_npz, allow_pickle=True)
        output_data = existing_data['output_data']
        skip_iterations = existing_data['i']
        a_priori_2d = existing_data['a_priori_2d']
        json_filename = str(existing_data['json_filename'])
        varname = str(existing_data['varname'])
    json_data = Plot.readJson(json_filename)

    
    #Generate output times
    start_time = json_data['emission_times'].min()
    end_time = json_data['emission_times'].max() + np.timedelta64(6,'D')
    hours = (end_time-start_time) / np.timedelta64(1,'h')
    output_times = np.arange(start_time, end_time, np.timedelta64(1,'h'))
    print(len(output_times))

    
    #Load latitude and longitude variables
    #with Dataset(sim_file) as nc_file:
    #    lat=nc_file['lat'][:]
    #    lon=nc_file['lon'][:]
    #bbox = [min(lon), max(lon), min(lat), max(lat)]
    bbox = [-30, 45, 30, 76]
    print(bbox)

    
    #Helper function to animate images
    def plot_solution(data, bbox, im, ax):

        #input data is given in ug/m2, convert to g/m2 for plotting, 1.0e-6
        data = np.ma.masked_where(data <= 0, data*1.0e-6)

        if (im is None):
            #cm = copy(plt.cm.hot_r)
            #cm = copy(plt.cm.gist_ncar_r)
            #cm = Plot.getColorMap("stohl")
            cm = Plot.getColorMap("ippc")
            cm.set_bad(alpha = 0.0)
            norm = colors.BoundaryNorm([0, 0.2, 2.0, 4.0], cm.N, extend='max')
            #im = plt.imshow(data, cmap=cm, vmin=1.0e-5, vmax=10, extent=bbox, origin='lower')
            im = plt.imshow(data, norm=norm, cmap=cm, vmin=0, vmax=4, extent=bbox, origin='lower')
            #im = plt.imshow(data.filled(), norm=colors.LogNorm(vmin=1.0e-6, vmax=20), vmin=1.0e-6, vmax=20, cmap=cm, extent=bbox, origin='lower')

            plt.plot(-19.608322, 63.631413, 'r*')
        else:
            im.set_data(data) 

        return im


    #Create plot figure
    fig=plt.figure(figsize=(7,5), dpi=200, constrained_layout=True)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[5, 1])

    ax = [[], [], []]

    ax[0] = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax[0].set_extent(bbox, crs=ccrs.PlateCarree())
    ax[0].coastlines()
    ax[0].set_aspect(1.8)
    gridlines = ax[0].gridlines(draw_labels=True)

    im = plot_solution(output_data[0,:,:], bbox, None, ax[0])

    #cbar = fig.colorbar(im, ax=ax[0], orientation='vertical', shrink=0.6)
    cbar = fig.colorbar(im, ax=ax[0], orientation='vertical')
    #cbar.set_label("$g / m^2$", rotation=270, labelpad=10)
    cbar.set_label("$g / m^2$", rotation=270)


    #Plot bar plot below
    ax[1] = fig.add_subplot(gs[1, 0])
    #ax[1] = plt.subplot(4, 2, 4)
    emis = (10**6)*a_priori_2d.sum(axis=0)/(3*3600)
    bars = ax[1].bar(json_data['emission_times'], emis, width=0.2, alpha=0.8)
    ax[1].set_ylabel('$1000~kg/s$')
    ax[1].set_xlim(json_data['emission_times'][0], json_data['emission_times'][-1])
    ax[1].scatter(json_data['emission_times'], emis, c=emis, cmap='jet', vmin=0, vmax=40, s=0.2)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(7))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
    patches = bars.patches
    for i in range(len(emis)):
        patches[i].set_facecolor(plt.cm.jet(emis[i]/40))
    today = plt.plot(json_data['emission_times'][0], 0, 'ro', markeredgecolor='k')

    def animate(idx):
        date = output_times[idx]

        print("{:d}/{:d}".format(idx, output_data.shape[0], end=''))
        plot_solution(output_data[idx,:,:], bbox, im, ax[0])
        today[0].set_data(date, 0)

        ax[0].set_title("{:s}".format(np.datetime_as_string(date, unit='s')))

        clear_output(wait = True)
        #print('.', end='')

    #anim = animation.FuncAnimation(fig, animate, range(10), interval=100)
    anim = animation.FuncAnimation(fig, animate, range(output_data.shape[0]), interval=100)

    anim.save(args.output_mp4)
    
    
    print("Done")